#!/usr/bin/env python
"""A20: replay of the registered E' router policy through the runtime loop GwenLaya.answer (docs/ANALYSIS_PLAN_E_TPU.md).

Stored generations, stored frozen-router probabilities, stored frozen-calibrator raw scores and stored gate verdicts
are fed to ``gwaya.gwenlaya.GwenLaya`` (route_policy="start_at", tau_route = 0.5) through injected replay components;
the per-task outcome is compared with ``scripts.analyze_laya.run_policy`` on the same stored arrays. No model runs.
The registered analysis is imported, never changed. Cost is the study's accelerator-seconds rule (answer seconds plus,
for math, program-of-thought seconds), delivered to the runtime through an injected clock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from gwaya.domains.task import STATUSES, CheckResult, Task  # noqa: E402
from gwaya.gwenlaya import ANSWERED, GwenLaya, Tier  # noqa: E402
from gwaya.laya_runtime import ISOTONIC_FILE, LayaRuntime  # noqa: E402
from scripts import analyze_cascades as AC  # noqa: E402
from scripts import analyze_laya as AL  # noqa: E402

TIERS = ("qwen3.5-2b-bf16", "qwen3.5-4b-bf16", "qwen3.5-9b-bf16")
WARMUP_CHUNK = 192
COST_TOL = 1e-5  # GwenLaya rounds cost_s to 6 decimals


# --------------------------------------------------------------------------- replay components
@dataclass
class StoredTier:
    """Everything stored for one (task, tier): the greedy answer, its signals, gate verdict, cost, raw Laya score."""
    text: str
    gate: str                    # VERIFIED / FAILED / UNVERIFIED
    gen_s: float                 # stored answer accelerator-seconds
    pot_s: float = 0.0           # stored program-of-thought seconds (math only)
    mean_logprob: float | None = None
    min_logprob: float | None = None
    raw: float | None = None     # stored raw calibrator sigmoid (None: no stored score)
    correct: bool = False        # stored hidden-test correctness of the candidate
    cal_row: dict[str, Any] | None = None   # the stored calibrator row the raw score belongs to (input integrity)


@dataclass
class StoredTask:
    domain: str
    task_id: str
    prompt: str
    router: dict[str, float]
    tiers: dict[str, StoredTier]


class ReplayClock:
    """Injected clock: ``advance`` is called by the replay tiers with the stored seconds of the call."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


class ReplayWorld:
    """Holds the stored tasks; ``current`` is set by the driver before each ``GwenLaya.answer``. The components below
    resolve the task from it (the Tier/Router/Scorer callables are not given the task id) and verify the prompt."""

    def __init__(self, tasks: dict[tuple[str, str], StoredTask], tier_names: tuple[str, ...] = TIERS,
                 clock: ReplayClock | None = None, check_features: bool = True) -> None:
        self.tasks, self.tier_names, self.clock = tasks, tuple(tier_names), clock or ReplayClock()
        self.current: tuple[str, str] | None = None
        self.check_features = check_features
        self.feature_mismatches: list[dict[str, Any]] = []
        self.calls = Counter()
        self._last_tier: str | None = None

    # -- plumbing
    def _task(self, prompt: str | None = None) -> StoredTask:
        if self.current is None:
            raise RuntimeError("ReplayWorld.current not set")
        t = self.tasks[self.current]
        if prompt is not None and prompt != t.prompt:
            raise RuntimeError(f"prompt does not match stored task {self.current}")
        return t

    def _mismatch(self, t: StoredTask, tier: str, what: str, got: Any, want: Any) -> None:
        self.feature_mismatches.append({"domain": t.domain, "task_id": t.task_id, "tier": tier, "field": what,
                                        "runtime": str(got)[:80], "stored": str(want)[:80]})

    # -- the runtime-facing callables
    def tier(self, name: str) -> Tier:
        def with_signals(prompt: str, domain: str) -> tuple[str, dict[str, Any]]:
            t = self._task(prompt)
            st = t.tiers[name]
            self._last_tier = name
            self.clock.advance(st.gen_s + st.pot_s)   # study cost rule: answer seconds + program seconds
            self.calls["generate"] += 1
            sig: dict[str, Any] = {}
            if st.mean_logprob is not None:
                sig["mean_logprob"] = st.mean_logprob
            if st.min_logprob is not None:
                sig["min_logprob"] = st.min_logprob
            return st.text, sig

        def plain(prompt: str, domain: str) -> str:
            return with_signals(prompt, domain)[0]
        return Tier(name, plain, with_signals)

    def tier_list(self) -> list[Tier]:
        return [self.tier(n) for n in self.tier_names]

    def router(self, prompt: str, domain: str) -> dict[str, float]:
        self.calls["router"] += 1
        return dict(self._task(prompt).router)

    def scorer(self, features: dict[str, Any]) -> float | None:
        """Stored raw sigmoid (the runtime applies the isotonic map itself). Also checks the features the runtime built."""
        t = self._task(features.get("prompt"))
        tier = str(features["tier"])
        st = t.tiers[tier]
        self.calls["scorer"] += 1
        if self.check_features and st.cal_row is not None:
            row = st.cal_row
            sig = row.get("signals") or {}
            gen = features.get("gen_signals") or {}
            if str(features.get("gate_status", "")).lower() != str(sig.get("gate")).lower():
                self._mismatch(t, tier, "gate_status", features.get("gate_status"), sig.get("gate"))
            if features.get("response") != row.get("candidate"):
                self._mismatch(t, tier, "response", features.get("response"), row.get("candidate"))
            if features.get("prompt") != row.get("prompt"):
                self._mismatch(t, tier, "prompt", features.get("prompt"), row.get("prompt"))
            for k in ("mean_logprob", "min_logprob"):
                a, b = gen.get(k), sig.get(k)
                if (a is None) != (b is None) or (a is not None and abs(float(a) - float(b)) > 0):
                    self._mismatch(t, tier, k, a, b)
        elif self.check_features:
            self._mismatch(t, tier, "cal_row", "scorer called", "no stored calibrator row")
        return st.raw

    def checker(self, task: Task, response: str) -> CheckResult:
        """Stored gate verdict of (task, tier). The tier is identified by the response text it produced."""
        t = self._task(task.prompt)
        self.calls["checker"] += 1
        tiers = [n for n in self.tier_names if t.tiers[n].text == response]
        if not tiers:
            raise RuntimeError(f"response matches no stored tier answer for {self.current}")
        # two tiers can produce identical text: the verdict is the one stored for the tier that generated last (the
        # runtime asks for a verdict right after that tier generated)
        name = self._last_tier if self._last_tier in tiers else tiers[0]
        g = t.tiers[name].gate
        return CheckResult(g if g in STATUSES else "UNVERIFIED", {"replayed_gate": g})


def isotonic_artifacts(domains: tuple[str, ...]) -> dict[str, Any]:
    """CalibrationArtifact per domain exactly as the runtime builds them (registered map; tau_hi/tau_lo = +/-inf)."""
    iso = json.loads(Path(ISOTONIC_FILE).read_text())
    return LayaRuntime(None, None, iso, domains=domains).calibration_artifacts()


def build_system(world: ReplayWorld, tier_names: tuple[str, ...], domains: tuple[str, ...], *, routed: bool,
                 tau: float = 0.5) -> GwenLaya:
    tiers = [world.tier(n) for n in tier_names]
    return GwenLaya(tiers, router=world.router if routed else None, scorer=world.scorer,
                    calibrations=isotonic_artifacts(domains), tau_route=tau if routed else None,
                    checker=world.checker, clock=world.clock, min_visible_tests={}, route_policy="start_at")


def run_runtime(world: ReplayWorld, system: GwenLaya, keys: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """One GwenLaya.answer per task; -> per-task summary from the returned evidence trajectory."""
    out = []
    for k in keys:
        world.current = k
        world.clock.t = 0.0
        t = world.tasks[k]
        res = system.answer(t.prompt, t.domain, checker_payload={}, task_id=t.task_id)
        traj = res["evidence"]["trajectory"]
        routing = res["evidence"].get("routing")
        out.append({
            "domain": t.domain, "task_id": t.task_id, "verdict": res["verdict"], "answered": bool(res["answered"]),
            "answer_returned": res["answer"] is not None, "tier": res["tier"], "cost_s": res["cost_s"],
            "p_correct": res["p_correct"],
            "routing": routing if isinstance(routing, str) else (routing or {}).get("policy"),
            "start_tier": traj[0]["tier"] if traj else None, "final_tier": traj[-1]["tier"] if traj else None,
            "path": [(s["tier"], s.get("gate", {}).get("status")) for s in traj],
            "route_p": (routing or {}).get("p_tier") if isinstance(routing, dict) else None})
    return out


# --------------------------------------------------------------------------- registered side
def registered_outcome(tau: float, arrays: list[dict[str, np.ndarray]], router_p: np.ndarray) -> dict[str, Any]:
    """analyze_laya.run_policy on the arrays, plus the consulted-tier set decoded from run_policy itself.

    run_policy returns only cor / answered / cost. To obtain its start and final tier without re-implementing it, it is
    called a second time on copies whose tier t costs 10**t (no program seconds): the returned cost is then
    sum(10**t) over exactly the tiers it consulted."""
    real = AL.run_policy(tau, arrays, router_p)
    probe = [{**a, "cost": np.full(len(a["cost"]), 10.0 ** t), "pot": np.zeros(len(a["cost"]))} for t, a in enumerate(arrays)]
    pc = AL.run_policy(tau, probe, router_p)["cost"]
    T = len(arrays)
    consulted = [[t for t in range(T) if int(round(c)) // (10 ** t) % 10 == 1] for c in pc]
    return {"cor": real["cor"], "answered": real["answered"], "cost": real["cost"], "consulted": consulted}


# --------------------------------------------------------------------------- loading
def _read(p: Path) -> list[dict[str, Any]]:
    return AL.read_jsonl(p)


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def load_stored(tasks_path: Path, tier_dirs: dict[str, Path], router_preds: Path, cal_preds: Path, cal_rows: Path,
                tiers: tuple[str, ...] = TIERS):
    """-> (stored tasks keyed (domain, task_id) in registered order, registered arrays per tier, router_p [n, T], keys)."""
    tasks = sorted(_read(tasks_path), key=lambda t: (t["domain"], t["task_id"]))
    keys = [(t["domain"], t["task_id"]) for t in tasks]
    rows = [r for m in tiers for r in _read(tier_dirs[m] / "rows.jsonl")]
    gens = [g for m in tiers for g in _read(tier_dirs[m] / "gens.jsonl")]
    arrays = [AC.tier_arrays(m, keys, rows, gens) for m in tiers]
    rp = {(p["key"][0], p["key"][1]): p["probs"] for p in _read(router_preds)}
    raw = {(p["key"][0], p["key"][1], p["tier"]): p["probs"][0] for p in _read(cal_preds)}
    crow = {(r["source"], r["id"], r["tier"]): r for r in _read(cal_rows)}
    ans = {(g["model"], g["domain"], g["task_id"]): g for g in gens if g.get("temperature") == 0.0 and g.get("kind") != "pot"}
    pot = {(g["model"], g["domain"], g["task_id"]): float(g.get("gpu_s") or 0.0) for g in gens if g.get("kind") == "pot"}
    gate_row: dict[tuple[str, str, str], str | None] = {}
    for r in rows:
        if r["arm"] == "gate_only":
            gate_row[(r["model"], r["domain"], r["task_id"])] = r.get("gate")  # later rows win, as tier_arrays
    base_row: dict[tuple[str, str, str], bool] = {}
    for r in rows:
        if r["arm"] == "base":
            base_row[(r["model"], r["domain"], r["task_id"])] = r["score"] == "VERIFIED"
    stored: dict[tuple[str, str], StoredTask] = {}
    router_p = np.zeros((len(tasks), len(tiers)))
    for i, t in enumerate(tasks):
        src, _, ident = t["task_id"].partition("/")
        probs = rp[(src, ident)]
        router_p[i] = probs
        tiers_d = {}
        for m in tiers:
            g = ans[(m, t["domain"], t["task_id"])]
            cr = crow.get((src, ident, m))
            sig = (cr or {}).get("signals") or {}
            tiers_d[m] = StoredTier(
                text=g["text"], gate=str(gate_row[(m, t["domain"], t["task_id"])] or "UNVERIFIED"),
                gen_s=float(g.get("gpu_s") or 0.0),
                pot_s=pot.get((m, t["domain"], t["task_id"]), 0.0) if t["domain"] == "math" else 0.0,
                mean_logprob=sig.get("mean_logprob"), min_logprob=sig.get("min_logprob"),
                raw=raw.get((src, ident, m)), correct=base_row[(m, t["domain"], t["task_id"])], cal_row=cr)
        stored[(t["domain"], t["task_id"])] = StoredTask(t["domain"], t["task_id"], t["prompt"],
                                                         dict(zip(tiers, (float(x) for x in probs))), tiers_d)
    return stored, arrays, router_p, keys, tasks


def zero_warmup(stored: dict[tuple[str, str], StoredTask], raw_files: dict[str, Path],
                chunk: int = WARMUP_CHUNK) -> dict[tuple[str, str], StoredTask]:
    """Copy of the stored tasks with answer seconds zeroed for the first ``chunk`` tasks of each tier's raw generation file
    (scripts/laya_router_warmup.py); program seconds stay."""
    first = {m: {(r["domain"], r["task_id"]) for r in _read(p)[:chunk]} for m, p in raw_files.items()}
    out = {}
    for k, t in stored.items():
        out[k] = StoredTask(t.domain, t.task_id, t.prompt, t.router,
                            {m: StoredTier(**{**st.__dict__, "gen_s": 0.0 if k in first[m] else st.gen_s})
                             for m, st in t.tiers.items()})
    return out


# --------------------------------------------------------------------------- comparison
def compare(runtime: list[dict[str, Any]], reg: dict[str, Any], keys: list[tuple[str, str]], tiers: tuple[str, ...],
            arrays: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    rows, dis = [], []
    for i, (r, k) in enumerate(zip(runtime, keys)):
        cons = reg["consulted"][i]
        reg_start, reg_final = tiers[cons[0]], tiers[cons[-1]]
        reg_path = [(tiers[t], "VERIFIED" if arrays[t]["gate_ok"][i] else "notVERIFIED") for t in cons]
        rt_path = [(t, "VERIFIED" if s == "VERIFIED" else "notVERIFIED") for t, s in r["path"]]
        checks = {
            "start_tier": r["start_tier"] == reg_start,
            "final_tier": r["final_tier"] == reg_final,
            "path": rt_path == reg_path,
            "cost": abs(r["cost_s"] - float(reg["cost"][i])) <= COST_TOL,
            "answered": bool(r["answered"]) == bool(reg["answered"][i]),
        }
        # registered accuracy quantity: stored correctness of the final candidate (verified one, else the last tier's)
        fin_idx = tiers.index(r["final_tier"]) if r["final_tier"] in tiers else None
        cor_final = bool(arrays[fin_idx]["cor"][i]) if fin_idx is not None else None
        checks["final_candidate_correct"] = cor_final == bool(reg["cor"][i])
        row = {"domain": k[0], "task_id": k[1], "reg_start": reg_start, "rt_start": r["start_tier"], "reg_final": reg_final,
               "rt_final": r["final_tier"], "reg_cost": float(reg["cost"][i]), "rt_cost": r["cost_s"],
               "reg_answered": bool(reg["answered"][i]), "rt_answered": bool(r["answered"]), "verdict": r["verdict"],
               "answer_returned": r["answer_returned"], "reg_cor": bool(reg["cor"][i]), "rt_final_cand_cor": cor_final,
               "rt_answered_correct": bool(r["answered"] and cor_final), **{f"agree_{c}": v for c, v in checks.items()}}
        rows.append(row)
        if not all(checks.values()):
            dis.append({**row, "failed_checks": [c for c, v in checks.items() if not v], "rt_path": rt_path, "reg_path": reg_path})
    n = len(rows)
    agree = {c: int(sum(r[f"agree_{c}"] for r in rows)) for c in
             ("start_tier", "final_tier", "path", "cost", "answered", "final_candidate_correct")}
    return {"n": n, "agree_counts": agree, "all_five_agree": int(sum(all(r[f"agree_{c}"] for c in
            ("start_tier", "final_tier", "path", "cost", "answered")) for r in rows)),
            "disagreements": dis, "rows": rows}


def aggregates(rows: list[dict[str, Any]], arrays9: dict[str, np.ndarray], run9: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    cost_pol = float(np.mean([r["rt_cost"] for r in rows]))
    cost_reg = float(np.mean([r["reg_cost"] for r in rows]))
    cost9_rt = float(np.mean([r["cost_s"] for r in run9]))
    cost9_reg = float(np.mean(arrays9["cost"] + arrays9["pot"]))
    acc_reg = float(np.mean([r["reg_cor"] for r in rows]))
    acc_rt_final = float(np.mean([bool(r["rt_final_cand_cor"]) for r in rows]))
    acc_rt_answered = float(np.mean([r["rt_answered_correct"] for r in rows]))
    acc9 = float(arrays9["cor"].mean())
    acc9_rt_answered = float(np.mean([bool(r["answered"]) and bool(arrays9["cor"][i]) for i, r in enumerate(run9)]))
    return {"n": n, "cost_per_task_policy_runtime": cost_pol, "cost_per_task_policy_registered": cost_reg,
            "cost_per_task_always9b_runtime": cost9_rt, "cost_per_task_always9b_registered": cost9_reg,
            "cost_ratio_vs_always9b_runtime": cost_pol / cost9_rt, "cost_ratio_vs_always9b_registered": cost_reg / cost9_reg,
            "acc_policy_registered_quantity": acc_reg, "acc_policy_runtime_final_candidate": acc_rt_final,
            "acc_policy_runtime_answered_only": acc_rt_answered, "acc_always9b_registered_quantity": acc9,
            "acc_always9b_runtime_answered_only": acc9_rt_answered,
            "d_acc_points_registered_quantity": 100 * (acc_reg - acc9),
            "d_acc_points_runtime_final_candidate": 100 * (acc_rt_final - acc9),
            "d_acc_points_runtime_answered_only": 100 * (acc_rt_answered - acc9_rt_answered),
            "answered_rate_policy_runtime": float(np.mean([r["rt_answered"] for r in rows])),
            "verdicts_policy_runtime": dict(Counter(r["verdict"] for r in rows))}


def replay_arm(stored, arrays, router_p, keys, tiers, tau, domains, raw_files=None) -> dict[str, Any]:
    world = ReplayWorld(stored, tiers)
    system = build_system(world, tiers, domains, routed=True, tau=tau)
    runtime = run_runtime(world, system, keys)
    reg = registered_outcome(tau, arrays, router_p)
    cmpd = compare(runtime, reg, keys, tiers, arrays)
    # always-largest: its own replay (single tier, no router), against the closed-form stored cost
    big = tiers[-1]
    w9 = ReplayWorld(stored, (big,))
    run9 = run_runtime(w9, build_system(w9, (big,), domains, routed=False), keys)
    c9 = [r["cost_s"] for r in run9]
    cost9_ok = bool(np.all(np.abs(np.array(c9) - (arrays[-1]["cost"] + arrays[-1]["pot"])) <= COST_TOL))
    out = {"agreement": {k: v for k, v in cmpd.items() if k not in ("rows",)},
           "aggregates": aggregates(cmpd["rows"], arrays[-1], run9), "always9b_cost_agrees_per_task": cost9_ok,
           "rows": cmpd["rows"], "feature_mismatches": world.feature_mismatches[:200],
           "n_feature_mismatches": len(world.feature_mismatches), "calls": dict(world.calls),
           "start_by_tier_runtime": dict(Counter(r["rt_start"] for r in cmpd["rows"])),
           "final_by_tier_runtime": dict(Counter(r["rt_final"] for r in cmpd["rows"]))}
    if raw_files:
        sw = zero_warmup(stored, raw_files)
        wn = ReplayWorld(sw, tiers)
        rn = run_runtime(wn, build_system(wn, tiers, domains, routed=True, tau=tau), keys)
        wn9 = ReplayWorld(sw, (big,))
        rn9 = run_runtime(wn9, build_system(wn9, (big,), domains, routed=False), keys)
        cost_pol, cost_big = float(np.mean([r["cost_s"] for r in rn])), float(np.mean([r["cost_s"] for r in rn9]))
        # registered no-warm: scripts/laya_router_warmup.py logic, on the registered arrays
        first = {m: {(r["domain"], r["task_id"]) for r in _read(p)[:WARMUP_CHUNK]} for m, p in raw_files.items()}
        nw = [{**a, "cost": np.where(np.array([k in first[m] for k in keys]), 0.0, a["cost"])} for m, a in zip(tiers, arrays)]
        pol = AL.run_policy(tau, nw, router_p)
        ratio_reg = float(pol["cost"].mean() / (nw[-1]["cost"] + nw[-1]["pot"]).mean())
        out["warmup_excluded"] = {"cost_ratio_runtime": cost_pol / cost_big, "cost_ratio_registered_recomputed": ratio_reg,
                                  "per_task_cost_agrees": bool(np.all(np.abs(np.array([r["cost_s"] for r in rn]) - pol["cost"]) <= COST_TOL)),
                                  "n_warm_tasks_per_tier": {m: len(s) for m, s in first.items()}}
    return out


def _paths(which: str, root: Path) -> dict[str, Any]:
    L = root / "laya_data"
    R = ROOT / "results" / "gwenlaya_v4" / "laya" / "data"
    if which == "eprime":
        return dict(tasks=L / "tasks_Eprime.jsonl", dirs={m: L / f"eprime_{m}" for m in TIERS},
                    router=R / "preds_Eprime_router.jsonl", cal=R / "preds_Eprime_calibrator.jsonl",
                    rows=L / "laya_eprime" / "calibrator.jsonl", raw={m: L / "eprime_raw" / f"raw_{m}.jsonl" for m in TIERS},
                    domains=("python", "math"))
    return dict(tasks=L / "tasks_Rust_fresh.jsonl", dirs={m: L / f"rfresh_{m}" for m in TIERS},
                router=R / "preds_Rfresh_router.jsonl", cal=R / "preds_Rfresh_calibrator.jsonl",
                rows=L / "laya_rfresh" / "calibrator.jsonl", raw=None, domains=("rust",))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", type=Path, default=Path("/mnt/data/home/xavkal/gwaya-data"))
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--sets", nargs="+", default=["eprime", "rfresh"], choices=["eprime", "rfresh"])
    ap.add_argument("--out", type=Path, default=ROOT / "results/gwenlaya_v4/laya_runtime/system_replay_eprime.json")
    a = ap.parse_args(argv)
    res: dict[str, Any] = {"meta": {"plan": "A20 (docs/ANALYSIS_PLAN_E_TPU.md)", "tau": a.tau, "tiers": list(TIERS),
                                    "route_policy": "start_at", "thresholds": "none (tau_hi=+inf, tau_lo=-inf)",
                                    "isotonic_map": str(ISOTONIC_FILE.relative_to(ROOT)), "isotonic_sha256": _sha(ISOTONIC_FILE),
                                    "inputs_sha256": {}}}
    for which in a.sets:
        p = _paths(which, a.data_root)
        stored, arrays, rp, keys, tasks = load_stored(p["tasks"], p["dirs"], p["router"], p["cal"], p["rows"])
        res["meta"]["inputs_sha256"][which] = {str(p[k].name): _sha(p[k]) for k in ("tasks", "router", "cal", "rows")}
        arm = replay_arm(stored, arrays, rp, keys, TIERS, a.tau, p["domains"], p["raw"])
        arm["domain_counts"] = dict(Counter(k[0] for k in keys))
        # registered numbers recomputed by the registered code (same arrays), for the comparison table
        big = arrays[-1]
        reg = AL.run_policy(a.tau, arrays, rp)
        arm["registered_recomputed"] = {"cost_ratio_vs_always9b": float(reg["cost"].mean() / (big["cost"] + big["pot"]).mean()),
                                        "d_acc_points": 100 * float(reg["cor"].mean() - big["cor"].mean())}
        res[which] = arm
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=lambda o: None if (isinstance(o, float) and math.isnan(o)) else str(o)))
    for which in a.sets:
        ag = res[which]["agreement"]
        print(which, "n", ag["n"], "all_five_agree", ag["all_five_agree"], ag["agree_counts"], "disagreements", len(ag["disagreements"]))
        print(" ", json.dumps(res[which]["aggregates"]))
        print("  feature mismatches", res[which]["n_feature_mismatches"], "registered", res[which]["registered_recomputed"],
              res[which].get("warmup_excluded"))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
