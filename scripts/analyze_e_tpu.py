#!/usr/bin/env python
"""Full-E baselines on TPU: the analyses A1-A7 and A9 of docs/ANALYSIS_PLAN_E_TPU.md.

Inputs (all produced by scripts/import_remote_gens.py + scripts/run_study.py --mode score):
  --tiers  dir with rows.jsonl/gens.jsonl holding the arms base, always_smallest, always_largest, gate_only,
           gwenlaya (static ladder = gate cascade, no router) for the two tiers
  --coder  dir with the continuity-anchor model's base rows/gens (optional)
  --tasks  tasks_E_primary.d25.jsonl (clusters = source problems)
  --night-cpu / --night-tasks  optional: CPU q4 night rows to compare on the 80 night tasks (A7)

Writes numbers JSON (every number with its source and CI), LaTeX tables and figures. Nothing is invented:
unknown values are null. Statistics reuse scripts/analyze_study.py (cluster bootstrap, ECE, AUROC, AURC).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from gwaya.metrics_selective import holm, mcnemar_exact  # noqa: E402
from scripts import analyze_study as A  # noqa: E402

DOMAINS = ("python", "rust", "math")
SEL = "pooled"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class Data:
    """Per-task arrays aligned on one sorted task list."""

    def __init__(self, tasks: list[dict], models: dict[str, dict[str, np.ndarray]], gate: dict[str, np.ndarray],
                 cascade: dict[str, np.ndarray], small: str, large: str) -> None:
        self.keys = [(t["domain"], t["task_id"]) for t in tasks]
        self.dom = np.array([k[0] for k in self.keys])
        self.clu = np.array([t.get("cluster") or f"{t['domain']}:{t['task_id']}" for t in tasks])
        self.n = len(self.keys)
        self.models, self.gate, self.cascade, self.small, self.large = models, gate, cascade, small, large
        self.is_dom = {d: self.dom == d for d in DOMAINS}

    def sel(self, idx: np.ndarray, d: str) -> np.ndarray:
        return idx if d == SEL else idx[self.is_dom[d][idx]]


def build(tasks: list[dict], tiers_rows: list[dict], tiers_gens: list[dict], small: str, large: str,
          coder: tuple[list[dict], list[dict], str] | None = None) -> Data:
    tasks = sorted(tasks, key=lambda t: (t["domain"], t["task_id"]))
    keys = [(t["domain"], t["task_id"]) for t in tasks]
    models: dict[str, dict[str, np.ndarray]] = {}

    def add_model(name: str, rows: list[dict], gens: list[dict]) -> None:
        base = {(r["model"], r["domain"], r["task_id"]): r for r in rows if r["arm"] == "base"}
        gen = {(g["model"], g["domain"], g["task_id"]): g for g in gens if g.get("temperature") == 0.0}
        miss = [k for k in keys if (name, *k) not in base or (name, *k) not in gen]
        if miss:
            raise SystemExit(f"{name}: {len(miss)} tasks without a scored row or generation, e.g. {miss[0]}")
        lp = np.array([gen[(name, *k)].get("mean_logprob") for k in keys], dtype=object)
        conf = np.array([math.exp(min(0.0, x)) if x is not None else float("nan") for x in lp], dtype=float)
        models[name] = {
            "cor": np.array([base[(name, *k)]["score"] == "VERIFIED" for k in keys]),
            "conf": conf,
            "trunc": np.array([gen[(name, *k)].get("done_reason") == "length" for k in keys]),
            "cost": np.array([float(gen[(name, *k)].get("gpu_s") or 0.0) for k in keys]),
            "toks": np.array([float(gen[(name, *k)].get("completion_tokens") or 0.0) for k in keys]),
        }

    add_model(small, tiers_rows, tiers_gens)
    add_model(large, tiers_rows, tiers_gens)
    if coder:
        add_model(coder[2], coder[0], coder[1])

    gate = {}
    for m in (small, large):
        g = {(r["model"], r["domain"], r["task_id"]): r for r in tiers_rows if r["arm"] == "gate_only"}
        gate[m] = np.array([g[(m, *k)]["answered"] is True for k in keys])

    casc = {(r["domain"], r["task_id"]): r for r in tiers_rows if r["arm"] == "gwenlaya"}
    miss = [k for k in keys if k not in casc]
    if miss:
        raise SystemExit(f"cascade: {len(miss)} tasks without a gwenlaya row, e.g. {miss[0]}")
    cascade = {
        "cor": np.array([casc[k]["score"] == "VERIFIED" for k in keys]),
        "answered": np.array([bool(casc[k]["answered"]) for k in keys]),
        "cost": np.array([float(casc[k].get("gpu_s") or 0.0) for k in keys]),
        "escalated": np.array([len(casc[k].get("tiers_invoked") or []) > 1 for k in keys]),
    }
    # consistency: the always_* arms must equal the corresponding base rows
    for arm, m in (("always_smallest", small), ("always_largest", large)):
        rr = {(r["domain"], r["task_id"]): r["score"] for r in tiers_rows if r["arm"] == arm}
        if rr and any((rr[k] == "VERIFIED") != models[m]["cor"][i] for i, k in enumerate(keys)):
            raise SystemExit(f"{arm} rows disagree with base rows of {m}")
    return Data(tasks, models, gate, cascade, small, large)


# ── metric registry (one bootstrap loop for everything) ───────────────────────────────────────

def _mean(a: np.ndarray) -> float:
    return float(a.mean()) if len(a) else float("nan")


def matched_cwr(conf: np.ndarray, cor: np.ndarray, k: int) -> float:
    """CWR of 'answer the k most confident tasks' (ties: wrong first, pessimistic), over all tasks."""
    n = len(conf)
    if n == 0:
        return float("nan")
    ok = ~np.isnan(conf)
    order = np.lexsort((cor.astype(int), -np.where(ok, conf, -np.inf)))
    top = order[:max(0, min(k, n))]
    return float((~cor[top]).sum() / n)


def registry(D: Data) -> list[tuple[str, Callable[[np.ndarray], float]]]:
    s, l = D.small, D.large
    M, G, C = D.models, D.gate, D.cascade
    specs: list[tuple[str, Callable[[np.ndarray], float]]] = []

    def add(name: str, fn: Callable[[np.ndarray], float]) -> None:
        specs.append((name, fn))

    for d in (*DOMAINS, SEL):
        for m in M:  # A1
            add(f"A1.acc.{m}.{d}", lambda i, m=m, d=d: _mean(M[m]["cor"][D.sel(i, d)]))
            add(f"A9.trunc.{m}.{d}", lambda i, m=m, d=d: _mean(M[m]["trunc"][D.sel(i, d)]))
        for m in M:  # A2 (never pooled across domains for a headline; SEL kept for completeness)
            def calib(i, m=m, d=d, which=""):
                j = D.sel(i, d)
                ok = ~np.isnan(M[m]["conf"][j])
                return M[m]["conf"][j][ok], M[m]["cor"][j][ok]
            add(f"A2.meanconf.{m}.{d}", lambda i, c=calib: _mean(c(i)[0]))
            add(f"A2.ece.{m}.{d}", lambda i, c=calib: A.ece_equal_mass(*c(i)))
            add(f"A2.brier.{m}.{d}", lambda i, c=calib: A.brier(c(i)[0], c(i)[1].astype(float)))
            add(f"A2.auroc.{m}.{d}", lambda i, c=calib: A.auroc(*c(i)))
            add(f"A2.aurc.{m}.{d}", lambda i, c=calib: A.aurc(*c(i)))
        for tier, g in ((s, G[s]), (l, G[l])):  # A3 gate-only per tier
            cor = M[tier]["cor"]
            add(f"A3.cov.{tier}.{d}", lambda i, g=g, d=d: _mean(g[D.sel(i, d)]))
            add(f"A3.prec.{tier}.{d}", lambda i, g=g, cor=cor, d=d: A.answered_acc(g[D.sel(i, d)], cor[D.sel(i, d)]))
            add(f"A3.cwr.{tier}.{d}", lambda i, g=g, cor=cor, d=d: A.cwr(g[D.sel(i, d)], cor[D.sel(i, d)]))
        add(f"A3.cwr_all.{l}.{d}", lambda i, d=d: _mean(~M[l]["cor"][D.sel(i, d)]))  # answer everything
        # A4 matched coverage: top-k by log-prob confidence with k = gate's answered count in the resample
        def a4(i, d=d):
            j = D.sel(i, d)
            return matched_cwr(M[l]["conf"][j], M[l]["cor"][j], int(G[l][j].sum()))
        add(f"A4.cwr_b3_matched.{l}.{d}", a4)
        add(f"A4.delta_cwr_gate_minus_b3.{l}.{d}",
            lambda i, d=d, a4=a4: A.cwr(G[l][D.sel(i, d)], M[l]["cor"][D.sel(i, d)]) - a4(i, d))
        # A5 cascade vs B1/B2
        cs, cl, cc = M[s]["cor"], M[l]["cor"], C["cor"]
        add(f"A5.acc_small.{d}", lambda i, d=d: _mean(cs[D.sel(i, d)]))
        add(f"A5.acc_large.{d}", lambda i, d=d: _mean(cl[D.sel(i, d)]))
        add(f"A5.acc_cascade.{d}", lambda i, d=d: _mean(cc[D.sel(i, d)]))
        add(f"A5.d_acc_cascade_minus_large.{d}", lambda i, d=d: _mean(cc[D.sel(i, d)]) - _mean(cl[D.sel(i, d)]))
        add(f"A5.d_acc_cascade_minus_small.{d}", lambda i, d=d: _mean(cc[D.sel(i, d)]) - _mean(cs[D.sel(i, d)]))
        add(f"A5.cost_small.{d}", lambda i, d=d: _mean(M[s]["cost"][D.sel(i, d)]))
        add(f"A5.cost_large.{d}", lambda i, d=d: _mean(M[l]["cost"][D.sel(i, d)]))
        add(f"A5.cost_cascade.{d}", lambda i, d=d: _mean(C["cost"][D.sel(i, d)]))
        add(f"A5.cost_ratio_cascade_over_large.{d}",
            lambda i, d=d: float(C["cost"][D.sel(i, d)].sum() / M[l]["cost"][D.sel(i, d)].sum()) if M[l]["cost"][D.sel(i, d)].sum() else float("nan"))
        add(f"A5.escalated.{d}", lambda i, d=d: _mean(C["escalated"][D.sel(i, d)]))
        add(f"A5.cost_per_correct_cascade.{d}",
            lambda i, d=d: float(C["cost"][D.sel(i, d)].sum() / cc[D.sel(i, d)].sum()) if cc[D.sel(i, d)].sum() else float("nan"))
        add(f"A5.cost_per_correct_large.{d}",
            lambda i, d=d: float(M[l]["cost"][D.sel(i, d)].sum() / cl[D.sel(i, d)].sum()) if cl[D.sel(i, d)].sum() else float("nan"))
        add(f"A5.cascade_selective_cov.{d}", lambda i, d=d: _mean(C["answered"][D.sel(i, d)]))
        add(f"A5.cascade_selective_cwr.{d}", lambda i, d=d: A.cwr(C["answered"][D.sel(i, d)], cc[D.sel(i, d)]))
        add(f"A6.acc_oracle_router.{d}", lambda i, d=d: _mean((cs | cl)[D.sel(i, d)]))
    return specs


def paired_tests(D: Data) -> dict[str, Any]:
    """A5: exact McNemar on pooled paired accuracy contrasts, Holm over m = 2."""
    cc, cl, cs = D.cascade["cor"], D.models[D.large]["cor"], D.models[D.small]["cor"]
    out = {}
    for name, other in (("cascade_vs_large", cl), ("cascade_vs_small", cs)):
        b, c = int((cc & ~other).sum()), int((~cc & other).sum())
        out[name] = {"cascade_only_correct": b, "other_only_correct": c, "p_two_sided": mcnemar_exact(b, c, "two-sided")}
    adj = holm([out["cascade_vs_large"]["p_two_sided"], out["cascade_vs_small"]["p_two_sided"]])
    out["cascade_vs_large"]["p_holm"], out["cascade_vs_small"]["p_holm"] = adj
    return out


def night_compare(D: Data, night_rows: list[dict], night_ids: set[tuple[str, str]]) -> dict[str, Any]:
    """A7: CPU q4 llama.cpp night arm vs TPU bf16 vLLM, same raw protocol, same tasks (small tier)."""
    cpu = {(r["domain"], r["task_id"]): r["score"] == "VERIFIED" for r in night_rows if r["arm"] == "base"}
    idx = [i for i, k in enumerate(D.keys) if k in cpu and k in night_ids]
    tpu = D.models[D.small]["cor"]
    # Math is excluded: the CPU night math generations went through the defective code prompt (D24/D28),
    # so a CPU-vs-TPU math difference measures that defect, not the engine. Amendment to the analysis plan.
    keep = ("python", "rust")
    idx = [i for i in idx if D.dom[i] in keep]
    out: dict[str, Any] = {"n_common": len(idx), "domains": list(keep), "by_domain": {},
                           "excluded": {"math": "CPU night math used the defective code prompt (D24/D28); not comparable"}}
    for d in (*keep, SEL):
        j = [i for i in idx if d == SEL or D.dom[i] == d]
        a = np.array([cpu[D.keys[i]] for i in j])
        b = np.array([tpu[i] for i in j])
        out["by_domain"][d] = {"n": len(j), "cpu_pass": int(a.sum()), "tpu_pass": int(b.sum()),
                               "agree": int((a == b).sum()), "cpu_only": int((a & ~b).sum()), "tpu_only": int((~a & b).sum()),
                               "p_mcnemar_two_sided": mcnemar_exact(int((a & ~b).sum()), int((~a & b).sum()), "two-sided")}
    return out


def overlay_corrections(orig: list[dict], overlay: list[dict]) -> list[list]:
    """Rows whose score the serial re-check changed: [arm, model, domain, task_id, was, now]."""
    was = {(r["arm"], r.get("model"), r["domain"], r["task_id"]): r["score"] for r in orig}
    out = set()
    for r in overlay:
        k = (r["arm"], r.get("model"), r["domain"], r["task_id"])
        if k in was and was[k] != r["score"]:
            out.add((*k, was[k], r["score"]))
    return [list(x) for x in sorted(out, key=str)]


def analyze(D: Data, n_boot: int, seed: int) -> dict[str, Any]:
    specs = registry(D)
    names = [n for n, _ in specs]
    full = np.arange(D.n)

    def stat(idx: np.ndarray) -> tuple[float, ...]:
        return tuple(fn(idx) for _, fn in specs)

    point = np.array([fn(full) for _, fn in specs], dtype=float)
    boot = A.cluster_boot(stat, D.dom, D.clu, n_boot=n_boot, seed=seed)
    res: dict[str, Any] = {}
    for j, name in enumerate(names):
        ci = A.pct_ci(float(point[j]), boot[:, j])
        res[name] = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in ci.items()}
    return {"metrics": res, "paired_tests": paired_tests(D)}


# ── tables and figures ────────────────────────────────────────────────────────────────────────

def f(res: dict, name: str, d: int = 3, pct: bool = False) -> str:
    m = res.get(name)
    if not m or m.get("point") is None:
        return "TBD"
    sc = 100.0 if pct else 1.0
    lo, hi = m.get("lo"), m.get("hi")
    p = f"{m['point'] * sc:.{d}f}"
    return p if lo is None or hi is None else f"{p} [{lo * sc:.{d}f}, {hi * sc:.{d}f}]"


def write_tables(path: Path, R: dict, D: Data, night: dict | None) -> None:
    r = R["metrics"]
    s, l = D.small, D.large
    L = ["% generated by scripts/analyze_e_tpu.py - do not edit"]
    L += ["\\begin{table}[t]\\centering\\small", "\\caption{Pass rate on the hidden checks, full E set, greedy, one sample "
          "(95\\% cluster bootstrap CI). TPU v5e bf16.}\\label{tab:e-pass}",
          "\\begin{tabular}{lcccc}\\toprule", "Model & Python (529) & Rust (507) & Math (500) & Pooled (1536)\\\\\\midrule"]
    for m in D.models:
        L.append(f"{m} & " + " & ".join(f(r, f'A1.acc.{m}.{d}') for d in (*DOMAINS, SEL)) + "\\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}", ""]
    L += ["\\begin{table}[t]\\centering\\small", "\\caption{Raw confidence $\\exp(\\overline{\\log p})$ as a correctness score "
          "(ECE: 15 equal-mass bins).}\\label{tab:e-calib}", "\\begin{tabular}{llcccc}\\toprule",
          "Model & Domain & mean conf. & ECE & AUROC & AURC\\\\\\midrule"]
    for m in D.models:
        for d in DOMAINS:
            L.append(f"{m} & {d} & {f(r, f'A2.meanconf.{m}.{d}')} & {f(r, f'A2.ece.{m}.{d}')} & "
                     f"{f(r, f'A2.auroc.{m}.{d}')} & {f(r, f'A2.aurc.{m}.{d}')}\\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}", ""]
    L += ["\\begin{table}[t]\\centering\\small", f"\\caption{{Gate-only on {l} (answer iff the executed gate verifies) "
          "versus answering everything, and versus a log-probability threshold matched to the gate's coverage "
          "(threshold tuned on E, which favours the baseline).}\\label{tab:e-gate}", "\\begin{tabular}{lccccc}\\toprule",
          "Domain & coverage & precision & CWR gate & CWR answer-all & $\\Delta$ vs matched log-prob\\\\\\midrule"]
    for d in DOMAINS:
        L.append(f"{d} & {f(r, f'A3.cov.{l}.{d}')} & {f(r, f'A3.prec.{l}.{d}')} & {f(r, f'A3.cwr.{l}.{d}')} & "
                 f"{f(r, f'A3.cwr_all.{l}.{d}')} & {f(r, f'A4.delta_cwr_gate_minus_b3.{l}.{d}')}\\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}", ""]
    L += ["\\begin{table}[t]\\centering\\small", f"\\caption{{Exploratory gate cascade ({s} $\\to$ gate $\\to$ {l}) versus always-"
          "smallest and always-largest. Cost = accelerator chip-seconds per task at the generation batch size "
          "(throughput, not latency).}\\label{tab:e-cascade}", "\\begin{tabular}{lccccc}\\toprule",
          "Domain & acc. small & acc. large & acc. cascade & cost ratio casc./large & escalated\\\\\\midrule"]
    for d in (*DOMAINS, SEL):
        L.append(f"{d} & {f(r, f'A5.acc_small.{d}')} & {f(r, f'A5.acc_large.{d}')} & {f(r, f'A5.acc_cascade.{d}')} & "
                 f"{f(r, f'A5.cost_ratio_cascade_over_large.{d}')} & {f(r, f'A5.escalated.{d}')}\\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}", ""]
    if night:
        L += ["\\begin{table}[t]\\centering\\small", "\\caption{Same raw protocol, night Python and Rust tasks (math excluded: the CPU math used a defective prompt): Qwen3.5-2B q4\\_K\\_M "
              "llama.cpp CPU versus bf16 vLLM TPU. Hardware, quantisation and engine all differ.}\\label{tab:e-night}",
              "\\begin{tabular}{lccccc}\\toprule", "Domain & $n$ & CPU pass & TPU pass & agree & McNemar $p$\\\\\\midrule"]
        for d in (*night["domains"], SEL):
            b = night["by_domain"][d]
            L.append(f"{d} & {b['n']} & {b['cpu_pass']} & {b['tpu_pass']} & {b['agree']} & {b['p_mcnemar_two_sided']:.3f}\\\\")
        L += ["\\bottomrule\\end{tabular}\\end{table}"]
    Path(path).write_text("\n".join(L) + "\n")


def make_figures(outdir: Path, D: Data) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True, exist_ok=True)
    made = []
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
    l = D.large
    for a, d in zip(ax, DOMAINS):
        j = np.where(D.is_dom[d])[0]
        p, y = D.models[l]["conf"][j], D.models[l]["cor"][j]
        ok = ~np.isnan(p)
        cov, risk = A.risk_coverage(p[ok], y[ok])
        a.plot(cov, risk, label="log-prob threshold")
        g = D.gate[l][j]
        if g.any():
            a.scatter([g.mean()], [A.answered_acc(g, y) and 1 - A.answered_acc(g, y)], c="C3", zorder=3, label="gate")
        a.axhline(1 - y.mean(), ls=":", c="gray")
        a.set_title(f"{d} ({l})"); a.set_xlabel("coverage"); a.set_ylabel("risk (error | answered)")
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(outdir / "e_risk_coverage.png", dpi=140); plt.close(fig); made.append("e_risk_coverage.png")
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for a, d in zip(ax, DOMAINS):
        j = D.is_dom[d]
        pts = {D.small: (D.models[D.small]["cost"][j].mean(), D.models[D.small]["cor"][j].mean()),
               D.large: (D.models[D.large]["cost"][j].mean(), D.models[D.large]["cor"][j].mean()),
               "gate cascade": (D.cascade["cost"][j].mean(), D.cascade["cor"][j].mean())}
        for k, (x, yv) in pts.items():
            a.scatter([x], [yv], label=k)
        a.set_title(d); a.set_xlabel("chip-seconds per task"); a.set_ylabel("pass rate")
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(outdir / "e_cascade_pareto.png", dpi=140); plt.close(fig); made.append("e_cascade_pareto.png")
    return made


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tiers", required=True, type=Path)
    ap.add_argument("--coder", type=Path)
    ap.add_argument("--coder-name", default="qwen2.5-coder-1.5b-bf16")
    ap.add_argument("--tasks", required=True, type=Path)
    ap.add_argument("--small", default="qwen3.5-2b-bf16")
    ap.add_argument("--large", default="qwen3.5-4b-bf16")
    ap.add_argument("--night-cpu", type=Path, help="night L2fix rows.jsonl (CPU q4 arm)")
    ap.add_argument("--night-tasks", type=Path)
    ap.add_argument("--overlay-tiers", type=Path, help="serial re-score rows (scripts/tpu/serial_overlay.py) that take precedence")
    ap.add_argument("--overlay-coder", type=Path)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-numbers", type=Path, required=True)
    ap.add_argument("--out-tables", type=Path)
    ap.add_argument("--out-figs", type=Path)
    a = ap.parse_args(argv)

    tasks = read_jsonl(a.tasks)
    tier_rows = read_jsonl(a.tiers / "rows.jsonl")
    coder_rows = read_jsonl(a.coder / "rows.jsonl") if a.coder else []
    corrections: dict[str, list] = {}
    for label, orig, ov_path in (("tiers", tier_rows, a.overlay_tiers), ("coder", coder_rows, a.overlay_coder)):
        if ov_path and Path(ov_path).exists():
            ov = read_jsonl(ov_path)
            corrections[label] = overlay_corrections(orig, ov)
            orig.extend(ov)  # later rows win in build()
    coder = (coder_rows, read_jsonl(a.coder / "gens.jsonl"), a.coder_name) if a.coder else None
    D = build(tasks, tier_rows, read_jsonl(a.tiers / "gens.jsonl"), a.small, a.large, coder)
    R = analyze(D, a.n_boot, a.seed)
    night = None
    if a.night_cpu and a.night_tasks:
        ids = {(t["domain"], t["task_id"]) for t in read_jsonl(a.night_tasks)}
        night = night_compare(D, read_jsonl(a.night_cpu), ids)
    out = {"meta": {"n_tasks": D.n, "n_boot": a.n_boot, "seed": a.seed, "small": a.small, "large": a.large,
                    "models": list(D.models), "plan": "docs/ANALYSIS_PLAN_E_TPU.md",
                    "cost_unit": "accelerator chip-seconds (chunk wall time split by token share)",
                    "sources": {"tiers": str(a.tiers), "coder": str(a.coder) if a.coder else None, "tasks": str(a.tasks)}},
           "serial_recheck_corrections": corrections,
           **R, "A7_night_cpu_vs_tpu": night}
    a.out_numbers.parent.mkdir(parents=True, exist_ok=True)
    a.out_numbers.write_text(json.dumps(out, indent=1))
    if a.out_tables:
        write_tables(a.out_tables, R, D, night)
    if a.out_figs:
        make_figures(a.out_figs, D)
    print(f"wrote {a.out_numbers} ({len(R['metrics'])} metrics)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
