#!/usr/bin/env python3
"""C0-lite stage 2: manifest_eval.json, E-night sizing, tasks JSONL, power. Reads items_sanity.json
(from run_c0_lite.py) and L0_env.json. Writes only under --night."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "data"))
import build_eval_manifest as B  # noqa: E402

LICENSE_OF = {"evalplus/humanevalplus": "evalplus/humanevalplus", "evalplus/mbppplus": "evalplus/mbppplus",
              "nuprl/MultiPL-E": "nuprl/MultiPL-E", "HuggingFaceH4/MATH-500": "HuggingFaceH4/MATH-500"}


def ntok(text: str, url: str) -> int:
    req = urllib.request.Request(url + "/tokenize", data=json.dumps({"content": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    return len(json.load(urllib.request.urlopen(req, timeout=60))["tokens"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--night", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8091")
    ap.add_argument("--margin", type=float, default=0.85)
    a = ap.parse_args()
    N = Path(a.night)
    d = json.loads((N / "datasets" / "items_sanity.json").read_text())
    items, meta = d["items"], d["meta"]
    for v in meta.values():
        if isinstance(v["license"], list):
            v["license"] = ",".join(v["license"])
    env = json.loads((N / "results" / "L0_env.json").read_text())
    tok = {"L2": env["tok_s"]["2B"]["tok_s_wall"], "L3": env["tok_s"]["4B"]["tok_s_wall"]}

    # --- manifest rows -------------------------------------------------------------------
    rows, counts = [], {}
    for it in items:
        repo = it["source"].split(":")[0]
        rows.append({
            "task_id": it["task_id"], "domain": it["domain"], "source": it["source"], "native_id": it["native_id"],
            "cluster": B.cluster_of(it["domain"], it["native_id"]),
            "sha256": B.item_sha(it["statement"], it["tests"]),
            "gate_tests_sha256": B.sha256_text(B.normalize(it["gate_tests"] or "")),
            "n_asserts": it.get("n_asserts"),
            "hf_revision": meta[repo]["hf_revision"], "license": meta[repo]["license"],
            "excluded": it["exclusion"],
        })
    for dom in B.PROPORTION:
        r = [x for x in rows if x["domain"] == dom]
        ex = [x for x in r if x["excluded"]]
        reasons: dict[str, int] = {}
        for x in ex:
            reasons[x["excluded"].split(":")[0]] = reasons.get(x["excluded"].split(":")[0], 0) + 1
        counts[dom] = {"rows": len(r), "excluded": len(ex), "included": len(r) - len(ex), "exclusion_reasons": reasons}

    # --- seeded order on included items ---------------------------------------------------
    by_id = {x["task_id"]: x for x in rows}
    ordered: dict[str, list[str]] = {}
    for dom in B.PROPORTION:
        ids = [x["task_id"] for x in rows if x["domain"] == dom and not x["excluded"]]
        ordered[dom] = B.seeded_order(ids, tag=dom)
        for k, tid in enumerate(ordered[dom]):
            by_id[tid]["eval_rank"] = k

    # --- token estimate (ASSUMED: no generation yet) ---------------------------------------
    py_tok: dict[str, int] = {}
    est: dict[str, list[float]] = {"python": [], "rust": [], "math": []}
    for it in items:
        if it["exclusion"]:
            continue
        if it["domain"] == "python":
            t = min(1024.0, 1.25 * ntok(it["reference"], a.server) + 20)
            py_tok[B.cluster_of("python", it["native_id"])] = t
            est["python"].append(t)
        elif it["domain"] == "math":
            est["math"].append(min(1024.0, float(ntok(it["reference"], a.server)) + 20))
    pm = statistics.mean(est["python"])
    for it in items:
        if it["domain"] == "rust" and not it["exclusion"]:
            base = py_tok.get(B.cluster_of("rust", it["native_id"]), pm)
            est["rust"].append(min(1024.0, 1.6 * base))
    mean_dom = {k: statistics.mean(v) for k, v in est.items()}
    w = B.PROPORTION
    mean_item = sum(mean_dom[k] * w[k] for k in w) / sum(w.values())
    sz = B.size_n_total(tok, mean_item, {"L2": 3.0, "L3": 4.0}, margin=a.margin)
    n_d = B.apportion(sz["n_total"], w)
    n_d = {k: min(v, counts[k]["included"]) for k, v in n_d.items()}

    # --- tasks -----------------------------------------------------------------------------
    def task_row(it):
        sc = {"tests": it["tests"]} if it["domain"] != "math" else {"answer": it["gold"]}
        if it["domain"] == "python":
            sc["timeout_s"] = 30.0
        gp = {"tests": it["gate_tests"], "timeout_s": 10.0} if it["domain"] != "math" else {}
        return {"domain": it["domain"], "task_id": it["task_id"], "prompt": it["prompt"],
                "checker_payload": sc, "gate_payload": gp, "cluster": B.cluster_of(it["domain"], it["native_id"])}

    it_by_id = {it["task_id"]: it for it in items}
    night_sel = {dom: ordered[dom][: n_d[dom]] for dom in w}
    for dom, ids in night_sel.items():
        for tid in ids:
            by_id[tid]["in_e_night"] = True
    night = B.interleave({dom: [task_row(it_by_id[t]) for t in ids] for dom, ids in night_sel.items()})
    primary = B.interleave({dom: [task_row(it_by_id[t]) for t in ids] for dom, ids in ordered.items()})
    (N / "results" / "tasks_E_night.jsonl").write_text("".join(json.dumps(r) + "\n" for r in night))
    (N / "results" / "tasks_E_primary.jsonl").write_text("".join(json.dumps(r) + "\n" for r in primary))

    manifest = {
        "name": "gwenlaya_v4 primary-E manifest (night 2026-10-07, C0-lite)",
        "prereg": "docs/GWENLAYA_PREREGISTRATION.md section 2",
        "eval_order_seed": B.EVAL_ORDER_SEED, "order_rule": "random.Random(f'{seed}:{domain}').shuffle(sorted(included task_ids))",
        "sources": {k: {**v, "rows": sum(1 for x in rows if x["source"].startswith(k))} for k, v in meta.items()},
        "per_domain": counts,
        "totals": {"rows": len(rows), "excluded": sum(c["excluded"] for c in counts.values()),
                   "included": sum(c["included"] for c in counts.values())},
        "n_d": n_d, "items": rows,
    }
    out = N / "results" / "manifest_eval.json"
    out.write_text(json.dumps(manifest, indent=1, sort_keys=True))

    # --- power at the sized n (same method as the registered table) ------------------------
    n = sum(n_d.values())
    scen = {"6 pts (8%/2%)": (0.08, 0.02), "4 pts (5%/1%)": (0.05, 0.01), "3 pts (4%/1%)": (0.04, 0.01), "2 pts (3%/1%)": (0.03, 0.01)}
    power = {k: {"alpha_0.025": round(B.mcnemar_power(n, *v, 0.025), 3),
                 "alpha_0.05/3": round(B.mcnemar_power(n, *v, 0.05 / 3), 3)} for k, v in scen.items()}
    sizing = {"tok_s_sequential_wall": tok, "mean_tokens_per_item_assumed": round(mean_item, 1),
              "mean_tokens_by_domain_assumed": {k: round(v, 1) for k, v in mean_dom.items()},
              "margin": a.margin, **sz, "n_d": n_d, "sum_n_d": n, "power_at_sum_n_d": power,
              "token_estimate_basis": "ASSUMED: 1.25 x Qwen-token length of the reference solution + 20 (python); 1.6 x its Python sibling (rust); reference-solution length + 20 (math); capped at 1024. Not measured; see addendum."}
    (N / "results" / "E_night_sizing.json").write_text(json.dumps(sizing, indent=1))
    print(json.dumps({"per_domain": counts, "sizing": sizing}, indent=1))
    print("manifest sha256:", B.sha256_text(out.read_text()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
