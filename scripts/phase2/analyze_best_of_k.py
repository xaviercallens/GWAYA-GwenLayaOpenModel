#!/usr/bin/env python3
"""A29 analysis: verifier-guided best-of-8 on F python (2B, 4B). Hidden checks only score, never select.
Arms: G greedy+gate; S1 = among greedy+8 samples, gate-verified candidate with highest mean log-prob (refuse if none);
S1r = S1 tasks ranked by number of gate-verified candidates then log-prob, top 50% coverage.
Writes JSON; prints a table. Comparison value D3 comes from lever_D_stacked_calibrated.json (F+E' rows, not identical)."""
import json, math, sys
import numpy as np
D = "/mnt/data/home/xavkal/gwaya-data"
SC, DEV, D3, OUT = f"{D}/eval_sets_F_sample/scored_samples.jsonl", f"{D}/trust_loop/dev_table.jsonl", \
    f"{D}/trust_loop/lever_D_stacked_calibrated.json", sys.argv[1]
def wil(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z*z/n; c = p + z*z/(2*n); a = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)); return [(c-a)/d, (c+a)/d]
samp = {}
for l in open(SC):
    r = json.loads(l); samp.setdefault((r["model"], r["task_id"]), []).append(r)
greedy = {}
for l in open(DEV):
    r = json.loads(l)
    if r["set"] == "F": greedy[(r["tier"], r["task_id"])] = r
def key_tid(t): return t if t.startswith("py/") else "py/" + t
d3 = json.load(open(D3))
res = {}
rng = np.random.default_rng(0)
for tier in ("qwen3.5-2b-bf16", "qwen3.5-4b-bf16"):
    rows = []
    for (m, tid), ss in samp.items():
        if m != tier: continue
        g = greedy.get((tier, tid)) or greedy.get((tier, tid.replace("py/", "")))
        cands = [(s["gate"].lower() == "verified", s["hidden"] == "VERIFIED", s["mean_logprob"] if s["mean_logprob"] is not None else -9.0) for s in ss]
        if g: cands.append((g["gate"] == "verified", bool(g["correct"]), g["mean_logprob"]))
        v = [c for c in cands if c[0]]
        rows.append(dict(tid=tid, n_v=len(v), best=max(v, key=lambda c: c[2]) if v else None,
                         g=(g["gate"] == "verified", bool(g["correct"])) if g else None, any_correct=any(c[1] for c in cands),
                         samp_pass=float(np.mean([c[1] for c in cands]))))
    n = len(rows)
    def summ(sel):
        k = sum(1 for r in sel if r["best"][1]); return dict(n=len(sel), cov=len(sel)/n, prec=k/len(sel) if sel else None, wilson=wil(k, len(sel)))
    acc = [r for r in rows if r["best"]]
    acc_rank = sorted(acc, key=lambda r: (-r["n_v"], -r["best"][2]))[:int(math.ceil(0.5*n))]
    gsel = [r for r in rows if r["g"] and r["g"][0]]
    gk = sum(1 for r in gsel if r["g"][1])
    boot = []
    for _ in range(1000):
        idx = rng.integers(0, len(acc_rank), len(acc_rank)); boot.append(np.mean([acc_rank[i]["best"][1] for i in idx]))
    d3c = d3.get("lr|core", {}).get(tier, {}).get("curve", {}).get("0.5", {})
    d3g = d3.get("gbm|core", {}).get(tier, {}).get("curve", {}).get("0.5", {})
    res[tier] = dict(n_tasks=n, greedy_base_rate=float(np.mean([r["g"][1] for r in rows if r["g"]])),
        sample_pass_rate=float(np.mean([r["samp_pass"] for r in rows])), oracle_any_correct=float(np.mean([r["any_correct"] for r in rows])),
        G_gate=dict(n=len(gsel), cov=len(gsel)/n, prec=gk/len(gsel) if gsel else None, wilson=wil(gk, len(gsel))),
        S1_all=summ(acc), S1_cov50=summ(acc_rank), S1_cov50_boot_ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        D3_core_cov50_FplusEprime=dict(lr=d3c.get("precision"), gbm=d3g.get("precision")))
json.dump(res, open(OUT, "w"), indent=1)
for t, v in res.items():
    print(t, "n", v["n_tasks"], "greedy %.3f sample-mean %.3f oracle-any %.3f" % (v["greedy_base_rate"], v["sample_pass_rate"], v["oracle_any_correct"]))
    for k in ("G_gate", "S1_all", "S1_cov50"):
        s = v[k]; print("  %-9s cov %.2f prec %s wilson %s n %d" % (k, s["cov"], None if s["prec"] is None else round(s["prec"], 3), None if s["wilson"][0] is None else [round(x, 3) for x in s["wilson"]], s["n"]))
    print("  S1_cov50 boot CI", [round(x, 3) for x in v["S1_cov50_boot_ci"]], "D3 core @cov50 (F+E')", v["D3_core_cov50_FplusEprime"])
