#!/usr/bin/env python3
"""A29 analysis: verifier-guided best-of-8 on F python (2B, 4B). Hidden checks only score, never select.
Arms: G greedy+gate; S1 = among greedy+8 samples, gate-verified candidate with highest mean log-prob (refuse if none);
S1r = S1 tasks ranked by number of gate-verified candidates then log-prob, top 50% coverage.
Writes JSON; prints a table. Comparison value D3 comes from lever_D_stacked_calibrated.json (F+E' rows, not identical)."""
import json, math, os, sys
from collections import Counter
import numpy as np
D = "/mnt/data/home/xavkal/gwaya-data"
# A30 (9B): GWAYA_SCORED=<scored_samples.jsonl of the 9B run>; default is the A29 (2B/4B) file.
SC, DEV, D3, OUT = os.environ.get("GWAYA_SCORED", f"{D}/eval_sets_F_sample/scored_samples.jsonl"), f"{D}/trust_loop/dev_table.jsonl", \
    f"{D}/trust_loop/lever_D_stacked_calibrated.json", sys.argv[1]
A30_PREC, A30_LB, A30_MIN_N, A30_MARGIN = 0.84, 0.80, 20, 0.05  # fixed in A30 before any 9B sample existed
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
def norm_text(t):
    """Text key for the identical-text arm: drop a surrounding markdown fence (greedy rows keep it, samples do not)."""
    t = (t or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"): t = t.rstrip()[:-3]
    return t.strip()
d3 = json.load(open(D3))
res = {}
rng = np.random.default_rng(0)
TIERS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["qwen3.5-2b-bf16", "qwen3.5-4b-bf16"]
for tier in TIERS:
    rows = []
    for (m, tid), ss in samp.items():
        if m != tier: continue
        g = greedy.get((tier, tid)) or greedy.get((tier, tid.replace("py/", "")))
        cands = [(s["gate"].lower() == "verified", s["hidden"] == "VERIFIED", s["mean_logprob"] if s["mean_logprob"] is not None else -9.0,
                  norm_text(s.get("text"))) for s in ss]
        if g: cands.append((g["gate"] == "verified", bool(g["correct"]), g["mean_logprob"], norm_text(g.get("text") or g.get("candidate"))))
        v = [c for c in cands if c[0]]
        # post hoc arm (A29 text, not part of any rule): among verified candidates, the one with the largest group of
        # textually identical siblings; ties broken by log-prob. Group size is the ranking key.
        grp = Counter(c[3] for c in v)
        best_txt = max(v, key=lambda c: (grp[c[3]], c[2])) if v else None
        rows.append(dict(tid=tid, n_v=len(v), best=max(v, key=lambda c: c[2]) if v else None,
                         best_txt=best_txt, grp_size=grp[best_txt[3]] if best_txt else 0,
                         g=(g["gate"] == "verified", bool(g["correct"])) if g else None, any_correct=any(c[1] for c in cands),
                         samp_pass=float(np.mean([c[1] for c in cands]))))
    n = len(rows)
    def summ(sel):
        k = sum(1 for r in sel if r["best"][1]); return dict(n=len(sel), cov=len(sel)/n, prec=k/len(sel) if sel else None, wilson=wil(k, len(sel)))
    acc = [r for r in rows if r["best"]]
    acc_rank = sorted(acc, key=lambda r: (-r["n_v"], -r["best"][2]))[:int(math.ceil(0.5*n))]
    gsel = [r for r in rows if r["g"] and r["g"][0]]
    gk = sum(1 for r in gsel if r["g"][1])
    def a30_largest(ranked, lab):
        """Largest rank prefix (n >= A30_MIN_N) with precision >= 0.84 and Wilson LB >= 0.80. Uses labels to pick the
        prefix, so it is the optimistic 'rank-curve' reading of A30, like the D3 rank-curve number it is compared with."""
        best, ks = None, 0
        for i, r in enumerate(ranked, 1):
            ks += int(lab(r))
            if i >= A30_MIN_N and ks / i >= A30_PREC and wil(ks, i)[0] >= A30_LB: best = dict(n=i, cov=i / n, prec=ks / i, wilson=wil(ks, i))
        return best
    ranked = sorted(acc, key=lambda r: (-r["n_v"], -r["best"][2]))
    best_a30 = a30_largest(ranked, lambda r: r["best"][1])
    ranked_txt = sorted(acc, key=lambda r: (-r["grp_size"], -r["best_txt"][2]))
    txt_a30 = a30_largest(ranked_txt, lambda r: r["best_txt"][1])
    txt_cov50 = ranked_txt[:int(math.ceil(0.5*n))]
    kt = sum(1 for r in txt_cov50 if r["best_txt"][1])
    boot = []
    for _ in range(1000):
        idx = rng.integers(0, len(acc_rank), len(acc_rank)); boot.append(np.mean([acc_rank[i]["best"][1] for i in idx]))
    d3t = d3.get("lr|core", {}).get(tier, {})
    d3c = d3t.get("curve", {}).get("0.5", {})
    d3g = d3.get("gbm|core", {}).get(tier, {}).get("curve", {}).get("0.5", {})
    # A30 reference for this tier: D3 lr|core largest 0.1-step rank-curve coverage meeting both conditions, and the
    # CV-selected threshold (0.85 target) coverage. Decision: S1 is a lever only if its A30 coverage >= rank-curve + 0.05.
    d3_rank = None
    for cv, pt in sorted(d3t.get("curve", {}).items(), key=lambda kv: float(kv[0])):
        if pt.get("precision", 0) >= A30_PREC and pt.get("wilson", [0])[0] >= A30_LB: d3_rank = dict(step=cv, **pt)
    d3_cvthr = d3t.get("at_cv_threshold", {}).get("0.85")
    s1cov = best_a30["cov"] if best_a30 else 0.0
    decision = dict(rule="S1 A30 coverage >= D3 lr|core rank-curve coverage + 0.05, both conditions holding",
                    S1_A30_cov=s1cov, D3_rank_curve_cov=d3_rank["coverage"] if d3_rank else None,
                    D3_cv_threshold_cov=d3_cvthr["coverage"] if d3_cvthr else None,
                    lever=bool(d3_rank and best_a30 and s1cov >= d3_rank["coverage"] + A30_MARGIN))
    res[tier] = dict(n_tasks=n, greedy_base_rate=float(np.mean([r["g"][1] for r in rows if r["g"]])),
        sample_pass_rate=float(np.mean([r["samp_pass"] for r in rows])), oracle_any_correct=float(np.mean([r["any_correct"] for r in rows])),
        n_samples_scored=sum(len(ss) for (m, _), ss in samp.items() if m == tier),
        verified_samples=dict(n=sum(1 for (m, _), ss in samp.items() if m == tier for s in ss if s["gate"].lower() == "verified"),
                              hidden_fail=sum(1 for (m, _), ss in samp.items() if m == tier for s in ss if s["gate"].lower() == "verified" and s["hidden"] != "VERIFIED")),
        G_gate=dict(n=len(gsel), cov=len(gsel)/n, prec=gk/len(gsel) if gsel else None, wilson=wil(gk, len(gsel))),
        S1_all=summ(acc), S1_cov50=summ(acc_rank), S1_A30_largest_coverage=best_a30, S1_cov50_boot_ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        posthoc_identical_text=dict(note="not part of any rule; A29 post hoc arm", cov50=dict(n=len(txt_cov50), cov=len(txt_cov50)/n,
                                    prec=kt/len(txt_cov50) if txt_cov50 else None, wilson=wil(kt, len(txt_cov50))), A30_largest_coverage=txt_a30,
                                    tasks_with_group_ge2=sum(1 for r in acc if r["grp_size"] >= 2)),
        D3_core_cov50_FplusEprime=dict(lr=d3c.get("precision"), gbm=d3g.get("precision")),
        D3_lr_core_A30_reference=dict(rank_curve=d3_rank, cv_threshold_085=d3_cvthr), A30_decision=decision)
json.dump(res, open(OUT, "w"), indent=1)
for t, v in res.items():
    print(t, "n", v["n_tasks"], "greedy %.3f sample-mean %.3f oracle-any %.3f" % (v["greedy_base_rate"], v["sample_pass_rate"], v["oracle_any_correct"]))
    for k in ("G_gate", "S1_all", "S1_cov50"):
        s = v[k]; print("  %-9s cov %.2f prec %s wilson %s n %d" % (k, s["cov"], None if s["prec"] is None else round(s["prec"], 3), None if s["wilson"][0] is None else [round(x, 3) for x in s["wilson"]], s["n"]))
    print("  verified samples", v["verified_samples"], "of", v["n_samples_scored"])
    print("  A30 largest coverage (prec>=0.84, LB>=0.80, n>=20):", v["S1_A30_largest_coverage"])
    print("  S1_cov50 boot CI", [round(x, 3) for x in v["S1_cov50_boot_ci"]], "D3 core @cov50 (F+E')", v["D3_core_cov50_FplusEprime"])
    pt = v["posthoc_identical_text"]
    print("  post hoc identical-text: cov50 prec %s wilson %s; A30 largest %s; tasks with group>=2: %d" % (
        None if pt["cov50"]["prec"] is None else round(pt["cov50"]["prec"], 3), [round(x, 3) for x in pt["cov50"]["wilson"]], pt["A30_largest_coverage"], pt["tasks_with_group_ge2"]))
    print("  D3 lr|core A30 reference:", v["D3_lr_core_A30_reference"])
    print("  A30 decision:", v["A30_decision"])
