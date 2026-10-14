#!/usr/bin/env python3
"""Independent recomputation for the 3.12.0 audit (docs/CLAIM_AUDIT_3120.md). Written by the auditor, not the author.
Reads only dev_table.jsonl (committed), the data-disk sample files and the committed sealed result; writes
results/gwenlaya_v4/trust_loop/audit3120_recomputed.json. Re-implements D3 lr|core grouped CV (same fold hash, C=0.3,
inner OOF isotonic, Wilson-LB threshold rule) to get subset figures the author's JSON does not hold."""
import collections, hashlib, json, math, os, sys, warnings
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get("GWAYA_DATA_ROOT", "/mnt/data/home/xavkal/gwaya-data")
OUT = os.path.join(ROOT, "results/gwenlaya_v4/trust_loop/audit3120_recomputed.json")
def wil(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return [(c - a) / d, (c + a) / d]
R = [json.loads(l) for l in open(os.path.join(ROOT, "results/gwenlaya_v4/trust_loop/dev_table.jsonl"))]
y = np.array([r["correct"] for r in R]); tier = np.array([r["tier"] for r in R]); st = np.array([r["set"] for r in R])
X = np.array([[r["gate"] == "verified", r["mean_logprob"], max(r["min_logprob"], -20), math.log1p(r["completion_tokens"]),
               r["finish_reason"] == "length", r["n_visible_tests"]] for r in R], float)
fo = np.array([int(hashlib.md5(("0|" + r["cluster"]).encode()).hexdigest(), 16) % 5 for r in R])
mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
def pick(sc, yy, t):
    o = np.argsort(-sc, kind="stable"); cs = np.cumsum(yy[o]); n = len(yy); b = None
    for i in range(math.ceil(.05 * n), n + 1):
        if wil(cs[i - 1], i)[0] >= t: b = i
    if b is None:
        for i in range(math.ceil(.05 * n), n + 1):
            if cs[i - 1] / i >= t: b = i
    if b is None: b = math.ceil(.5 * n)
    return sc[o[b - 1]]
sc = np.zeros(len(y)); acc = np.zeros(len(y), bool)
for j in range(5):
    tr = np.where(fo != j)[0]; te = np.where(fo == j)[0]; f = fo[tr]; raw = np.zeros(len(tr))
    for q in range(5):
        m = f == q
        if m.sum(): raw[m] = mk().fit(X[tr[~m]], y[tr[~m]]).predict_proba(X[tr[m]])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw, y[tr]); cal = iso.predict(raw) + 1e-6 * raw
    r = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]; sc[te] = iso.predict(r) + 1e-6 * r
    acc[te] = sc[te] >= pick(cal, y[tr], 0.85)
def block(m):
    yy = y[m]; s = sc[m]; N = len(yy); o = np.argsort(-s, kind="stable"); cs = np.cumsum(yy[o]); a = acc[m]; k = int(yy[a].sum()); n = int(a.sum())
    n5 = math.ceil(.5 * N)
    grid = None
    for c in [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]:
        i = math.ceil(c * N)
        if cs[i - 1] / i >= .84 and wil(cs[i - 1], i)[0] >= .8: grid = dict(coverage=c, precision=float(cs[i - 1] / i))
    fine = None
    for i in range(math.ceil(.05 * N), N + 1):
        if cs[i - 1] / i >= .84 and wil(cs[i - 1], i)[0] >= .8: fine = dict(coverage=i / N, precision=float(cs[i - 1] / i))
    return dict(n=N, base_rate=float(yy.mean()), cv085=dict(coverage=n / N, precision=k / n, wilson=wil(k, n)),
                rank05=dict(precision=float(cs[n5 - 1] / n5), wilson=wil(int(cs[n5 - 1]), n5)), a30_grid=grid, a30_every_prefix=fine)
out = {"d3_lr_core_recomputed": {}}
for sname in [None, "F", "Eprime"]:
    for tname in [None] + sorted(set(tier)):
        m = np.ones(len(y), bool)
        if sname: m &= st == sname
        if tname: m &= tier == tname
        out["d3_lr_core_recomputed"][f"{sname or 'ALL'}|{tname or 'pooled'}"] = block(m)
# 2B hand rule vs D3 at equal coverage
m = tier == "qwen3.5-2b-bf16"; g = np.array([r["gate"] == "verified" for r in R]); o = np.argsort(-sc[m], kind="stable")
ng = int((g & m).sum())
out["2B_equal_coverage"] = dict(gate_verified_rows=ng, coverage=ng / int(m.sum()), hand_rule_precision=float(y[m & g].mean()),
                                d3_top_same_n_precision=float(y[m][o[:ng]].mean()), d3_top_same_n_wilson=wil(int(y[m][o[:ng]].sum()), ng))
# sample pools: nine-candidate vs eight-sample pass rates, truncation, verified-but-wrong
G = {(r["tier"], r["task_id"]): r for r in R if r["set"] == "F"}
S = [json.loads(l) for f in ("eval_sets_F_sample", "eval_sets_F_sample_9b") for l in open(f"{DATA}/{f}/scored_samples.jsonl")]
out["pools"] = {}
for t in sorted(set(tier)):
    ss = [s for s in S if s["model"] == t]; by = collections.defaultdict(list)
    for s in ss: by[s["task_id"]].append(s)
    gr = [G[k] for k in G if k[0] == t]
    ver = [s for s in ss if s["gate"].lower() == "verified"]
    out["pools"][t] = dict(n_samples=len(ss), n_tasks=len(by), verified=len(ver), verified_hidden_fail=sum(s["hidden"] != "VERIFIED" for s in ver),
        truncated=sum(s["finish"] == "length" for s in ss), truncated_gate_pass=sum(s["finish"] == "length" and s["gate"].lower() == "verified" for s in ss),
        truncated_hidden_pass=sum(s["finish"] == "length" and s["hidden"] == "VERIFIED" for s in ss),
        greedy_truncated=sum(r["finish_reason"] == "length" for r in gr), n_greedy=len(gr),
        mean_pass_eight_samples=float(np.mean([np.mean([s["hidden"] == "VERIFIED" for s in v]) for v in by.values()])),
        mean_pass_nine_candidates=float(np.mean([(sum(s["hidden"] == "VERIFIED" for s in v) + bool(G[(t, k)]["correct"] if (t, k) in G else G[(t, k.replace("py/", ""))]["correct"])) / (len(v) + 1) for k, v in by.items()])))
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out, indent=1)[:3000])
