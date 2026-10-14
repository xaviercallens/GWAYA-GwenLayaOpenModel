#!/usr/bin/env python3
"""A32: learned selector over sampled candidate pools on F python (dev only). Arms B0 (greedy only) and B1 (full pool),
same folds. See docs/ANALYSIS_PLAN_E_TPU.md A32 for the rule. Usage: selector_over_pool.py OUT.json"""
import hashlib, json, math, re, sys, warnings
import numpy as np
from collections import Counter, defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
D = "/mnt/data/home/xavkal/gwaya-data"
POOLS = {"qwen3.5-2b-bf16": f"{D}/eval_sets_F_sample/scored_samples.jsonl", "qwen3.5-4b-bf16": f"{D}/eval_sets_F_sample/scored_samples.jsonl",
         "qwen3.5-9b-bf16": f"{D}/eval_sets_F_sample_9b/scored_samples.jsonl"}
def wil(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return [(c - a) / d, (c + a) / d]
def norm(t):
    t = re.sub(r"```(?:python)?", "", t or ""); return re.sub(r"\s+", "", re.sub(r"#.*", "", t))
def fold_of(c, k=5): return int(hashlib.md5(("0|" + c).encode()).hexdigest(), 16) % k
def mk(kind):
    if kind == "lr": return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
    return HistGradientBoostingClassifier(max_depth=2, max_iter=80, learning_rate=0.05, min_samples_leaf=20, random_state=0)
greedy = {}
for l in open(f"{D}/trust_loop/dev_table.jsonl"):
    r = json.loads(l)
    if r["set"] == "F": greedy[(r["tier"], r["task_id"])] = r
def build(tier):
    pool = defaultdict(list)
    for l in open(POOLS[tier]):
        s = json.loads(l)
        if s["model"] == tier: pool[s["task_id"]].append(dict(gate=s["gate"].lower() == "verified", ok=s["hidden"] == "VERIFIED",
            mlp=s["mean_logprob"] if s["mean_logprob"] is not None else -9.0, ntok=s["completion_tokens"] or 0, trunc=s["finish"] == "length", txt=norm(s["text"]), kind="s"))
    tasks = []
    for tid, ss in pool.items():
        g = greedy.get((tier, tid)) or greedy.get((tier, tid.replace("py/", "")))
        if g is None: continue
        gc = dict(gate=g["gate"] == "verified", ok=bool(g["correct"]), mlp=g["mean_logprob"], ntok=g["completion_tokens"], trunc=g["finish_reason"] == "length", txt=norm(g["candidate"]), kind="g")
        cands = ss + [gc]; grp = Counter(c["txt"] for c in cands); nv = sum(c["gate"] for c in cands) / len(cands)
        m = np.array([c["mlp"] for c in cands]); rk = m.argsort().argsort() / max(len(m) - 1, 1)
        for c, r_ in zip(cands, rk):
            c["f"] = [int(c["gate"]), c["mlp"], math.log1p(c["ntok"]), int(c["trunc"]), grp[c["txt"]] / len(cands), nv, r_, float(m.mean()), g["n_visible_tests"]]
        tasks.append(dict(tid=tid, cluster=g["cluster"], cands=cands, greedy=gc))
    return tasks
def pick_thr(sc, yy, target=0.85):
    o = np.argsort(-sc, kind="stable"); cs = np.cumsum(yy[o]); n = len(yy); best = None
    for i in range(int(math.ceil(0.05 * n)), n + 1):
        if wil(cs[i - 1], i)[0] >= target: best = i
    if best is None:
        for i in range(int(math.ceil(0.05 * n)), n + 1):
            if cs[i - 1] / i >= target: best = i
    if best is None: best = int(math.ceil(0.5 * n))
    return float(sc[o[best - 1]])
def flat(tasks, idx, arm):
    X, y, tix = [], [], []
    for j in idx:
        for c in (tasks[j]["cands"] if arm == "B1" else [tasks[j]["greedy"]]):
            X.append(c["f"]); y.append(int(c["ok"])); tix.append(j)
    return np.array(X, float), np.array(y), np.array(tix)
def task_scores(model, tasks, idx, arm):
    """per task: (max candidate score, correctness of that candidate)"""
    X, y, tix = flat(tasks, idx, arm); p = model.predict_proba(X)[:, 1]; out = {}
    for pi, yi, ti in zip(p, y, tix):
        if ti not in out or pi > out[ti][0]: out[ti] = (pi, yi)
    return np.array([out[j][0] for j in idx]), np.array([out[j][1] for j in idx])
def run(tasks, arm, kind, K=5):
    n = len(tasks); fo = np.array([fold_of(t["cluster"], K) for t in tasks]); sc = np.zeros(n); yy = np.zeros(n, int); acc = np.zeros(n, bool)
    for j in range(K):
        tr, te = np.where(fo != j)[0], np.where(fo == j)[0]
        # inner OOF task-level scores on the training fold, for isotonic + threshold
        fi = np.array([fold_of("i" + tasks[i]["cluster"], K) for i in tr]); raw = np.zeros(len(tr)); ytr = np.zeros(len(tr), int)
        for q in range(K):
            a, b = tr[fi != q], tr[fi == q]
            if len(b) == 0: continue
            Xa, ya, _ = flat(tasks, a, arm); m = mk(kind).fit(Xa, ya)
            r, yb = task_scores(m, tasks, b, arm); raw[fi == q] = r; ytr[fi == q] = yb
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw, ytr)
        thr = pick_thr(iso.predict(raw) + 1e-6 * raw, ytr)
        Xt, yt, _ = flat(tasks, tr, arm); m = mk(kind).fit(Xt, yt)
        r, yb = task_scores(m, tasks, te, arm); s = iso.predict(r) + 1e-6 * r
        sc[te] = s; yy[te] = yb; acc[te] = s >= thr
    return sc, yy, acc
def summ(mask, yy):
    k = int(yy[mask].sum()); n = int(mask.sum()); return dict(n_accept=n, coverage=n / len(yy), precision=k / n if n else None, wilson=wil(k, n))
def a30_rank(sc, yy):
    o = np.argsort(-sc, kind="stable"); cs = np.cumsum(yy[o]); best = None
    for i in range(20, len(yy) + 1):
        if cs[i - 1] / i >= 0.84 and wil(int(cs[i - 1]), i)[0] >= 0.80: best = dict(n=i, coverage=i / len(yy), precision=float(cs[i - 1] / i), wilson=wil(int(cs[i - 1]), i))
    return best
def correct_at_cov(sc, yy, cov):
    o = np.argsort(-sc, kind="stable"); return int(yy[o[:int(math.ceil(cov * len(yy)))]].sum())
if __name__ == "__main__":
    out = {}; rng = np.random.default_rng(0)
    for tier in POOLS:
        tasks = build(tier); n = len(tasks); out[tier] = dict(n_tasks=n, oracle_any=float(np.mean([any(c["ok"] for c in t["cands"]) for t in tasks])),
                                                              greedy=float(np.mean([t["greedy"]["ok"] for t in tasks])))
        clus = sorted(set(t["cluster"] for t in tasks)); cidx = {c: [i for i, t in enumerate(tasks) if t["cluster"] == c] for c in clus}
        res = {}
        for kind in ("lr", "gbm"):
            for arm in ("B0", "B1"): res[(kind, arm)] = run(tasks, arm, kind)
        for kind in ("lr", "gbm"):
            s0, y0, a0 = res[(kind, "B0")]; s1, y1, a1 = res[(kind, "B1")]
            d = [correct_at_cov(s1[np.concatenate([cidx[c] for c in cs])], y1[np.concatenate([cidx[c] for c in cs])], 0.30)
                 - correct_at_cov(s0[np.concatenate([cidx[c] for c in cs])], y0[np.concatenate([cidx[c] for c in cs])], 0.30)
                 for cs in (rng.choice(clus, len(clus)) for _ in range(1000))]
            cv1, cv0 = summ(a1, y1), summ(a0, y0)
            ok_a = cv1["precision"] is not None and cv1["precision"] >= 0.84 and cv1["wilson"][0] >= 0.80
            out[tier][kind] = dict(B0=dict(cv_threshold=cv0, a30_rank=a30_rank(s0, y0), prec_cov50=summ(s0 >= np.sort(s0)[::-1][int(math.ceil(.5 * n)) - 1], y0)),
                                   B1=dict(cv_threshold=cv1, a30_rank=a30_rank(s1, y1), prec_cov50=summ(s1 >= np.sort(s1)[::-1][int(math.ceil(.5 * n)) - 1], y1)),
                                   paired_correct_diff_cov30=dict(mean=float(np.mean(d)), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]),
                                   rule=dict(a_criterion=bool(ok_a), b_cov_gain=cv1["coverage"] - cv0["coverage"], b_ok=bool(cv1["coverage"] - cv0["coverage"] >= 0.05),
                                             c_ok=bool(np.percentile(d, 2.5) > 0), lever=bool(kind == "lr" and ok_a and cv1["coverage"] - cv0["coverage"] >= 0.05 and np.percentile(d, 2.5) > 0)))
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    for t, v in out.items():
        print(t, "n", v["n_tasks"], "greedy %.3f oracle %.3f" % (v["greedy"], v["oracle_any"]))
        for kind in ("lr", "gbm"):
            for arm in ("B0", "B1"):
                c = v[kind][arm]["cv_threshold"]; r = v[kind][arm]["a30_rank"]; p5 = v[kind][arm]["prec_cov50"]
                print("  %-3s %s cv-thr cov %.2f prec %s %s | cov0.5 prec %.3f | A30 rank %s" % (kind, arm, c["coverage"], None if c["precision"] is None else round(c["precision"], 3),
                      None if c["wilson"][0] is None else [round(x, 3) for x in c["wilson"]], p5["precision"], None if r is None else (round(r["coverage"], 2), round(r["precision"], 3))))
            print("  %-3s paired dCorrect@cov0.30 %s rule %s" % (kind, v[kind]["paired_correct_diff_cov30"], v[kind]["rule"]))
