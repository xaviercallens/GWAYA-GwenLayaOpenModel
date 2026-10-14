#!/usr/bin/env python3
"""Lever U1: prompt-only accept/refuse classifier (A26). Reads dev_table.jsonl ONLY.

Step 1 (venv python, once):  lever_U_prompt_only.py --embed   -> caches frozen Qwen2.5-0.5B mean-pooled
                              prompt embeddings (no labels, no network).
Step 2 (system python3):     lever_U_prompt_only.py          -> grouped-CV evaluation, JSON result.
Prompt-only: no candidate, log-probs or gate result. router_p (Laya router, label-free) is a comparison
reference (F only). laya_p is NOT prompt-only (calibrator on generation features) and is reported only
as a labelled non-prompt reference.
"""
import json, math, re, sys, os
import numpy as np

ROOT = "/mnt/data/home/xavkal/gwaya-data/trust_loop"
DEV = f"{ROOT}/dev_table.jsonl"
EMB = f"{ROOT}/lever_U_prompt_emb_qwen25_05b.npz"
OUT = f"{ROOT}/lever_U_prompt_only.json"
TIERS = ["qwen3.5-4b-bf16", "qwen3.5-9b-bf16", "qwen3.5-2b-bf16"]
SEED = 0


def load():
    return [json.loads(l) for l in open(DEV)]


def tasks_of(rows):
    """unique (set, cluster) -> prompt, benchmark"""
    d = {}
    for r in rows:
        d.setdefault((r["set"], r["cluster"]), (r["prompt"], r["benchmark"]))
    return d


def embed():
    import torch
    from transformers import AutoTokenizer, AutoModel
    import glob
    path = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/*"))[0]
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    m = AutoModel.from_pretrained(path, local_files_only=True, torch_dtype=torch.float32).eval()
    T = tasks_of(load())
    keys = sorted(T)
    out = []
    with torch.no_grad():
        for k in keys:
            x = tok(T[k][0], return_tensors="pt", truncation=True, max_length=1024)
            h = m(**x).last_hidden_state[0]
            out.append(h.mean(0).numpy())
    np.savez(EMB, keys=np.array([f"{a}|{b}" for a, b in keys]), emb=np.array(out))
    print("embedded", len(keys))


# ---------- handcrafted prompt features ----------
def hand(p):
    body = p.split("```python block.", 1)[-1]
    L = body.splitlines()
    f = {}
    f["len_chars"] = math.log1p(len(body))
    f["n_lines"] = math.log1p(len(L))
    f["n_words"] = math.log1p(len(body.split()))
    f["n_def"] = body.count("def ")
    f["n_class"] = body.count("class ")
    f["n_import"] = len(re.findall(r"^\s*(?:import|from)\s", body, re.M))
    for lib in ["numpy", "pandas", "requests", "matplotlib", "sklearn", "scipy", "os", "re", "json", "random",
                "datetime", "collections", "itertools", "math", "heapq", "typing", "subprocess", "socket", "flask",
                "seaborn", "csv", "sqlite3", "urllib", "bs4", "tkinter", "PIL", "nltk"]:
        f["imp_" + lib] = int(bool(re.search(r"\b(?:import|from)\s+" + lib + r"\b", body)))
    f["n_example"] = len(re.findall(r"(?i)example|>>>|input:|output:", body))
    f["has_doctest"] = int(">>>" in body)
    f["n_assert"] = body.count("assert")
    f["n_constraint"] = len(re.findall(r"(?i)\b(must|should|raise|exception|constraint|at most|at least|exactly|only)\b", body))
    f["n_raise"] = len(re.findall(r"(?i)raise|exception|error", body))
    f["has_return_spec"] = int(bool(re.search(r"(?i)returns?:", body)))
    f["has_param_spec"] = int(bool(re.search(r"(?i)param|args?:", body)))
    f["n_bullets"] = len(re.findall(r"^\s*[-*]\s", body, re.M))
    f["n_digits"] = math.log1p(len(re.findall(r"\d+", body)))
    f["frac_upper"] = sum(c.isupper() for c in body) / max(1, len(body))
    f["frac_symbols"] = sum(not c.isalnum() and not c.isspace() for c in body) / max(1, len(body))
    f["n_params"] = max(0, (re.search(r"def\s+\w+\(([^)]*)\)", body) or [None, ""])[1].count(",") + 1) if re.search(r"def\s+\w+\(", body) else 0
    f["n_typehint"] = body.count("->") + len(re.findall(r":\s*(?:int|str|float|list|List|dict|Dict|bool)\b", body))
    f["leetcode"] = int("class Solution" in body)
    f["n_return_kw"] = body.count("return")
    f["mentions_file"] = int(bool(re.search(r"(?i)\bfile|path|directory|csv|json|url\b", body)))
    f["mentions_plot"] = int(bool(re.search(r"(?i)plot|figure|axes|chart|histogram", body)))
    f["mentions_df"] = int(bool(re.search(r"(?i)dataframe|pandas|\bdf\b", body)))
    f["mentions_random"] = int(bool(re.search(r"(?i)random|seed", body)))
    f["mentions_time"] = int(bool(re.search(r"(?i)time|date|timezone", body)))
    f["max_line_len"] = math.log1p(max([len(x) for x in L] + [0]))
    return f


# ---------- metrics ----------
def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(c - a) / d, (c + a) / d]


def bal(y, pred):
    y = np.asarray(y); pred = np.asarray(pred)
    P = (y == 1).sum(); N = (y == 0).sum()
    tp = ((pred == 1) & (y == 1)).sum(); tn = ((pred == 0) & (y == 0)).sum()
    tpr = tp / P if P else float("nan"); tnr = tn / N if N else float("nan")
    return dict(n=int(len(y)), acc=float((pred == y).mean()), acc_wilson=wilson(int((pred == y).sum()), len(y)),
                bal_acc=float((tpr + tnr) / 2), tpr=float(tpr), tnr=float(tnr),
                tpr_wilson=wilson(int(tp), int(P)), tnr_wilson=wilson(int(tn), int(N)),
                accept_rate=float(pred.mean()))


def cluster_boot_bal(y, pred, groups, B=1000, seed=SEED):
    rng = np.random.RandomState(seed)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    v = []
    for _ in range(B):
        s = rng.choice(ug, len(ug))
        ii = np.concatenate([idx[g] for g in s])
        yy, pp = y[ii], pred[ii]
        if (yy == 1).any() and (yy == 0).any():
            v.append(((pp[yy == 1] == 1).mean() + (pp[yy == 0] == 0).mean()) / 2)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def prec_report(y, pred):
    y = np.asarray(y); pred = np.asarray(pred)
    n = int(pred.sum()); k = int(y[pred == 1].sum())
    return dict(n_accepted=n, k_correct=k, precision=(k / n if n else None), wilson=wilson(k, n),
                coverage=float(pred.mean()))


def best_bal_thr(y, s):
    cand = np.unique(s)
    best, bt = -1, 0.5
    for t in cand:
        pr = (s >= t).astype(int)
        P = (y == 1).sum(); N = (y == 0).sum()
        b = (((pr == 1) & (y == 1)).sum() / max(1, P) + ((pr == 0) & (y == 0)).sum() / max(1, N)) / 2
        if b > best + 1e-12:
            best, bt = b, t
    return bt


def prec_thr(y, s, target=0.90, min_n=15):
    """lowest threshold (max coverage) whose accepted precision >= target with >= min_n accepted"""
    o = np.argsort(-s, kind="stable")
    ys = y[o]; ss = s[o]
    cs = np.cumsum(ys); n = np.arange(1, len(y) + 1)
    ok = np.where((cs / n >= target) & (n >= min_n))[0]
    if len(ok) == 0:
        return np.inf
    j = ok[-1]
    return ss[j]


# ---------- models ----------
def make_models(Xtxt_tr, Xtxt_te, Xh_tr, Xh_te, E_tr, E_te, ytr, name):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from scipy.sparse import hstack, csr_matrix
    if name == "prior":
        return np.full(len(Xh_te), ytr.mean())
    if name == "tfidf_lr":
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20000,
                            token_pattern=r"[A-Za-z_][A-Za-z_0-9]+")
        A = v.fit_transform(Xtxt_tr); B = v.transform(Xtxt_te)
        m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(A, ytr)
        return m.predict_proba(B)[:, 1]
    if name == "hand_gbm":
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=10,
                                           l2_regularization=1.0, random_state=SEED).fit(Xh_tr, ytr)
        return m.predict_proba(Xh_te)[:, 1]
    if name == "hand_lr":
        s = StandardScaler().fit(Xh_tr)
        m = LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced").fit(s.transform(Xh_tr), ytr)
        return m.predict_proba(s.transform(Xh_te))[:, 1]
    if name == "emb_lr":
        s = StandardScaler().fit(E_tr)
        m = LogisticRegression(C=0.003, max_iter=3000, class_weight="balanced").fit(s.transform(E_tr), ytr)
        return m.predict_proba(s.transform(E_te))[:, 1]
    if name == "ensemble":  # mean of tfidf_lr, hand_lr, emb_lr probabilities (rank-free average)
        ps = [make_models(Xtxt_tr, Xtxt_te, Xh_tr, Xh_te, E_tr, E_te, ytr, n) for n in ("tfidf_lr", "hand_lr", "emb_lr")]
        return np.mean(ps, axis=0)
    raise ValueError(name)


def folds(groups, k, seed):
    ug = np.unique(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(ug)
    a = {g: i % k for i, g in enumerate(ug)}
    return np.array([a[g] for g in groups])


def nested_predict(model, T, X, y, groups, outer_k=5, inner_k=4, seed=SEED, fixed_folds=None):
    """Return OOF score + thresholds chosen on inner-OOF of the training part (never on the test fold)."""
    n = len(y)
    s = np.zeros(n); t_bal = np.zeros(n); t_p90 = np.zeros(n)
    fo = fixed_folds if fixed_folds is not None else folds(groups, outer_k, seed)
    for f in np.unique(fo):
        tr = np.where(fo != f)[0]; te = np.where(fo == f)[0]
        args = lambda a, b: (T[a], T[b], X["hand"][a], X["hand"][b], X["emb"][a], X["emb"][b], y[a])
        s[te] = make_models(*args(tr, te), model)
        fi = folds(groups[tr], inner_k, seed + 1)
        inner = np.zeros(len(tr))
        for g in np.unique(fi):
            a = tr[fi != g]; b = tr[fi == g]
            if len(np.unique(y[a])) < 2:
                inner[fi == g] = y[a].mean(); continue
            inner[fi == g] = make_models(*args(a, b), model)
        t_bal[te] = best_bal_thr(y[tr], inner)
        t_p90[te] = prec_thr(y[tr], inner)
    return s, t_bal, t_p90


def main():
    from sklearn.metrics import roc_auc_score
    R = load()
    Tk = tasks_of(R)
    z = np.load(EMB, allow_pickle=True)
    emb = {tuple(k.split("|", 1)): e for k, e in zip(z["keys"], z["emb"])}
    result = dict(config=dict(seed=SEED, outer_folds=5, inner_folds=4, cv="grouped by source cluster",
                              note="thresholds chosen on inner OOF of the training folds only"), tiers={})
    for tier in TIERS:
        rows = [r for r in R if r["tier"] == tier]
        rows.sort(key=lambda r: (r["set"], r["cluster"]))
        y = np.array([r["correct"] for r in rows], int)
        groups = np.array([f'{r["set"]}|{r["cluster"]}' for r in rows])
        bench = np.array([r["benchmark"] for r in rows])
        T = np.array([r["prompt"] for r in rows], dtype=object)
        H = [hand(r["prompt"]) for r in rows]
        names = sorted(H[0])
        X = dict(hand=np.array([[h[k] for k in names] for h in H], float),
                 emb=np.array([emb[(r["set"], r["cluster"])] for r in rows]))
        router = np.array([np.nan if r["router_p"] is None else r["router_p"] for r in rows])
        laya = np.array([np.nan if r["laya_p"] is None else r["laya_p"] for r in rows])
        te = dict(n=len(y), base_rate=float(y.mean()), base_rate_wilson=wilson(int(y.sum()), len(y)),
                  per_benchmark={b: dict(n=int((bench == b).sum()), base_rate=float(y[bench == b].mean()))
                                 for b in np.unique(bench)}, models={})
        fo = folds(groups, 5, SEED)
        for model in ["prior", "tfidf_lr", "hand_lr", "hand_gbm", "emb_lr", "ensemble"]:
            s, tb, tp = nested_predict(model, T, X, y, groups, fixed_folds=fo)
            e = dict(auroc=float(roc_auc_score(y, s)) if model != "prior" else 0.5)
            pb = (s >= tb).astype(int)
            e["max_bal_acc_point"] = bal(y, pb)
            e["max_bal_acc_point"]["bal_acc_cluster_boot95"] = cluster_boot_bal(y, pb, groups)
            e["max_bal_acc_point"]["precision_of_accepted"] = prec_report(y, pb)
            pp = (s >= tp).astype(int)
            e["precision90_point"] = prec_report(y, pp)
            e["precision90_point"]["bal"] = bal(y, pp)
            e["per_benchmark_at_max_bal"] = {b: bal(y[bench == b], pb[bench == b]) for b in np.unique(bench)
                                             if len(np.unique(y[bench == b])) == 2}
            # oracle (non-nested, optimistic) upper bound of what any threshold could do on these scores
            to = best_bal_thr(y, s)
            e["optimistic_best_thr_bal_acc"] = bal(y, (s >= to).astype(int))["bal_acc"]
            # precision-coverage curve on OOF scores
            o = np.argsort(-s, kind="stable")
            curve = []
            for cov in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                k = int(math.ceil(cov * len(y)))
                kk = int(y[o[:k]].sum())
                curve.append(dict(coverage=cov, precision=kk / k, wilson=wilson(kk, k)))
            e["precision_coverage_curve"] = curve
            te["models"][model] = e
        # router_p reference (F only, threshold chosen by inner CV the same way, within F tasks only)
        mF = ~np.isnan(router)
        if mF.sum() > 30:
            yF, sF, gF = y[mF], router[mF], groups[mF]
            foF = folds(gF, 5, SEED)
            tb = np.zeros(len(yF))
            for f in np.unique(foF):
                tr = foF != f
                tb[foF == f] = best_bal_thr(yF[tr], sF[tr])
            pbF = (sF >= tb).astype(int)
            te["router_p_reference_F_only"] = dict(auroc=float(roc_auc_score(yF, sF)), point=bal(yF, pbF),
                                                   bal_acc_cluster_boot95=cluster_boot_bal(yF, pbF, gF))
            # same-subset comparison for our best prompt model: ensemble OOF restricted to F
            # (computed from the full OOF above is fair: same grouped folds)
        mL = ~np.isnan(laya)
        if mL.sum() > 30:
            te["laya_p_NOT_prompt_only_reference"] = dict(auroc=float(roc_auc_score(y[mL], laya[mL])),
                                                          note="calibrator uses generation features; not a U input")
        # leave-one-benchmark-out transfer for the ensemble (closest proxy to unseen HumanEval+/MBPP+)
        lobo = {}
        for b in np.unique(bench):
            trm = bench != b; tem = bench == b
            if len(np.unique(y[tem])) < 2:
                continue
            a = np.where(trm)[0]; c = np.where(tem)[0]
            s = make_models(T[a], T[c], X["hand"][a], X["hand"][c], X["emb"][a], X["emb"][c], y[a], "ensemble")
            fi = folds(groups[a], 4, SEED + 1)
            inner = np.zeros(len(a))
            for g in np.unique(fi):
                aa = a[fi != g]; bb = a[fi == g]
                inner[fi == g] = make_models(T[aa], T[bb], X["hand"][aa], X["hand"][bb], X["emb"][aa], X["emb"][bb], y[aa], "ensemble")
            thr = best_bal_thr(y[a], inner)
            lobo[b] = dict(auroc=float(roc_auc_score(y[c], s)), point=bal(y[c], (s >= thr).astype(int)),
                           base_rate=float(y[c].mean()))
        te["leave_one_benchmark_out_ensemble"] = lobo
        result["tiers"][tier] = te
        m = te["models"]
        print(tier, "n", len(y), "base %.3f" % y.mean())
        for k, e in m.items():
            b = e["max_bal_acc_point"]
            print("  %-9s auroc %.3f  bal %.3f [%.3f,%.3f] acc %.3f  | p90pt prec %s cov %.2f" % (
                k, e["auroc"], b["bal_acc"], *b["bal_acc_cluster_boot95"], b["acc"],
                None if e["precision90_point"]["precision"] is None else round(e["precision90_point"]["precision"], 3),
                e["precision90_point"]["coverage"]))
        print("  lobo", {b: (round(v["auroc"], 3), round(v["point"]["bal_acc"], 3)) for b, v in lobo.items()})
        if "router_p_reference_F_only" in te:
            print("  router F:", round(te["router_p_reference_F_only"]["auroc"], 3), round(te["router_p_reference_F_only"]["point"]["bal_acc"], 3))
    json.dump(result, open(OUT, "w"), indent=1, default=float)


if __name__ == "__main__":
    if "--embed" in sys.argv:
        embed()
    else:
        main()
