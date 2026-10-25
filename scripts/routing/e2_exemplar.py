#!/usr/bin/env python3
"""E2 exemplar-retrieval arm (dev only), same held-out-source split as e2_heldout_source.py.

Three subcommands, so the encoder can run in a torch environment and the scoring in the sklearn one:
  texts  --out DIR           write train/test texts and labels (JSON)
  embed  --dir DIR --model M write embeddings (npy) with a small CPU encoder (BAAI/bge-small-en-v1.5, MIT), mean of CLS, L2-normalised
  score  --dir DIR           top-k cosine retrieval vote and LR on embeddings vs the keyword arm, on HumanEval+ docstrings and stubs
Not the sealed read.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path


def cmd_texts(a) -> int:
    import pandas as pd
    from scripts.routing.build_python_stratum import MARK
    from scripts.routing.e2_confusers import CAP, rust_text, sol_text
    from scripts.routing.e2_heldout_source import docstring
    b = pd.read_parquet(a.bcb)
    X, y = [], []
    for x in b["instruct_prompt"]: X.append(x.split(MARK)[0].strip()); y.append(1)
    for x in b["complete_prompt"]: X.append(x[:CAP]); y.append(1)
    for q in pd.read_parquet(a.gsm8k)["question"]: X.append(q); y.append(0)
    for d in pd.read_parquet(a.rust)["input_data"]: X.append(rust_text(d)); y.append(0)
    for t in sol_text(a.sol): X.append(t); y.append(0)
    h = pd.read_parquet(a.heplus)["prompt"].tolist()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump({"train": X, "y": y, "doc": [docstring(p) for p in h], "stub": h}, open(Path(a.out) / "texts.json", "w"))
    print(len(X), len(h))
    return 0


def cmd_embed(a) -> int:
    import numpy as np, torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(8)
    t = json.load(open(Path(a.dir) / "texts.json"))
    tok, m = AutoTokenizer.from_pretrained(a.model), AutoModel.from_pretrained(a.model).eval()
    for k in ("train", "doc", "stub"):
        out = []
        for i in range(0, len(t[k]), 32):
            enc = tok(t[k][i:i + 32], padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                e = m(**enc).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(e, dim=-1).numpy())
        np.save(Path(a.dir) / f"{k}.npy", np.concatenate(out))
    return 0


def cmd_score(a) -> int:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from scripts.routing.build_python_stratum import hits
    from scripts.routing.route_eval import wilson
    d = Path(a.dir)
    t = json.load(open(d / "texts.json"))
    Xtr, y = np.load(d / "train.npy"), np.array(t["y"])
    lr = LogisticRegression(C=10, max_iter=3000, class_weight="balanced").fit(Xtr, y)
    res = {"n_train": len(y)}
    for view in ("doc", "stub"):
        E = np.load(d / f"{view}.npy")
        sims = E @ Xtr.T
        res[view] = {"keyword": sum(hits(x) > 0 for x in t[view])}
        for k in (1, 5, 15):
            idx = np.argsort(-sims, axis=1)[:, :k]
            vote = (y[idx].mean(axis=1) > 0.5)
            res[view][f"knn{k}"] = int(vote.sum())
        res[view]["lr_emb"] = int(lr.predict(E).sum())
        n = len(E)
        res[view] = {kk: [v, n, wilson(v, n)] for kk, v in res[view].items()}
    # off-domain check: how often does each arm claim a GSM8K/Rust/Solidity train row when held out is not available -> leave-one-out kNN
    neg = np.where(y == 0)[0]
    s = Xtr[neg] @ Xtr.T
    s[np.arange(len(neg)), neg] = -1
    for k in (1, 5, 15):
        idx = np.argsort(-s, axis=1)[:, :k]
        res[f"loo_negative_false_route_knn{k}"] = [int((y[idx].mean(axis=1) > 0.5).sum()), len(neg)]
    # leave-one-source-out: drop a whole negative source from the exemplars, then route its own prompts (held-out off-domain)
    bounds = {"gsm8k": (2280, 3599), "rust": (3599, 3824), "solidity": (3824, len(y))}
    res["leave_source_out_false_route"] = {}
    for name, (lo, hi) in bounds.items():
        keep = np.ones(len(y), bool); keep[lo:hi] = False
        s2 = Xtr[lo:hi] @ Xtr[keep].T
        yk = y[keep]
        res["leave_source_out_false_route"][name] = {}
        for k in (1, 5, 15):
            idx = np.argsort(-s2, axis=1)[:, :k]
            v = int((yk[idx].mean(axis=1) > 0.5).sum())
            res["leave_source_out_false_route"][name][f"knn{k}"] = [v, hi - lo, wilson(v, hi - lo)]
    # similarity floor: tau = 5th percentile of each Python exemplar's best similarity to OTHER Python exemplars
    # (the same task's description/code pair, indices i and i+1140, is masked). Uses Python training rows only; no negative or test data.
    py = np.where(y == 1)[0]
    sp = Xtr[py] @ Xtr[py].T
    n_task = len(py) // 2
    for i in range(len(py)):
        sp[i, i] = sp[i, (i + n_task) % len(py)] = -1
    tau = float(np.percentile(sp.max(axis=1), 5))
    res["floor_tau"] = round(tau, 4)
    floor = {"doc": {}, "stub": {}, "leave_source_out": {}}
    pyidx = np.where(y == 1)[0]
    for view in ("doc", "stub"):
        E = np.load(d / f"{view}.npy")
        sm = E @ Xtr.T
        top = np.argsort(-sm, axis=1)[:, :15]
        vote = y[top].mean(axis=1) > 0.5
        near = (E @ Xtr[pyidx].T).max(axis=1) >= tau
        floor[view] = {"knn15": int(vote.sum()), "knn15_with_floor": int((vote & near).sum()), "n": len(E)}
    for name, (lo, hi) in bounds.items():
        keep = np.ones(len(y), bool); keep[lo:hi] = False
        s2 = Xtr[lo:hi] @ Xtr[keep].T
        top = np.argsort(-s2, axis=1)[:, :15]
        vote = y[keep][top].mean(axis=1) > 0.5
        near = (Xtr[lo:hi] @ Xtr[pyidx].T).max(axis=1) >= tau
        floor["leave_source_out"][name] = {"knn15": int(vote.sum()), "knn15_with_floor": int((vote & near).sum()), "n": hi - lo}
    res["floor"] = floor
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("texts")
    for n in ("bcb", "gsm8k", "rust", "sol", "heplus", "out"):
        p.add_argument("--" + n, required=True)
    p = sub.add_parser("embed"); p.add_argument("--dir", required=True); p.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    p = sub.add_parser("score"); p.add_argument("--dir", required=True); p.add_argument("--out", default="results/routing/e2_exemplar.json")
    a = ap.parse_args()
    return {"texts": cmd_texts, "embed": cmd_embed, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
