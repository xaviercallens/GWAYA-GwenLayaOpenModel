#!/usr/bin/env python3
"""E2 with hard confusers (dev only): non-Python code as negatives for the Python expert.

Negatives added to E2: Strandset-Rust-v1 test split (Apache-2.0, Rust code tasks) and Offlin33er/solidity-audit-instruct eval
split (MIT, Solidity audit prompts). Arms: keyword; TF-IDF+LR trained on Python-vs-GSM8K only (the E2 arm, to test the
style-artifact hypothesis); TF-IDF+LR trained on Python vs GSM8K+Rust+Solidity (half-split by hash). Not the sealed read.
"""
from __future__ import annotations
import argparse, json, sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from scripts.routing.build_python_stratum import MARK, hits
from scripts.routing.e2_python_dev import half
from scripts.routing.route_eval import wilson

CAP = 1500
POS = ("py", "pycode")  # Python as English description, and Python as code (complete_prompt)


def rust_text(d) -> str:
    return " ".join(str(v) for v in (d.values() if isinstance(d, dict) else [d]))[:CAP]


def sol_text(path: str) -> list[str]:
    out = []
    for line in open(path):
        m = json.loads(line)["messages"]
        out.append(" ".join(x["content"] for x in m if x["role"] == "user")[:CAP])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    for n in ("bcb", "gsm8k", "rust", "sol"):
        ap.add_argument("--" + n, required=True)
    ap.add_argument("--out", default="results/routing/e2_confusers.json")
    a = ap.parse_args()
    b = pd.read_parquet(a.bcb)
    pos = [(f"b{t}", x.split(MARK)[0].strip(), "py") for t, x in zip(b["task_id"], b["instruct_prompt"])]
    pc = [(f"c{t}", x[:CAP], "pycode") for t, x in zip(b["task_id"], b["complete_prompt"])]
    gs = [(f"g{i}", q, "gsm8k") for i, q in enumerate(pd.read_parquet(a.gsm8k)["question"])]
    ru = [(f"r{i}", rust_text(d), "rust") for i, d in enumerate(pd.read_parquet(a.rust)["input_data"])]
    so = [(f"s{i}", t, "solidity") for i, t in enumerate(sol_text(a.sol))]
    rows = pos + pc + gs + ru + so
    tr = [r for r in rows if half(r[0]) == 1]
    dv = [r for r in rows if half(r[0]) == 0]
    mk = lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), LogisticRegression(C=4, max_iter=2000, class_weight="balanced"))
    easy = mk().fit([r[1] for r in tr if r[2] in ("py", "pycode", "gsm8k")], [r[2] in POS for r in tr if r[2] in ("py", "pycode", "gsm8k")])
    hard = mk().fit([r[1] for r in tr], [r[2] in POS for r in tr])
    X = [r[1] for r in dv]
    arms = {"keyword": [hits(t) > 0 for t in X], "lr_py_vs_gsm8k": list(easy.predict(X)), "lr_py_vs_all": list(hard.predict(X))}
    res = {"n_dev": {s: sum(r[2] == s for r in dv) for s in ("py", "pycode", "gsm8k", "rust", "solidity")}}
    for name, pred in arms.items():
        res[name] = {}
        for s in ("py", "pycode", "gsm8k", "rust", "solidity"):
            idx = [i for i, r in enumerate(dv) if r[2] == s]
            k = sum((bool(pred[i]) if s in POS else not pred[i]) for i in idx)
            res[name][("recall_" + s if s in POS else "reject_" + s)] = [k, len(idx), round(k / len(idx), 4), wilson(k, len(idx))]
    json.dump(res, open(a.out, "w"), indent=1)
    for n, v in res.items():
        print(n, v if n == "n_dev" else {k: f"{x[0]}/{x[1]}" for k, x in v.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
