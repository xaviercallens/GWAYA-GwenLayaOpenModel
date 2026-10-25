#!/usr/bin/env python3
"""E2 held-out-source check (dev only): train on BigCodeBench + GSM8K + Rust + Solidity, test on HumanEval+ (Apache-2.0).

HumanEval+ is a different Python source, so a learned arm cannot lean on BigCodeBench-specific style. Positives: the docstring text
alone ("doc", keyword-blind where possible) and the full code stub ("stub"). The train set is the whole of the four E2 sources (no
half split needed: the test source is disjoint). Compared with the keyword arm. Not the sealed read.
"""
from __future__ import annotations
import argparse, ast, json, sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from scripts.routing.build_python_stratum import MARK, hits
from scripts.routing.e2_confusers import CAP, POS, rust_text, sol_text
from scripts.routing.route_eval import wilson


def docstring(prompt: str) -> str:
    """Docstring of the last function in a HumanEval stub, else the whole prompt."""
    try:
        fns = [n for n in ast.walk(ast.parse(prompt)) if isinstance(n, ast.FunctionDef)]
        d = ast.get_docstring(fns[-1]) if fns else None
        return d or prompt
    except SyntaxError:
        return prompt


def main() -> int:
    ap = argparse.ArgumentParser()
    for n in ("bcb", "gsm8k", "rust", "sol", "heplus"):
        ap.add_argument("--" + n, required=True)
    ap.add_argument("--out", default="results/routing/e2_heldout_source.json")
    a = ap.parse_args()
    b = pd.read_parquet(a.bcb)
    X, y = [], []
    for x in b["instruct_prompt"]: X.append(x.split(MARK)[0].strip()); y.append(1)
    for x in b["complete_prompt"]: X.append(x[:CAP]); y.append(1)
    for q in pd.read_parquet(a.gsm8k)["question"]: X.append(q); y.append(0)
    for d in pd.read_parquet(a.rust)["input_data"]: X.append(rust_text(d)); y.append(0)
    for t in sol_text(a.sol): X.append(t); y.append(0)
    clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), LogisticRegression(C=4, max_iter=2000, class_weight="balanced"))
    clf.fit(X, y)
    h = pd.read_parquet(a.heplus)["prompt"].tolist()
    views = {"doc": [docstring(p) for p in h], "stub": h}
    res = {"n_train": len(X), "n_test": len(h)}
    for v, T in views.items():
        kw = sum(hits(t) > 0 for t in T)
        lr = int(sum(clf.predict(T)))
        res[v] = {"keyword_recall": [kw, len(T), wilson(kw, len(T))], "lr_recall": [lr, len(T), wilson(lr, len(T))]}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
