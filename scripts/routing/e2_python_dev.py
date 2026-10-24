#!/usr/bin/env python3
"""E2 on dev only (Python first): keyword arm vs TF-IDF + logistic regression, Python-vs-rest, split by a fixed hash.

Positives: BigCodeBench descriptions (scaffold stripped). Negatives: GSM8K test questions. Train on the odd-hash half, report on
the even-hash half. LIMITATION: these negatives are plain-language maths, so any text classifier separates them easily; this run
does not include hard confusers (Rust/Solidity/Lean code, other-language prompts) and is NOT the sealed read.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from scripts.routing.build_python_stratum import MARK, hits
from scripts.routing.route_eval import mcnemar_exact, wilson


def half(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bcb", required=True)
    ap.add_argument("--gsm8k", required=True)
    ap.add_argument("--out", default="results/routing/e2_python_dev.json")
    a = ap.parse_args()
    b = pd.read_parquet(a.bcb)
    pos = [(f"b{t}", x.split(MARK)[0].strip()) for t, x in zip(b["task_id"], b["instruct_prompt"])]
    neg = [(f"g{i}", q) for i, q in enumerate(pd.read_parquet(a.gsm8k)["question"])]
    rows = [(k, t, 1) for k, t in pos] + [(k, t, 0) for k, t in neg]
    tr = [r for r in rows if half(r[0]) == 1]
    dv = [r for r in rows if half(r[0]) == 0]
    clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), LogisticRegression(C=4, max_iter=2000, class_weight="balanced"))
    clf.fit([r[1] for r in tr], [r[2] for r in tr])
    kw = [int(hits(r[1]) > 0) for r in dv]
    lr = [int(p) for p in clf.predict([r[1] for r in dv])]
    y = [r[2] for r in dv]
    blind = [r[2] == 1 and hits(r[1]) == 0 for r in dv]

    def rate(pred, sel):
        k = sum(p == t for p, t, s in zip(pred, y, sel) if s); n = sum(sel)
        return [k, n, round(k / n, 4) if n else None, wilson(k, n)]

    allm, posm, negm = [True] * len(dv), [t == 1 for t in y], [t == 0 for t in y]
    b_only = sum(l == t and k != t for k, l, t in zip(kw, lr, y))
    a_only = sum(k == t and l != t for k, l, t in zip(kw, lr, y))
    res = {"n_train": len(tr), "n_dev": len(dv),
           "keyword": {"python_recall": rate(kw, posm), "blind_recall": rate(kw, blind), "negatives_correct": rate(kw, negm)},
           "tfidf_lr": {"python_recall": rate(lr, posm), "blind_recall": rate(lr, blind), "negatives_correct": rate(lr, negm)},
           "paired_overall": {"lr_only_right": b_only, "keyword_only_right": a_only, "mcnemar_p": round(mcnemar_exact(b_only, a_only), 6)}}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
