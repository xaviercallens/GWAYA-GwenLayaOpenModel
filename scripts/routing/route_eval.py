#!/usr/bin/env python3
"""Evaluation harness for HiveMind prompt-level expert routing (docs/PROPOSAL_OPENLAYA_HIVEMIND_ROUTING.md, part B bar).

A router is ``module:function`` taking ``(prompt)`` and returning ``(domains, weights)`` as ``route_with_serving_experts`` /
``Season3DomainExpertRegistry().route_domain_query`` do; ``([], {})`` means abstain (use the base model). Rows are JSONL
``{"prompt": str, "domain": str | "none", "source": str}``. Reported, per the pre-registered bar:
  top1_in_domain   share of in-domain prompts whose best expert is the labelled domain (Wilson 95%)
  abstain_off      share of off-domain ("none") prompts that abstain (Wilson 95%)
  false_route_off  share of off-domain prompts sent to some expert
  abstain_in       share of in-domain prompts that abstain (lost coverage)
The paired comparison of two routers on the same rows uses an exact McNemar test (``compare``).
The fixture ``routing_fixture_synthetic.jsonl`` is hand-written for regression/smoke use: it is NOT evidence about routing
quality (same author as the fix, small, no Solidity/biomedical data from real sources).
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p, d, c = k / n, 1 + z * z / n, k / n + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - h) / d, 4), round((c + h) / d, 4)


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    for r in rows:
        if not isinstance(r.get("prompt"), str) or not isinstance(r.get("domain"), str):
            raise ValueError(f"bad row: {r}")
    return rows


def predictions(router, rows: list[dict]) -> list[str]:
    """Best expert per prompt, or "none" when the router abstains."""
    out = []
    for r in rows:
        doms, _ = router(r["prompt"])
        out.append(doms[0] if doms else "none")
    return out


def evaluate(pred: list[str], rows: list[dict]) -> dict:
    inn = [(p, r) for p, r in zip(pred, rows) if r["domain"] != "none"]
    off = [(p, r) for p, r in zip(pred, rows) if r["domain"] == "none"]
    t1 = sum(p == r["domain"] for p, r in inn)
    ab_off = sum(p == "none" for p, _ in off)
    ab_in = sum(p == "none" for p, _ in inn)
    return {
        "n_in_domain": len(inn), "n_off_domain": len(off),
        "top1_in_domain": [t1, len(inn), round(t1 / len(inn), 4) if inn else None, wilson(t1, len(inn))],
        "abstain_off": [ab_off, len(off), round(ab_off / len(off), 4) if off else None, wilson(ab_off, len(off))],
        "false_route_off": [len(off) - ab_off, len(off)],
        "abstain_in": [ab_in, len(inn)],
    }


def compare(pred_a: list[str], pred_b: list[str], rows: list[dict]) -> dict:
    """Paired exact McNemar on 'router output is right' (right = labelled domain, or abstain on an off-domain prompt)."""
    ok = lambda p, r: p == r["domain"]  # noqa: E731  (domain "none" == abstain)
    b = sum(ok(pb, r) and not ok(pa, r) for pa, pb, r in zip(pred_a, pred_b, rows))
    c = sum(ok(pa, r) and not ok(pb, r) for pa, pb, r in zip(pred_a, pred_b, rows))
    return {"b_only_right": b, "a_only_right": c, "mcnemar_p": round(mcnemar_exact(b, c), 4)}


def resolve(spec: str):
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", required=True, help="module:function, e.g. hivemind.core.serving_experts:route_with_serving_experts")
    ap.add_argument("--rows", default=str(Path(__file__).with_name("routing_fixture_synthetic.jsonl")))
    ap.add_argument("--dump-preds", default=None, help="write predictions as JSON (for a later paired compare)")
    a = ap.parse_args()
    rows = load_rows(Path(a.rows))
    pred = predictions(resolve(a.router), rows)
    if a.dump_preds:
        Path(a.dump_preds).write_text(json.dumps(pred))
    print(json.dumps(evaluate(pred, rows), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
