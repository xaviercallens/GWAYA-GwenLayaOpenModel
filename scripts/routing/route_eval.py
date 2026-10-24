#!/usr/bin/env python3
"""Evaluation harness for HiveMind prompt-level expert routing (docs/PROPOSAL_OPENLAYA_HIVEMIND_ROUTING.md, part B bar).

Optional ``"tier"`` (int, 0 = cheapest) marks the cheapest tier that solves the prompt, judged from tier solve rates and never
from prompt length; ``evaluate_tiers`` scores a tier router against it. A router is ``module:function`` taking ``(prompt)`` and returning ``(domains, weights)`` as ``route_with_serving_experts`` /
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


def cascade_cost(pred: int, label: int, costs: list[float]) -> float:
    """Cost of serving a request whose cheapest sufficient tier is ``label`` when the router starts at ``pred``.

    Starting too low pays every failed attempt on the way up (escalation on a failed check); starting at or above the label
    pays the starting tier only. ``costs[t]`` is the cost of one attempt at tier ``t`` (tokens, generations or seconds).
    """
    return float(sum(costs[pred:label + 1])) if pred < label else float(costs[pred])


def evaluate_tiers(pred: list[int], rows: list[dict], costs: list[float]) -> dict:
    """Complexity-tier routing on rows that carry an integer ``tier`` label (cheapest tier that solves the prompt).

    under = routed below the label (needs escalation), over = routed above it (waste). Costs are compared with the oracle
    (always the label) and with always-top; the oracle is the ceiling for any tier router on these rows.
    """
    pairs = [(p, r["tier"]) for p, r in zip(pred, rows) if isinstance(r.get("tier"), int)]
    n = len(pairs)
    if not n:
        return {"n_tiered": 0}
    top = len(costs) - 1
    cost = sum(cascade_cost(p, t, costs) for p, t in pairs) / n
    return {
        "n_tiered": n,
        "exact": round(sum(p == t for p, t in pairs) / n, 4),
        "under": round(sum(p < t for p, t in pairs) / n, 4),
        "over": round(sum(p > t for p, t in pairs) / n, 4),
        "mean_cost": round(cost, 4),
        "oracle_cost": round(sum(costs[t] for _, t in pairs) / n, 4),
        "always_top_cost": round(float(costs[top]), 4),
        "always_lowest_cost": round(sum(cascade_cost(0, t, costs) for _, t in pairs) / n, 4),
    }


BAND = (0.3, 0.7)  # registered before any sealed read (docs/PROPOSAL_ROUTING_DATASET_AND_STRATEGY.md, bar 5); do not tune on test


def discrimination(preds_by_arm: dict[str, list[str]], rows: list[dict], strata: dict[str, list[bool]],
                   band: tuple[float, float] = BAND) -> dict:
    """Computed band check per stratum: pooled in-domain top-1 over ALL arms; outside ``band`` the stratum cannot rank the arms.

    ``strata[name][i]`` says whether row i belongs to the stratum (e.g. the keyword-blind rows). Returned as data so the check
    is asserted by code, not left to a reader.
    """
    out = {}
    for name, mask in strata.items():
        hit = tot = 0
        for pred in preds_by_arm.values():
            for p, r, m in zip(pred, rows, mask):
                if m and r["domain"] != "none":
                    tot += 1
                    hit += p == r["domain"]
        pooled = hit / tot if tot else None
        out[name] = {"pooled_top1": None if pooled is None else round(pooled, 4), "n": tot,
                     "in_band": pooled is not None and band[0] <= pooled <= band[1], "band": list(band)}
    return out


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
