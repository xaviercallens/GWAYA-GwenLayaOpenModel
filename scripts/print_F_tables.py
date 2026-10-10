#!/usr/bin/env python
"""Markdown tables from numbers_F_pm.json / numbers_F_lean.json (output of scripts/analyze_F.py) for docs/RESULTS_F.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

TIERS = ("qwen3.5-2b-bf16", "qwen3.5-4b-bf16", "qwen3.5-9b-bf16")
SHORT = {"qwen3.5-2b-bf16": "2B", "qwen3.5-4b-bf16": "4B", "qwen3.5-9b-bf16": "9B"}


def f(ci: dict | None, d: int = 3) -> str:
    if not ci or ci.get("point") is None:
        return "n/a"
    return f"{ci['point']:.{d}f} [{ci['lo']:.{d}f}, {ci['hi']:.{d}f}]" if ci.get("lo") is not None else f"{ci['point']:.{d}f}"


def table(res: dict, metrics: list[str], slices: list[str], title: str) -> str:
    out = [f"**{title}**", "", "| slice | n | tier | " + " | ".join(metrics) + " |", "|---|---|---|" + "---|" * len(metrics)]
    for s in slices:
        for t in TIERS:
            d = res["per_slice"][s][t]
            out.append(f"| {s} | {int(d['n']['point'])} | {SHORT[t]} | " + " | ".join(f(d[m]) for m in metrics) + " |")
    return "\n".join(out)


def main() -> None:
    root = Path(sys.argv[1])
    res = json.loads((root / "numbers_F_pm.json").read_text())
    sl = list(res["per_slice"])
    print(table(res, ["acc", "truncated"], sl, "Accuracy (hidden suite / gold answer) and truncation"))
    print()
    print(table(res, ["gate_coverage", "gate_precision", "gate_recall", "gate_false_accept", "gate_confident_wrong_rate",
                      "logprob_matched_confident_wrong_rate"], sl, "Gate vs hidden correctness"))
    print()
    print(table(res, ["auroc_mean_logprob", "aurc_mean_logprob", "auroc_min_logprob", "aurc_min_logprob", "auroc_gate_binary"], sl,
                "Raw log-prob signals and gate as scores for correctness"))
    c = res["cascade"]["ci"]
    print()
    print("**Cascade 2B -> 4B -> 9B vs always-9B** (cost = chip-seconds; 'warm' = compile warm-up chunk included)\n")
    ms = ["acc_cascade3", "acc_always_9b", "d_acc_cascade3_minus_9b", "oracle_acc", "cost_ratio_cascade3_over_9b",
          "cost_ratio_cascade3_over_9b_with_program", "cost_per_correct_cascade3", "cost_per_correct_always_9b", "escalated_share",
          "answered_share", "confident_wrong_rate_cascade3"]
    print("| slice | reading | " + " | ".join(ms) + " |\n|---|---|" + "---|" * len(ms))
    seen = []
    for k in c:
        s, rd, _ = k.split("|")
        if (s, rd) not in seen:
            seen.append((s, rd))
    for s, rd in seen:
        print(f"| {s} | {rd} | " + " | ".join(f(c.get(f"{s}|{rd}|{m}")) for m in ms) + " |")
    print("\nMcNemar (pooled python+math, cascade vs 9B):", json.dumps(res["cascade"]["mcnemar_vs_9b_pooled"]))
    print("Total chip-seconds:", json.dumps(res["cascade"]["total_cost_chip_s"]))
    lp = root / "numbers_F_lean.json"
    if lp.exists():
        L = json.loads(lp.read_text())["lean"]
        print("\n**Lean (miniF2F-valid, n=%d, greedy)**\n" % L["n_tasks"])
        print("| tier | proved | proof rate | truncated | chip-s per proof | USD total |\n|---|---|---|---|---|---|")
        for t in TIERS:
            cps = L["chip_seconds_per_proof"][t]
            print(f"| {SHORT[t]} | {L['proved_n'][t]} | {f(L['ci'][t + '|proof_rate'])} | {f(L['ci'][t + '|truncated'])} | "
                  f"{'n/a' if cps is None else round(cps, 1)} | {L['usd_total'][t]:.3f} |")
        print("\nany-tier proof rate:", f(L["ci"]["any_tier_proof_rate"]), "union", L["union_n"], "overlap", L["overlap"],
              "match types", L["verified_match_type"], "only-largest", L["only_largest_n"],
              "small-tier proves not proved by 9B", L["smaller_tier_proves_n_not_proved_by_9b"])


if __name__ == "__main__":
    main()
