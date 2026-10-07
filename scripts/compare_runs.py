#!/usr/bin/env python3
"""Compare two GWAYA benchmark runs and report deltas.

Usage:
  compare_runs.py results/run1.json results/run2.json
  compare_runs.py --baseline cpu_results.json --compare gpu_results.json

The script reads two results.json files and prints:
  - Pass rate delta (percentage points)
  - Average tokens/second delta
  - VRAM usage delta (if available)
  - Gate rule status (e.g., "PASS: delta > -5%")
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def load_results(path: str | Path) -> dict[str, Any]:
    """Load and validate a results.json file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return data


def extract_metrics(results: dict[str, Any]) -> dict[str, float]:
    """Extract key metrics from results."""
    metrics = {}

    # Pass rate: percentage of solved problems
    rows = results.get("rows", [])
    if rows:
        solved = sum(1 for row in rows if row.get("solved", False))
        metrics["pass_rate"] = 100.0 * solved / len(rows)
        metrics["total_problems"] = len(rows)
    else:
        metrics["pass_rate"] = 0.0
        metrics["total_problems"] = 0

    # Tokens per second (aggregate from rows)
    tok_per_sec_values = []
    for row in rows:
        tok_per_sec = row.get("tok_per_sec")
        if tok_per_sec is not None:
            tok_per_sec_values.append(tok_per_sec)

    if tok_per_sec_values:
        metrics["avg_tok_per_sec"] = statistics.mean(tok_per_sec_values)
        metrics["min_tok_per_sec"] = min(tok_per_sec_values)
        metrics["max_tok_per_sec"] = max(tok_per_sec_values)

    # VRAM usage (if available)
    vram_gb_values = []
    for row in rows:
        vram_gb = row.get("peak_vram_gb")
        if vram_gb is not None:
            vram_gb_values.append(vram_gb)

    if vram_gb_values:
        metrics["avg_vram_gb"] = statistics.mean(vram_gb_values)
        metrics["max_vram_gb"] = max(vram_gb_values)

    # Summary from metadata
    summary = results.get("summary", {})
    if "total_duration_sec" in summary:
        metrics["total_duration_sec"] = summary["total_duration_sec"]
    if "wall_time_sec" in summary:
        metrics["wall_time_sec"] = summary["wall_time_sec"]

    return metrics


def compare_runs(
    baseline: dict[str, Any], compare: dict[str, Any], gate_rule: str | None = None
) -> dict[str, Any]:
    """Compare two runs and compute deltas."""
    base_metrics = extract_metrics(baseline)
    comp_metrics = extract_metrics(compare)

    deltas = {}

    # Pass rate delta (percentage points)
    if "pass_rate" in base_metrics and "pass_rate" in comp_metrics:
        base_pr = base_metrics["pass_rate"]
        comp_pr = comp_metrics["pass_rate"]
        deltas["pass_rate_pct"] = comp_pr - base_pr
        deltas["pass_rate_pct_relative"] = (
            ((comp_pr - base_pr) / base_pr * 100) if base_pr > 0 else 0
        )

    # Tokens/sec delta
    if "avg_tok_per_sec" in base_metrics and "avg_tok_per_sec" in comp_metrics:
        base_tps = base_metrics["avg_tok_per_sec"]
        comp_tps = comp_metrics["avg_tok_per_sec"]
        deltas["tok_per_sec_delta"] = comp_tps - base_tps
        deltas["tok_per_sec_relative"] = (
            ((comp_tps - base_tps) / base_tps * 100) if base_tps > 0 else 0
        )

    # VRAM delta
    if "max_vram_gb" in base_metrics and "max_vram_gb" in comp_metrics:
        base_vram = base_metrics["max_vram_gb"]
        comp_vram = comp_metrics["max_vram_gb"]
        deltas["vram_gb_delta"] = comp_vram - base_vram
        deltas["vram_gb_relative"] = (
            ((comp_vram - base_vram) / base_vram * 100) if base_vram > 0 else 0
        )

    # Apply gate rule if provided
    gate_status = "PASS"
    gate_reason = "no rule applied"
    if gate_rule:
        # Parse rule like: "pass_rate_pct >= -5" or "tok_per_sec_delta > 0"
        try:
            # Simple rule parser
            if ">=" in gate_rule:
                metric, threshold = gate_rule.split(">=")
                metric = metric.strip()
                threshold = float(threshold.strip())
                if metric in deltas and deltas[metric] < threshold:
                    gate_status = "FAIL"
                    gate_reason = f"{metric}={deltas.get(metric, 'N/A'):.2f} < {threshold}"
                else:
                    gate_reason = f"{metric}={deltas.get(metric, 'N/A'):.2f} >= {threshold}"
            elif ">" in gate_rule:
                metric, threshold = gate_rule.split(">")
                metric = metric.strip()
                threshold = float(threshold.strip())
                if metric in deltas and deltas[metric] <= threshold:
                    gate_status = "FAIL"
                    gate_reason = f"{metric}={deltas.get(metric, 'N/A'):.2f} <= {threshold}"
                else:
                    gate_reason = f"{metric}={deltas.get(metric, 'N/A'):.2f} > {threshold}"
        except Exception as e:
            gate_status = "ERROR"
            gate_reason = str(e)

    return {
        "baseline_metrics": base_metrics,
        "compare_metrics": comp_metrics,
        "deltas": deltas,
        "gate_status": gate_status,
        "gate_reason": gate_reason,
    }


def print_comparison(result: dict[str, Any]) -> None:
    """Pretty-print comparison results."""
    print("\n=== Baseline Metrics ===")
    for k, v in sorted(result["baseline_metrics"].items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    print("\n=== Compare Metrics ===")
    for k, v in sorted(result["compare_metrics"].items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    print("\n=== Deltas ===")
    for k, v in sorted(result["deltas"].items()):
        if isinstance(v, float):
            sign = "+" if v > 0 else ""
            print(f"  {k}: {sign}{v:.2f}")
        else:
            print(f"  {k}: {v}")

    print(f"\n=== Gate Result ===")
    print(f"  Status: {result['gate_status']}")
    print(f"  Reason: {result['gate_reason']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two GWAYA benchmark runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  compare_runs.py run1.json run2.json
  compare_runs.py --baseline cpu.json --compare gpu.json
  compare_runs.py --baseline cpu.json --compare gpu.json --gate "pass_rate_pct >= -5"
""",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Two results.json files (positional: baseline, compare)",
    )
    parser.add_argument(
        "--baseline",
        help="Baseline results file (takes precedence over first positional arg)",
    )
    parser.add_argument(
        "--compare",
        help="Compare results file (takes precedence over second positional arg)",
    )
    parser.add_argument(
        "--gate",
        help="Gate rule for pass/fail decision (e.g. 'pass_rate_pct >= -5')",
    )

    args = parser.parse_args()

    # Resolve baseline and compare files
    baseline_path = args.baseline
    compare_path = args.compare

    if not baseline_path and args.files:
        baseline_path = args.files[0]
    if not compare_path and len(args.files) > 1:
        compare_path = args.files[1]

    if not baseline_path or not compare_path:
        parser.print_help()
        return 1

    try:
        baseline = load_results(baseline_path)
        compare = load_results(compare_path)
        result = compare_runs(baseline, compare, args.gate)
        print_comparison(result)

        if result["gate_status"] != "PASS":
            return 1
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
