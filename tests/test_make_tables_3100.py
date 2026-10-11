"""Formatting helpers and invariants of scripts/make_tables_3100.py (no data root needed for the pure helpers)."""
from __future__ import annotations

import json
from pathlib import Path

from scripts import make_tables_3100 as M

ROOT = Path(__file__).resolve().parents[1]


def test_number_formatting_keeps_the_sign_of_the_value():
    assert M.n(-0.1044) == "$-0.104$"
    assert M.n(0.0, sign=True) == "$+0.000$"
    assert M.n(9.8e-05, sign=True) == "$+0.000$"
    assert M.n(None) == "n/a"


def test_ci_formatting():
    c = {"point": -0.002, "lo": -0.022, "hi": 0.016}
    assert M.ci(c, sign=True) == "$-0.002$ [$-0.022$, $+0.016$]"
    assert M.ci({"point": 0.5, "lo": None, "hi": None}) == "0.500"
    assert M.ci(None) == "n/a"


def test_committed_f_results_have_the_headline_values():
    L = json.loads((ROOT / "results/gwenlaya_v4/eval_sets_F/numbers_F_laya.json").read_text())
    p = L["primary"]["inference"]
    assert p["H1_supported"] is True and p["H3_supported"] is False
    assert round(p["H1_dAURC_laya_minus_logprob"]["point"], 3) == -0.104
    assert round(p["H3_dRisk_at_gate_coverage_laya_minus_gate"]["point"], 3) == -0.002


def test_lean_rows_all_verified_matches_are_exact():
    for d in ("results/gwenlaya_v4/lean_a18/scored_greedy_v3", "results/gwenlaya_v4/eval_sets_F/lean_scored"):
        s = json.loads((ROOT / d / "summary_a18.json").read_text())
        for t in s["tiers"].values():
            assert t["verified_match"]["defeq"] == 0 and t["verified_match"]["unknown"] == 0
            assert t["verified_match"]["exact"] == len(t["greedy_proved"])
