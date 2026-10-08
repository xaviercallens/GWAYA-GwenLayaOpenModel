"""Offline tests for scripts/analyze_study.py on synthetic data with known answers."""
import json
import math

import numpy as np
import pytest

from gwaya import metrics_selective as ms
from scripts import analyze_study as an

NB = 300


def _write(path, recs):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


# ── metric kernels ───────────────────────────────────────────────────────────────────

def test_auroc_known_values():
    assert an.auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])) == 1.0
    assert an.auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([1, 1, 0, 0])) == 0.0
    assert an.auroc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([1, 0, 1, 0])) == 0.5
    assert math.isnan(an.auroc(np.array([0.5, 0.6]), np.array([1, 1])))


def test_ece_known_values():
    assert an.ece_equal_mass(np.array([0.5] * 30), np.array([1, 0] * 15, dtype=float)) == pytest.approx(0.0)
    # always claims 0.9 but is right 50% of the time -> ECE 0.4, Brier 0.41
    p, y = np.array([0.9] * 30), np.array([1, 0] * 15, dtype=float)
    assert an.ece_equal_mass(p, y) == pytest.approx(0.4)
    assert an.brier(p, y) == pytest.approx(0.41)


def test_kernels_match_reference_implementation():
    rng = np.random.default_rng(3)
    p = rng.uniform(size=300)
    y = (rng.uniform(size=300) < p)
    assert an.ece_equal_mass(p, y.astype(float)) == pytest.approx(ms.ece(list(p), list(y.astype(float)))[0])
    assert an.auroc(p, y) == pytest.approx(ms.auroc(list(p), list(y.astype(float))))
    assert an.aurc(p, y) == pytest.approx(ms.aurc(list(p), list(y.astype(float))))
    assert an.brier(p, y.astype(float)) == pytest.approx(ms.brier(list(p), list(y.astype(float))))


def test_aurc_perfect_vs_inverted():
    y = np.array([1, 1, 0, 0])
    assert an.aurc(np.array([0.9, 0.8, 0.2, 0.1]), y) == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4)
    assert an.aurc(np.array([0.1, 0.2, 0.8, 0.9]), y) > an.aurc(np.array([0.9, 0.8, 0.2, 0.1]), y)


def test_cwr_and_cost_per_correct():
    ans = np.array([1, 1, 1, 0, 0], dtype=bool)
    cor = np.array([1, 0, 0, 0, 1], dtype=bool)
    assert an.cwr(ans, cor) == pytest.approx(2 / 5)
    assert an.answered_acc(ans, cor) == pytest.approx(1 / 3)
    assert an.cost_per_correct(ans, cor, np.array([2.0, 2, 2, 2, 2])) == pytest.approx(10.0)  # 1 correct answered
    assert math.isnan(an.cost_per_correct(~ans, np.zeros(5, bool), np.ones(5)))


def test_cluster_bootstrap_resamples_whole_clusters_within_domain():
    dom = np.array(["a"] * 4 + ["b"] * 4)
    cl = np.array(["c1", "c1", "c2", "c2", "c3", "c3", "c4", "c4"])
    sizes = []
    an.cluster_boot(lambda i: sizes.append((np.sum(dom[i] == "a"), np.sum(dom[i] == "b"))) or 0.0, dom, cl, 50, 1)
    assert all(s == (4, 4) for s in sizes)  # stratified: domain sizes preserved
    # cluster-level: a resample of domain a contains items in pairs
    seen = []
    an.cluster_boot(lambda i: seen.append(sorted(i.tolist())) or 0.0, dom, cl, 20, 2)
    for idx in seen:
        cnt = np.bincount(idx, minlength=8)
        assert cnt[0] == cnt[1] and cnt[2] == cnt[3]


def test_holm_and_mcnemar_known():
    assert an.holm([0.01, 0.04]) == pytest.approx([0.02, 0.04])
    assert an.mcnemar_exact(20, 0, "greater") == pytest.approx(2.0 ** -20)
    assert an.mcnemar_exact(5, 5) == 1.0


# ── end to end ──────────────────────────────────────────────────────────────────────

def _synthetic(tmp_path, n=200):
    """GL and B5 are right on the same items; B3 is wrong on a superset (20 extra confident-wrong).
    GL costs 1 per item, B5 costs 2 per item. Two items per cluster, single domain."""
    tasks, rows = [], []
    for i in range(n):
        tid = f"py/T{i}"
        tasks.append({"task_id": tid, "domain": "python", "cluster": f"T{i // 2}"})
        gl_wrong = i < 5
        b3_wrong = i < 25
        base = {"domain": "python", "task_id": tid, "quant": "q4"}
        rows.append({**base, "arm": "gwenlaya", "model": "big", "answered": True, "gate": "VERIFIED", "p_correct": 0.9 if not gl_wrong else 0.3,
                     "score": "FAILED" if gl_wrong else "VERIFIED", "gpu_s": 1.0, "tiers_invoked": ["small"] if i % 4 else ["small", "big"]})
        rows.append({**base, "arm": "gate_only", "model": "big", "answered": True, "gate": "VERIFIED",
                     "score": "FAILED" if gl_wrong else "VERIFIED", "gpu_s": 2.0})
        rows.append({**base, "arm": "always_largest", "model": "big", "answered": True,
                     "score": "FAILED" if gl_wrong else "VERIFIED", "gpu_s": 2.0})
        rows.append({**base, "arm": "raw_confidence", "model": "big", "answered": True, "confidence": -0.1,
                     "score": "FAILED" if b3_wrong else "VERIFIED", "gpu_s": 2.0})
    _write(tmp_path / "rows.jsonl", rows)
    _write(tmp_path / "tasks.jsonl", tasks)
    _write(tmp_path / "ledger.jsonl", [{"item": "x", "seconds": 5}, {"item": "y", "usd_estimate": 1.5, "seconds": 1}])
    return tmp_path


def _run(tmp_path, scores=None, rows=True):
    return an.analyze([tmp_path / "rows.jsonl"] if rows else [], [], tmp_path / "tasks.jsonl", scores, tmp_path / "ledger.jsonl",
                      NB, 0, tmp_path / "n.json", tmp_path / "t.tex", tmp_path / "figs")


def test_h1_h3_known_answers(tmp_path):
    _synthetic(tmp_path)
    N = _run(tmp_path)
    h1, h3 = N.hyp["H1"], N.hyp["H3"]
    assert h1["discordant_b3_only"] == 20 and h1["discordant_gl_only"] == 0
    assert h1["delta_cwr"]["point"] == pytest.approx(-20 / 200)
    assert h1["p_raw"] == pytest.approx(2.0 ** -20)
    assert h1["coverage_matched"] and h1["holm_m"] == 2
    assert h1["p_holm"] == pytest.approx(2 * h1["p_raw"])  # smaller raw p of m = 2
    assert h1["verdict"].startswith("PASS (meaningful)")
    assert h3["ratio"]["point"] == pytest.approx(0.5)
    assert h3["ratio"]["lo"] == pytest.approx(0.5) and h3["ratio"]["hi"] == pytest.approx(0.5)
    assert h3["p_raw"] == pytest.approx(1 / (NB + 1))
    assert h3["cost_unit"] == "gpu_seconds"
    assert h3["escalation_rate"]["point"] == pytest.approx(50 / 200)
    assert h3["verdict"].startswith("PASS")
    assert "underpowered" in h1["verdict"]  # n = 200 < 500
    # H2: GL and B5 have identical errors
    assert N.hyp["H2"]["delta_cwr"]["point"] == 0.0
    assert "no confident-wrong reduction" in N.hyp["H2"]["statement"]
    assert N.hyp["H7"]["per_domain"]["python"]["point"] == pytest.approx(195 / 200)


def test_matching_failure_is_reported(tmp_path):
    _synthetic(tmp_path)
    rows = [json.loads(l) for l in (tmp_path / "rows.jsonl").read_text().splitlines()]
    for r in rows:  # B3 abstains on 20% of items -> coverage difference far outside +-3 pts
        if r["arm"] == "raw_confidence" and int(r["task_id"].split("T")[1]) % 5 == 0:
            r["answered"] = False
    _write(tmp_path / "rows.jsonl", rows)
    assert _run(tmp_path).hyp["H1"]["verdict"].startswith("matching failed")


def test_holm_over_kept_family_only(tmp_path):
    _synthetic(tmp_path)
    rows = [json.loads(l) for l in (tmp_path / "rows.jsonl").read_text().splitlines() if '"raw_confidence"' not in l]
    _write(tmp_path / "rows.jsonl", rows)  # no B3 -> H1 not run -> H3 alone, Holm m = 1
    N = _run(tmp_path)
    assert N.hyp["H1"]["status"] == "not_run"
    assert N.hyp["H3"]["holm_m"] == 1 and N.hyp["H3"]["p_holm"] == N.hyp["H3"]["p_raw"]


def test_no_rows_is_not_run_not_a_number(tmp_path):
    _synthetic(tmp_path)
    N = _run(tmp_path, rows=False)
    for h in ("H1", "H2", "H3", "H7"):
        assert N.hyp[h]["status"] == "not_run"
    assert N.hyp["H10"]["status"] == "dropped" and N.hyp["H9"]["status"] == "deferred"
    out = json.loads((tmp_path / "n.json").read_text())
    assert "NaN" not in (tmp_path / "n.json").read_text()
    assert out["hypotheses"]["H13"]["lines_without_usd_estimate"] == 1
    assert out["hypotheses"]["H13"]["usd_known_total"] == 1.5
    assert (tmp_path / "t.tex").read_text().count("\\begin{table}") == 3


def test_every_number_has_a_source_and_is_finite_or_null(tmp_path):
    _synthetic(tmp_path)
    _run(tmp_path)
    out = json.loads((tmp_path / "n.json").read_text())
    assert out["numbers"]
    for k, v in out["numbers"].items():
        assert v["source"], k
        assert v["value"] is None or isinstance(v["value"], (int, float)), k
    assert "H1.delta_cwr.point" in out["numbers"] and "H3.ratio.point" in out["numbers"]


def test_secondary_hypotheses_known_answers(tmp_path):
    _synthetic(tmp_path, n=400)
    rng = np.random.default_rng(0)
    scores = []
    for i in range(400):
        tid = f"py/T{i}"
        correct = bool(i % 2)
        scores.append({"task_id": tid, "arm": "laya", "p_correct": 0.95 if correct else 0.05, "correct": correct, "tau_hi": 0.5})
        scores.append({"task_id": tid, "arm": "b3_ts", "p_correct": float(rng.uniform(0.2, 0.8)), "correct": correct})
        scores.append({"task_id": tid, "arm": "lr", "p_correct": float(rng.uniform()), "correct": correct})
        scores.append({"task_id": tid, "arm": "sh", "p_correct": float(rng.uniform()), "correct": correct})
    _write(tmp_path / "scores.jsonl", scores)
    N = _run(tmp_path, scores=tmp_path / "scores.jsonl")
    assert N.hyp["H5"]["auroc_laya"] == 1.0
    assert N.hyp["H5"]["auroc_diff"]["lo"] > 0.3 and N.hyp["H5"]["verdict"].startswith("PASS")
    assert N.hyp["H5"]["sh_leak_flag"] is False  # shuffled-label arm AUROC CI includes 0.5
    yl = np.array([bool(i % 2) for i in range(400)])
    pl = np.where(yl, 0.95, 0.05)
    assert N.hyp["H4"]["ece_laya"]["point"] == pytest.approx(ms.ece(list(pl), list(yl.astype(float)))[0])
    assert N.hyp["H4"]["ece_laya"]["point"] < 0.05
    assert N.hyp["H4"]["ece_diff"]["hi"] < 0
    assert N.hyp["H6"]["per_domain"]["python"]["point"] == 0.0  # selective error 0 at tau 0.5
    assert N.hyp["H6"]["per_domain"]["python"]["verdict"] == "not refuted"


def test_cli_smoke(tmp_path):
    _synthetic(tmp_path)
    rc = an.main(["--results", str(tmp_path), "--tasks", str(tmp_path / "tasks.jsonl"), "--ledger", str(tmp_path / "ledger.jsonl"),
                  "--n-boot", "50", "--out-numbers", str(tmp_path / "o.json"), "--out-tables", str(tmp_path / "o.tex"),
                  "--out-figs", str(tmp_path / "f")])
    assert rc == 0 and (tmp_path / "o.json").exists()
