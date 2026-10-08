"""Known-answer tests for scripts/analyze_e_tpu.py on a tiny hand-built study."""
import math

import numpy as np
import pytest

from scripts import analyze_e_tpu as E

S, L = "small-bf16", "large-bf16"
# (domain, task_id, small_correct, large_correct, gate_large_ok, gate_small_ok, large_logprob)
SPEC = [
    ("python", "p0", 1, 1, 1, 1, -0.05), ("python", "p1", 0, 1, 1, 0, -0.10), ("python", "p2", 0, 0, 0, 0, -0.90),
    ("python", "p3", 1, 1, 1, 1, -0.02), ("python", "p4", 0, 0, 1, 0, -0.60), ("python", "p5", 1, 0, 0, 1, -0.70),
    ("rust", "r0", 1, 1, 1, 1, -0.10), ("rust", "r1", 0, 1, 0, 0, -0.30), ("rust", "r2", 0, 0, 0, 0, -0.80),
    ("math", "m0", 0, 1, 0, 0, -0.40), ("math", "m1", 1, 1, 0, 0, -0.20), ("math", "m2", 0, 0, 0, 0, -0.95),
]


def make_inputs():
    tasks = [{"domain": d, "task_id": t, "cluster": f"c-{t}"} for d, t, *_ in SPEC]
    rows, gens = [], []
    for d, t, cs, cl, gl, gs, lp in SPEC:
        for model, cor, gate, lpv in ((S, cs, gs, -0.3), (L, cl, gl, lp)):
            rows.append({"arm": "base", "model": model, "domain": d, "task_id": t, "score": "VERIFIED" if cor else "FAILED"})
            rows.append({"arm": "gate_only", "model": model, "domain": d, "task_id": t, "answered": bool(gate),
                         "score": "VERIFIED" if cor else "FAILED"})
            gens.append({"model": model, "domain": d, "task_id": t, "temperature": 0.0, "mean_logprob": lpv,
                         "done_reason": "stop", "gpu_s": 1.0 if model == S else 3.0, "completion_tokens": 10})
        # cascade: small answers; escalate to large iff the small tier's gate does not verify
        esc = not gs
        final_cor, final_gate = (cl, gl) if esc else (cs, gs)
        rows.append({"arm": "gwenlaya", "model": L if esc else S, "domain": d, "task_id": t,
                     "score": "VERIFIED" if final_cor else "FAILED", "answered": bool(final_gate),
                     "gpu_s": 4.0 if esc else 1.0, "tiers_invoked": [S, L] if esc else [S]})
    return tasks, rows, gens


def make():
    tasks, rows, gens = make_inputs()
    return E.build(tasks, rows, gens, S, L)


def point(res, name):
    return res["metrics"][name]["point"]


@pytest.fixture(scope="module")
def res():
    return E.analyze(make(), n_boot=200, seed=0)


def test_pass_rates(res):
    assert point(res, f"A1.acc.{S}.python") == pytest.approx(3 / 6)
    assert point(res, f"A1.acc.{L}.python") == pytest.approx(3 / 6)
    assert point(res, f"A1.acc.{L}.pooled") == pytest.approx(7 / 12)
    assert point(res, f"A1.acc.{S}.math") == pytest.approx(1 / 3)


def test_gate_metrics_on_the_large_tier(res):
    # python: gate_large_ok = p0,p1,p3,p4 -> coverage 4/6; correct among them p0,p1,p3 -> precision 3/4; wrong&answered = p4
    assert point(res, f"A3.cov.{L}.python") == pytest.approx(4 / 6)
    assert point(res, f"A3.prec.{L}.python") == pytest.approx(3 / 4)
    assert point(res, f"A3.cwr.{L}.python") == pytest.approx(1 / 6)
    assert point(res, f"A3.cwr_all.{L}.python") == pytest.approx(3 / 6)
    assert point(res, f"A3.cov.{L}.math") == pytest.approx(0.0)  # coverage 0 is reported, not dropped


def test_matched_coverage_baseline_uses_the_gates_answer_count(res):
    # python: k=4 most confident by large logprob: p3(-.02),p0(-.05),p1(-.10),p4(-.60)? order: p3,p0,p1,p4 -> wrong: p4 -> 1/6
    assert point(res, f"A4.cwr_b3_matched.{L}.python") == pytest.approx(1 / 6)
    assert point(res, f"A4.delta_cwr_gate_minus_b3.{L}.python") == pytest.approx(0.0)
    # rust: gate answers r0 only (k=1); most confident is r0 (correct) -> 0
    assert point(res, f"A4.cwr_b3_matched.{L}.rust") == pytest.approx(0.0)


def test_cascade_known_answers(res):
    # escalated iff small gate fails: python gs=[1,0,0,1,0,1] -> escalate p1,p2,p4; final correct: p0 cs=1, p1 cl=1, p2 cl=0,
    # p3 cs=1, p4 cl=0, p5 cs=1 -> 4/6
    assert point(res, "A5.acc_cascade.python") == pytest.approx(4 / 6)
    assert point(res, "A5.escalated.python") == pytest.approx(3 / 6)
    assert point(res, "A5.cost_cascade.python") == pytest.approx((1 + 4 + 4 + 1 + 4 + 1) / 6)
    assert point(res, "A5.cost_large.python") == pytest.approx(3.0)
    assert point(res, "A5.cost_ratio_cascade_over_large.python") == pytest.approx((15 / 6) / 3.0)
    assert point(res, "A5.d_acc_cascade_minus_large.python") == pytest.approx(4 / 6 - 3 / 6)
    assert point(res, "A6.acc_oracle_router.python") == pytest.approx(4 / 6)  # small|large correct: p0, p1, p3, p5


def test_paired_mcnemar_and_holm(res):
    t = res["paired_tests"]
    D = make()
    cc, cl = D.cascade["cor"], D.models[L]["cor"]
    assert t["cascade_vs_large"]["cascade_only_correct"] == int((cc & ~cl).sum())
    assert t["cascade_vs_large"]["other_only_correct"] == int((~cc & cl).sum())
    ps = [t["cascade_vs_large"]["p_two_sided"], t["cascade_vs_small"]["p_two_sided"]]
    adj = [t["cascade_vs_large"]["p_holm"], t["cascade_vs_small"]["p_holm"]]
    assert adj[int(np.argmin(ps))] == pytest.approx(min(1.0, 2 * min(ps)))
    assert all(a >= p - 1e-12 for a, p in zip(adj, ps))


def test_bootstrap_ci_brackets_the_point(res):
    for name in (f"A1.acc.{L}.pooled", "A5.acc_cascade.pooled", f"A3.cov.{L}.python"):
        m = res["metrics"][name]
        assert m["lo"] <= m["point"] <= m["hi"]


def test_calibration_metrics_match_the_shared_primitives(res):
    D = make()
    j = np.where(D.is_dom["python"])[0]
    p, y = D.models[L]["conf"][j], D.models[L]["cor"][j]
    assert point(res, f"A2.auroc.{L}.python") == pytest.approx(E.A.auroc(p, y))
    assert point(res, f"A2.ece.{L}.python") == pytest.approx(E.A.ece_equal_mass(p, y))
    assert point(res, f"A2.meanconf.{L}.python") == pytest.approx(float(p.mean()))
    assert 0 <= point(res, f"A2.brier.{L}.python") <= 1


def test_night_comparison_counts():
    D = make()
    rows = [{"arm": "base", "domain": d, "task_id": t, "score": "VERIFIED" if cs else "FAILED"} for d, t, cs, *_ in SPEC]
    rows[0]["score"] = "FAILED"  # p0: CPU fails, TPU small passes
    out = E.night_compare(D, rows, {(d, t) for d, t, *_ in SPEC})
    py = out["by_domain"]["python"]
    assert py["n"] == 6 and py["tpu_only"] == 1 and py["cpu_only"] == 0 and py["agree"] == 5


def test_build_rejects_missing_rows():
    tasks = [{"domain": "python", "task_id": "x", "cluster": "c"}]
    with pytest.raises(SystemExit):
        E.build(tasks, [], [], S, L)


def test_night_comparison_excludes_math_and_says_why():
    D = make()
    rows = [{"arm": "base", "domain": d, "task_id": t, "score": "VERIFIED" if cs else "FAILED"} for d, t, cs, *_ in SPEC]
    out = E.night_compare(D, rows, {(d, t) for d, t, *_ in SPEC})
    assert "math" not in out["by_domain"] and out["domains"] == ["python", "rust"]
    assert out["n_common"] == 9 and "defective" in out["excluded"]["math"]



def test_overlay_rows_take_precedence_and_are_reported():
    tasks, rows, gens = make_inputs()
    overlay = [{"arm": "base", "model": S, "domain": "python", "task_id": "p1", "score": "VERIFIED"}]
    before = E.build(tasks, rows, gens, S, L)
    after = E.build(tasks, rows + overlay, gens, S, L)
    i = before.keys.index(("python", "p1"))
    assert not before.models[S]["cor"][i] and after.models[S]["cor"][i]
    assert E.overlay_corrections(rows, overlay) == [["base", S, "python", "p1", "FAILED", "VERIFIED"]]
    assert E.overlay_corrections(rows, [dict(overlay[0], score="FAILED")]) == []  # unchanged score is not a correction


def test_conservative_sensitivity_counts_unstable_items_as_wrong():
    D = make()
    # p0 is correct for both tiers and the gate-verified answer of the large tier; mark it unstable for the large tier
    assert D.mark_unstable(L, {("python", "p0"), ("python", "nope")}) == 1
    res = E.analyze(D, n_boot=50, seed=0)
    assert point(res, f"A1.acc.{L}.python") == pytest.approx(3 / 6)  # headline unchanged
    assert point(res, f"S1.acc_conservative.{L}.python") == pytest.approx(2 / 6)
    assert point(res, f"S1.n_unstable.{L}.python") == pytest.approx(1.0)
    # p0 was answered by the gate and correct; unstable -> now confident-wrong: cwr 1/6 -> 2/6
    assert point(res, f"S1.cwr_gate_conservative.{L}.python") == pytest.approx(2 / 6)
    # cascade: p0 is answered by the small tier (its gate verifies), so the large tier's instability does not touch it
    assert point(res, "S1.acc_cascade_conservative.python") == pytest.approx(point(res, "A5.acc_cascade.python"))


def test_a4_has_no_cross_domain_pooled_threshold_and_domain_matched_pool_is_exact():
    D = make()
    # make the large tier's confidence anti-informative so the baseline is clearly worse than the gate
    for i, (d, t, *_rest) in enumerate(SPEC):
        D.models[L]["conf"][D.keys.index((d, t))] = 0.5 + 0.01 * (1 if not D.models[L]["cor"][D.keys.index((d, t))] else 0) + 1e-4 * i
    res = E.analyze(D, n_boot=50, seed=0)
    assert f"A4.cwr_b3_matched.{L}.pooled" not in res["metrics"]  # the misleading single-threshold pooled value is gone
    gate_wrong = b3_wrong = 0
    for d in ("python", "rust", "math"):
        idx = [i for i, k in enumerate(D.keys) if k[0] == d]
        k = int(D.gate[L][idx].sum())
        gate_wrong += int((D.gate[L][idx] & ~D.models[L]["cor"][idx]).sum())
        order = sorted(idx, key=lambda i: (-D.models[L]["conf"][i], D.models[L]["cor"][i]))
        b3_wrong += sum(1 for i in order[:k] if not D.models[L]["cor"][i])
    assert point(res, f"A4.cwr_gate.{L}.domain_matched") == pytest.approx(gate_wrong / 12)
    assert point(res, f"A4.cwr_b3_matched.{L}.domain_matched") == pytest.approx(b3_wrong / 12)
    assert point(res, f"A4.delta_cwr_gate_minus_b3.{L}.domain_matched") == pytest.approx((gate_wrong - b3_wrong) / 12)
    assert b3_wrong > gate_wrong


def test_warmup_exclusion_uses_the_first_chunk_in_file_order():
    tasks, rows, gens = make_inputs()
    D = E.build(tasks, rows, gens, S, L, warmup_chunk=3)
    flagged = {D.keys[i] for i in np.where(D.warm)[0]}
    assert flagged == {(t["domain"], t["task_id"]) for t in tasks[:3]}
    res = E.analyze(D, n_boot=20, seed=0)
    # excluding warm tasks changes the cost means but keeps the ratio defined
    assert point(res, "S2.cost_ratio_cascade_over_large_nowarm.pooled") > 0
    assert point(res, "S2.cost_large_nowarm.pooled") == pytest.approx(3.0)
