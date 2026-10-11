"""Offline tests for scripts/replay_system_eprime.py (replay components and comparison) on synthetic rows."""
import json

import numpy as np
import pytest

from gwaya.domains.task import Task
from gwaya.gwenlaya import GwenLaya
from scripts import replay_system_eprime as RP

T = ("t2", "t4", "t9")
ISO = ("python",)


def _st(text, gate, gen_s, raw=0.5, cor=False, pot=0.0, prompt="p"):
    sig = {"gate": gate.lower(), "mean_logprob": -0.1, "min_logprob": -1.0}
    return RP.StoredTier(text=text, gate=gate, gen_s=gen_s, pot_s=pot, mean_logprob=-0.1, min_logprob=-1.0, raw=raw,
                         correct=cor, cal_row={"prompt": prompt, "candidate": text, "signals": sig})


def _world(items):
    tasks = {}
    for i, (router, tiers) in enumerate(items):
        k = ("python", f"py/{i}")
        tasks[k] = RP.StoredTask("python", f"py/{i}", f"prompt{i}", dict(zip(T, router)),
                                 {n: _st(*a, prompt=f"prompt{i}") for n, a in zip(T, tiers)})
    return RP.ReplayWorld(tasks, T), list(tasks)


ITEMS = [
    # router: 4B is the cheapest acceptable; 4B verifies -> start 4B, stop at 4B
    ((0.2, 0.7, 0.9), [("a", "FAILED", 1.0), ("b", "VERIFIED", 2.0, 0.5, True), ("c", "VERIFIED", 4.0)]),
    # no tier acceptable -> start at the last tier; its gate does not verify -> answer withheld
    ((0.1, 0.2, 0.3), [("a", "FAILED", 1.0), ("b", "FAILED", 2.0), ("c", "UNVERIFIED", 4.0, 0.5, True)]),
    # 2B acceptable but unverified; 4B has identical text with a different stored verdict
    ((0.9, 0.9, 0.9), [("same", "UNVERIFIED", 1.0), ("same", "VERIFIED", 2.0, 0.5, True), ("c", "VERIFIED", 4.0)]),
]


def _arrays_and_router(world, keys):
    arrays = []
    for t in T:
        arrays.append({"cor": np.array([world.tasks[k].tiers[t].correct for k in keys]),
                       "gate_ok": np.array([world.tasks[k].tiers[t].gate == "VERIFIED" for k in keys]),
                       "cost": np.array([world.tasks[k].tiers[t].gen_s for k in keys]),
                       "pot": np.array([world.tasks[k].tiers[t].pot_s for k in keys])})
    rp = np.array([[world.tasks[k].router[t] for t in T] for k in keys])
    return arrays, rp


def test_clock_and_tier_return_stored_text_signals_and_charge_stored_seconds():
    w, keys = _world(ITEMS)
    w.current = keys[0]
    tier = w.tier("t4")
    t0 = w.clock()
    text, sig = tier.generate_with_signals("prompt0", "python")
    assert text == "b" and sig == {"mean_logprob": -0.1, "min_logprob": -1.0}
    assert w.clock() - t0 == 2.0
    with pytest.raises(RuntimeError):
        tier.generate_with_signals("not the stored prompt", "python")


def test_scorer_returns_stored_raw_and_flags_feature_mismatch():
    w, keys = _world(ITEMS)
    w.current = keys[0]
    f = {"prompt": "prompt0", "domain": "python", "tier": "t4", "response": "b", "gate_status": "VERIFIED",
         "gen_signals": {"mean_logprob": -0.1, "min_logprob": -1.0}}
    assert w.scorer(dict(f)) == 0.5 and not w.feature_mismatches
    assert w.scorer({**f, "gate_status": "FAILED", "response": "zzz"}) == 0.5
    assert {m["field"] for m in w.feature_mismatches} == {"gate_status", "response"}


def test_checker_uses_the_stored_verdict_of_the_tier_that_just_generated():
    w, keys = _world(ITEMS)
    w.current = keys[2]
    task = Task("python", "py/2", "prompt2", {})
    w.tier("t2").generate_with_signals("prompt2", "python")
    assert w.checker(task, "same").status == "UNVERIFIED"
    w.tier("t4").generate_with_signals("prompt2", "python")
    assert w.checker(task, "same").status == "VERIFIED"


def test_registered_outcome_decodes_consulted_tiers():
    w, keys = _world(ITEMS)
    arrays, rp = _arrays_and_router(w, keys)
    reg = RP.registered_outcome(0.5, arrays, rp)
    assert reg["consulted"] == [[1], [2], [0, 1]]
    assert list(reg["answered"]) == [True, False, True]
    assert reg["cost"].tolist() == [2.0, 4.0, 3.0]


def test_runtime_replay_agrees_with_registered_policy_on_synthetic_rows():
    w, keys = _world(ITEMS)
    arrays, rp = _arrays_and_router(w, keys)
    rt = RP.run_runtime(w, RP.build_system(w, T, ISO, routed=True, tau=0.5), keys)
    cmpd = RP.compare(rt, RP.registered_outcome(0.5, arrays, rp), keys, T, arrays)
    assert cmpd["all_five_agree"] == 3 and not cmpd["disagreements"]
    assert not w.feature_mismatches
    # the runtime withholds the last tier's unverified answer; the registered quantity still counts it (plan A20)
    assert rt[1]["answer_returned"] is False and rt[1]["verdict"] in ("ABSTAIN", "LIKELY_WRONG")
    assert cmpd["rows"][1]["rt_final_cand_cor"] is True and cmpd["rows"][1]["rt_answered_correct"] is False


def test_a_runtime_with_the_filter_policy_is_detected_as_disagreeing():
    w, keys = _world(ITEMS)
    arrays, rp = _arrays_and_router(w, keys)
    system = RP.build_system(w, T, ISO, routed=True, tau=0.5)
    assert isinstance(system, GwenLaya)
    system.route_policy = "filter"  # tasks with no tier >= tau get no tier at all, instead of starting at the last
    rt = RP.run_runtime(w, system, keys)
    cmpd = RP.compare(rt, RP.registered_outcome(0.5, arrays, rp), keys, T, arrays)
    assert cmpd["disagreements"] and cmpd["disagreements"][0]["task_id"] == "py/1"


def test_zero_warmup_keeps_program_seconds(tmp_path):
    w, keys = _world([((0.9, 0.9, 0.9), [("a", "VERIFIED", 1.0, 0.5, True, 3.0), ("b", "VERIFIED", 1.0), ("c", "VERIFIED", 1.0)])])
    raw = {t: tmp_path / f"{t}.jsonl" for t in T}
    for p in raw.values():
        p.write_text(json.dumps({"domain": "python", "task_id": "py/0"}) + "\n")
    z = RP.zero_warmup(w.tasks, raw, chunk=1)
    st = z[keys[0]].tiers["t2"]
    assert st.gen_s == 0.0 and st.pot_s == 3.0
