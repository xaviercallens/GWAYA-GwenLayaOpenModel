import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_refusal as R  # noqa: E402


def mk(n=20):
    data = {}
    for t in ("a", "b"):
        data[t] = {}
        for i in range(n):
            ok = int(i < n // 2)  # tasks 0..9 correct, 10..19 wrong, in both tiers
            data[t][("s", str(i))] = {"correct": ok, "laya": 0.9 - i * 0.01, "logprob": -0.1 - i * 0.05, "gate": float(ok)}
    return data


def test_perfect_ranker_gets_full_accuracy_and_refusal_precision():
    res = R.analyse(mk(), n_boot=50)
    assert res["n_hard_all_tiers_wrong"] == 10
    s = res["by_tier"]["a"]["scorers"]["laya"]
    assert s["1.0"]["accuracy_answered"] == 0.5 and s["0.4"]["accuracy_answered"] == 1.0
    assert abs(s["0.4"]["refusal_precision"] - 10 / 12) < 1e-9  # refused 12 tasks, 10 of them wrong
    assert s["0.4"]["hard_recall"] is not None and s["0.4"]["hard_recall"] == 1.0
    assert "gate_then_logprob" in res["by_tier"]["a"]["scorers"]
