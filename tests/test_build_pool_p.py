"""Offline tests for scripts/data/build_pool_p.py (small fixtures, no network, no model)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))
import build_pool_p as P  # noqa: E402


def test_math_row_last_boxed_and_schema():
    r = P.math_row("algebra", 3, {"problem": " Find $x$. ", "level": "Level 2",
                                  "solution": r"first \boxed{1} then \boxed{\frac{3}{4}}."})
    assert r["task_id"] == "ma/train/algebra/3" and r["gate_payload"] == {}
    assert r["checker_payload"] == {"answer": r"\frac{3}{4}"}
    assert r["prompt"] == "Find $x$." + "\n\nPut the final answer in \\boxed{}."
    assert r["cluster"] == "MATH/train/algebra/3"
    assert P.math_row("algebra", 4, {"problem": "p", "solution": "no box"}) is None
    assert P.math_row("algebra", 5, {"problem": "p", "solution": r"\boxed{ }"}) is None


def test_gsm8k():
    assert P.gsm8k_answer("a\n#### 1,234") == "1234"
    assert P.gsm8k_answer("no marker") is None
    r = P.gsm8k_row(7, {"question": "How many?", "answer": "x\n#### 42"})
    assert r["task_id"] == "ma/gsm8k/7" and r["checker_payload"] == {"answer": "42"}
    assert r["prompt"].endswith("\n\nPut the final answer in \\boxed{}.")


def test_stratified_sample_deterministic_and_proportional():
    rows = [{"task_id": f"ma/train/a/{i}", "_stratum": "a|1"} for i in range(80)] + \
           [{"task_id": f"ma/train/b/{i}", "_stratum": "b|1"} for i in range(20)]
    s1 = P.stratified_sample(rows, 10, 1, lambda r: r["_stratum"])
    s2 = P.stratified_sample(list(reversed(rows)), 10, 1, lambda r: r["_stratum"])
    assert s1 == s2 and len(s1) == 10
    assert sum(r["_stratum"] == "a|1" for r in s1) == 8
    assert len(P.stratified_sample(rows, 500, 1, lambda r: r["_stratum"])) == 100


MB = {"task_id": 11, "text": "Write f.", "code": "def f(x):\r\n    return x\r\n",
      "test_list": ["assert f(1)==1", "assert f(2)==2"], "test_setup_code": ""}


def test_mbpp_row_visible_vs_hidden():
    r = P.mbpp_row(MB)
    assert r["task_id"] == "py/MBPP/11" and r["cluster"] == "MBPP/11"
    assert r["gate_payload"]["tests"] == "assert f(1)==1"
    assert "assert f(2)==2" not in r["gate_payload"]["tests"]
    assert r["checker_payload"]["tests"] == "assert f(1)==1\nassert f(2)==2"
    assert "assert f(1)==1\nReply with the full Python function." in r["prompt"]
    assert "\r" not in r["_reference"]
    assert r["checker_payload"]["timeout_s"] == 30.0 and r["gate_payload"]["timeout_s"] == 10.0


def test_mbpp_overlap_by_id_and_hash():
    assert P.mbpp_overlap_with_plus(MB, {11}, set()) == "task_id_in_mbppplus"
    assert P.mbpp_overlap_with_plus(MB, set(), {P.phash("Write   f.")}) == "prompt_hash_in_mbppplus"
    assert P.mbpp_overlap_with_plus(MB, {1}, {"x"}) is None


def test_decontaminate():
    e = [{"prompt": "Q one\n\nPut the final answer in \\boxed{}.", "cluster": "MATH-500/x"},
         {"prompt": "Write f.\nYour code should pass this test:\nassert f(1)==1\nReply.", "cluster": "MBPP/11"}]
    rows = [{"task_id": "ma/train/a/1", "prompt": "Q   one\n\nPut the final answer in \\boxed{}.", "cluster": "c1"},
            {"task_id": "py/MBPP/11", "prompt": "z", "cluster": "MBPP/11"},
            {"task_id": "py/MBPP/12", "prompt": "Write f.\nYour code should pass this test:\nassert g()\nReply.",
             "cluster": "MBPP/12"},
            {"task_id": "ma/gsm8k/1", "prompt": "fresh", "cluster": "G/1"}]
    kept, rep = P.decontaminate(rows, e)
    assert [r["task_id"] for r in kept] == ["ma/gsm8k/1"]
    assert rep["by_reason"] == {"prompt_hash_in_E": 1, "cluster_in_E": 1, "statement_hash_in_E": 1}


def test_split_rule():
    import hashlib
    for c in ["MBPP/1", "GSM8K/train/5", "MATH/train/algebra/9"]:
        b = int(hashlib.sha256(c.encode()).hexdigest(), 16) % 10
        assert P.bucket_of(c) == b
        assert P.split_of(c) == ("train" if b <= 5 else "calib" if b == 6 else "eval")
    assert {P.split_of(f"c{i}") for i in range(200)} == {"train", "calib", "eval"}


def test_sanity_python_real_oracle_and_build():
    good = P.mbpp_row(MB)
    bad = P.mbpp_row({**MB, "task_id": 12, "code": "def f(x):\n    return 0\n"})
    kept, drops = P.run_sanity([good, bad], workers=2)
    assert [r["task_id"] for r in kept] == ["py/MBPP/11"]
    assert drops[0]["task_id"] == "py/MBPP/12" and drops[0]["reason"].startswith("reference_fails")


def test_build_end_to_end_no_sanity():
    src = {"problems": [], "plus_ids": {5}, "plus_hashes": set(),
           "math_all": [P.math_row("algebra", i, {"problem": f"p{i}", "level": "Level 1",
                                                  "solution": r"\boxed{%d}" % i}) for i in range(20)],
           "gsm_all": [P.gsm8k_row(i, {"question": f"q{i}", "answer": f"#### {i}"}) for i in range(10)],
           "mbpp_raw": [MB, {**MB, "task_id": 5, "text": "other"}]}
    rows, rep = P.build(src, [], do_sanity=False, math_n=8, gsm_n=4)
    assert len(rows) == 8 + 4 + 1 and rep["mbpp"]["dropped_vs_mbppplus"] == {"task_id_in_mbppplus": 1}
    assert all(set(r) == {"domain", "task_id", "prompt", "checker_payload", "gate_payload", "cluster", "split"}
               for r in rows)
