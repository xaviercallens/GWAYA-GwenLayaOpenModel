"""Offline tests for scripts/data/build_pool_py_oci.py (no network, no sandbox run)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "data")]
import build_pool_py_oci as B  # noqa: E402


def src(**kw):
    r = {"id": "abc", "input": "Write add(a, b) returning the sum of two ints.", "output": "```python\ndef add(a, b):\n    return a + b\n```",
         "unit_tests": '["assert add(1,2)==3", "assert add(0,0)==0", "assert add(-1,1)==0", "assert add(2,2)==4"]',
         "tests_execution_status": '["pass","pass","pass","pass"]'}
    r.update(kw)
    return r


def test_row_schema_and_gate_is_first_assert():
    row, why = B.to_row(src(), 3, 1200)
    assert why is None and row["domain"] == "python" and row["cluster"] == "OCI/abc"
    assert row["gate_payload"]["tests"] == "assert add(1,2)==3"
    assert row["checker_payload"]["tests"].count("assert") == 4
    assert row["prompt"].endswith("Reply with the full Python function.")


def test_failing_source_tests_are_not_trusted_and_few_tests_drop():
    row, why = B.to_row(src(tests_execution_status='["pass","fail","fail","pass"]'), 3, 1200)
    assert row is None and why == "too_few_passing_tests"
    assert B.to_row(src(input="x" * 5000), 3, 1200)[1] == "prompt_too_long_or_empty"
    assert B.to_row(src(output="no code here"), 3, 1200)[1] == "no_reference_code"


def test_holdout_filter_fires_on_exact_and_near_duplicates():
    held = [{"prompt": "Write add(a, b) returning the sum of two ints.\nYour code should pass this test:\nassert add(1,2)==3\nReply with the full Python function.",
             "cluster": "MBPP/1"}]
    h = B.Holdout(held)
    row, _ = B.to_row(src(), 3, 1200)
    assert h.why(row["prompt"], row["_task_text"]) is not None
    other, _ = B.to_row(src(input="Compute the median of a list of floats, ignoring NaN values entirely."), 3, 1200)
    assert h.why(other["prompt"], other["_task_text"]) is None


def test_split_buckets_cover_all_and_are_stable():
    names = {B.split_of(f"OCI/{i}") for i in range(200)}
    assert names == {"train", "calib", "dev"} and B.split_of("OCI/7") == B.split_of("OCI/7")
