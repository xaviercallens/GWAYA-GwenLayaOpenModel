"""Offline tests for scripts/import_remote_gens.py and the pure helpers of scripts/tpu/gen_batch.py."""
import json
import types

import pytest

from gwaya.domains.task import CheckResult
from scripts import import_remote_gens as irg
from scripts import run_study as rs
from scripts.tpu import gen_batch as gb

PLAN = {"stages": [], "seeds": {"eval_order": 7}, "generation_settings": {"max_new_tokens": {"python": 64, "math": 64}}}
STAGE = {"id": "T", "models": ["small-bf16"], "quant": ["bf16"], "arms": []}


def write_tasks(path, n=4):
    rows = [{"domain": "python", "task_id": f"p{i}", "prompt": f"write f{i}", "checker_payload": {"want": "ok"}} for i in range(n)]
    rows.append({"domain": "math", "task_id": "m0", "prompt": "2+3?", "checker_payload": {"answer": "5"}})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def write_raw(path, rows, chunk_size=3):
    out = []
    for i, r in enumerate(rows):
        text = "ok" if r["domain"] == "python" else "so \\boxed{5}"
        out.append({"domain": r["domain"], "task_id": r["task_id"], "prompt_tokens": 10, "completion_tokens": 20,
                    "chunk": i // chunk_size, "chunk_wall_s": 6.0, "chunk_size": chunk_size, "text": text,
                    "finish_reason": "stop", "mean_logprob": -0.25, "token_logprobs": [-0.25] * 20})
    path.write_text("".join(json.dumps(r) + "\n" for r in out))


def checker(task, text):
    if task.domain == "math":
        return CheckResult("VERIFIED" if "boxed{5}" in text else "FAILED")
    return CheckResult("VERIFIED" if "\nok\n" in text else "FAILED")


def test_import_then_score_has_zero_cache_misses(tmp_path):
    rows = write_tasks(tmp_path / "tasks.jsonl")
    write_raw(tmp_path / "raw.jsonl", rows)
    out = tmp_path / "run"
    meta = irg.import_gens(tmp_path / "tasks.jsonl", tmp_path / "raw.jsonl", "small-bf16", "bf16", out)
    assert meta["added"] == 5 and meta["tasks_without_generation"] == 0
    tasks = rs.load_tasks_jsonl(tmp_path / "tasks.jsonl")
    st = rs.Study(stage=STAGE, plan=PLAN, tasks=tasks, models=["small-bf16"], quants=["bf16"], arms=["base"],
                  backend=types.SimpleNamespace(), out_dir=out, mode="score", checker=checker)
    st.run()  # a cache miss would raise CacheMiss / report PARTIAL
    assert st.new_gens == 0 and len(st.rows.records) == 5
    assert all(r["score"] == "VERIFIED" for r in st.rows.records)
    # the imported chain is a valid hash chain that run_study re-reads without error
    assert len(rs.ChainedLog(out / "gens.jsonl").records) == 5


def test_seeds_follow_the_sorted_rank(tmp_path):
    rows = write_tasks(tmp_path / "tasks.jsonl")
    write_raw(tmp_path / "raw.jsonl", rows)
    irg.import_gens(tmp_path / "tasks.jsonl", tmp_path / "raw.jsonl", "small-bf16", "bf16", tmp_path / "run")
    recs = {r["task_id"]: r for r in rs.ChainedLog(tmp_path / "run" / "gens.jsonl").records}
    # sorted by (domain, task_id): math/m0 = 0, python/p0..p3 = 1..4
    assert recs["m0"]["seed"] == 1000 and recs["p0"]["seed"] == 2000 and recs["p3"]["seed"] == 5000
    assert recs["p1"]["key"] == "small-bf16|bf16|python|p1|3000|t0.0"


def test_code_is_fenced_and_math_stays_prose(tmp_path):
    rows = write_tasks(tmp_path / "tasks.jsonl")
    write_raw(tmp_path / "raw.jsonl", rows)
    irg.import_gens(tmp_path / "tasks.jsonl", tmp_path / "raw.jsonl", "small-bf16", "bf16", tmp_path / "run")
    recs = {r["task_id"]: r for r in rs.ChainedLog(tmp_path / "run" / "gens.jsonl").records}
    assert recs["p0"]["text"] == "```python\nok\n```" and recs["m0"]["text"] == "so \\boxed{5}"


def test_accelerator_seconds_are_split_by_token_share_and_sum_to_chunk_wall(tmp_path):
    rows = write_tasks(tmp_path / "tasks.jsonl")
    write_raw(tmp_path / "raw.jsonl", rows, chunk_size=5)  # one chunk of 5 equal items, wall 6.0 s
    irg.import_gens(tmp_path / "tasks.jsonl", tmp_path / "raw.jsonl", "small-bf16", "bf16", tmp_path / "run")
    recs = rs.ChainedLog(tmp_path / "run" / "gens.jsonl").records
    assert sum(r["gpu_s"] for r in recs) == pytest.approx(6.0, abs=1e-4)
    assert all(r["gpu_s"] == pytest.approx(1.2, abs=1e-4) for r in recs)


def test_second_import_is_idempotent(tmp_path):
    rows = write_tasks(tmp_path / "tasks.jsonl")
    write_raw(tmp_path / "raw.jsonl", rows)
    a = irg.import_gens(tmp_path / "tasks.jsonl", tmp_path / "raw.jsonl", "small-bf16", "bf16", tmp_path / "run")
    b = irg.import_gens(tmp_path / "tasks.jsonl", tmp_path / "raw.jsonl", "small-bf16", "bf16", tmp_path / "run")
    assert a["added"] == 5 and b["added"] == 0 and b["skipped_existing"] == 5


def test_unknown_raw_task_is_rejected(tmp_path):
    rows = write_tasks(tmp_path / "tasks.jsonl")
    write_raw(tmp_path / "raw.jsonl", rows + [{"domain": "python", "task_id": "ghost", "prompt": "x", "checker_payload": {}}])
    with pytest.raises(SystemExit):
        irg.import_gens(tmp_path / "tasks.jsonl", tmp_path / "raw.jsonl", "small-bf16", "bf16", tmp_path / "run")


def test_gen_batch_resume_keys_ignore_a_torn_last_line(tmp_path):
    p = tmp_path / "raw.jsonl"
    p.write_text(json.dumps({"domain": "python", "task_id": "p0"}) + "\n" + '{"domain": "python", "task_id": "p1"')
    assert gb.done_keys(p) == {("python", "p0")}
    assert gb.done_keys(tmp_path / "missing.jsonl") == set()


def test_gen_batch_mean_logprob_uses_sampled_tokens():
    LP = types.SimpleNamespace
    out = types.SimpleNamespace(token_ids=[5, 6], logprobs=[{5: LP(logprob=-1.0), 9: LP(logprob=-0.1)}, {6: LP(logprob=-3.0)}])
    mean, lst = gb.mean_and_list(out)
    assert mean == pytest.approx(-2.0) and lst == [-1.0, -3.0]
    assert gb.mean_and_list(types.SimpleNamespace(token_ids=[], logprobs=None)) == (None, [])


def test_require_logprobs_fails_fast_when_engine_returns_none():
    rows = [{"completion_tokens": 20, "mean_logprob": None}, {"completion_tokens": 5, "mean_logprob": None}]
    with pytest.raises(gb.NoLogprobsError):
        gb.require_logprobs(rows)
    gb.require_logprobs(rows + [{"completion_tokens": 7, "mean_logprob": -0.3}])  # one row with logprobs is enough
    gb.require_logprobs([{"completion_tokens": 0, "mean_logprob": None}])  # prompt_too_long rows are ignored


def test_subset_reimport_keeps_original_attributed_seconds(tmp_path):
    rows = write_tasks(tmp_path / "tasks.jsonl")
    write_raw(tmp_path / "raw.jsonl", rows, chunk_size=5)  # chunk wall 6.0 s over 5 tasks: 1.2 s each
    irg.import_gens(tmp_path / "tasks.jsonl", tmp_path / "raw.jsonl", "small-bf16", "bf16", tmp_path / "full")
    # subset of 2 tasks: naive attribution would give 3.0 s each; the original 1.2 s must be kept
    sub = [r for r in (json.loads(l) for l in (tmp_path / "raw.jsonl").read_text().splitlines()) if r["task_id"] in ("p0", "p1")]
    (tmp_path / "raw_sub.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sub))
    irg.import_gens(tmp_path / "tasks.jsonl", tmp_path / "raw_sub.jsonl", "small-bf16", "bf16", tmp_path / "naive")
    irg.import_gens(tmp_path / "tasks.jsonl", tmp_path / "raw_sub.jsonl", "small-bf16", "bf16", tmp_path / "kept",
                    seconds_from=tmp_path / "full" / "gens.jsonl")
    naive = [r["gpu_s"] for r in rs.ChainedLog(tmp_path / "naive" / "gens.jsonl").records]
    kept = [r["gpu_s"] for r in rs.ChainedLog(tmp_path / "kept" / "gens.jsonl").records]
    assert naive == pytest.approx([3.0, 3.0], abs=1e-4)  # the bug this guards against
    assert kept == pytest.approx([1.2, 1.2], abs=1e-4)
