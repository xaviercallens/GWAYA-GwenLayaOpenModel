"""Offline tests for scripts/run_study.py: fake backend, fake checker, no GPU/network/sandbox."""
import json
import types

import pytest

from gwaya.domains.task import CheckResult, Task
from scripts import run_study as rs

PLAN = {"stages": [], "seeds": {"eval_order": 7, "generation": "g"},
        "generation_settings": {"max_new_tokens": {"python": 64, "math": 64, "rust": 64, "lean4": 64}}}
STAGE = {"id": "T", "models": ["small", "big"], "quant": ["q4_K_M"], "arms": []}


class FakeBackend(rs.Backend):
    kind, url = "fake", "http://localhost:1"

    def __init__(self, logprob=-0.1):
        self.calls, self.logprob = [], logprob

    def generate(self, model, prompt, domain, temperature, max_tokens, seed):
        self.calls.append((model, seed, temperature))
        good = model == "big" or temperature > 0
        return {"text": f"```python\n{'ok' if good else 'bad'}\n```", "prompt_tokens": 3, "completion_tokens": 10,
                "eval_s": 0.5, "gpu_s": 1.0, "mean_logprob": self.logprob, "done_reason": "stop"}

    def info(self):
        return {"kind": "fake", "version": "0"}


def checker(task, text):
    want = task.checker_payload.get("want", "ok")
    return CheckResult("VERIFIED" if f"\n{want}\n" in text else "FAILED")


def mk_tasks(n=3, gate=True):
    out = []
    for i in range(n):
        payload = {"want": "ok", "__gate__": {"want": "ok"}} if gate else {"want": "ok"}
        out.append(Task("python", f"t{i}", f"p{i}", payload))
    return out


def make(tmp_path, arms, backend=None, mode="all", tasks=None, **kw):
    return rs.Study(stage=STAGE, plan=PLAN, tasks=tasks or mk_tasks(), models=["small", "big"], quants=["q4_K_M"],
                    arms=arms, backend=backend or FakeBackend(), out_dir=tmp_path, mode=mode, checker=checker, **kw)


def summ(study):
    return rs.summarize_rows(study.rows.records)


def test_arms_and_one_cache(tmp_path):
    be = FakeBackend()
    st = make(tmp_path, ["base", "gate_only", "always_smallest", "always_largest", "raw_confidence"], be)
    st.run()
    # greedy once per (model, task): 2 models x 3 tasks, shared by five arms
    assert len(be.calls) == 6 and st.new_gens == 6
    s = summ(st)
    assert s["base|small|q4_K_M|python"]["final_verified"] == 0
    assert s["base|big|q4_K_M|python"]["final_verified"] == 3
    assert s["always_largest|big|q4_K_M|python"]["final_verified"] == 3
    assert s["gate_only|small|q4_K_M|python"]["answered"] == 0       # gate refutes the bad code
    assert s["gate_only|big|q4_K_M|python"]["answered_and_verified"] == 3
    assert s["raw_confidence|big|q4_K_M|python"]["answered_unknown"] == 3   # --tau-b3 unset


def test_seed_scheme_and_resume(tmp_path):
    be = FakeBackend()
    st = make(tmp_path, ["base"], be)
    st.run()
    seeds = {c[1] for c in be.calls}
    assert seeds == {1000, 2000, 3000}                      # 1000*(item_index+1)+0
    be2 = FakeBackend()
    st2 = make(tmp_path, ["base"], be2)
    st2.run()
    assert be2.calls == [] and st2.new_rows == 0           # fully resumed


def test_resume_after_kill_mid_run(tmp_path):
    be = FakeBackend()
    st = make(tmp_path, ["base"], be)
    t = st.tasks[0]
    st.run_task(t)                                          # simulate a run that died after one item
    with open(tmp_path / "rows.jsonl", "ab") as fh:         # and a torn write
        fh.write(b'{"key": "x", "pre')
    be2 = FakeBackend()
    st2 = make(tmp_path, ["base"], be2)
    assert st2.rows.torn_tail_dropped
    st2.run()
    assert len(be2.calls) == 4                              # 2 remaining items x 2 models
    assert len(st2.rows.records) == 6


def test_hash_chain_detects_tampering(tmp_path):
    st = make(tmp_path, ["base"])
    st.run()
    p = tmp_path / "rows.jsonl"
    lines = p.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["score"] = "TAMPERED"
    lines[1] = json.dumps(rec)
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(rs.ChainError):
        rs.ChainedLog(p)
    # dropping a middle record also breaks the chain
    del lines[1]
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(rs.ChainError):
        rs.ChainedLog(p)


def test_self_consistency(tmp_path):
    be = FakeBackend()
    st = make(tmp_path, ["self_consistency"], be, sc_k=3, tau_b4=0.6)
    st.run()
    assert len(be.calls) == 9 and all(c[2] == 0.8 for c in be.calls)
    row = st.rows.records[0]
    assert row["confidence"] == 1.0 and row["answered"] and row["score"] == "VERIFIED" and row["k"] == 3


def test_raw_confidence_threshold(tmp_path):
    st = make(tmp_path, ["raw_confidence"], FakeBackend(logprob=-0.1), tau_b3=-0.5)
    st.run()
    assert all(r["answered"] is True for r in st.rows.records)
    d = tmp_path / "b"
    st2 = make(d, ["raw_confidence"], FakeBackend(logprob=None), tau_b3=-0.5)
    st2.run()
    assert all(r["answered"] is None and "no logprobs" in r["unavailable_reason"] for r in st2.rows.records)


def test_gwenlaya_arm_escalates_and_counts_cost(tmp_path):
    st = make(tmp_path, ["gwenlaya"])
    st.run()
    r = st.rows.records[0]
    assert r["gate"] == "VERIFIED" and r["answered"] and r["model"] == "big"
    assert r["tiers_invoked"] == ["small", "big"] and r["gpu_s"] == 2.0


def test_gate_payload_absent_gives_unverified(tmp_path):
    st = make(tmp_path, ["gate_only"], tasks=mk_tasks(2, gate=False), mode="all",
              )
    # real gate checker with empty payload: stub says FAILED/VERIFIED by text, so use the real one
    from gwaya.domains import checkers
    st.checker = lambda t, r: checkers.check(t, r) if not t.checker_payload.get("want") else checker(t, r)
    st.run()
    big = [r for r in st.rows.records if r["model"] == "big"]
    assert all(r["gate"] == "UNVERIFIED" and r["answered"] is False for r in big)


def test_generate_then_score_mode(tmp_path):
    be = FakeBackend()
    gen = make(tmp_path, ["gate_only", "gwenlaya", "self_consistency"], be, mode="generate", sc_k=2)
    gen.run()
    assert gen.rows.records == []                           # no scoring in generate mode
    n = len(be.calls)
    assert n == 2 * 3 + 2 * 3                               # greedy (2 models x 3) + 2 samples x 3
    be2 = FakeBackend()
    sc = make(tmp_path, ["gate_only", "gwenlaya", "self_consistency"], be2, mode="score", sc_k=2)
    sc.run()
    assert be2.calls == [] and len(sc.rows.records) > 0
    miss = make(tmp_path / "other", ["base"], FakeBackend(), mode="score")
    with pytest.raises(rs.CacheMiss):
        miss.run()


def test_throughput_and_results(tmp_path):
    import time
    from datetime import datetime, timezone
    st = make(tmp_path, ["base"])
    st.run()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}")
    res = rs.build_results(st, started=datetime.now(timezone.utc), t0=time.perf_counter(), backend_info={"k": 1},
                           skipped=[], args_echo={}, plan_path=plan_path, status="COMPLETE", prior=None)
    assert res["throughput"]["big|q4_K_M"]["tokens_per_s"] == 20.0
    assert res["gpu_seconds_total"] == 6.0
    assert res["peak_vram_mib"]["overall"] is None          # sampler disabled -> null, not 0
    assert res["chain_heads"]["rows"] == st.rows.head
    assert "git" in res and "wall_clock" in res and res["seeds"]["eval_order"] == 7


def test_vram_sampler_peak_and_absent():
    vals = iter([[100.0], [300.0], [200.0], [200.0]])
    vs = rs.VramSampler(reader=lambda: next(vals), enabled=True)
    vs.label = "m|q"
    for _ in range(3):
        vs.sample()
    assert vs.peak["m|q"] == 300.0 and vs.overall == 300.0
    none = rs.VramSampler(reader=lambda: None)
    none.sample()
    assert none.overall is None
    fake = lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="1024\n2048\n", stderr="")  # noqa: E731
    assert rs.nvidia_smi_used_mib(runner=fake) == [1024.0, 2048.0]
    bad = lambda *a, **k: types.SimpleNamespace(returncode=9, stdout="", stderr="x")  # noqa: E731
    assert rs.nvidia_smi_used_mib(runner=bad) is None


def test_served_name_and_arm_resolution():
    assert rs.served_name("qwen3.5:9b-q4_K_M", "q4_K_M", {}) == ("qwen3.5:9b-q4_K_M", "tag")
    assert rs.served_name("qwen3.5:9b-q4_K_M", "q8_0", {}) == ("qwen3.5:9b-q8_0", "tag_rewritten")
    assert rs.served_name("Qwen/Qwen3.5-9B", "bf16", {}) == ("Qwen/Qwen3.5-9B", "label_only")
    assert rs.served_name("m", "nf4", {"m@nf4": "x"}) == ("x", "map")
    arms, skipped = rs.resolve_arms(["B1", "B2", "B3", "B4", "B5", "GL", "A0", "LR", "continuity"])
    assert arms == ["always_smallest", "always_largest", "raw_confidence", "self_consistency", "gate_only",
                    "gwenlaya", "base"]
    assert {s["arm"] for s in skipped} == {"LR", "continuity"}


def test_lake_dest_is_confined_and_sync_uses_runner(tmp_path):
    assert rs.lake_dest("S2", "rows.jsonl") == rs.LAKE_PREFIX + "runs/S2/rows.jsonl"
    for bad in ("../x", "a/b", "", "-x"):
        with pytest.raises(ValueError):
            rs.lake_dest(bad, "rows.jsonl")
    (tmp_path / "rows.jsonl").write_text("{}\n")
    cmds = []

    def runner(cmd, **k):
        cmds.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    out = rs.lake_sync(tmp_path, "S2", runner)
    assert out["rows.jsonl"]["ok"] and cmds == [["gcloud", "storage", "cp", str(tmp_path / "rows.jsonl"),
                                                 rs.LAKE_PREFIX + "runs/S2/rows.jsonl"]]


def write_plan(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({**PLAN, "stages": [{**STAGE, "arms": ["B1", "B2", "LR"]}]}))
    t = tmp_path / "tasks.jsonl"
    t.write_text("\n".join(json.dumps({"domain": "python", "task_id": f"t{i}", "prompt": "p",
                                       "checker_payload": {"want": "ok"}, "gate_payload": {"want": "ok"}})
                           for i in range(2)))
    return p, t


def test_main_refuses_scoring_without_sandbox(tmp_path, monkeypatch, capsys):
    from gwaya import sandbox
    monkeypatch.setattr(sandbox, "is_sandbox_available", lambda: False)
    p, t = write_plan(tmp_path)
    be = FakeBackend()
    rc = rs.main(["--plan", str(p), "--stage", "T", "--tasks", str(t), "--out", str(tmp_path / "o")], backend=be)
    assert rc == 2 and be.calls == [] and "REFUSED" in capsys.readouterr().err
    monkeypatch.setattr(sandbox, "is_sandbox_available", lambda: True)
    monkeypatch.setenv(sandbox.ALLOW_UNISOLATED_ENV, "1")
    assert rs.main(["--plan", str(p), "--stage", "T", "--tasks", str(t), "--out", str(tmp_path / "o")], backend=be) == 2


def test_main_end_to_end_with_lake_sync(tmp_path, monkeypatch):
    from gwaya import sandbox
    monkeypatch.setattr(sandbox, "is_sandbox_available", lambda: True)
    monkeypatch.delenv(sandbox.ALLOW_UNISOLATED_ENV, raising=False)
    p, t = write_plan(tmp_path)
    cmds = []

    def runner(cmd, **k):
        cmds.append(cmd)
        return types.SimpleNamespace(returncode=1 if "rows.jsonl" in cmd[3] and cmd[3].startswith("gs://") else 0,
                                     stdout="", stderr="")

    out = tmp_path / "o"
    rc = rs.main(["--plan", str(p), "--stage", "T", "--tasks", str(t), "--out", str(out), "--lake-sync", "--no-vram"],
                 backend=FakeBackend(), checker=checker, runner=runner)
    assert rc == 0
    res = json.loads((out / "results.json").read_text())
    assert res["status"] == "COMPLETE" and res["arms"] == ["always_smallest", "always_largest"]
    assert res["arms_skipped"][0]["arm"] == "LR"
    assert res["counts"]["rows"] == 4 and res["peak_vram_mib"]["overall"] is None
    uploads = [c for c in cmds if c[:3] == ["gcloud", "storage", "cp"] and c[4].startswith("gs://")]
    assert uploads and all(c[4].startswith(rs.LAKE_PREFIX + "runs/T/") for c in uploads)


def test_main_generate_mode_needs_no_sandbox(tmp_path, monkeypatch):
    from gwaya import sandbox
    monkeypatch.setattr(sandbox, "is_sandbox_available", lambda: False)
    p, t = write_plan(tmp_path)
    rc = rs.main(["--plan", str(p), "--stage", "T", "--tasks", str(t), "--out", str(tmp_path / "o"),
                  "--mode", "generate", "--no-vram"], backend=FakeBackend())
    assert rc == 0 and not (tmp_path / "o" / "rows.jsonl").exists()


def test_main_rejects_cpu_stage_and_unknown_stage(tmp_path, capsys):
    plan = json.loads((rs.ROOT / "experiments" / "plan.json").read_text())
    assert plan["stages"]
    t = tmp_path / "t.jsonl"
    t.write_text("")
    assert rs.main(["--stage", "C0", "--tasks", str(t)]) == 2
    assert rs.main(["--stage", "NOPE", "--tasks", str(t)]) == 2


def test_openai_backend_parsing(monkeypatch):
    be = rs.OpenAICompatBackend("http://localhost:8000/v1")
    seen = {}

    def post(path, payload):
        seen.update(path=path, payload=payload)
        return {"choices": [{"text": "x = 1\n", "finish_reason": "stop",
                             "logprobs": {"token_logprobs": [-1.0, -3.0]}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2}}

    monkeypatch.setattr(be, "_post", post)
    out = be.generate("m", "write x", "python", 0.0, 32, 1000)
    assert seen["path"] == "/completions" and seen["payload"]["seed"] == 1000 and seen["payload"]["stop"] == rs.STOP
    assert out["text"] == "```python\nx = 1\n```" and out["mean_logprob"] == -2.0 and out["completion_tokens"] == 2


def test_openai_backend_records_logprobs_and_cpu_seconds(monkeypatch):
    import os
    be = rs.OpenAICompatBackend("http://localhost:8000/v1", cpu_pid=os.getpid())
    monkeypatch.setattr(be, "_post", lambda path, payload: {
        "choices": [{"text": "x", "finish_reason": "stop", "logprobs": {"token_logprobs": [-1.0, -3.0]}}],
        "usage": {}})
    out = be.generate("m", "p", "python", 0.0, 8, 1)
    assert out["token_logprobs"] == [-1.0, -3.0]
    assert isinstance(out["cpu_seconds"], float) and out["cpu_seconds"] >= 0.0
    assert rs.proc_cpu_seconds(None) is None and rs.proc_cpu_seconds(2 ** 30) is None


def test_openai_backend_parses_llamacpp_logprobs_content(monkeypatch):
    be = rs.OpenAICompatBackend("http://localhost:8000/v1")
    monkeypatch.setattr(be, "_post", lambda path, payload: {
        "choices": [{"text": "x", "finish_reason": "stop",
                     "logprobs": {"content": [{"token": "a", "logprob": -1.0}, {"token": "b", "logprob": -3.0}]}}],
        "usage": {}})
    out = be.generate("m", "p", "python", 0.0, 8, 1)
    assert out["token_logprobs"] == [-1.0, -3.0] and out["mean_logprob"] == -2.0
