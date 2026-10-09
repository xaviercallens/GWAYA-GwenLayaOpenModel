import json

from scripts.data import build_laya_data as B


def _w(p, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_builds_router_and_calibrator_rows(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    row = lambda arm, m, tid, **kw: {"arm": arm, "model": m, "domain": "math", "task_id": tid, **kw}
    _w(d / "rows.jsonl", [row("base", "t1", "ma/a", score="VERIFIED"), row("gate_only", "t1", "ma/a", gate="VERIFIED"),
                          row("base", "t2", "ma/a", score="FAILED"), row("gate_only", "t2", "ma/a", gate="FAILED")])
    gen = lambda m, **kw: {"model": m, "domain": "math", "task_id": "ma/a", "temperature": 0.0, "text": "ans", **kw}
    _w(d / "gens.jsonl", [gen("t1", mean_logprob=-0.1, token_logprobs=[-0.1, -0.9]),
                          gen("t2", mean_logprob=-0.5, token_logprobs=[-0.2]),
                          gen("t1", kind="pot", text="program", token_logprobs=[-5.0])])
    tasks = [{"domain": "math", "task_id": "ma/a", "prompt": "p"}, {"domain": "math", "task_id": "ma/zz", "prompt": "q"}]
    router, calib, counts = B.build(tasks, B.index_study([d]), ["t1", "t2", "t3"])
    assert router == [{"source": "ma", "id": "a", "domain": "math", "prompt": "p", "tiers": {"t1": 1, "t2": 0}}]
    assert counts["tasks_without_any_tier"] == 1 and counts["calibrator_rows"] == 2
    c1 = next(c for c in calib if c["tier"] == "t1")
    assert c1["signals"] == {"gate": "verified", "mean_logprob": -0.1, "min_logprob": -0.9}  # pot program ignored
    assert c1["candidate"] == "ans" and c1["correct"] == 1
