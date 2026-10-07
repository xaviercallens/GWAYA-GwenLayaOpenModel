"""Offline tests for gwaya.data_build and scripts/data/build_{sft_dataset,dpo_pairs}.py.

Everything uses the FakeGenerator / fake checker or patched transport: no network, GPU, Ollama,
Rust or Lean toolchain. One test runs the real python checker (isolated subprocess harness)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "data"))

from gwaya import data_build as db  # noqa: E402
from gwaya.domains.task import CheckResult, Task  # noqa: E402

import build_dpo_pairs  # noqa: E402
import build_sft_dataset  # noqa: E402
import make_dataset_card  # noqa: E402

FIX = ROOT / "scripts" / "data" / "fixtures"


def _gate(tmp_path, items=()):
    idx = tmp_path / "idx.jsonl"
    idx.write_text("".join(json.dumps(i) + "\n" for i in items) or
                   json.dumps({"eval_set": "e", "item_id": "z", "domain": "math",
                               "fields": {"prompt": "unrelated eval problem about prime numbers and nothing else here at all"}}) + "\n")
    return db.DecontamGate.from_index(idx, check_plan=False, manifest_path=None)


def _bt(tid="p_ok_0", domain="python", payload=None, prompt="Write f.", source="s", lic="mit"):
    return db.BuildTask(Task(domain, tid, prompt, payload or {"tests": "assert 1", "entry_point": "f"}), source, lic)


def _cfg(tmp_path, ladder, **kw):
    return db.BuildConfig(db.SampleConfig(**kw.pop("sample", {})), ladder, [], dict(db.DEFAULT_CAPS),
                          db.DOMAINS, kw.pop("check_fn", db.fake_check), "t", _gate(tmp_path), True)


# ---- plumbing -------------------------------------------------------------------------------

def test_parse_rung_kinds():
    o = db.parse_rung("ollama@http://h:11434#qwen2.5-coder:7b")
    assert isinstance(o, db.OllamaGenerator) and o.model == "qwen2.5-coder:7b" and o.base_url == "http://h:11434"
    v = db.parse_rung("openai@http://h:8000/v1#Qwen/Qwen3.5-9B")
    assert isinstance(v, db.OpenAICompatGenerator) and v.model == "Qwen/Qwen3.5-9B"
    with pytest.raises(ValueError):
        db.parse_rung("nonsense")
    with pytest.raises(ValueError):
        db.parse_rung("grpc@x#m")


def test_split_bucket_matches_plan_rule():
    import hashlib
    assert db.split_bucket("a", "b") == int(hashlib.sha256(b"ab").hexdigest(), 16) % 20


def test_license_rules():
    assert db.license_problem(_bt(lic="")) == "no_license"
    assert db.license_problem(_bt(lic="cc-by-nc-4.0")) == "license_not_permitted"
    assert db.license_problem(_bt(source="KodCode/x", lic="mit")) == "excluded_source"
    assert db.license_problem(_bt(lic="derived")) is None
    assert db.license_problem(_bt(lic="apache-2.0")) is None


def test_ollama_and_openai_payloads(monkeypatch):
    seen = []

    def fake_post(url, payload, headers, timeout):
        seen.append((url, payload, headers))
        return ({"message": {"content": "A"}} if "/api/chat" in url
                else {"choices": [{"message": {"content": "B"}}]})
    monkeypatch.setattr(db, "_post_json", fake_post)
    monkeypatch.setenv("GWAYA_OPENAI_API_KEY", "k")
    assert db.OllamaGenerator("m", base_url="http://h:1").generate("p", seed=7, temperature=.8, top_p=.95, max_tokens=5) == "A"
    assert db.OpenAICompatGenerator("m", base_url="http://h:2").generate("p", seed=7, temperature=.8, top_p=.95, max_tokens=5) == "B"
    (u1, p1, _), (u2, p2, h2) = seen
    assert u1 == "http://h:1/api/chat" and p1["think"] is False and p1["options"]["seed"] == 7
    assert u2 == "http://h:2/v1/chat/completions" and h2["Authorization"] == "Bearer k"
    assert p2["chat_template_kwargs"] == {"enable_thinking": False} and p2["seed"] == 7


def test_unreachable_endpoint_is_not_a_failed_candidate(tmp_path):
    class Down(db.Generator):
        def generate(self, *a, **k):
            raise db.GeneratorUnavailable("down")
    st = db.TaskState(0, _bt())
    cfg = db.SampleConfig()
    db.sample_stage([st], [Down("d")], cfg, db.fake_check)
    assert st.cands == [] and st.gen_errors


# ---- sampling, seeds, escalation ------------------------------------------------------------

def test_seeds_escalation_and_early_stop():
    ok = lambda i: f"```python\n# FIXTURE_OK\ndef f():\n    return {i}\n```"
    small = db.FakeGenerator("small", default="```python\n# bad\n```")
    large = db.FakeGenerator("large", responses={"Write f.": [None] * 4 + [ok(1), ok(1), ok(2), ok(3)]})
    st = db.TaskState(4, _bt())
    db.sample_stage([st], [small, large], db.SampleConfig(k={**db.DEFAULT_K, "python": 4}), db.fake_check)
    assert [c.seed for c in st.cands][:3] == [5000, 5001, 5002]            # 1000*(idx+1)+call
    assert [c.model for c in st.cands[:4]] == ["small"] * 4                 # cheap rung first
    assert st.cands[4].model == "large"
    assert len(st.verified) >= 2 and len(st.cands) <= 8
    assert len({c.norm_hash for c in db._dedup_pick(st.verified, 2)}) == 2


# ---- verdict hygiene ------------------------------------------------------------------------

def test_infra_and_unverified_classification():
    assert db.is_infra(CheckResult("UNVERIFIED", {}))
    assert db.is_infra(CheckResult("FAILED", {"reason": "rust_linker_missing"}))
    assert db.is_infra(CheckResult("FAILED", {"error": "Command timed out"}))
    assert db.is_infra(CheckResult("FAILED", {"error": "No such file or directory: rustc"}))
    assert not db.is_infra(CheckResult("FAILED", {"error": "assertion failed"}))
    assert not db.is_infra(CheckResult("VERIFIED", {}))


def _cand(c, status, resp, cat=None, infra=False, audit=(), seed=0):
    return db.Candidate(task_key=c.key, task_id=c.task.task_id, domain=c.task.domain, source=c.source,
                        license=c.license, prompt=c.task.prompt, response=resp, rung="fake:m", model="m",
                        endpoint_kind="fake", seed=seed, call_index=seed, round=1, status=status, infra=infra,
                        reason="", error="", audit=list(audit), failure_category=cat, gate_pass=None,
                        norm_hash=db._norm_hash(c.task.domain, resp))


def test_dpo_never_uses_unverified_or_infra_as_rejected(tmp_path):
    bt = _bt()
    cs = [_cand(bt, "VERIFIED", "good answer here"), _cand(bt, "UNVERIFIED", "u", infra=True, seed=1),
          _cand(bt, "FAILED", "infra", infra=True, seed=2)]
    drops: dict = {}
    pairs, excl = db.build_pairs({bt.key: cs}, {bt.key: bt}, _cfg(tmp_path, []), drops)
    assert pairs == [] and "no_pair_possible" in drops and excl["unverified_or_infra_ignored"] == 2
    cs.append(_cand(bt, "FAILED", "genuinely wrong", cat="wrong", seed=3))
    pairs, _ = db.build_pairs({bt.key: cs}, {bt.key: bt}, _cfg(tmp_path, []), {})
    assert len(pairs) == 1
    p = pairs[0]
    assert p["rejected"][0]["content"] == "genuinely wrong" and p["provenance"]["rejected"]["checker_status"] == "FAILED"
    assert p["provenance"]["chosen"]["checker_status"] == "VERIFIED"
    assert p["prompt"][0]["content"].startswith("Write f.")
    assert p["length"]["ratio"] == round(len("good answer here") / len("genuinely wrong"), 4)


def test_dpo_prefers_confident_wrong_and_caps_shallow(tmp_path):
    bts = [_bt(f"t_{i}_0", prompt=f"Write f{i}.") for i in range(8)]
    by, tasks = {}, {}
    for i, bt in enumerate(bts):
        shallow = i < 6
        by[bt.key] = [_cand(bt, "VERIFIED", f"ok {i}"),
                      _cand(bt, "FAILED", f"syn {i}", cat="compile", seed=1),
                      *([] if shallow else [_cand(bt, "FAILED", f"cw {i}", cat="confident_wrong", seed=2)])]
        tasks[bt.key] = bt
    drops: dict = {}
    pairs, _ = db.build_pairs(by, tasks, _cfg(tmp_path, []), drops)
    cats = [p["provenance"]["rejected_category"] for p in pairs]
    assert cats.count("confident_wrong") == 2
    assert cats.count("compile") <= 0.25 * len(pairs)
    assert len(drops["shallow_pair_over_25pct_cap"]) == 6 - cats.count("compile")


def test_length_stats():
    s = db.length_stats([0.5, 1.0, 2.0])
    assert s["n"] == 3 and s["min"] == 0.5 and s["max"] == 2.0 and s["median"] == 1.0
    assert abs(s["frac_chosen_longer"] - 1 / 3) < 1e-3
    assert db.length_stats([]) == {"n": 0}


# ---- decontamination ------------------------------------------------------------------------

def test_decontam_gate_flags_planted_overlap_and_fails_closed(tmp_path):
    text = ("Return the sum of the squares of all odd numbers in a list of integers while ignoring "
            "negative values and any non integer entries and return zero for an empty list")
    gate = _gate(tmp_path, [{"eval_set": "ev", "item_id": "1", "domain": "python", "fields": {"prompt": text}}])
    assert gate.check_task(_bt(prompt=text))
    assert not gate.check_task(_bt(tid="other_0", prompt="Compute the nth Fibonacci number modulo a prime using fast doubling"))
    with pytest.raises(db.DecontamAborted):
        db.DecontamGate.from_index(_write(tmp_path, {"hf_id": "x/y", "domain": "python", "role": "eval"}),
                                   check_plan=False, manifest_path=None)
    ok = db.DecontamGate.from_index(_write(tmp_path, {"hf_id": "x/y", "domain": "python", "role": "eval"}),
                                    allow_missing=True, check_plan=False, manifest_path=None)
    assert ok.report["complete"] is False and ok.report["missing_eval_sets"]


def _write(tmp_path, rec):
    p = tmp_path / "desc.jsonl"
    p.write_text(json.dumps(rec) + "\n")
    return p


def test_prefilter_rules(tmp_path):
    cfg = _cfg(tmp_path, [])
    ts = [_bt("a_0", lic=""), _bt("b_0", lic="mit"), _bt("c_0", lic="mit")]
    ts[2].split = "test"
    ts.append(_bt("t_triv", payload={"tests": "x", "entry_point": "g", "fixture_trivial": True}))
    drops: dict = {}
    kept = db.prefilter(ts, cfg, drops)
    reasons = {k for k in drops}
    assert "no_license" in reasons and "test_split" in reasons and "tests_accept_trivial_baseline" in reasons
    assert all(k.task.task_id != "a_0" for k in kept)


# ---- translation ----------------------------------------------------------------------------

def test_translation_guards():
    row = {"task_id": "p0", "source": "s", "license": "mit", "prompt": "Add.", "solution": "def add(a,b): return a+b",
           "tests": "assert add(1, 2) == 3"}
    good = "```rust\n// FIXTURE_OK\nfn add(a: i64, b: i64) -> i64 { a + b }\n```\n```rust_tests\nassert_eq!(add(1, 2), 3);\n```"
    halluc = good.replace("3);", "99);")
    bad_rust = good.replace("FIXTURE_OK", "nothing")
    for text, key, n in [(good, "accepted", 1), (halluc, "low_literal_overlap", 0), (bad_rust, "rust_not_verified", 0),
                         ("no blocks", "unparseable", 0)]:
        gen = db.FakeGenerator("t", responses={"Source task id: p0": [text]})
        out, st = db.translate_python_rows([row], gen, db.fake_check)
        assert len(out) == n and st[key] == 1, (key, st)
    t = out if out else db.translate_python_rows([row], db.FakeGenerator("t", responses={"Source task id: p0": [good]}), db.fake_check)[0]
    assert t[0].task.domain == "rust" and t[0].source == "translated:s" and t[0].extra["translated_from"] == "p0"
    assert "fn add(a: i64, b: i64) -> i64" in t[0].task.prompt


# ---- lean expert iteration ------------------------------------------------------------------

def test_lean_expert_iteration_only_resamples_unsolved(tmp_path):
    stmt = "theorem q : 1 = 1 := by"
    mk = lambda tid: db.BuildTask(Task("lean4", tid, "Prove it.\n" + stmt + tid, {"formal_statement": stmt}), "s", "mit")
    solved = f"```lean4\n-- FIXTURE_OK\n{stmt}\n  rfl\n```"
    r1 = db.FakeGenerator("r1", responses={"qa": [solved, solved.replace("rfl", "simp")], "qb": ["```lean4\n" + stmt + "\n  skip\n```"]})
    r2 = db.FakeGenerator("r2", responses={"qb": [None] * 3 + [solved, solved.replace("rfl", "simp")]})
    cfg = db.BuildConfig(db.SampleConfig(k={**db.DEFAULT_K, "lean4": 3}), [r1], [[r1], [r2]], dict(db.DEFAULT_CAPS),
                         ["lean4"], db.fake_check, "t", _gate(tmp_path), True)
    # task ids chosen so the prompt substrings qa / qb select the right canned lists
    tasks = [mk("qa"), mk("qb")]
    states, info = db.run_sampling(tasks, cfg)
    by = {s.bt.task.task_id: s for s in states}
    assert {c.round for c in by["qa"].cands} == {1}                    # solved in round 1: not resampled
    assert {c.round for c in by["qb"].cands} == {1, 2}
    assert info["rounds"]["lean4:1"]["solved_total"] == 1 and info["rounds"]["lean4:2"]["solved_total"] == 2
    assert [c.model for c in by["qb"].verified] == ["r2", "r2"]


# ---- end to end -----------------------------------------------------------------------------

def _rows(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines()]


def test_sft_dry_run_end_to_end(tmp_path):
    out = tmp_path / "sft"
    assert build_sft_dataset.main(["--dry-run", "--out-dir", str(out)]) == 0
    rows = _rows(out / "sft.jsonl")
    assert rows and all(r["provenance"]["checker_status"] == "VERIFIED" and r["provenance"]["dry_run"] for r in rows)
    for r in rows:
        assert db.provenance_complete(r["provenance"])
        assert len(r["provenance"]["sha256"]) == 64 and r["provenance"]["checker_version"] == "fake-dryrun-checker"
        assert r["messages"][1]["content"] and "FIXTURE_OK" in r["messages"][1]["content"]
    ids = {r["task_id"] for r in rows}
    assert not any(i.startswith(p) for i in ids for p in ("py_leak", "py_trivial", "py_nolicense", "py_reserved", "py_never"))
    assert {r["domain"] for r in rows} == set(db.DOMAINS)
    per_task = {}
    for r in rows:
        per_task[r["task_id"]] = per_task.get(r["task_id"], 0) + 1
    assert max(per_task.values()) <= 2
    rep = json.loads((out / "build_report.json").read_text())
    assert rep["decontam_verification"]["ok"] and rep["decontam_task_removed"] == 1 and rep["dry_run"]
    assert rep["lean"]["eval_only_rule_triggered"] is True and rep["output_sha256"] == db.file_sha256(out / "sft.jsonl")
    card = (out / "DATASET_CARD.md").read_text()
    assert "DRY RUN" in card and "Decontamination" in card and "fake-dryrun-checker" in card
    assert (out / "candidates.jsonl").exists() and (out / "flagged_decontam.jsonl").read_text().strip()


def test_sft_dry_run_is_deterministic_and_translates(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    build_sft_dataset.main(["--dry-run", "--out-dir", str(a)])
    build_sft_dataset.main(["--dry-run", "--out-dir", str(b)])
    assert (a / "sft.jsonl").read_bytes() == (b / "sft.jsonl").read_bytes()
    assert build_sft_dataset.main(["--dry-run", "--out-dir", str(c), "--rust-translate-from", str(a / "sft.jsonl")]) == 0
    rep = json.loads((c / "build_report.json").read_text())
    assert rep["rust_translation"]["accepted"] == 1 and rep["rust_translation"]["low_literal_overlap"] == 1
    tr = [r for r in _rows(c / "sft.jsonl") if r["source"].startswith("translated:")]
    assert tr and tr[0]["provenance"]["translated_from"].startswith("py_add") and tr[0]["domain"] == "rust"


def test_dpo_dry_run_end_to_end_and_candidate_reuse(tmp_path):
    out = tmp_path / "dpo"
    assert build_dpo_pairs.main(["--dry-run", "--out-dir", str(out)]) == 0
    pairs = _rows(out / "dpo_pairs.jsonl")
    assert pairs
    keys = [(p["source"], p["task_id"]) for p in pairs]
    assert len(keys) == len(set(keys))                                   # one pair per prompt
    for p in pairs:
        assert p["provenance"]["chosen"]["checker_status"] == "VERIFIED"
        assert p["provenance"]["rejected"]["checker_status"] == "FAILED"
        assert p["chosen"] != p["rejected"] and db.provenance_complete(p["provenance"]["rejected"])
    assert not any(p["task_id"].startswith("rs_infra") and "FIXTURE_INFRA" in p["rejected"][0]["content"] for p in pairs)
    rep = json.loads((out / "build_report.json").read_text())
    assert rep["length_ratio"]["n"] == len(pairs) and rep["decontam_verification"]["ok"]
    out2 = tmp_path / "dpo2"
    assert build_dpo_pairs.main(["--dry-run", "--candidates", str(out / "candidates.jsonl"), "--out-dir", str(out2)]) == 0
    assert {p["id"] for p in _rows(out2 / "dpo_pairs.jsonl")} == {p["id"] for p in pairs}


def test_cli_fails_closed_without_decontam_index(tmp_path):
    with pytest.raises(SystemExit):
        build_sft_dataset.main(["--tasks", str(FIX / "dryrun_tasks.jsonl"), "--ladder", "fake@#x", "--out-dir", str(tmp_path)])


def test_cli_aborts_when_eval_set_missing(tmp_path, capsys):
    idx = _write(tmp_path, {"hf_id": "x/y", "domain": "python", "role": "eval"})
    rc = build_sft_dataset.main(["--tasks", str(FIX / "dryrun_tasks.jsonl"), "--ladder", "fake@#x",
                                 "--decontam-index", str(idx), "--no-plan-check", "--out-dir", str(tmp_path / "o")])
    assert rc == 2 and not (tmp_path / "o" / "sft.jsonl").exists()


def test_card_regeneration_matches(tmp_path):
    out = tmp_path / "o"
    build_sft_dataset.main(["--dry-run", "--out-dir", str(out)])
    orig = (out / "DATASET_CARD.md").read_text()
    make_dataset_card.main([str(out / "build_report.json"), "--out", str(tmp_path / "card.md")])
    assert (tmp_path / "card.md").read_text() == orig


# ---- real python checker (offline subprocess harness) ----------------------------------------

def test_real_python_checker_through_pipeline(tmp_path):
    from gwaya.domains import check
    t = db.BuildTask(Task("python", "real_sq_0", "Write sq(x) returning x squared.",
                          {"tests": "assert sq(3) == 9\nassert sq(-2) == 4", "entry_point": "sq"}), "s", "mit")
    good = "```python\ndef sq(x):\n    return x * x\n```"
    bad = "```python\ndef sq(x):\n    return x + x\n```"
    gen = db.FakeGenerator("g", responses={"Write sq": [bad, good, good.replace("x * x", "x ** 2")]})
    cfg = db.BuildConfig(db.SampleConfig(), [gen], [], dict(db.DEFAULT_CAPS), ["python"], check, "real", _gate(tmp_path), False)
    assert check(t.task, db.baseline_response(t.task)).status != "VERIFIED"
    states, _ = db.run_sampling([t], cfg)
    st = states[0]
    assert [c.status for c in st.cands][:3] == ["FAILED", "VERIFIED", "VERIFIED"]
    pairs, _ = db.build_pairs({t.key: st.cands}, {t.key: t}, cfg, {})
    assert len(pairs) == 1 and pairs[0]["rejected"][0]["content"] == bad
