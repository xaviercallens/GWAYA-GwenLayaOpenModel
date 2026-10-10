"""Offline tests for gwaya.laya_runtime (tiny random encoder, synthetic heads; sha verification bypassed) plus a
real-model PARITY test that is skipped unless the backbone, transformers/peft and the frozen files are present."""
import hashlib
import json
import math
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from gwaya import gwenlaya as g  # noqa: E402
from gwaya import laya_runtime as L  # noqa: E402
from gwaya.domains.task import CheckResult  # noqa: E402
from gwaya.laya_features import N_FEATURES, gate_features, render_calibrator_text  # noqa: E402

TIERS = ["qwen3.5-2b-bf16", "qwen3.5-4b-bf16", "qwen3.5-9b-bf16"]
H = 8


class FakeTok:
    """Whitespace tokenizer: id = 2 + hash(word) % 50; [CLS]=0 ... [SEP]=1; HF-like truncation (keeps the final SEP)."""

    def __init__(self, pad_to: int | None = None):
        self.pad_to = pad_to

    def _ids(self, text):
        return [0] + [2 + int(hashlib.md5(w.encode()).hexdigest(), 16) % 50 for w in text.split()] + [1]

    def __call__(self, text, truncation=False, max_length=None, return_tensors=None):
        ids = self._ids(text)
        if truncation and max_length and len(ids) > max_length:
            ids = ids[:max_length - 1] + [1]
        if return_tensors is None:
            return {"input_ids": ids}
        mask = [1] * len(ids)
        if self.pad_to and len(ids) < self.pad_to:
            mask += [0] * (self.pad_to - len(ids))
            ids += [1] * (self.pad_to - len(ids))
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])}


def make_encoder(seed=0):
    torch.manual_seed(seed)
    emb = torch.nn.Embedding(64, H)

    def enc(ids, mask):
        return emb(ids) * 3.0
    return enc


def const_head(n_in, n_out, bias):
    lin1, lin2 = torch.nn.Linear(n_in, H // 2), torch.nn.Linear(H // 2, n_out)
    torch.nn.init.zeros_(lin2.weight)
    with torch.no_grad():
        lin2.bias.copy_(torch.tensor(bias, dtype=torch.float32))
    return torch.nn.Sequential(lin1, torch.nn.GELU(), torch.nn.Dropout(0.1), lin2).eval()


def rand_head(n_in, n_out, seed=1):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(n_in, H // 2), torch.nn.GELU(), torch.nn.Dropout(0.1),
                               torch.nn.Linear(H // 2, n_out)).eval()


ISO = {"x": [0.0, 0.5, 1.0], "y": [0.0, 0.25, 1.0]}


def sig(z):
    return 1 / (1 + math.exp(-z))


def runtime(router_bias=(2.0, 0.0, -2.0), cal_bias=3.0, max_len=512, domains=("python", "math"), tier_map=None,
            router_head=None, cal_head=None, tok=None):
    tok = tok or FakeTok()
    r = L.HeadModel(tok, make_encoder(), router_head or const_head(H, 3, list(router_bias)), max_len, 0, TIERS)
    c = L.HeadModel(tok, make_encoder(), cal_head or const_head(H + N_FEATURES, 1, [cal_bias]), max_len, N_FEATURES, TIERS)
    return L.LayaRuntime(r, c, ISO, domains, tier_map, model_sha="test0000")


# ---- router / scorer units --------------------------------------------------------------------------------

def test_router_returns_tier_ordered_probabilities_and_evidence():
    rt = runtime()
    out = rt.router("write f", "python")
    assert list(out) == TIERS
    assert out[TIERS[0]] == pytest.approx(sig(2.0)) and out[TIERS[2]] == pytest.approx(sig(-2.0))
    assert out.evidence["applied"] and out.evidence["model_sha"] == "test0000" and out.evidence["p_tier"] == dict(out)


def test_router_uses_rendered_text_and_mean_pooling():
    rh = rand_head(H, 3)
    rt = runtime(router_head=rh)
    tok, enc = FakeTok(), make_encoder()
    ids = tok("[domain] python\n[prompt]\nwrite f", return_tensors="pt")["input_ids"]
    want = torch.sigmoid(rh(enc(ids, None).mean(1)))[0].tolist()
    got = rt.router("write f", "python")
    assert [got[t] for t in TIERS] == pytest.approx(want, abs=1e-6)


def test_masked_mean_pooling_ignores_padding():
    rh = rand_head(H, 3)
    a = runtime(router_head=rh, tok=FakeTok()).router("alpha beta", "python")
    b = runtime(router_head=rh, tok=FakeTok(pad_to=40)).router("alpha beta", "python")
    assert [a[t] for t in TIERS] == pytest.approx([b[t] for t in TIERS], abs=1e-6)


def test_tier_map_renames_and_unmapped_tiers_get_nothing():
    rt = runtime(tier_map={"local:small": TIERS[0]})
    out = rt.router("p", "python")
    assert out["local:small"] == pytest.approx(out[TIERS[0]])
    f = {"domain": "python", "tier": "local:small", "prompt": "p", "response": "c", "gate_status": "UNVERIFIED"}
    assert rt.scorer(f) is not None and f["laya"]["tier"] == TIERS[0]
    f2 = dict(f, tier="mystery:7b")
    assert rt.scorer(f2) is None and f2["laya"]["reason"].startswith("tier_not_in_laya_training")


def test_scorer_is_raw_sigmoid_and_isotonic_applied_by_artifact():
    rt = runtime(cal_bias=0.4)
    f = {"domain": "math", "tier": TIERS[1], "prompt": "q", "response": "\\boxed{4}", "gate_status": "UNVERIFIED",
         "gen_signals": {"mean_logprob": -0.2, "min_logprob": -3.0}}
    raw = rt.scorer(f)
    assert raw == pytest.approx(sig(0.4))
    iso = 0.25 + (raw - 0.5) * (0.75 / 0.5)
    assert f["laya"]["p_isotonic"] == pytest.approx(iso)
    art = rt.calibration_artifacts()["math"]
    assert art.model().predict(raw)[0] == pytest.approx(iso)
    assert math.isinf(art.tau_hi) and art.tau_hi > 0 and art.tau_lo == -math.inf  # no registered Laya thresholds


def test_scorer_input_layout_matches_training_format():
    seen = {}

    class Spy(torch.nn.Module):
        def forward(self, x):
            seen["x"] = x
            return torch.zeros(1, 1)
    rt = runtime(cal_head=Spy())
    texts = []
    orig = rt.cal.tok

    class Tok(FakeTok):
        def __call__(self, text, **kw):
            texts.append(text)
            return orig(text, **kw)
    rt.cal.tok = Tok()
    f = {"domain": "python", "tier": TIERS[2], "prompt": "P", "response": "C", "gate_status": "VERIFIED",
         "gen_signals": {"mean_logprob": -0.1, "min_logprob": -2.0}}
    rt.scorer(f)
    sg = {"gate": "verified", "mean_logprob": -0.1, "min_logprob": -2.0}
    want = render_calibrator_text({"domain": "python", "tier": TIERS[2], "signals": sg, "prompt": "P", "candidate": "C"})
    assert want in texts and "[signals] gate=verified mean_logprob=-0.1 min_logprob=-2.0" in want
    assert seen["x"][0, H:].tolist() == pytest.approx(gate_features(sg))


def test_missing_logprobs_flag_and_veto():
    rt = runtime()
    f = {"domain": "python", "tier": TIERS[0], "prompt": "p", "response": "c", "gate_status": "UNVERIFIED"}
    raw = rt.scorer(f)
    ev = f["laya"]
    assert raw is not None and ev["signals_incomplete"] and ev["missing_signals"] == ["mean_logprob", "min_logprob"]
    assert "signals_incomplete" in ev["likely_correct_vetoed"]
    f2 = dict(f, gen_signals={"mean_logprob": -0.1, "min_logprob": float("nan")})
    rt.scorer(f2)
    assert f2["laya"]["missing_signals"] == ["min_logprob"]
    f3 = dict(f, gen_signals={"mean_logprob": -0.1, "min_logprob": -1.0})
    rt.scorer(f3)
    assert not f3["laya"]["signals_incomplete"] and f3["laya"]["likely_correct_vetoed"] == []


def test_truncation_flag_and_candidate_visibility():
    rt = runtime(max_len=30)
    base = {"domain": "python", "tier": TIERS[0], "gate_status": "UNVERIFIED",
            "gen_signals": {"mean_logprob": -0.1, "min_logprob": -1.0}}
    short = dict(base, prompt="a b c", response="x y z")
    rt.scorer(short)
    assert not short["laya"]["truncated_candidate"] and short["laya"]["likely_correct_vetoed"] == []
    cut = dict(base, prompt="a b c", response=" ".join(["w"] * 60))
    rt.scorer(cut)
    ev = cut["laya"]
    assert ev["truncated_candidate"] and "truncated_candidate" in ev["likely_correct_vetoed"]
    assert not ev["candidate_fully_outside_window"] and 0 < ev["candidate_tokens_visible"] < 60
    gone = dict(base, prompt=" ".join(["p"] * 80), response="x y z")
    rt.scorer(gone)
    assert gone["laya"]["candidate_fully_outside_window"] and gone["laya"]["candidate_tokens_visible"] == 0


def test_domain_switch_and_lean_refused():
    rt = runtime()
    assert rt.router("p", "rust") == {} and rt.router("p", "rust").evidence["reason"] == "domain_not_enabled:rust"
    assert rt.router("p", "lean4") == {}
    f = {"domain": "lean4", "tier": TIERS[0], "prompt": "p", "response": "c", "gate_status": "VERIFIED"}
    assert rt.scorer(f) is None and f["laya"]["applied"] is False
    assert L.parse_domains(None) == ("python", "math") and L.parse_domains("python, rust") == ("python", "rust")
    with pytest.raises(ValueError):
        L.parse_domains("python,lean4")
    with pytest.raises(ValueError):
        runtime(domains=("lean4",))
    assert set(rt.calibration_artifacts()) == {"python", "math"}


def test_thresholds_file_roundtrip(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"thresholds": {"python": {"tau_hi": 0.8, "tau_lo": 0.3}, "math": {"tau_hi": 0.6}}}))
    th = L.load_thresholds(p)
    assert th == {"python": {"tau_hi": 0.8, "tau_lo": 0.3}}
    arts = runtime().calibration_artifacts(th)
    assert arts["python"].tau_hi == 0.8 and math.isinf(arts["math"].tau_hi)


# ---- freeze verification ----------------------------------------------------------------------------------

def _fake_laya_dir(tmp_path):
    lines = []
    for rel in L.FROZEN_FILES:
        p = tmp_path / "laya" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(rel.encode())
        lines.append(f"{hashlib.sha256(rel.encode()).hexdigest()}  {rel}")
    iso = tmp_path / "iso.json"
    iso.write_text(json.dumps(ISO))
    lines.append(f"{hashlib.sha256(iso.read_bytes()).hexdigest()}  laya_isotonic_C.json  (note)")
    fz = tmp_path / "FREEZE.txt"
    fz.write_text("# header\n" + "\n".join(lines) + "\n")
    return tmp_path / "laya", iso, fz


def test_verify_freeze_accepts_and_refuses_mismatch(tmp_path):
    d, iso, fz = _fake_laya_dir(tmp_path)
    assert len(L.verify_freeze(d, iso, fz)) == len(L.FROZEN_FILES) + 1
    (d / "calibrator" / "heads.safetensors").write_bytes(b"tampered")
    with pytest.raises(L.LayaFreezeError, match="sha256 mismatch"):
        L.verify_freeze(d, iso, fz)
    with pytest.raises(L.LayaFreezeError, match="missing"):
        L.verify_freeze(tmp_path / "nowhere", iso, fz)
    # the repo's real freeze file lists every file the runtime checks
    assert set(L.FROZEN_FILES) <= set(L.parse_freeze())


def test_load_refuses_before_touching_torch_on_mismatch(tmp_path):
    d, iso, fz = _fake_laya_dir(tmp_path)
    (d / "router" / "head_config.json").write_bytes(b"{}")
    with pytest.raises(L.LayaFreezeError):
        L.LayaRuntime.load(d, isotonic_path=iso, freeze_path=fz)


# ---- GwenLaya integration ---------------------------------------------------------------------------------

def checker(status_by_resp):
    return lambda task, response: CheckResult(status_by_resp.get(response, "UNVERIFIED"), {})


def tiers(calls=None, signals=None):
    def mk(name, resp):
        def gen(prompt, domain):
            calls is not None and calls.append(name)
            return resp
        if signals is None:
            return g.Tier(name, gen)
        return g.Tier(name, gen, generate_with_signals=lambda p, d: (gen(p, d), dict(signals)))
    return [mk(TIERS[0], "cand-small"), mk(TIERS[1], "cand-mid"), mk(TIERS[2], "cand-big")]


def system(rt, tl, status=None, thresholds=None, tau=0.5, policy="start_at", **kw):
    th = thresholds if thresholds is not None else {"python": {"tau_hi": 0.7, "tau_lo": 0.05},
                                                    "math": {"tau_hi": 0.7, "tau_lo": 0.05}}
    return g.GwenLaya(tl, router=rt.router, scorer=rt.scorer, calibrations=rt.calibration_artifacts(th), tau_route=tau,
                      route_policy=policy, checker=checker(status or {}), **kw)


def test_end_to_end_likely_correct_needs_signals():
    rt = runtime(router_bias=(-3.0, 2.0, 2.0), cal_bias=3.0)  # iso(sig(3)=0.95) = 0.25 + 0.45*1.5 = 0.925
    calls = []
    s = system(rt, tiers(calls, signals={"mean_logprob": -0.1, "min_logprob": -1.0}))
    out = s.answer("write f", "python")
    assert calls == [TIERS[1]]  # start_at: cheapest tier with p >= 0.5 is the 4B
    assert out["verdict"] == "LIKELY_CORRECT" and out["answered"] and out["tier"] == TIERS[1]
    assert out["p_correct"] == pytest.approx(0.925, abs=0.01)
    lay = out["evidence"]["laya"]
    assert lay["router"]["applied"] and lay["scorer"]["raw"] == pytest.approx(sig(3.0))
    assert lay["truncated_candidate"] is False and lay["signals_incomplete"] is False and lay["model_sha"] == "test0000"
    assert out["evidence"]["routing"]["policy"] == "start_at" and out["evidence"]["routing"]["start_tier"] == TIERS[1]


def test_end_to_end_no_logprobs_never_likely_correct():
    rt = runtime(router_bias=(2.0, 2.0, 2.0), cal_bias=3.0)
    s = system(rt, tiers())  # plain generate=: no log-probs, as with Ollama
    out = s.answer("write f", "python", allow_escalation=False)
    assert out["verdict"] != "LIKELY_CORRECT" and not out["answered"]
    assert out["p_correct"] is not None and out["evidence"]["laya"]["signals_incomplete"] is True
    step = out["evidence"]["trajectory"][0]
    assert step["decision"].startswith("likely_correct_vetoed:signals_incomplete")
    assert out["evidence"]["laya"]["scorer"]["likely_correct_vetoed"] == ["signals_incomplete"]


def test_end_to_end_truncation_vetoes_but_gate_pass_still_verified():
    rt = runtime(router_bias=(2.0, 2.0, 2.0), cal_bias=3.0, max_len=10)
    sg = {"mean_logprob": -0.1, "min_logprob": -1.0}
    long_prompt = " ".join(["word"] * 30)
    s = system(rt, tiers(signals=sg))
    out = s.answer(long_prompt, "python", allow_escalation=False)
    assert out["verdict"] != "LIKELY_CORRECT" and out["evidence"]["laya"]["truncated_candidate"] is True
    assert out["evidence"]["trajectory"][0]["decision"].startswith("likely_correct_vetoed:truncated_candidate")
    v = system(rt, tiers(signals=sg), status={"cand-small": "VERIFIED"}).answer(long_prompt, "python")
    assert v["verdict"] == "VERIFIED" and v["evidence"]["laya"]["truncated_candidate"] is True


def test_end_to_end_disabled_domain_is_gate_only():
    rt = runtime(router_bias=(-5.0, -5.0, -5.0), cal_bias=3.0)  # would route to the last tier / abstain if applied
    calls = []
    s = system(rt, tiers(calls, signals={"mean_logprob": -0.1, "min_logprob": -1.0}), status={"cand-mid": "VERIFIED"})
    out = s.answer("write f", "rust")
    assert calls == [TIERS[0], TIERS[1]] and out["verdict"] == "VERIFIED"  # static ladder, gate decides
    assert out["evidence"]["routing"] == "static_ladder (router_not_applicable)"
    assert out["evidence"]["laya"]["router"]["reason"] == "domain_not_enabled:rust"
    assert out["p_correct"] is None
    lean = system(rt, tiers(), status={}).answer("theorem t : True := trivial", "lean4")
    assert lean["verdict"] in ("ABSTAIN", "ESCALATE", "LIKELY_WRONG") and lean["p_correct"] is None


def test_unmatched_tier_names_fall_back_to_static_ladder():
    rt = runtime()
    local = [g.Tier("qwen2.5-coder:1.5b", lambda p, d: "a"), g.Tier("qwen2.5-coder:7b", lambda p, d: "b")]
    out = system(rt, local, status={"b": "VERIFIED"}).answer("p", "python")
    assert out["verdict"] == "VERIFIED" and out["tier"] == "qwen2.5-coder:7b"
    assert out["evidence"]["routing"] == "static_ladder (router_tiers_unmatched)"
    assert out["evidence"]["laya"]["scorer"]["reason"].startswith("tier_not_in_laya_training")


def test_default_filter_policy_unchanged_by_the_new_field():
    probs = lambda p, d: {TIERS[0]: 0.1, TIERS[1]: 0.9, TIERS[2]: 0.8}  # noqa: E731
    s = g.GwenLaya(tiers(), router=probs, tau_route=0.5, checker=checker({}))
    assert s.route_policy == "filter"
    assert [t.name for t in s._eligible("p", "python", None, {})] == TIERS[1:]
    s2 = g.GwenLaya(tiers(), router=probs, tau_route=0.95, checker=checker({}))
    assert s2.answer("p", "python")["verdict"] == "ABSTAIN"
    with pytest.raises(ValueError):
        g.GwenLaya(tiers(), route_policy="nope")


def test_start_at_matches_registered_run_policy():
    """scripts/analyze_laya.py::run_policy: cheapest tier with p >= tau, else the last."""
    calls = []
    s = g.GwenLaya(tiers(calls), router=lambda p, d: {TIERS[0]: 0.2, TIERS[1]: 0.3, TIERS[2]: 0.1}, tau_route=0.5,
                   route_policy="start_at", checker=checker({}))
    assert [t.name for t in s._eligible("p", "python", None, {})] == [TIERS[2]]
    s = g.GwenLaya(tiers(), router=lambda p, d: {TIERS[0]: 0.2, TIERS[1]: 0.6, TIERS[2]: 0.1}, tau_route=0.5,
                   route_policy="start_at", checker=checker({}))
    assert [t.name for t in s._eligible("p", "python", None, {})] == TIERS[1:]


def test_no_laya_dir_means_unchanged_behaviour(monkeypatch):
    monkeypatch.delenv("GWENLAYA_LAYA_DIR", raising=False)
    monkeypatch.delenv("GWENLAYA_CALIBRATION", raising=False)
    s = g.system_from_env()
    assert s.router is None and s.scorer is None and s.calibrations == {} and s.tau_route is None
    assert s.route_policy == "filter" and [t.name for t in s.tiers] == [
        "qwen2.5-coder:1.5b", "qwen2.5-coder:3b", "qwen2.5-coder:7b"]
    assert all(t.generate_with_signals is None for t in s.tiers)
    # an ordinary run yields the same output as before: no "laya" key in the evidence
    plain = g.GwenLaya([g.Tier("a", lambda p, d: "x")], checker=checker({"x": "VERIFIED"}))
    out = plain.answer("p", "python")
    assert out["verdict"] == "VERIFIED" and "laya" not in out["evidence"]
    assert set(out["evidence"]) == {"calibrated", "trajectory", "policy", "routing", "escalated"}


def test_system_from_env_wires_runtime(monkeypatch, tmp_path):
    rt = runtime()
    seen = {}

    def fake_load(d, device="cpu", **kw):
        seen.update(dir=d, device=device, **kw)
        return rt
    monkeypatch.setattr(L.LayaRuntime, "load", staticmethod(fake_load))
    th = tmp_path / "th.json"
    th.write_text(json.dumps({"thresholds": {"python": {"tau_hi": 0.9, "tau_lo": 0.1}}}))
    for k, v in {"GWENLAYA_LAYA_DIR": "/x/laya_v1", "GWENLAYA_LAYA_DEVICE": "cpu", "GWENLAYA_LAYA_THRESHOLDS": str(th),
                 "GWENLAYA_LAYA_TIER_MAP": "qwen2.5-coder:1.5b=qwen3.5-2b-bf16", "GWENLAYA_LAYA_THREADS": "3",
                 "GWENLAYA_TIERS": "qwen2.5-coder:1.5b"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("GWENLAYA_CALIBRATION", raising=False)
    s = g.system_from_env()
    assert seen["dir"] == "/x/laya_v1" and seen["tier_map"] == {"qwen2.5-coder:1.5b": "qwen3.5-2b-bf16"} and seen["threads"] == 3
    assert s.router == rt.router and s.scorer == rt.scorer and s.tau_route == 0.5 and s.route_policy == "start_at"
    assert s.calibrations["python"].tau_hi == 0.9 and math.isinf(s.calibrations["math"].tau_hi)
    assert "rust" not in s.calibrations and "lean4" not in s.calibrations


# ---- parity against the stored TPU predictions (skipped without the real model) ---------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("GWAYA_DATA_ROOT", "/mnt/data/home/xavkal/gwaya-data"))


def _real_available():
    try:
        import peft  # noqa: F401
        import transformers  # noqa: F401
        root = Path(str(L.resolve_backbone("answerdotai/ModernBERT-large")))
        if not any((root / n).is_file() for n in ("model.safetensors", "pytorch_model.bin")):
            return False  # a snapshot directory without weights (partial cache)
    except Exception:  # noqa: BLE001
        return False
    return (DATA / "laya_data" / "laya_v1" / "router" / "heads.safetensors").is_file() and \
        (DATA / "laya_data" / "laya_calib" / "router.jsonl").is_file()


@pytest.mark.skipif(not _real_available(), reason="needs ModernBERT-large in the HF cache, transformers, peft and the frozen Laya files")
def test_parity_with_stored_tpu_predictions():
    import importlib.util
    spec = importlib.util.spec_from_file_location("laya_runtime_parity", ROOT / "scripts" / "laya_runtime_parity.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    res = m.parity(str(DATA / "laya_data" / "laya_v1"), str(DATA), n=20, threads=3)
    print("parity", {k: {a: b for a, b in v.items() if a != "rows_detail"} for k, v in res.items()})
    assert res["router"]["max_abs_diff"] < 0.02 and res["calibrator"]["max_abs_diff"] < 0.02
