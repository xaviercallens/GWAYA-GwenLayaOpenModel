"""A22: candidate-visible layouts (gwaya.laya_features), the trainer flag, and the runtime's layout handling."""
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from gwaya import laya_features as F

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("train_laya_for_layout", ROOT / "scripts" / "train_laya.py")
T = importlib.util.module_from_spec(spec)
sys.modules["train_laya_for_layout"] = T
spec.loader.exec_module(T)


class WordTok:
    """Fast-tokenizer stand-in: one token per whitespace-delimited word, with character offsets; CLS/SEP specials."""

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, truncation=False, max_length=None):
        spans = [m.span() for m in re.finditer(r"\S+", text)]
        ids = list(range(len(spans)))
        offs = list(spans)
        if add_special_tokens:
            ids = [-1] + ids + [-2]
        if truncation and max_length and len(ids) > max_length:
            ids = ids[:max_length - 1] + [-2]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out


ROW = {"domain": "rust", "tier": "qwen3.5-4b-bf16", "prompt": "Write f.\nassert f(1) == 2", "candidate": "fn f(x: i32) -> i32 { x + 1 }",
       "signals": {"gate": "verified", "mean_logprob": -0.05, "min_logprob": -1.25}}


def long_row(n_prompt=900, n_cand=60):
    return dict(ROW, prompt=" ".join(f"p{i}" for i in range(n_prompt)), candidate=" ".join(f"c{i}" for i in range(n_cand)))


def test_default_layout_is_byte_identical_and_explicit_matches():
    want = ("[domain] rust [tier] qwen3.5-4b-bf16\n[signals] gate=verified mean_logprob=-0.05 min_logprob=-1.25\n"
            "[prompt]\nWrite f.\nassert f(1) == 2\n[candidate]\nfn f(x: i32) -> i32 { x + 1 }")
    assert F.render_calibrator_text(ROW) == want == F.render_calibrator_text(ROW, "prompt_first")
    only = F.render_calibrator_text_only(ROW)
    assert "[signals]" not in only and only == F.render_calibrator_text_only(ROW, "prompt_first")
    assert F.check_layout(None) == "prompt_first"
    with pytest.raises(ValueError):
        F.check_layout("candidate_last")


def test_candidate_first_text():
    t = F.render_calibrator_text(ROW, "candidate_first")
    assert t == ("[domain] rust [tier] qwen3.5-4b-bf16\n[signals] gate=verified mean_logprob=-0.05 min_logprob=-1.25\n"
                 "[candidate]\nfn f(x: i32) -> i32 { x + 1 }\n[prompt]\nWrite f.\nassert f(1) == 2")
    t2 = F.render_calibrator_text_only(ROW, "candidate_first")
    assert "[signals]" not in t2 and t2.startswith("[domain] rust [tier] qwen3.5-4b-bf16\n[candidate]\n")
    long = dict(ROW, candidate="x" * 2500)
    assert "x" * 2000 + "\n[prompt]" in F.render_calibrator_text(long, "candidate_first")
    assert "x" * 2001 not in F.render_calibrator_text(long, "candidate_first")


def test_middle_truncate_needs_tokenizer_and_is_identity_when_it_fits():
    with pytest.raises(ValueError):
        F.render_calibrator_text(ROW, "middle_truncate")
    assert F.render_calibrator_text(ROW, "middle_truncate", WordTok(), 512) == F.render_calibrator_text(ROW)


def test_candidate_always_inside_first_512_tokens_for_a_long_prompt():
    tok, row = WordTok(), long_row()
    base = F.candidate_window(row, "prompt_first", tok, 512)
    assert base["visible_tokens"] == 0 and not base["any_visible"]          # v1: candidate entirely outside the window
    for layout in ("candidate_first", "middle_truncate"):
        w = F.candidate_window(row, layout, tok, 512)
        assert w["any_visible"] and w["fully_visible"] and w["visible_tokens"] == w["candidate_tokens"] == 60, layout
        text = F.render_calibrator_text(row, layout, tok, 512)
        cut = tok(text, truncation=True, max_length=512)["input_ids"]
        assert len(cut) <= 512
        assert "c59" in text
    # middle_truncate renders a text that needs no truncation at all
    assert len(tok(F.render_calibrator_text(row, "middle_truncate", tok, 512))["input_ids"]) <= 512


def test_middle_truncate_keeps_head_and_tail_and_fits_exactly():
    tok, row = WordTok(), long_row(900, 60)
    text = F.render_calibrator_text(row, "middle_truncate", tok, 512)
    assert len(tok(text)["input_ids"]) <= 512
    assert "p0 " in text and "p899" in text and "[...]" in text and "p450" not in text
    assert text.endswith("c59") and text == F.render_calibrator_text(row, "middle_truncate", tok, 512)  # deterministic


def test_middle_truncate_shortens_a_huge_candidate_but_keeps_some_prompt():
    tok = WordTok()
    row = dict(long_row(900, 1), candidate=" ".join("x" for _ in range(900)))  # 1,799 chars: under CAND_CHARS, 900 tokens
    text = F.render_calibrator_text(row, "middle_truncate", tok, 512)
    assert len(tok(text)["input_ids"]) <= 512
    w = F.candidate_window(row, "middle_truncate", tok, 512)
    assert 0 < w["visible_tokens"] < 900 and not w["fully_visible"]
    assert "p0 " in text and "p899" in text
    wc = F.candidate_window(row, "candidate_first", tok, 512)  # candidate fills the window, the prompt is cut by design
    assert wc["any_visible"] and not wc["fully_visible"]


def test_candidate_first_keeps_the_candidate_start_even_with_a_huge_prompt():
    assert F.candidate_window(long_row(5000, 40), "candidate_first", WordTok(), 512)["fully_visible"]


def test_trainer_layout_flag_and_examples():
    assert T.build_parser().parse_args(["--mode", "calibrator", "--data", "x"]).layout == "prompt_first"
    assert T.build_parser().parse_args(["--mode", "calibrator", "--data", "x", "--layout", "candidate_first"]).layout == "candidate_first"
    row = dict(ROW, source="rs", id="1", correct=1)
    a = T.make_examples([row], "calibrator", ["t"])[0]
    b = T.make_examples([row], "calibrator", ["t"], layout="candidate_first")[0]
    assert a["text"] == F.render_calibrator_text(row) and b["text"] == F.render_calibrator_text(row, "candidate_first")
    assert a["feats"] == b["feats"] and a["labels"] == b["labels"]
    c = T.make_examples([row], "calibrator", ["t"], text_only=True, layout="candidate_first")[0]
    assert c["text"] == F.render_calibrator_text_only(row, "candidate_first") and c["feats"] == []
    m = T.make_examples([row], "calibrator", ["t"], layout="middle_truncate", tokenizer=WordTok())[0]
    assert m["text"] == F.render_calibrator_text(row)
    r = T.make_examples([dict(row, tiers={"t": 1})], "router", ["t"], layout="candidate_first")[0]
    assert r["text"] == F.render_router_text(row)  # the router never sees a candidate


def test_print_plan_records_layout(capsys, tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text("")
    assert T.main(["--mode", "calibrator", "--data", str(p), "--layout", "candidate_first", "--print-plan", "--no-slice-filter"]) == 0
    assert '"layout": "candidate_first"' in capsys.readouterr().out


# ---- runtime: layout read from head_config, renderer matched --------------------------------------------------

def test_runtime_uses_the_layout_of_the_head_model():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(ROOT / "tests"))
    import test_laya_runtime as R
    from gwaya import laya_runtime as L

    texts = []

    class Tok(WordTok):
        def __call__(self, text, **kw):
            texts.append(text)
            rt_ = kw.pop("return_tensors", None)
            out = super().__call__(text, **kw)
            if rt_:
                ids = out["input_ids"]
                out = {"input_ids": torch.tensor([[abs(i) % 50 + 2 for i in ids]]),
                       "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
            return out

    rt = R.runtime()
    rt.cal.tok = Tok()
    assert rt.cal.layout == "prompt_first"
    rt.cal.layout = "candidate_first"
    f = {"domain": "python", "tier": R.TIERS[2], "prompt": " ".join(f"p{i}" for i in range(900)), "response": "C0 C1 C2",
         "gate_status": "VERIFIED", "gen_signals": {"mean_logprob": -0.1, "min_logprob": -2.0}}
    rt.scorer(f)
    ev = f["laya"]
    assert ev["layout"] == "candidate_first" and not ev["candidate_fully_outside_window"] and not ev["truncated_candidate"]
    assert ev["candidate_tokens_visible"] == 3
    assert any(len(t.split("\n")) > 2 and t.split("\n")[2] == "[candidate]" for t in texts)
    rt.cal.layout = "prompt_first"
    f2 = dict(f)
    rt.scorer(f2)
    assert f2["laya"]["candidate_fully_outside_window"] and f2["laya"]["layout"] == "prompt_first"
    with pytest.raises(ValueError):
        L.HeadModel(Tok(), None, None, 512, 0, [], layout="nope")


def test_runtime_loader_reads_layout_and_rejects_unknown(tmp_path):
    from gwaya import laya_runtime as L
    from gwaya.laya_features import GATE_VOCAB, N_FEATURES, SIGNAL_KEYS
    sub = tmp_path / "calibrator"
    sub.mkdir()
    base = {"mode": "calibrator", "tiers": ["a"], "backbone": "x", "hidden_size": 8, "n_features": N_FEATURES,
            "gate_vocab": list(GATE_VOCAB), "signal_keys": list(SIGNAL_KEYS), "max_len": 512}
    (sub / "head_config.json").write_text(json.dumps({**base, "layout": "sideways"}))
    with pytest.raises(ValueError, match="unknown layout"):
        L._load_head_model(sub, "calibrator", "cpu", None)
    router = tmp_path / "router"
    router.mkdir()
    (router / "head_config.json").write_text(json.dumps({**base, "mode": "router", "n_features": 0, "layout": "candidate_first"}))
    with pytest.raises(L.LayaFreezeError, match="router"):
        L._load_head_model(router, "router", "cpu", None)


def test_real_modernbert_tokenizer_window_if_cached():
    tf = pytest.importorskip("transformers")
    from gwaya.laya_runtime import resolve_backbone
    try:
        path = resolve_backbone("answerdotai/ModernBERT-large")
    except FileNotFoundError:
        pytest.skip("backbone not cached")
    tok = tf.AutoTokenizer.from_pretrained(path, local_files_only=True)
    row = dict(ROW, prompt="Implement the exercise.\n" + "fn helper() { let x = 1; } // docs\n" * 400,
               candidate="pub fn solve(v: &[i32]) -> i32 {\n    v.iter().sum()\n}\n")
    assert not F.candidate_window(row, "prompt_first", tok)["any_visible"]
    for layout in ("candidate_first", "middle_truncate"):
        w = F.candidate_window(row, layout, tok)
        assert w["fully_visible"] and w["visible_tokens"] > 0, layout
    text = F.render_calibrator_text(row, "middle_truncate", tok, 512)
    assert len(tok(text)["input_ids"]) <= 512
