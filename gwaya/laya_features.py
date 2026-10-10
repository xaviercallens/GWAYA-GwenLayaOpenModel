"""Pure-python Laya input construction (no torch): text rendering and gate/log-prob features.

Moved verbatim from scripts/train_laya.py so the training script and the runtime (gwaya/laya_runtime.py) share ONE
implementation. The frozen Laya (docs/LAYA_FREEZE.txt) was trained on exactly this layout; changing any constant or
format here changes the model's input and invalidates every registered evaluation. scripts/train_laya.py re-exports
these names; tests/test_laya_features.py pins them byte-for-byte against known renderings.
"""
from __future__ import annotations

import math
from typing import Any

GATE_VOCAB = ("verified", "failed", "unverified", "compile_error", "none")
SIGNAL_KEYS = ("tests_passed_frac", "selftest_agreement", "mean_logprob", "min_logprob", "repair_rounds")
N_FEATURES = len(GATE_VOCAB) + 1 + 2 * len(SIGNAL_KEYS)  # gate one-hot (+other), value+missing per signal
CAND_CHARS = 2000


def gate_features(signals: dict[str, Any] | None) -> list[float]:
    s = signals or {}
    g = str(s.get("gate", "none")).lower()
    onehot = [1.0 if g == v else 0.0 for v in GATE_VOCAB] + [1.0 if g not in GATE_VOCAB else 0.0]
    vec = list(onehot)
    for k in SIGNAL_KEYS:
        v = s.get(k)
        ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        scale = 0.1 if k == "repair_rounds" else (0.1 if k.endswith("logprob") else 1.0)
        vec += [float(v) * scale if ok else 0.0, 0.0 if ok else 1.0]
    return vec


def render_router_text(r: dict[str, Any]) -> str:
    return f"[domain] {r.get('domain', '?')}\n[prompt]\n{r['prompt']}"


LAYOUTS = ("prompt_first", "candidate_first", "middle_truncate")
DEFAULT_LAYOUT = "prompt_first"
MAX_LEN = 512
PROMPT_MIN_FRAC = 0.25  # middle_truncate: the prompt keeps at least this share of the free token budget
_ELLIPSIS = "\n[...]\n"


def check_layout(layout: str | None) -> str:
    """Validated layout name; None / missing means the frozen v1 layout."""
    layout = DEFAULT_LAYOUT if layout is None else layout
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")
    return layout


def _header(r: dict[str, Any], with_signals: bool) -> str:
    h = f"[domain] {r.get('domain', '?')} [tier] {r.get('tier', '?')}\n"
    if with_signals:
        s = r.get("signals") or {}
        h += "[signals] " + " ".join(f"{k}={s[k]}" for k in ("gate",) + SIGNAL_KEYS if k in s) + "\n"
    return h


def _cand(r: dict[str, Any]) -> str:
    return str(r.get("candidate", ""))[:CAND_CHARS]


def _tok_ids(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    t = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return list(t["input_ids"]), [tuple(o) for o in t["offset_mapping"]]


def _n_special(tokenizer: Any) -> int:
    return len(tokenizer("", add_special_tokens=True)["input_ids"])


def _render_middle(r: dict[str, Any], with_signals: bool, tokenizer: Any, max_len: int) -> str:
    """Prompt-first text, but when it does not fit, keep the head and tail of the prompt around the candidate.

    Deterministic and tokenizer-aware (offset mapping, so cuts fall on token boundaries of the real text). A text that
    already fits is returned unchanged (identical to prompt_first)."""
    head, prompt, cand = _header(r, with_signals), str(r["prompt"]), _cand(r)
    plain = f"{head}[prompt]\n{prompt}\n[candidate]\n{cand}"
    if len(tokenizer(plain)["input_ids"]) <= max_len:
        return plain
    budget = max_len - _n_special(tokenizer)
    p_ids, p_off = _tok_ids(tokenizer, prompt)
    c_ids, c_off = _tok_ids(tokenizer, cand)
    skeleton = len(_tok_ids(tokenizer, f"{head}[prompt]\n\n[candidate]\n")[0]) + len(_tok_ids(tokenizer, _ELLIPSIS)[0])
    avail = max(0, budget - skeleton)
    p_keep = min(len(p_ids), max(avail - len(c_ids), int(avail * PROMPT_MIN_FRAC)))
    out = plain
    for _ in range(64):  # token boundaries across the joins can add a token or two: shrink until it fits
        c_keep = min(len(c_ids), max(0, avail - p_keep))
        h, t = (p_keep + 1) // 2, p_keep // 2
        if p_keep >= len(p_ids):
            ptxt = prompt
        else:
            ptxt = (prompt[:p_off[h - 1][1]] if h else "") + _ELLIPSIS + (prompt[p_off[len(p_ids) - t][0]:] if t else "")
        ctxt = cand if c_keep >= len(c_ids) else (cand[:c_off[c_keep - 1][1]] if c_keep > 0 else "")
        out = f"{head}[prompt]\n{ptxt}\n[candidate]\n{ctxt}"
        over = len(tokenizer(out)["input_ids"]) - max_len
        if over <= 0:
            return out
        if p_keep > 0:
            p_keep = max(0, p_keep - over)
        else:
            avail = max(0, avail - over)
    return out


def _render(r: dict[str, Any], with_signals: bool, layout: str | None, tokenizer: Any, max_len: int) -> str:
    layout = check_layout(layout)
    if layout == "prompt_first":
        return f"{_header(r, with_signals)}[prompt]\n{r['prompt']}\n[candidate]\n{_cand(r)}"
    if layout == "candidate_first":
        return f"{_header(r, with_signals)}[candidate]\n{_cand(r)}\n[prompt]\n{r['prompt']}"
    if tokenizer is None:
        raise ValueError("layout middle_truncate needs a tokenizer")
    return _render_middle(r, with_signals, tokenizer, max_len)


def render_calibrator_text(r: dict[str, Any], layout: str | None = None, tokenizer: Any = None, max_len: int = MAX_LEN) -> str:
    """Calibrator input text. ``layout`` None / "prompt_first" is the frozen v1 rendering (byte-identical).

    candidate_first: header, [signals], [candidate] block, then [prompt]: truncation at max_len cuts the prompt's tail.
    middle_truncate (needs ``tokenizer``): v1 order, but an over-long prompt is cut in the middle so the candidate stays."""
    return _render(r, True, layout, tokenizer, max_len)


def render_calibrator_text_only(r: dict[str, Any], layout: str | None = None, tokenizer: Any = None, max_len: int = MAX_LEN) -> str:
    """A16 ablation: no [signals] line (gate verdict, log-probs); only domain, tier, prompt and candidate."""
    return _render(r, False, layout, tokenizer, max_len)


def candidate_window(r: dict[str, Any], layout: str | None, tokenizer: Any, max_len: int = MAX_LEN,
                     text_only: bool = False) -> dict[str, Any]:
    """How much of the candidate the encoder sees under ``layout`` (tokenizer needs fast-tokenizer offsets).

    -> {candidate_tokens (of the 2,000-char candidate), visible_tokens, any_visible, fully_visible}. Content slots are
    max_len minus the special tokens (CLS, SEP)."""
    layout = check_layout(layout)
    text = (render_calibrator_text_only if text_only else render_calibrator_text)(r, layout, tokenizer, max_len)
    cand = _cand(r)
    n_cand = len(_tok_ids(tokenizer, cand)[0]) if cand else 0
    if layout == "candidate_first":
        start = text.index("[candidate]\n") + len("[candidate]\n")
        end = start + len(cand)
    else:  # the candidate is the suffix (possibly shortened by middle_truncate)
        marker = "\n[candidate]\n"
        start = text.rindex(marker) + len(marker)
        end = len(text)
    _, off = _tok_ids(tokenizer, text)
    slots = max_len - _n_special(tokenizer)
    vis = sum(1 for i, (a, b) in enumerate(off) if i < slots and a >= start and b <= end and b > a)
    return {"candidate_tokens": n_cand, "visible_tokens": vis, "any_visible": bool(vis > 0 or n_cand == 0),
            "fully_visible": bool(vis >= n_cand)}
