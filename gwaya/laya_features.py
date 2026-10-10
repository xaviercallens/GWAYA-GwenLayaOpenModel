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


def render_calibrator_text(r: dict[str, Any]) -> str:
    s = r.get("signals") or {}
    sig = " ".join(f"{k}={s[k]}" for k in ("gate",) + SIGNAL_KEYS if k in s)
    return (f"[domain] {r.get('domain', '?')} [tier] {r.get('tier', '?')}\n[signals] {sig}\n"
            f"[prompt]\n{r['prompt']}\n[candidate]\n{str(r.get('candidate', ''))[:CAND_CHARS]}")


def render_calibrator_text_only(r: dict[str, Any]) -> str:
    """A16 ablation: no [signals] line (gate verdict, log-probs); only domain, tier, prompt and candidate."""
    return (f"[domain] {r.get('domain', '?')} [tier] {r.get('tier', '?')}\n"
            f"[prompt]\n{r['prompt']}\n[candidate]\n{str(r.get('candidate', ''))[:CAND_CHARS]}")
