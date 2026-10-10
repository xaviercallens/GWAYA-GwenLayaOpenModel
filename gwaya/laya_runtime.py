"""Laya at runtime: the frozen ModernBERT router (pre-generation) and calibrator (post-generation) as plain callables.

``LayaRuntime.load(dir)`` verifies the sha256 of every frozen file against docs/LAYA_FREEZE.txt (the authority; a
mismatch refuses to load), then exposes ``router(prompt, domain) -> {tier: p_correct}`` and
``scorer(features) -> raw score`` matching gwaya.gwenlaya.Router / Scorer. The model, its input layout and the
isotonic map (docs/laya_isotonic_C.json, fitted on C, n=849) are exactly the registered ones; nothing is retrained.
Rendering and gate features come from gwaya.laya_features (shared with scripts/train_laya.py). The scorer returns the
RAW sigmoid output; GwenLaya applies the isotonic map through the CalibrationArtifact from ``calibration_artifacts``.

What this is NOT: an end-to-end validated system. Only the model's offline numbers exist (docs/LAYA_RUNTIME.md).

Honest failure modes (all fail closed; evidence is written to evidence["laya"]):
  * domains: Laya was trained on python, rust and math only; Lean 4 was never seen, so Lean is refused outright
    (router and scorer return None and the gate-only policy of gwenlaya.py stands). Rust was in training, but on the
    fresh Rust set R' Laya is at chance (docs/CLAIM_AUDIT_380.md; results/gwenlaya_v4/laya/laya_rfresh.json), so the
    default is GWENLAYA_LAYA_DOMAINS="python,math" and Rust stays gate-only unless explicitly enabled.
  * truncation: the text is rendered prompt first and cut at max_len=512 tokens. If any of the candidate is cut,
    ``truncated_candidate`` is recorded and LIKELY_CORRECT is vetoed (measured on R': the candidate is wholly
    outside the window for 103 of 207 rows; results/gwenlaya_v4/laya/laya_rfresh.json). VERIFIED from the gate is
    unaffected.
  * signals: the calibrator was trained on rows that always carried mean_logprob and min_logprob. When the tier
    cannot supply them (Ollama has no log-probs), ``signals_incomplete`` is recorded and LIKELY_CORRECT is vetoed.
  * tiers: the heads were trained for the tier names in head_config.json (Qwen3.5 2B/4B/9B bf16). A tier with another
    name gets no probability unless an explicit ``tier_map`` says which trained tier it stands in for.

torch / transformers / peft are imported lazily; ``import gwaya`` stays light.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from gwaya.laya_features import GATE_VOCAB, N_FEATURES, SIGNAL_KEYS, gate_features, render_calibrator_text, render_router_text

ROOT = Path(__file__).resolve().parents[1]
FREEZE_FILE = ROOT / "docs" / "LAYA_FREEZE.txt"
ISOTONIC_FILE = ROOT / "docs" / "laya_isotonic_C.json"
LAYA_TRAINED_DOMAINS = ("python", "rust", "math")  # lean4 was never in the training data
DEFAULT_DOMAINS = ("python", "math")  # rust is at chance on fresh data: gate-only by default
FROZEN_FILES = ("router/adapter/adapter_model.safetensors", "router/heads.safetensors", "router/head_config.json",
                "calibrator/adapter/adapter_model.safetensors", "calibrator/heads.safetensors",
                "calibrator/head_config.json")
DOMAINS_ENV, DIR_ENV, DEVICE_ENV = "GWENLAYA_LAYA_DOMAINS", "GWENLAYA_LAYA_DIR", "GWENLAYA_LAYA_DEVICE"


class LayaFreezeError(RuntimeError):
    """A frozen Laya file is missing or its sha256 differs from docs/LAYA_FREEZE.txt."""


class RouterResult(dict):
    """{tier: p}; ``.evidence`` is what GwenLaya records under evidence["laya"]["router"]."""
    evidence: dict[str, Any]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_freeze(path: str | Path = FREEZE_FILE) -> dict[str, str]:
    """-> {relative path: sha256} from the freeze file (lines '<64 hex>  <path>[  (note)]')."""
    out: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 64 and all(c in "0123456789abcdef" for c in parts[0]):
            out[parts[1]] = parts[0]
    return out


def verify_freeze(laya_dir: str | Path, isotonic: str | Path = ISOTONIC_FILE,
                  freeze: str | Path = FREEZE_FILE) -> dict[str, str]:
    """Check the frozen files (and the isotonic map) against the freeze file; -> {path: sha256}. Raises LayaFreezeError."""
    frozen = parse_freeze(freeze)
    d = Path(laya_dir)
    todo = {rel: d / rel for rel in FROZEN_FILES} | {"laya_isotonic_C.json": Path(isotonic)}
    seen: dict[str, str] = {}
    for rel, p in todo.items():
        want = frozen.get(rel)
        if want is None:
            raise LayaFreezeError(f"{rel} is not listed in {freeze}")
        if not p.is_file():
            raise LayaFreezeError(f"missing frozen file {p}")
        got = _sha256(p)
        if got != want:
            raise LayaFreezeError(f"sha256 mismatch for {rel}: {got} != frozen {want}")
        seen[rel] = got
    return seen


def parse_domains(spec: str | Sequence[str] | None) -> tuple[str, ...]:
    """'python,math' -> ('python','math'). Unknown or never-trained domains (lean4, ...) raise ValueError."""
    if spec is None:
        return DEFAULT_DOMAINS
    names = [x.strip() for x in (spec.split(",") if isinstance(spec, str) else spec) if str(x).strip()]
    bad = [n for n in names if n not in LAYA_TRAINED_DOMAINS]
    if bad:
        raise ValueError(f"{DOMAINS_ENV}: {bad} not in the frozen Laya's training domains {LAYA_TRAINED_DOMAINS}")
    return tuple(dict.fromkeys(names))


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


class HeadModel:
    """Encoder + masked-mean pooling + MLP head, as in scripts/train_laya.py::train.forward (batch size 1).

    ``encoder(input_ids, attention_mask) -> last_hidden_state [1, T, H]`` and ``tokenizer(text, ...)`` are injected,
    so tests can use a tiny random encoder; ``_load_head_model`` builds the real ones."""

    def __init__(self, tokenizer: Any, encoder: Callable[[Any, Any], Any], head: Any, max_len: int, n_feat: int,
                 tiers: Sequence[str], device: str = "cpu") -> None:
        self.tok, self.encoder, self.head = tokenizer, encoder, head
        self.max_len, self.n_feat, self.tiers, self.device = int(max_len), int(n_feat), list(tiers), device

    def n_tokens(self, text: str) -> int:
        """Tokens of the untruncated text, special tokens included."""
        return len(self.tok(text, truncation=False)["input_ids"])

    def logits(self, text: str, feats: Sequence[float] | None = None) -> list[float]:
        import torch
        t = self.tok(text, truncation=True, max_length=self.max_len, return_tensors="pt")
        ids, mask = t["input_ids"].to(self.device), t["attention_mask"].to(self.device)
        with torch.no_grad():
            h = self.encoder(ids, mask).float()
            m = mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
            if self.n_feat:
                if feats is None or len(feats) != self.n_feat:
                    raise ValueError(f"expected {self.n_feat} features, got {None if feats is None else len(feats)}")
                pooled = torch.cat([pooled, torch.tensor([list(feats)], dtype=pooled.dtype, device=self.device)], 1)
            return [float(x) for x in self.head(pooled)[0].cpu().tolist()]

    def probs(self, text: str, feats: Sequence[float] | None = None) -> list[float]:
        return [_sigmoid(z) for z in self.logits(text, feats)]


def resolve_backbone(name: str) -> str:
    """Local path of the backbone: an existing directory, else the HF cache under $HF_HOME or $GWAYA_DATA_ROOT/hf-home.
    No network is used; raises FileNotFoundError when the weights are not cached."""
    if Path(name).is_dir():
        return name
    cands = []
    for home in (os.environ.get("HF_HOME"), os.path.join(os.environ.get("GWAYA_DATA_ROOT", ""), "hf-home")):
        if home:
            cands.append(Path(home) / "hub" / ("models--" + name.replace("/", "--")) / "snapshots")
    cands.append(Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + name.replace("/", "--")) / "snapshots")
    for c in cands:
        if c.is_dir():
            snaps = sorted(p for p in c.iterdir() if (p / "config.json").is_file())
            if snaps:
                return str(snaps[-1])
    raise FileNotFoundError(f"backbone {name!r} is not cached locally (no network is used); set HF_HOME or pass backbone=")


def _load_head_model(sub: Path, mode: str, device: str, backbone: str | None) -> HeadModel:
    """Real loader: ModernBERT + LoRA adapter (merged for speed) + head. Heavy imports are local."""
    import torch
    import transformers as tf
    from safetensors.torch import load_file

    cfg = json.loads((sub / "head_config.json").read_text())
    if cfg.get("mode") != mode:
        raise LayaFreezeError(f"{sub}/head_config.json mode {cfg.get('mode')!r} != {mode!r}")
    if list(cfg["gate_vocab"]) != list(GATE_VOCAB) or list(cfg["signal_keys"]) != list(SIGNAL_KEYS):
        raise LayaFreezeError("head_config gate_vocab/signal_keys differ from gwaya.laya_features (input layout drift)")
    want_feat = 0 if mode == "router" else N_FEATURES
    if int(cfg["n_features"]) != want_feat:
        raise LayaFreezeError(f"head_config n_features {cfg['n_features']} != {want_feat}")
    path = resolve_backbone(backbone or cfg["backbone"])
    tok = tf.AutoTokenizer.from_pretrained(path, local_files_only=True)
    enc = tf.AutoModel.from_pretrained(path, local_files_only=True)
    import peft
    enc = peft.PeftModel.from_pretrained(enc, str(sub / "adapter")).merge_and_unload()
    enc.eval().to(device)
    hidden = int(cfg["hidden_size"])
    n_out = len(cfg["tiers"]) if mode == "router" else 1
    head = torch.nn.Sequential(torch.nn.Linear(hidden + want_feat, hidden // 2), torch.nn.GELU(), torch.nn.Dropout(0.1),
                               torch.nn.Linear(hidden // 2, n_out))
    head.load_state_dict(load_file(str(sub / "heads.safetensors")))
    head.eval().to(device)
    return HeadModel(tok, lambda ids, m: enc(input_ids=ids, attention_mask=m).last_hidden_state, head,
                     cfg["max_len"], want_feat, cfg["tiers"], device)


class LayaRuntime:
    """The frozen Laya as Router / Scorer callables. Build with ``load``; tests inject ``HeadModel``s directly."""

    def __init__(self, router: HeadModel | None, calibrator: HeadModel | None, isotonic: Mapping[str, Sequence[float]],
                 domains: Sequence[str] = DEFAULT_DOMAINS, tier_map: Mapping[str, str] | None = None,
                 model_sha: str = "unverified") -> None:
        bad = [d for d in domains if d not in LAYA_TRAINED_DOMAINS]
        if bad:
            raise ValueError(f"domains {bad} not in {LAYA_TRAINED_DOMAINS}")
        self.rt, self.cal, self.isotonic = router, calibrator, {"x": list(isotonic["x"]), "y": list(isotonic["y"])}
        self.domains, self.tier_map = tuple(domains), dict(tier_map or {})
        self.model_sha = model_sha
        self.tiers = list((router or calibrator).tiers) if (router or calibrator) else []  # type: ignore[union-attr]

    # -- construction -----------------------------------------------------------------------------------------
    @classmethod
    def load(cls, laya_dir: str | Path, device: str = "cpu", *, verify: bool = True, domains: str | Sequence[str] | None = None,
             tier_map: Mapping[str, str] | None = None, backbone: str | None = None, threads: int | None = None,
             isotonic_path: str | Path = ISOTONIC_FILE, freeze_path: str | Path = FREEZE_FILE) -> "LayaRuntime":
        d = Path(laya_dir)
        shas = verify_freeze(d, isotonic_path, freeze_path) if verify else {}
        if threads:
            import torch
            torch.set_num_threads(int(threads))
        iso = json.loads(Path(isotonic_path).read_text())
        sha = hashlib.sha256("\n".join(f"{k}:{v}" for k, v in sorted(shas.items())).encode()).hexdigest()[:12] if shas else "unverified"
        return cls(_load_head_model(d / "router", "router", device, backbone),
                   _load_head_model(d / "calibrator", "calibrator", device, backbone), iso,
                   parse_domains(domains if domains is not None else os.environ.get(DOMAINS_ENV)), tier_map, sha)

    # -- helpers ----------------------------------------------------------------------------------------------
    def _laya_tier(self, name: str) -> str | None:
        t = self.tier_map.get(name, name)
        return t if t in self.tiers else None

    def isotonic_map(self, raw: float) -> float:
        """Registered map raw -> P(correct): np.interp semantics (linear between knots, clamped at the ends)."""
        xs, ys = self.isotonic["x"], self.isotonic["y"]
        if raw <= xs[0]:
            return ys[0]
        if raw >= xs[-1]:
            return ys[-1]
        import bisect
        j = bisect.bisect_right(xs, raw)
        return ys[j - 1] + (ys[j] - ys[j - 1]) * (raw - xs[j - 1]) / (xs[j] - xs[j - 1])

    def _base_ev(self, **kw: Any) -> dict[str, Any]:
        return {"model_sha": self.model_sha, **kw}

    # -- Router -----------------------------------------------------------------------------------------------
    def router(self, prompt: str, domain: str) -> RouterResult:
        """{tier_name: p_correct}; empty (with evidence.applied False) when the domain is not enabled or unseen."""
        res = RouterResult()
        if self.rt is None or domain not in self.domains:
            reason = "no_router" if self.rt is None else f"domain_not_enabled:{domain}"
            res.evidence = self._base_ev(applied=False, reason=reason, domains=list(self.domains))
            return res
        t0 = time.perf_counter()
        ps = self.rt.probs(render_router_text({"domain": domain, "prompt": prompt}))
        by_laya = dict(zip(self.rt.tiers, ps))
        names = {**{t: t for t in self.rt.tiers}, **self.tier_map}  # runtime name -> trained tier
        res.update({rn: by_laya[lt] for rn, lt in names.items() if lt in by_laya})
        res.evidence = self._base_ev(applied=True, p_tier=dict(res), n_tokens=self.rt.n_tokens(render_router_text(
            {"domain": domain, "prompt": prompt})), latency_s=round(time.perf_counter() - t0, 3))
        return res

    # -- Scorer -----------------------------------------------------------------------------------------------
    def scorer(self, features: dict[str, Any]) -> float | None:
        """Raw calibrator sigmoid in [0, 1], or None (not applicable). Annotates ``features["laya"]`` with evidence.

        The isotonic map is applied by the caller (CalibrationArtifact from ``calibration_artifacts``); the mapped value
        is also recorded as ``p_isotonic`` for audit."""
        domain, tier = str(features.get("domain")), str(features.get("tier"))
        if self.cal is None or domain not in self.domains:
            features["laya"] = self._base_ev(applied=False, reason="no_calibrator" if self.cal is None else f"domain_not_enabled:{domain}",
                                             truncated_candidate=False, signals_incomplete=False)
            return None
        lt = self._laya_tier(tier)
        if lt is None:
            features["laya"] = self._base_ev(applied=False, reason=f"tier_not_in_laya_training:{tier}",
                                             truncated_candidate=False, signals_incomplete=False)
            return None
        gen = features.get("gen_signals") or {}
        signals: dict[str, Any] = {"gate": str(features.get("gate_status", "none")).lower()}
        for k in ("mean_logprob", "min_logprob"):  # the only numeric signals the frozen rows carried
            if _finite(gen.get(k)):
                signals[k] = float(gen[k])
        missing = [k for k in ("mean_logprob", "min_logprob") if k not in signals]
        row = {"domain": domain, "tier": lt, "signals": signals, "prompt": str(features.get("prompt", "")),
               "candidate": str(features.get("response", ""))}
        text = render_calibrator_text(row)
        t0 = time.perf_counter()
        n = self.cal.n_tokens(text)
        prefix_n = self.cal.n_tokens(render_calibrator_text({**row, "candidate": ""}))  # everything before the candidate
        # content slots in the window: max_len - 2 (CLS, SEP); prefix content tokens: prefix_n - 2
        visible = max(0, min(n - prefix_n, self.cal.max_len - prefix_n))
        truncated = n > self.cal.max_len
        raw = self.cal.probs(text, gate_features(signals))[0]
        veto = (["truncated_candidate"] if truncated else []) + (["signals_incomplete"] if missing else [])
        features["laya"] = self._base_ev(
            applied=True, tier=lt, raw=raw, p_isotonic=self.isotonic_map(raw), n_tokens=n, max_len=self.cal.max_len,
            truncated_candidate=truncated, candidate_fully_outside_window=bool(prefix_n >= self.cal.max_len),
            candidate_tokens_visible=visible, signals_incomplete=bool(missing), missing_signals=missing,
            likely_correct_vetoed=veto, latency_s=round(time.perf_counter() - t0, 3))
        return raw

    # -- calibration artifact ---------------------------------------------------------------------------------
    def calibration_artifacts(self, thresholds: Mapping[str, Mapping[str, float]] | None = None) -> dict[str, Any]:
        """{domain: CalibrationArtifact} for the enabled domains: the registered isotonic map plus tau_hi / tau_lo.

        No Laya tau_hi/tau_lo was registered, so without ``thresholds`` ({domain: {"tau_hi","tau_lo"}}) they are
        +inf / -inf: p is reported but Laya alone never yields LIKELY_CORRECT or LIKELY_WRONG."""
        from gwaya.gwenlaya import CalibrationArtifact
        th = thresholds or {}
        out = {}
        for d in self.domains:
            t = th.get(d, {})
            out[d] = CalibrationArtifact(
                calibrator={"kind": "isotonic", "xs": self.isotonic["x"], "ys": self.isotonic["y"]},
                tau_hi=float(t.get("tau_hi", math.inf)), tau_lo=float(t.get("tau_lo", -math.inf)),
                n_cal=849, selection={"source": "docs/laya_isotonic_C.json", "thresholds": "given" if t else "none (inf/-inf)"})
        return out


def load_thresholds(path: str | Path) -> dict[str, dict[str, float]]:
    """JSON {domain: {"tau_hi": f, "tau_lo": f}} (see scripts/laya_runtime_thresholds.py); other keys are ignored."""
    raw = json.loads(Path(path).read_text())
    src = raw.get("thresholds", raw)
    return {d: {"tau_hi": float(v["tau_hi"]), "tau_lo": float(v["tau_lo"])} for d, v in src.items()
            if isinstance(v, dict) and "tau_hi" in v and "tau_lo" in v}


def parse_tier_map(spec: str | None) -> dict[str, str]:
    """'qwen3.5:2b=qwen3.5-2b-bf16,...' -> {runtime tier: trained tier}. Explicit only; there is no fuzzy matching."""
    out: dict[str, str] = {}
    for part in (spec or "").split(","):
        if part.strip():
            k, sep, v = part.partition("=")
            if not sep or not k.strip() or not v.strip():
                raise ValueError(f"malformed tier map entry {part!r}")
            out[k.strip()] = v.strip()
    return out
