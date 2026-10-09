#!/usr/bin/env python3
"""LoRA training of the Laya / ModernBERT encoder as GwenLaya's router and calibrator (prereg sections 1, 5, 6).

Two heads on one LoRA-adapted encoder (r=16):
  router      predict per-tier P(correct) from the domain tag + prompt           (pre-generation head)
  calibrator  predict P(correct) from prompt + candidate + gate signals          (post-generation head)

Row schemas (JSONL; `source` + `id` identify the source problem and drive the split hash):
  router      {"source","id","domain","prompt","tiers": {"<tier>": 0|1, ...}}   (missing tier = no label)
  calibrator  {"source","id","domain","prompt","tier","candidate","correct": 0|1,
               "signals": {"gate": "verified|failed|...", "tests_passed_frac", "selftest_agreement",
                           "mean_logprob", "min_logprob", "repair_rounds"}}   (absent signal = missing flag)

By default only the router-train slice (sha256(source||id) mod 20 in {1, 2}) is used; calibration (0) and
eval rows are refused. No temperature/isotonic map is fitted here: that is stage C6 on the C split.
`--shuffle-labels` trains the SH negative control (expected AUROC about 0.5).

Heavy imports (torch, transformers, peft, safetensors) are lazy; `--print-plan` and the pure helpers need none.
Output: adapter (safetensors via peft) + heads.safetensors + head_config.json. Nothing here is a measured result.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gwaya.data_build import split_bucket  # noqa: E402

# Encoder named in the data-lake rl_agent_config.json. Whether convaiinnovations/laya (a custom
# LayaTypedDecisions model) loads through AutoModel was NOT verified; pass it via --backbone to try.
DEFAULT_BACKBONE = "answerdotai/ModernBERT-large"
# ModernBERT fused attention/MLP linears (per its modeling code); validated against named_modules() at load.
MODERNBERT_TARGETS = ("Wqkv", "Wo", "Wi")
ROUTER_TRAIN_BUCKETS = (1, 2)
GATE_VOCAB = ("verified", "failed", "unverified", "compile_error", "none")
SIGNAL_KEYS = ("tests_passed_frac", "selftest_agreement", "mean_logprob", "min_logprob", "repair_rounds")
N_FEATURES = len(GATE_VOCAB) + 1 + 2 * len(SIGNAL_KEYS)  # gate one-hot (+other), value+missing per signal
SEED = 1234
CPU_HOURS_LIMIT = 12.0  # prereg section 5: CPU if projected <= 12 h, else inside S5 on L4
CAND_CHARS = 2000


# ---- pure helpers ------------------------------------------------------------------------------

def row_key(r: dict[str, Any]) -> tuple[str, str]:
    return str(r["source"]), str(r["id"])


def in_router_train(r: dict[str, Any]) -> bool:
    return split_bucket(*row_key(r)) in ROUTER_TRAIN_BUCKETS


def is_val(r: dict[str, Any], pct: int = 10) -> bool:
    """Deterministic, problem-grouped hold-out inside the router-train slice."""
    h = hashlib.sha256(("val|" + "|".join(row_key(r))).encode()).hexdigest()
    return int(h, 16) % 100 < pct


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


def make_examples(rows: Sequence[dict[str, Any]], mode: str, tiers: Sequence[str]) -> list[dict[str, Any]]:
    """-> [{key, text, feats, labels, mask}]; router has len(tiers) labels (masked), calibrator has one."""
    out = []
    for r in rows:
        if mode == "router":
            lab = [float(r["tiers"][t]) if t in r.get("tiers", {}) else 0.0 for t in tiers]
            mask = [1.0 if t in r.get("tiers", {}) else 0.0 for t in tiers]
            if not any(mask):
                continue
            ex = dict(text=render_router_text(r), feats=[], labels=lab, mask=mask)
        else:
            if "correct" not in r:
                continue
            ex = dict(text=render_calibrator_text(r), feats=gate_features(r.get("signals")),
                      labels=[float(bool(r["correct"]))], mask=[1.0])
        ex["key"] = row_key(r)
        ex["tier"] = r.get("tier")
        out.append(ex)
    return out


def select_rows(rows: Sequence[dict[str, Any]], enforce_slice: bool = True) -> tuple[list, list, dict[str, int]]:
    """Router-train slice -> (train, val, counts). Rows outside the slice are dropped and counted."""
    train, val, dropped = [], [], 0
    for r in rows:
        if enforce_slice and not in_router_train(r):
            dropped += 1
            continue
        (val if is_val(r) else train).append(r)
    return train, val, {"train": len(train), "val": len(val), "dropped_outside_slice": dropped}


def shuffle_labels(examples: list[dict[str, Any]], seed: int = SEED) -> list[dict[str, Any]]:
    """SH negative control: permute the (labels, mask) pairs across examples."""
    rng = random.Random(seed)
    pairs = [(e["labels"], e["mask"]) for e in examples]
    rng.shuffle(pairs)
    return [dict(e, labels=l, mask=m) for e, (l, m) in zip(examples, pairs)]


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    rp = sum(r for r, y in zip(ranks, labels) if y)
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def ece_equal_mass(probs: Sequence[float], labels: Sequence[int], bins: int = 15) -> float | None:
    n = len(probs)
    if n == 0:
        return None
    idx = sorted(range(n), key=lambda i: probs[i])
    tot = 0.0
    for b in range(bins):
        sl = idx[b * n // bins:(b + 1) * n // bins]
        if sl:
            conf = sum(probs[i] for i in sl) / len(sl)
            acc = sum(labels[i] for i in sl) / len(sl)
            tot += len(sl) / n * abs(conf - acc)
    return tot


def brier(probs: Sequence[float], labels: Sequence[int]) -> float | None:
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs) if probs else None


def decide_where(projected_hours: float) -> str:
    return "cpu" if projected_hours <= CPU_HOURS_LIMIT else "l4 (inside S5)"


def project_timing(seconds_per_step: float, n_train: int, batch: int, epochs: int) -> dict[str, Any]:
    steps = math.ceil(n_train / batch) * epochs
    hours = seconds_per_step * steps / 3600
    return {"seconds_per_step_measured": round(seconds_per_step, 3), "total_steps": steps,
            "projected_hours": round(hours, 2), "cpu_limit_hours": CPU_HOURS_LIMIT, "run_on": decide_where(hours)}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def metrics_for(mode: str, tiers: Sequence[str], probs: list[list[float]], examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Held-out metrics (computed from predictions, never fabricated); None when a class is absent."""
    res: dict[str, Any] = {"n": len(examples)}
    cols = range(len(tiers)) if mode == "router" else [0]
    for c in cols:
        ps = [p[c] for p, e in zip(probs, examples) if e["mask"][c]]
        ys = [int(e["labels"][c]) for e in examples if e["mask"][c]]
        res[tiers[c] if mode == "router" else "p_correct"] = {
            "n": len(ys), "auroc": auroc(ps, ys), "ece15": ece_equal_mass(ps, ys), "brier": brier(ps, ys)}
    return res


# ---- training (lazy heavy imports) -------------------------------------------------------------

def _need(mod: str) -> Any:
    try:
        return importlib.import_module(mod)
    except ImportError as e:
        raise SystemExit(f"missing dependency {mod!r}: pip install -e '.[train]'  ({e})") from e


def train(args: argparse.Namespace, tr: list[dict], va: list[dict], tiers: Sequence[str]) -> dict[str, Any]:
    torch = _need("torch")
    tf = _need("transformers")
    peft = _need("peft")
    st = _need("safetensors.torch")
    torch.manual_seed(SEED)
    xm = None
    if args.device == "xla":  # TPU via torch_xla: fixed shapes (pad to max_len) so the graph compiles once
        xm = _need("torch_xla.core.xla_model")
        dev = xm.xla_device()
    else:
        dev = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    pad = "max_length" if xm is not None else True

    tok = tf.AutoTokenizer.from_pretrained(args.backbone)
    enc = tf.AutoModel.from_pretrained(args.backbone)
    names = [n for n, m in enc.named_modules() if isinstance(m, torch.nn.Linear)]
    targets = [t for t in MODERNBERT_TARGETS if any(n.endswith("." + t) or n == t for n in names)]
    if not targets:
        raise SystemExit(f"none of {MODERNBERT_TARGETS} match Linear modules of {args.backbone}; "
                         f"edit MODERNBERT_TARGETS (found e.g. {names[:6]})")
    if args.load_dir:
        enc = peft.PeftModel.from_pretrained(enc, str(Path(args.load_dir) / "adapter"))
    else:
        enc = peft.get_peft_model(enc, peft.LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05,
                                                       target_modules=targets, bias="none"))
    hidden = enc.config.hidden_size
    n_out = len(tiers) if args.mode == "router" else 1
    n_feat = 0 if args.mode == "router" else N_FEATURES
    head = torch.nn.Sequential(torch.nn.Linear(hidden + n_feat, hidden // 2), torch.nn.GELU(),
                               torch.nn.Dropout(0.1), torch.nn.Linear(hidden // 2, n_out))
    if args.load_dir:
        head.load_state_dict(st.load_file(str(Path(args.load_dir) / "heads.safetensors")))
    enc.to(dev)
    head.to(dev)
    params = [p for p in enc.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    def forward(batch: list[dict]) -> Any:
        t = tok([e["text"] for e in batch], padding=pad, truncation=True, max_length=args.max_len, return_tensors="pt").to(dev)
        with torch.autocast("xla" if xm is not None else "cpu", dtype=torch.bfloat16, enabled=bool(args.bf16 and xm is not None)):
            h = enc(**t).last_hidden_state
        h = h.float()
        m = t["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        if n_feat:
            pooled = torch.cat([pooled, torch.tensor([e["feats"] for e in batch], dtype=pooled.dtype, device=dev)], 1)
        return head(pooled)

    def loss_fn(logits: Any, batch: list[dict]) -> Any:
        y = torch.tensor([e["labels"] for e in batch], dtype=logits.dtype, device=dev)
        m = torch.tensor([e["mask"] for e in batch], dtype=logits.dtype, device=dev)
        l = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
        return (l * m).sum() / m.sum().clamp(min=1)

    rng = random.Random(SEED)
    steps_per_epoch = math.ceil(len(tr) / args.batch_size)
    total = 0 if (args.load_dir or args.init_only) else (args.timing_steps or steps_per_epoch * args.epochs)
    enc.train(); head.train()
    step, t0, losses = 0, time.time(), []
    while step < total:
        order = list(range(len(tr))); rng.shuffle(order)
        for i in range(0, len(order), args.batch_size):
            if step >= total:
                break
            batch = [tr[j] for j in order[i:i + args.batch_size]]
            loss = loss_fn(forward(batch), batch)
            opt.zero_grad(); loss.backward(); opt.step()
            if xm is not None:
                xm.mark_step()
            losses.append(loss.item()); step += 1
            if step % 25 == 0:
                print(f"[train_laya] step {step}/{total} loss {sum(losses[-25:]) / 25:.4f} {time.time() - t0:.0f}s", flush=True)
    wall = time.time() - t0
    result: dict[str, Any] = {"steps": step, "wall_seconds": round(wall, 1),
                              "final_train_loss_last10_mean": round(sum(losses[-10:]) / max(1, len(losses[-10:])), 4)}
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if args.timing_steps and not args.load_dir:
        result["timing"] = project_timing(wall / max(1, step), len(tr), args.batch_size, args.epochs)
        (out / "timing.json").write_text(json.dumps(result, indent=1))
        return result

    enc.eval(); head.eval()
    if not args.load_dir and not args.init_only:  # save first, from CPU: safetensors cannot read XLA-device storage, and a late failure would lose the run
        enc.to("cpu"); head.to("cpu")
        enc.save_pretrained(str(out / "adapter"))  # peft: adapter_model.safetensors
        tok.save_pretrained(str(out / "adapter"))
        st.save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}, str(out / "heads.safetensors"))
        enc.to(dev); head.to(dev)

    def predict(examples: list[dict]) -> list[list[float]]:
        res: list[list[float]] = []
        with torch.no_grad():
            for i in range(0, len(examples), args.batch_size):
                res += torch.sigmoid(forward(examples[i:i + args.batch_size])).float().cpu().tolist()
        return res

    probs = predict(va) if va else []
    result["val_metrics"] = metrics_for(args.mode, tiers, probs, va) if va else "TBD (no validation rows)"
    for spec in args.predict or []:  # NAME=PATH: raw scores for other row files (calibration split, evaluation set)
        name, _, path = spec.partition("=")
        ex = make_examples(read_jsonl(path), args.mode, tiers)
        pr = predict(ex)
        with open(out / f"preds_{name}.jsonl", "w") as f:
            for e, p_ in zip(ex, pr):
                f.write(json.dumps({"key": list(e["key"]), "tier": e.get("tier"), "probs": p_}) + "\n")
        result.setdefault("predicted", {})[name] = len(ex)
    if args.load_dir or args.init_only:
        (out / "predict_manifest.json").write_text(json.dumps(result, indent=1, default=str))
        return result
    (out / "head_config.json").write_text(json.dumps({
        "mode": args.mode, "tiers": list(tiers), "backbone": args.backbone, "lora_r": args.r, "lora_targets": targets,
        "hidden_size": hidden, "n_features": n_feat, "gate_vocab": list(GATE_VOCAB), "signal_keys": list(SIGNAL_KEYS),
        "max_len": args.max_len, "shuffled_labels": args.shuffle_labels,
        "calibration": "none fitted here (stage C6 fits temperature/isotonic on the C split)"}, indent=1))
    result["versions"] = {m: getattr(importlib.import_module(m), "__version__", "?") for m in ("torch", "transformers", "peft")}
    (out / "run_manifest.json").write_text(json.dumps(result, indent=1, default=str))
    return result


# ---- CLI ---------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["router", "calibrator"])
    ap.add_argument("--data", required=True, help="JSONL, schema in --help")
    ap.add_argument("--tiers", default="qwen3.5:2b,qwen3.5:4b,qwen3.5:9b,qwen3.8:27b",
                    help="ordered tier names for the router head (cheapest first)")
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | xla (TPU, needs torch_xla)")
    ap.add_argument("--load-dir", default=None, help="predict-only: load adapter/ + heads.safetensors from a previous --out-dir")
    ap.add_argument("--init-only", action="store_true", help="zero training steps: score with the untrained (seeded) head, the 'twin' of a trained run")
    ap.add_argument("--predict", nargs="*", help="NAME=PATH row files to score after training/loading (writes preds_NAME.jsonl)")
    ap.add_argument("--bf16", action="store_true", help="bfloat16 autocast (xla only)")
    ap.add_argument("--timing-steps", type=int, default=0, help="e.g. 50: time N steps, project hours, decide CPU vs L4")
    ap.add_argument("--shuffle-labels", action="store_true", help="SH negative control")
    ap.add_argument("--no-slice-filter", action="store_true", help="dev only: skip the router-train hash filter")
    ap.add_argument("--print-plan", action="store_true", help="resolved plan, no torch import")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tiers = [t for t in args.tiers.split(",") if t]
    rows = read_jsonl(args.data)
    tr_rows, va_rows, counts = select_rows(rows, not args.no_slice_filter)
    tr, va = make_examples(tr_rows, args.mode, tiers), make_examples(va_rows, args.mode, tiers)
    if args.shuffle_labels:
        tr = shuffle_labels(tr)
    plan = {"mode": args.mode, "backbone": args.backbone, "lora_r": args.r, "tiers": tiers if args.mode == "router" else None,
            "rows_read": len(rows), "selection": counts, "examples": {"train": len(tr), "val": len(va)},
            "steps_per_epoch": math.ceil(len(tr) / args.batch_size) if tr else "TBD", "epochs": args.epochs,
            "shuffle_labels": args.shuffle_labels, "slice": f"sha256(source||id) mod 20 in {list(ROUTER_TRAIN_BUCKETS)}",
            "projected_hours": "TBD (run --timing-steps 50)"}
    if args.print_plan:
        print(json.dumps(plan, indent=1))
        return 0
    if args.load_dir and not tr:
        tr = [{"text": "x", "feats": [0.0] * (0 if args.mode == "router" else N_FEATURES),
               "labels": [0.0] * (len(tiers) if args.mode == "router" else 1), "mask": [1.0] * (len(tiers) if args.mode == "router" else 1)}]
    if not tr or not args.out_dir:
        print("error: need training examples and --out-dir", file=sys.stderr)
        return 2
    print(json.dumps(train(args, tr, va, tiers), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
