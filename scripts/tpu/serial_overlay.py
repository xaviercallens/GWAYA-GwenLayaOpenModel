#!/usr/bin/env python
"""Re-score tasks whose parallel-run check flipped on a serial re-check, into an OVERLAY run.

The original hash-chained rows/gens are never edited. For every (model, domain, task_id) that
recheck_failures.py found VERIFIED when run alone, this re-imports the stored generations of the
affected tasks into a fresh study cache and re-scores ALL arms for those tasks serially (one check at a
time). analyze_e_tpu.py --overlay-* then lets these rows take precedence for those tasks and reports
the corrections in the numbers file.

  python scripts/tpu/serial_overlay.py --audits a.json b.json c.json --raw-dir RAW --tasks TASKS --out OUT \
      --tiers qwen3.5-2b-bf16 qwen3.5-4b-bf16 --coder qwen2.5-coder-1.5b-bf16
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
ARMS = ["base", "always_smallest", "always_largest", "gate_only", "raw_confidence", "gwenlaya"]


def collect_flips(audits: list[Path]) -> dict[str, set[tuple[str, str]]]:
    flips: dict[str, set[tuple[str, str]]] = {}
    for a in audits:
        d = json.loads(Path(a).read_text())
        for f in d["flips"]:
            flips.setdefault(d["model"], set()).add((f["domain"], f["task_id"]))
    return flips


def subset_files(tasks_path: Path, raw_dir: Path, models: list[str], keys: set[tuple[str, str]], out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(l) for l in tasks_path.read_text().splitlines() if l.strip()]
    keep = [t for t in tasks if (t["domain"], t["task_id"]) in keys]
    tfile = out / "tasks_subset.jsonl"
    tfile.write_text("".join(json.dumps(t) + "\n" for t in keep))
    for m in models:
        rows = [json.loads(l) for l in (raw_dir / f"raw_{m}.jsonl").read_text().splitlines() if l.strip()]
        sub = [r for r in rows if (r["domain"], r["task_id"]) in keys]
        (out / f"raw_{m}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sub))
    return tfile


def run_group(name: str, models: list[str], arms: list[str], keys, tasks_path: Path, raw_dir: Path, out: Path) -> dict:
    if not keys:
        return {"group": name, "tasks": 0}
    gdir = out / name
    tfile = subset_files(tasks_path, raw_dir, models, keys, gdir)
    for m in models:
        subprocess.run([PY, str(ROOT / "scripts/import_remote_gens.py"), "--tasks", str(tfile), "--raw", str(gdir / f"raw_{m}.jsonl"),
                        "--served", m, "--quant", "bf16", "--out", str(gdir), "--accelerator", "TPU v5e x1"], check=True, cwd=ROOT)
    subprocess.run([PY, str(ROOT / "scripts/run_study.py"), "--plan", str(ROOT / "experiments/night/plan_night.json"),
                    "--stage", "night_L2", "--backend", "openai", "--backend-url", "http://127.0.0.1:1/v1", "--tasks", str(tfile),
                    "--models", *models, "--quants", "bf16", "--arms", *arms, "--mode", "score", "--no-vram",
                    "--check-workers", "1", "--out", str(gdir)], check=True, cwd=ROOT)
    return {"group": name, "tasks": len(keys), "models": models, "arms": arms}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--audits", nargs="+", required=True, type=Path)
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--tasks", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tiers", nargs=2, required=True)
    ap.add_argument("--coder", required=True)
    a = ap.parse_args(argv)
    flips = collect_flips(a.audits)
    tier_keys = flips.get(a.tiers[0], set()) | flips.get(a.tiers[1], set())
    meta = {"flips": {m: sorted(v) for m, v in flips.items()},
            "tiers": run_group("tiers", list(a.tiers), ARMS, tier_keys, a.tasks, a.raw_dir, a.out),
            "coder": run_group("coder", [a.coder], ["base"], flips.get(a.coder, set()), a.tasks, a.raw_dir, a.out)}
    (a.out / "overlay_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: (v if k != "flips" else {m: len(x) for m, x in v.items()}) for k, v in meta.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
