"""Shared CLI plumbing for build_sft_dataset.py / build_dpo_pairs.py (see gwaya/data_build.py)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gwaya import data_build as db  # noqa: E402
from gwaya.domains.task import DOMAINS  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"

LIMITATIONS = [
    "Verified means the checker accepted the output against the available tests / kernel; it does not "
    "prove the tests are complete.",
    "Decontamination thresholds are conventional and were not validated on this data.",
    "Only eval sets present in the decontamination index were checked; see the Decontamination section "
    "for completeness.",
    "Token lengths were not measured; length ratios are in characters.",
    "No downstream training or evaluation result is implied by this dataset card.",
]


def add_common_args(ap: argparse.ArgumentParser, kind: str) -> None:
    ap.add_argument("--tasks", help="tasks JSONL {domain, task_id, prompt, checker_payload, source, license?, split?}")
    ap.add_argument("--ladder", action="append", default=[], metavar="KIND@URL#MODEL",
                    help="generator rung, cheap to strong, repeatable. ollama@http://host:11434#model or "
                         "openai@http://host:8000/v1#model (vLLM / llama.cpp server)")
    ap.add_argument("--lean-round", action="append", default=[], metavar="RUNG|RUNG",
                    help="Lean expert-iteration round ladder ('|' separated rungs), repeatable: round 2+ "
                         "normally points at the model fine-tuned on round 1 data. Default: one round on --ladder")
    ap.add_argument("--domains", default=",".join(DOMAINS))
    ap.add_argument("--k", action="append", default=[], metavar="DOMAIN=N", help="samples per rung (default 4; lean4 8)")
    ap.add_argument("--cap", action="append", default=[], metavar="DOMAIN=N", help="rows cap per domain (prereg planning caps)")
    ap.add_argument("--workers", type=int, default=1, help="concurrent tasks sampled")
    ap.add_argument("--decontam-index", help="eval index JSONL for gwaya.decontaminate (required unless --dry-run)")
    ap.add_argument("--allow-incomplete-decontam", action="store_true",
                    help="proceed when eval sets are not materialized (the card records complete=false)")
    ap.add_argument("--no-plan-check", action="store_true", help="do not require every plan eval set in the index")
    ap.add_argument("--rust-translate-from", help="SFT JSONL with VERIFIED python rows; adds verified Rust translations")
    ap.add_argument("--translator", help="rung used for translation (default: first --ladder rung)")
    ap.add_argument("--out-dir")
    ap.add_argument("--dry-run", action="store_true",
                    help="fake generator + fake checker + bundled fixtures; contacts nothing, outputs are marked dry_run")
    if kind == "dpo":
        ap.add_argument("--candidates", help="reuse a candidates.jsonl from a previous run instead of sampling")


def _kv(items: list[str], base: dict[str, int]) -> dict[str, int]:
    out = dict(base)
    for it in items:
        d, _, n = it.partition("=")
        if d not in DOMAINS or not n.isdigit():
            raise SystemExit(f"bad DOMAIN=N: {it!r}")
        out[d] = int(n)
    return out


def fake_ladder(tasks: list[db.BuildTask]) -> dict[str, list[str]]:
    resp = {}
    for t in tasks:
        if t.extra.get("fake_responses"):
            resp[t.task.prompt] = list(t.extra["fake_responses"])
        if t.extra.get("fake_translation"):
            resp[f"Source task id: {t.task.task_id}"] = [t.extra["fake_translation"]]
    return resp


def build_config(a: argparse.Namespace, tasks: list[db.BuildTask], kind: str):
    domains = [d for d in a.domains.split(",") if d]
    sc = db.SampleConfig(k=_kv(a.k, db.DEFAULT_K), workers=a.workers, need_failure=(kind == "dpo"))
    caps = _kv(a.cap, db.DEFAULT_CAPS)
    if a.dry_run:
        resp = fake_ladder(tasks)
        ladder = [db.FakeGenerator("fake-small", responses=resp), db.FakeGenerator("fake-large", responses=resp)]
        lean = [[db.FakeGenerator("fake-lean-r1", responses=resp)], [db.FakeGenerator("fake-lean-r2", responses=resp)]]
        gate = db.DecontamGate.from_index(a.decontam_index or FIXTURES / "dryrun_eval_index.jsonl",
                                          check_plan=False, manifest_path=None)
        return db.BuildConfig(sc, ladder, lean, caps, domains, db.fake_check, db.checker_version(fake=True), gate, True)
    if not a.ladder and not (kind == "dpo" and a.candidates):
        raise SystemExit("--ladder is required (or --dry-run)")
    if not a.decontam_index:
        raise SystemExit("--decontam-index is required: refusing to write data without decontamination")
    gate = db.DecontamGate.from_index(a.decontam_index, allow_missing=a.allow_incomplete_decontam,
                                      check_plan=not a.no_plan_check)
    from gwaya.domains import check
    return db.BuildConfig(sc, [db.parse_rung(s) for s in a.ladder],
                          [[db.parse_rung(s) for s in r.split("|")] for r in a.lean_round],
                          caps, domains, check, db.checker_version(), gate, False)


def load_tasks(a: argparse.Namespace) -> list[db.BuildTask]:
    path = a.tasks or (FIXTURES / "dryrun_tasks.jsonl" if a.dry_run else None)
    if not path:
        raise SystemExit("--tasks is required (or --dry-run)")
    return db.load_build_tasks(path, db.plan_licenses())


def maybe_translate(a, cfg, tasks, all_py_tasks) -> tuple[list[db.BuildTask], dict[str, int] | None]:
    if not a.rust_translate_from or "rust" not in cfg.domains:
        return [], None
    by_id = {t.task.task_id: t for t in all_py_tasks if t.task.domain == "python"}
    rows, seen = [], set()
    for r in db.read_jsonl(Path(a.rust_translate_from)):
        bt = by_id.get(r["task_id"])
        if r["task_id"] in seen:       # several accepted traces per prompt: translate each parent once
            continue
        seen.add(r["task_id"])
        if r["domain"] != "python" or bt is None or not bt.task.checker_payload.get("tests"):
            continue
        rows.append({"task_id": r["task_id"], "source": r["source"], "license": r["license"],
                     "prompt": bt.task.prompt, "tests": bt.task.checker_payload["tests"],
                     "solution": db.extract_code(r["messages"][1]["content"], "python")})
    if a.dry_run:
        resp = fake_ladder(tasks)
        translator = db.FakeGenerator("fake-translator", responses=resp)
    else:
        translator = db.parse_rung(a.translator or a.ladder[0])
    out, stats = db.translate_python_rows(rows, translator, cfg.check_fn)
    if a.dry_run:      # fixture behavior: the fake rungs answer translated tasks with the reference solution
        for bt in out:
            for g in [*cfg.ladder, *(g for r in cfg.lean_rounds for g in r)]:
                g.responses[bt.task.prompt] = [f"```rust\n{bt.extra['reference_solution']}\n```"]
    return out, stats


def out_dir(a: argparse.Namespace, kind: str) -> Path:
    d = Path(a.out_dir) if a.out_dir else Path(db.DEFAULT_DATA_ROOT) / ("dryrun" if a.dry_run else "") / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def common_report(a, cfg, drops, tasks_used, states_info) -> dict[str, Any]:
    gate = cfg.gate
    return {
        "dry_run": bool(a.dry_run), "checker_version": cfg.checker_ver,
        "generation": {"rungs": [g.rung for g in cfg.ladder],
                       "lean_round_rungs": [[g.rung for g in r] for r in cfg.lean_rounds],
                       "k": cfg.sample.k, "temperature": cfg.sample.temperature, "top_p": cfg.sample.top_p,
                       "max_tokens": cfg.sample.max_tokens, "keep_per_prompt": cfg.sample.keep},
        "decontamination": gate.report if gate else {},
        "drops": {k: len(v) for k, v in sorted(drops.items())},
        "decontam_task_removed": len(drops.get("decontam_task", [])),
        "decontam_solution_removed": len(drops.get("decontam_solution", [])),
        "sources": {t.source: t.license for t in tasks_used},
        "sampling_rounds": states_info.get("rounds", {}),
        "limitations": LIMITATIONS + (["Dry run: fixtures and a fake checker, nothing here is a real result."]
                                      if a.dry_run else []),
    }


def finish(kind: str, outdir: Path, cfg, rows: list[dict[str, Any]], report: dict[str, Any],
           drops: dict[str, list[str]], states) -> int:
    name = "sft.jsonl" if kind == "sft" else "dpo_pairs.jsonl"
    path = outdir / name
    db.write_jsonl(path, rows)
    (outdir / "dropped.jsonl").write_text("".join(
        json.dumps({"reason": k, "key": x}) + "\n" for k, v in sorted(drops.items()) for x in v))
    if states is not None:
        db.write_jsonl(outdir / "candidates.jsonl", (asdict(c) for s in states for c in s.cands))
    db.write_jsonl(outdir / "flagged_decontam.jsonl", cfg.gate.flagged)
    ver = (db.verify_written_sft if kind == "sft" else db.verify_written_dpo)(cfg.gate, path)
    prov_ok = all(db.provenance_complete(r["provenance"] if kind == "sft" else r["provenance"]["chosen"])
                  and (kind == "sft" or db.provenance_complete(r["provenance"]["rejected"])) for r in rows)
    report.update({"kind": kind, "output": str(path), "output_sha256": db.file_sha256(path),
                   "n_rows": len(rows), "decontam_verification": ver, "provenance_complete": prov_ok})
    (outdir / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (outdir / "DATASET_CARD.md").write_text(db.render_card(kind, report))
    print(f"{kind}: {len(rows)} rows -> {path}")
    print(f"decontam re-check ok={ver['ok']} (n={ver['n_rechecked']}); provenance complete={prov_ok}")
    print(f"report: {outdir / 'build_report.json'}  card: {outdir / 'DATASET_CARD.md'}")
    return 0 if ver["ok"] and prov_ok else 1


def per_domain(rows_by_domain: Counter, states, ids_by_domain: dict[str, set[str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for d in DOMAINS:
        sts = [s for s in (states or []) if s.bt.task.domain == d]
        if sts or rows_by_domain.get(d):
            out[d] = {"tasks_sampled": len(sts), "tasks_with_output": len(ids_by_domain.get(d, ())),
                      "rows": rows_by_domain.get(d, 0)}
    return out
