#!/usr/bin/env python3
"""Stage helper for deploy/run_on_gcp.sh --stage <id> (GwenLaya v4, work item G2).

Single source of truth for a stage's cap, max run time, zones, expected outputs and entry commands,
all read from experiments/plan.json. Stdlib only; never calls gcloud, never touches the network.

  stage_launch.py info S2 [--gpu t4|l4]   shell KEY='value' lines for `eval` (exit 3 if not a spot-VM stage)
  stage_launch.py entry S2                the commands deploy/stage_entry.sh runs, one per line
  stage_launch.py ollama-models S2        ollama tags to pull before the stage
  stage_launch.py budget                  sum of stage caps + endpoint cap vs the hard/planned totals
  stage_launch.py packet [--out FILE]     (re)generate experiments/launch_packet.md from real dry-runs
  stage_launch.py check-quota             stdin: `gcloud compute project-info describe --format=json`
  stage_launch.py check-accel             stdin: `gcloud compute instances list --format=json`
  stage_launch.py ledger-total            stdin: LEDGER.jsonl
  stage_launch.py ledger-line --stage S --seconds N --rate R --attempt K --status X

No number is measured here. Costs are caps from plan.json (planning rates, list prices read
2026-10-07 per plan.json price_source); est_hours stays TBD until S1 measures throughput.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess  # nosec B404
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "experiments" / "plan.json"
PRICES_PATH = Path(__file__).resolve().parent / "gpu_prices.json"
PACKET_PATH = ROOT / "experiments" / "launch_packet.md"
LAKE = "gs://socrateai-datalake-gen-lang-client-0625573011/gwenlaya_v4"
STAGE_IMAGE = "us-east4-docker.pkg.dev/gen-lang-client-0625573011/gwenlaya/runner:v4"  # not built yet: human checklist
HARD_TOTAL_USD = 50.0
SPOT_WHERE = {"l4": "l4", "t4": "t4"}
T4_MAX_PARAMS_B = 4.7  # scripts/train_lora.py GPUS["t4"]["max_params_b"]
_STAGE_RE = re.compile(r"^[A-Za-z0-9]{1,8}$")
_PARAMS_RE = re.compile(r"(\d+(?:\.\d+)?)b\b", re.I)
RUN_STUDY_STAGES = {"S1", "S2", "S5", "S6"}
TRAIN_STAGES = {  # stage -> recipes (scripts/train_lora.py RECIPES)
    "S4a": ["qwen3.5-4b-sft", "qwen3.5-4b-dpo"],
    "S4b": ["qwen3.5-9b-sft"],
    "S4c": ["qwen3.8-27b-sft"],
    "S4d": ["qwen3.8-27b-sft"],
}
RECIPE_BASE = {"qwen3.5-4b-sft": "Qwen/Qwen3.5-4B", "qwen3.5-4b-dpo": "Qwen/Qwen3.5-4B",
               "qwen3.5-9b-sft": "Qwen/Qwen3.5-9B", "qwen3.8-27b-sft": "Qwen/Qwen3.8-27B"}


def load_plan(path: Path = PLAN_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def get_stage(plan: dict[str, Any], sid: str) -> dict[str, Any]:
    if not _STAGE_RE.match(sid):
        raise ValueError(f"unsafe stage id {sid!r}")
    for s in plan["stages"]:
        if s["id"] == sid:
            return s
    raise ValueError(f"unknown stage {sid!r}; plan has {[s['id'] for s in plan['stages']]}")


def is_ollama_tag(m: str) -> bool:
    return ":" in m and "/" not in m and " " not in m


def max_params_b(stage: dict[str, Any]) -> float | None:
    """Largest parameter count named in the stage's models; None when a model's size is unknown."""
    best = 0.0
    for m in stage.get("models") or []:
        found = [float(x) for x in _PARAMS_RE.findall(m)]
        if "Qwen3.5-4B" in m:
            found.append(4.7)  # 4B is 4.7B parameters (train_lora PARAMS_B)
        if not found:
            return None
        best = max(best, *found)
    return best


def expected_outputs(sid: str, stage: dict[str, Any]) -> list[str]:
    if sid in RUN_STUDY_STAGES:
        return [f"{LAKE}/runs/{sid}/gens.jsonl", f"{LAKE}/runs/{sid}/rows.jsonl", f"{LAKE}/runs/{sid}/results.json",
                f"{LAKE}/runs/{sid}/DONE", f"{LAKE}/results/{sid}/"]
    if sid == "S3":
        return [f"{LAKE}/datasets/S3/ (sft.jsonl, candidates.jsonl, build_report.json, DATASET_CARD.md)",
                f"{LAKE}/runs/{sid}/DONE"]
    if sid in TRAIN_STAGES:
        outs = [f"{LAKE}/adapters/{r}/" for r in TRAIN_STAGES[sid]] if sid != "S4c" else [
            f"{LAKE}/runs/S4c/probe/ (run_manifest.json: peak VRAM, tokens/s; no adapter is kept)"]
        return outs + [f"{LAKE}/results/{sid}/", f"{LAKE}/runs/{sid}/DONE"]
    return [f"{LAKE}/runs/{sid}/DONE"]


def entry_commands(plan: dict[str, Any], sid: str) -> list[str]:
    st = get_stage(plan, sid)
    gpu = "${GWAYA_GPU:-l4}"
    res = f"/data/results/{sid}"
    cmds = [f"mkdir -p {res}"]
    if sid in RUN_STUDY_STAGES:
        tags = [m for m in st.get("models", []) if is_ollama_tag(m)]
        models = " ".join(shlex.quote(m) for m in tags) if tags else "${GWAYA_STUDY_MODELS:?set GWAYA_STUDY_MODELS (served tuned tier names, cheapest first)}"
        lim = f" --limit {st['n']}" if isinstance(st.get("n"), int) else ""
        cmds.append(f"python scripts/run_study.py --stage {sid} --out /data/out --mode generate --backend ollama "
                    f"--lake-sync --tasks /data/datasets/tasks_{sid}.jsonl --models {models} --quants q4_K_M{lim} "
                    "${GWAYA_STUDY_EXTRA:-}")
        if sid == "S1":
            for r in ("qwen3.5-4b-sft", "qwen3.5-9b-sft"):
                cmds.append(f"python scripts/train_lora.py --recipe {r} --gpu {gpu} --train-file /data/datasets/pilot_sft.jsonl "
                            f"--out-dir /data/adapters/S1_pilot_{r} --max-steps 20 --no-resume")
            cmds.append(f"python scripts/quantize.py plan --base Qwen/Qwen3.5-9B --out-dir /data/quantized/S1_fit_check > {res}/quantize_plan_9b.json")
        cmds.append(f"cp /data/out/results.json {res}/ 2>/dev/null || true")
    elif sid == "S3":
        cmds.append("python scripts/data/build_sft_dataset.py --tasks /data/datasets/tasks_S3.jsonl --decontam-index /data/datasets/decontam_index.jsonl "
                    "--ladder ${GWAYA_S3_LADDER:-openai@http://127.0.0.1:8000/v1#Qwen/Qwen3.5-9B} --out-dir /data/out/S3 ${GWAYA_S3_EXTRA:-}")
        cmds.append(f"cp -r /data/out/S3/. {res}/")
    elif sid in TRAIN_STAGES:
        for r in TRAIN_STAGES[sid]:
            kind = "dpo_pairs" if r.endswith("-dpo") else "sft"
            probe = sid == "S4c"
            out = f"/data/out/S4c/probe" if probe else f"/data/adapters/{r}"
            sync = f"{LAKE}/runs/S4c/probe" if probe else f"{LAKE}/adapters/{r}"
            extra = " --max-steps 100" if probe else ""
            base = " ${GWAYA_DPO_BASE:+--base-model-path $GWAYA_DPO_BASE}" if r.endswith("-dpo") else ""
            cmds.append(f"python scripts/train_lora.py --recipe {r} --gpu {gpu} --train-file /data/datasets/{kind}.jsonl "
                        f"--out-dir {out} --gcs-sync {sync}{extra}{base}")
            if not probe and sid in ("S4a", "S4b") and not r.endswith("-dpo"):
                cmds.append(f'[ "${{GWAYA_QUANTIZE_ON_VM:-0}}" != 1 ] || python scripts/quantize.py run --base {RECIPE_BASE[r]} '
                            f"--adapter /data/adapters/{r} --out-dir /data/quantized/{r} --install-llama-cpp")
        cmds.append(f"cp -r /data/out/{sid}/. {res}/ 2>/dev/null || true")
    return cmds


def stage_info(plan: dict[str, Any], sid: str, gpu: str | None = None) -> dict[str, Any]:
    st = get_stage(plan, sid)
    where = st.get("where")
    if where not in SPOT_WHERE:
        raise ValueError(f"stage {sid} runs on {where!r}, not on a spot GPU VM"
                         + (" (Cloud Run endpoint: needs the user's approval of exact commands; see deploy/cloudrun)" if where == "cloudrun-l4" else ""))
    gpu = gpu or SPOT_WHERE[where]
    if gpu not in ("l4", "t4"):
        raise ValueError(f"gpu must be l4 or t4 (A100/H100 quota is 0 and the user declined requests), got {gpu!r}")
    if gpu == "t4":
        mp = max_params_b(st)
        if mp is None or mp > T4_MAX_PARAMS_B:
            raise ValueError(f"stage {sid} does not fit a 16 GB T4 (largest model {mp}B, limit {T4_MAX_PARAMS_B}B)")
    rates = plan["planning_rates_usd_per_h"]
    rate = float(rates["l4_spot_vm"] if gpu == "l4" else rates["t4_spot_vm_fallback"])
    cap = float(st["cap_usd"])
    seconds = int(cap / rate * 3600)
    prices = json.loads(PRICES_PATH.read_text())["gpus"][gpu]
    zones = []
    for region in st.get("region_pref") or ["us-east4", "us-central1"]:
        zones += [z for z in prices["zones"] if z.startswith(region + "-")]
    return {
        "id": sid, "gpu": gpu, "where": where, "cap_usd": cap, "rate_usd_per_h": rate, "max_run_seconds": seconds,
        "max_hours": round(seconds / 3600, 2), "worst_usd": round(seconds / 3600 * rate, 2), "zones": zones,
        "machine_type": prices["machine_type"], "depends_on": st.get("depends_on", []),
        "conditional": st.get("conditional"), "outputs": expected_outputs(sid, st),
        "price_source": plan.get("price_source", "unverified"),
    }


def budget(plan: dict[str, Any]) -> dict[str, Any]:
    stages = {s["id"]: float(s["cap_usd"]) for s in plan["stages"] if float(s.get("cap_usd", 0)) > 0}
    endpoint = sum(v for k, v in stages.items() if get_stage(plan, k).get("where") == "cloudrun-l4")
    vm = round(sum(stages.values()) - endpoint, 2)
    misc = float(plan["misc_cap_usd"])
    total = round(vm + endpoint + misc, 2)
    return {"vm_stage_caps": vm, "endpoint_cap": endpoint, "stage_caps_plus_endpoint": round(vm + endpoint, 2),
            "misc_cap": misc, "total_with_misc": total, "planned_cap": float(plan["planned_cap_usd"]),
            "hard_cap": HARD_TOTAL_USD, "ok": round(vm + endpoint, 2) <= HARD_TOTAL_USD
            and total <= float(plan["planned_cap_usd"]) and total <= HARD_TOTAL_USD, "per_stage": stages}


def check_quota(doc: dict[str, Any]) -> tuple[bool, str]:
    for q in doc.get("quotas", []):
        if q.get("metric") == "GPUS_ALL_REGIONS":
            used, lim = float(q.get("usage", 0)), float(q.get("limit", 0))
            return used == 0, f"GPUS_ALL_REGIONS usage={used:g} limit={lim:g}"
    return False, "GPUS_ALL_REGIONS not found in project-info (fail closed)"


def check_accel(instances: list[dict[str, Any]]) -> tuple[bool, str]:
    busy = [i.get("name", "?") for i in instances
            if i.get("guestAccelerators") and i.get("status") in ("RUNNING", "STAGING", "PROVISIONING", "REPAIRING")]
    return not busy, ("running accelerator VMs: " + ", ".join(busy)) if busy else "no running accelerator VM"


def ledger_total(text: str) -> float:
    tot = 0.0
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                tot += float(json.loads(line).get("cost_usd_upper_bound", 0))
            except (ValueError, TypeError, AttributeError):
                raise ValueError(f"unreadable ledger line (fail closed): {line[:80]!r}")
    return round(tot, 4)


# ---- launch packet ------------------------------------------------------------------------------

CHECKLIST = """\
## Human checklist (nothing here has been done by the tooling)

1. **Billing budget alert.** None exists (creating one is a new cloud resource and needs your approval).
   Until then the ledger and the per-stage `--max-run-duration` are the only guards.
2. **GPU slot.** `GPUS_ALL_REGIONS` limit is 1. plan.json records a blocker read 2026-10-07 19:11 UTC:
   `socreateai-agora-hermes-node1` (T4 spot, us-east4-b) was RUNNING. Not GwenLaya's; you stop it or schedule
   around it. The launcher refuses to start while usage != 0 (read-only check, re-done at launch).
3. **Quotas** (read 2026-10-07 per the task context, re-check before each stage): spot L4 us-central1=3,
   us-east4=1, europe-west4=1; spot T4=1 per region; A100/H100=0 (not used, no request made).
   Cloud Run L4 quota check for S7 is separate and read-only.
4. **Runner image.** `{image}` does not exist yet. The old `deploy/Dockerfile` is a CPU-torch Ollama image and
   cannot train. A CUDA image with torch, transformers, peft, bitsandbytes, trl, vllm (S3) and llama.cpp must be
   built (Cloud Build spend goes on the `misc: build` line, 1.50 USD cap) and pushed; its tag and contents are TBD.
   The launcher's preflight fails closed if the image is missing.
5. **HF token.** Qwen3.5/Qwen3.8 are Apache-2.0 and ungated per the task context, so none is expected; export
   `HF_TOKEN` in the VM metadata only if a download is refused. Not wired by default.
6. **Inputs in the lake** (`{lake}/datasets/`): `tasks_<STAGE>.jsonl` for run_study stages, `pilot_sft.jsonl` (S1),
   `sft.jsonl` / `dpo_pairs.jsonl` (S4*), `decontam_index.jsonl` (S3). C0/C3 produce them; names are this tooling's
   convention. The C0 addendum hash must exist before S2.
7. **Per-stage authorization.** Each launch needs your explicit approval of the exact command, then
   `GWAYA_CONFIRM_SPEND=1 deploy/run_on_gcp.sh --stage <id> --yes-spend`.
8. **After each stage.** Read the release check output (no `purpose=gwenlaya` VM or disk), and read the ledger.

## Known gaps (stated, not hidden)

- No script measures S1 tokens/s at batch 1/8, peak VRAM per engine, or the vLLM/llama.cpp support matrix.
  The S1 entry runs the run_study generation pass, two 20-step QLoRA memory pilots and `quantize.py plan`;
  the throughput/support matrix still needs a measurement script (TBD).
- S3 runs `build_sft_dataset.py`, which also verifies candidates on the GPU VM, while plan.json puts verification
  on CPU (C3). It needs an OpenAI-compatible server at `GWAYA_S3_LADDER`; the image must provide vLLM (TBD).
- S4a DPO trains on the base unless `GWAYA_DPO_BASE` points to an SFT-merged model; merging is a CPU step (C4).
- S5/S6 need `GWAYA_STUDY_MODELS` / `GWAYA_STUDY_EXTRA` (tuned tier names, `--lora-map`, `--quant-map`,
  `--calibrations`) from the C4/C6 outputs. Tuned GGUFs found in `/data/quantized/*/Modelfile` are registered with Ollama.
- Spot preemption relaunch resumes run_study (hash-chained logs) and train_lora (`--gcs-sync` checkpoints);
  S3 has no resume of its own (restarts from the synced `candidates.jsonl` only if the builder supports it: unchecked).
- Prices: planning rates are carried from plan.json (`{price_source}`); NOT re-read from the billing catalog in
  this session. `deploy/gpu_prices.json` rates (self-labelled unverified) are not used for stage caps.
"""


def build_packet(plan: dict[str, Any], runner=subprocess.run) -> str:
    b = budget(plan)
    out = ["# GwenLaya v4 launch packet", "",
           "Generated by `python deploy/stage_launch.py packet` from `experiments/plan.json` "
           f"(docs/GWENLAYA_PREREGISTRATION.md section 11). Generated {time.strftime('%Y-%m-%d')}. "
           "Every dry-run block below is real output of `deploy/run_on_gcp.sh --stage <id>` (dry-run default; "
           "no gcloud call is made). Nothing was launched; no number below is measured. `est_hours` is TBD until S1.", "",
           "## Budget", "",
           f"- VM stage caps: {b['vm_stage_caps']:.2f} USD; endpoint (S7) cap: {b['endpoint_cap']:.2f} USD; "
           f"sum: **{b['stage_caps_plus_endpoint']:.2f}** (must be <= {b['hard_cap']:.0f}): {'OK' if b['stage_caps_plus_endpoint'] <= b['hard_cap'] else 'VIOLATED'}.",
           f"- Plus misc caps {b['misc_cap']:.2f} = **{b['total_with_misc']:.2f}** vs planned cap {b['planned_cap']:.2f} and hard cap "
           f"{b['hard_cap']:.0f}: {'OK' if b['ok'] else 'VIOLATED'}. Reserve {plan['reserve_usd']:.2f} is never pre-allocated.",
           f"- Planning rates (USD/h, all-in with margin): L4 spot VM {plan['planning_rates_usd_per_h']['l4_spot_vm']}, "
           f"T4 spot VM {plan['planning_rates_usd_per_h']['t4_spot_vm_fallback']}, Cloud Run L4 4 CPU/16 GiB "
           f"{plan['planning_rates_usd_per_h']['cloudrun_l4_4cpu_16gib']}. Source: {plan.get('price_source', 'unverified')}; "
           "carried from plan.json, not re-read this session.",
           "- Worst-case USD per stage = floor(cap / rate) seconds x rate, never above the cap. One GPU at a time.", "",
           "| Stage | Where | Cap USD | Max run (s) | Worst-case USD | Depends on |", "|---|---|---|---|---|---|"]
    for s in plan["stages"]:
        if float(s.get("cap_usd", 0)) <= 0:
            continue
        if s.get("where") in SPOT_WHERE:
            i = stage_info(plan, s["id"])
            out.append(f"| {s['id']} | {i['gpu']} spot VM | {i['cap_usd']:.2f} | {i['max_run_seconds']} | {i['worst_usd']:.2f} | {', '.join(i['depends_on']) or '-'} |")
        else:
            out.append(f"| {s['id']} | {s['where']} | {float(s['cap_usd']):.2f} | n/a (<= {s.get('max_instance_hours', 'TBD')} instance-h) | "
                       f"{float(s['cap_usd']):.2f} (cap) | {', '.join(s.get('depends_on', [])) or '-'} |")
    out += ["", "S7 (Cloud Run endpoint) is not launched by `run_on_gcp.sh`: it needs your approval of the exact commands, "
            "a read-only Cloud Run L4 quota check, and its image build (misc: build). Its cap is counted above.", ""]
    for s in plan["stages"]:
        if s.get("where") not in SPOT_WHERE or float(s.get("cap_usd", 0)) <= 0:
            continue
        sid = s["id"]
        r = runner(["bash", str(ROOT / "deploy" / "run_on_gcp.sh"), "--stage", sid, "--dry-run"],
                   capture_output=True, text=True, check=False)
        i = stage_info(plan, sid)
        out += [f"## {sid}", "", f"- Models: {', '.join(s.get('models') or []) or 'none'}",
                f"- Cap {i['cap_usd']:.2f} USD, max run {i['max_run_seconds']} s ({i['max_hours']} h), worst-case {i['worst_usd']:.2f} USD "
                f"at {i['rate_usd_per_h']} USD/h; est_hours TBD.",
                f"- Depends on: {', '.join(i['depends_on']) or '-'}"
                + (f"; conditional: {i['conditional']}" if i["conditional"] else ""), "- Expected outputs:"]
        out += [f"  - `{o}`" if " (" not in o else f"  - {o}" for o in i["outputs"]]
        out += ["", "Dry-run output:", "", "```", (r.stdout + (r.stderr if r.returncode else "")).rstrip(), "```", ""]
    out.append(CHECKLIST.format(image=STAGE_IMAGE, lake=LAKE, price_source=plan.get("price_source", "unverified")))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default=str(PLAN_PATH))
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("info", "entry", "ollama-models"):
        p = sub.add_parser(n)
        p.add_argument("stage")
        if n == "info":
            p.add_argument("--gpu")
    sub.add_parser("budget")
    pk = sub.add_parser("packet")
    pk.add_argument("--out", default=str(PACKET_PATH))
    for n in ("check-quota", "check-accel", "ledger-total"):
        sub.add_parser(n)
    ll = sub.add_parser("ledger-line")
    ll.add_argument("--stage", required=True)
    ll.add_argument("--seconds", type=int, required=True)
    ll.add_argument("--rate", type=float, required=True)
    ll.add_argument("--attempt", type=int, default=1)
    ll.add_argument("--status", default="UNKNOWN")
    a = ap.parse_args(argv)
    try:
        plan = load_plan(Path(a.plan))
        if a.cmd == "info":
            i = stage_info(plan, a.stage, a.gpu)
            for k, v in (("S_GPU", i["gpu"]), ("S_CAP", i["cap_usd"]), ("S_RATE", i["rate_usd_per_h"]),
                         ("S_MAX_SECONDS", i["max_run_seconds"]), ("S_WORST", i["worst_usd"]), ("S_ZONES", " ".join(i["zones"])),
                         ("S_MACHINE", i["machine_type"]), ("S_PRICE_SOURCE", i["price_source"]),
                         ("S_OUTPUTS", "; ".join(i["outputs"])), ("S_DEPENDS", " ".join(i["depends_on"]))):
                print(f"{k}={shlex.quote(str(v))}")
        elif a.cmd == "entry":
            print("\n".join(entry_commands(plan, a.stage)))
        elif a.cmd == "ollama-models":
            print("\n".join(m for m in get_stage(plan, a.stage).get("models", []) if is_ollama_tag(m)))
        elif a.cmd == "budget":
            b = budget(plan)
            print(json.dumps(b, indent=1))
            return 0 if b["ok"] else 1
        elif a.cmd == "packet":
            Path(a.out).write_text(build_packet(plan) + "\n")
            print(f"wrote {a.out}")
        elif a.cmd == "check-quota":
            ok, msg = check_quota(json.load(sys.stdin))
            print(msg)
            return 0 if ok else 1
        elif a.cmd == "check-accel":
            ok, msg = check_accel(json.load(sys.stdin))
            print(msg)
            return 0 if ok else 1
        elif a.cmd == "ledger-total":
            print(ledger_total(sys.stdin.read()))
        elif a.cmd == "ledger-line":
            print(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": a.stage, "attempt": a.attempt,
                              "seconds": a.seconds, "rate_usd_per_h": a.rate, "status": a.status,
                              "cost_usd_upper_bound": round(a.seconds / 3600 * a.rate, 4)}))
    except (ValueError, KeyError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 3 if a.cmd == "info" else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
