---
name: gwaya-tpu-stage-launch
description: Use before and while running any TPU generation stage in the GWAYA repo (launch_*.sh, drive.sh, run_e_gen.sh) - pre-flight checks, duplicate-VM guard, monitoring, hung-SSH recovery, cleanup and cost accounting. Written from the phase-3 incidents (duplicate T5 VMs, idle VM, empty task file, AWQ failures).
---

# Launching a TPU stage safely

Spend rules and the $50 cap live in `gwaya-study-workflow`; the lessons are in `LL.md` (items 1 to 11). Never launch without the user's explicit spend approval for THIS run, stated with its worst case and the margin left under the ceiling.

## Pre-flight (all read-only)
1. `bash scripts/restart.sh` (ledger, VMs that bill). Trust the ledger over notes.
2. `gcloud compute tpus tpu-vm list --zone -` and `pgrep -af "launch_|drive.sh"`: abort if a driver or a `gwenlaya-tpu-*` VM for this stage exists. Our VMs are `gwenlaya-tpu-*`; `gwenlaya-tpu-1` and others are not ours, never delete them.
3. Dry run the launcher (no `--yes-spend`). It prints the plan and worst case. Check: every task file it uploads is non-empty (`wc -l`), model list and TP are what you intend, worst case + ledger <= ceiling.
4. A checkpoint you have not run before gets a one-model, one-step smoke first. Known: vLLM-TPU needs `logprobs=1`; Qwen3.5 needs `SKIP_JAX_PRECOMPILE=1`; AWQ needs `dtype=bfloat16` (fixed in `gen_batch.py`) and the 35B-A3B AWQ does NOT load at TP=4 (weight shape not aligned); 4 v5e chips per zone and project is the quota, 8-chip slices are unavailable.

## Launch (in the user's terminal if the classifier blocks you)
`GWAYA_CONFIRM_SPEND=1 setsid nohup scripts/tpu/launch_<stage>.sh --yes-spend > $GWAYA_DATA_ROOT/<stage>.log 2>&1 < /dev/null &` once, from one shell line. Never chain the dry run and the real run in one line with `&&` after a pipe to `tail`.

## Monitor
- Use Monitor with an `until` loop on the log or on the GCS partial `status.jsonl` (`gs://.../gwenlaya_v4/tpu_runs/partial/<name>/status.jsonl`; `SMOKE_DONE` is the end marker). Do not sleep-poll. Re-arm expired monitors.
- Each step in `status.jsonl` has `rc` and `secs`; a step with rc=1 in < 60 s is a config error, read its log on the VM (`gcloud compute tpus tpu-vm ssh <vm> --zone <z> --command 'grep -h ERROR ~/smoke/*.log | tail'`).

## When it is done or hung
- If `SMOKE_DONE` is posted and the VM is still READY after ~10 minutes, the driver's SSH is hung: `gcloud compute tpus tpu-vm scp` the results (`raw_*.jsonl`, `status.jsonl`, logs) into `$GWAYA_DATA_ROOT/<stage>/out`, check row counts (tasks x samples), then delete OUR VM with one plain `gcloud compute tpus tpu-vm delete <vm> --zone <z> --quiet` (do not chain `pkill`; kill drivers by PID).
- Confirm with `tpu-vm list` and re-run `restart.sh`; the driver books the ledger when it finishes (estimate = list price, booked from the timestamp in the VM name, so retries can over-attribute).
- Generate on TPU, score LOCALLY (`scripts/phase3/build_pool.py`, `exec_tests.py`), detached with `setsid nohup`, one writer per output file.
