# GCP Spot GPU Benchmarking Runbook

## Overview

This runbook describes how to run the GWAYA benchmark on low-cost GCP spot GPU instances to evaluate performance across different hardware and model sizes. The tooling supports **T4** (16GB VRAM) and **L4** (24GB VRAM) GPUs on spot instances.

## GPU Options

### Supported GPUs

| GPU   | Machine Type    | VRAM  | Price/Hour | Suggested Models               | Use Case                           |
|-------|-----------------|-------|------------|--------------------------------|------------------------------------|
| **T4**  | n1-standard-8   | 16GB  | ~$0.29    | qwen2.5-coder:7b              | Cost-optimized smaller models     |
| **L4**  | g2-standard-8   | 24GB  | ~$0.35    | qwen2.5-coder:{7b,14b}         | Production; supports larger models |

### Unavailable Options

- **H4**: Not available on GCP; use L4 as alternative (similar performance)
- **A100/H100**: Out-of-budget for spot instances; use L4 instead

## Prerequisites

1. **GCP Project Setup**
   - Set `GCP_PROJECT` environment variable or default to `gen-lang-client-0625573011`
   - Ensure service account has permissions:
     - `compute.instances.create`, `compute.instances.delete`
     - `storage.buckets.list`, `storage.objects.get`, `storage.objects.create`
     - `artifactregistry.dockerimages.get`

2. **Data Lake Assets**
   - Model tarball: `gs://<lake>/model/ollama_models.tar`
   - Dataset: `gs://<lake>/datasets/mbpp-sanitized/`
   - Docker image: `gs://<lake>/images/gwaya-bench.tar.gz` (fallback)
   - Registry image: `us-east4-docker.pkg.dev/<project>/gwaya-bench/runner:v1`

3. **Local Environment**
   - `gcloud` CLI configured with default project
   - `python3` installed
   - Bash 4.0+

## What to Benchmark on GPU

The benchmark measures:

1. **Models**: Small-to-medium code LLMs
   - `qwen2.5-coder:7b` (fits T4 16GB, preferred on L4)
   - `qwen2.5-coder:14b` (L4 only; 24GB needed)
   - `qwen2.5-coder:32b` (not recommended; requires quantization or larger GPU)

2. **Dataset**: MBPP subset for reproducible evaluation
   - Default: 257 problems (`--n 257`)
   - Smoke test: 5 problems (`--smoke`)
   - Full replication: 1000+ (adjust `--n`)

3. **Metrics per Problem**
   - **Solved**: Problem code passes all public + hidden tests in bwrap sandbox
   - **Pass Rate**: Percentage of solved problems
   - **Tokens/sec**: Generation speed (tok/s)
   - **Peak VRAM**: Memory usage during generation
   - **Cascade Router**: Measured token throughput and VRAM for multi-pass repair

4. **Cascade Router Measurement**
   - Cascade router selects models based on token-budget constraints
   - Measure on both T4 and L4 to find the crossover point
   - Record: tokens generated, VRAM used, wall-time

## Example Runs

### Dry-Run (Default; No Costs)

```bash
# Print gcloud commands, cost estimate, and zones to try (no resources created)
deploy/run_on_gcp.sh --gpu l4 --n 257

# With model override
deploy/run_on_gcp.sh --gpu t4 --n 257 --model-ladder qwen2.5-coder:7b

# Smoke test on L4
deploy/run_on_gcp.sh --smoke
```

Output shows:
- Exact gcloud commands for each zone
- Worst-case cost estimate
- Zones to try (listed in order)

### Real Run (Requires Spend Confirmation)

```bash
# Launch the benchmark (requires --yes-spend AND env var)
GWAYA_CONFIRM_SPEND=1 deploy/run_on_gcp.sh --gpu l4 --n 257 --yes-spend

# Custom parameters
GWAYA_CONFIRM_SPEND=1 deploy/run_on_gcp.sh \
  --gpu t4 \
  --n 100 \
  --model-ladder qwen2.5-coder:7b \
  --max-hours 1 \
  --cap-usd 2 \
  --yes-spend
```

### Parameters

| Parameter         | Default      | Description                                      |
|-------------------|--------------|--------------------------------------------------|
| `--gpu {t4,l4}`   | `l4`         | GPU type                                        |
| `--n <num>`       | `257`        | Number of MBPP problems to solve                |
| `--model-ladder`  | Auto         | Model name (e.g., `qwen2.5-coder:7b`)          |
| `--max-hours`     | `2`          | Max instance runtime (includes OS/setup time)   |
| `--cap-usd`       | `5`          | Abort if worst-case cost exceeds this           |
| `--dry-run`       | Default      | Print commands without executing                |
| `--yes-spend`     | Off          | Enable actual resource creation                 |
| `--tag`           | Auto         | Custom run tag for tracking                     |
| `--zone`          | All zones    | Specific zone (default tries multiple zones)    |
| `--smoke`         | Off          | 5 problems, 1 hour (quick test)                 |

### Cost Estimation

Worst-case cost = (max_hours × price_per_hour × 1.25) + 0.15

- **1.25 multiplier**: 25% margin for spot preemption overhead
- **$0.15 allowance**: Disk + egress costs

Example (L4, 2 hours):
```
2 hours × $0.35/hour × 1.25 + $0.15 = $0.90 worst-case
```

## CPU-vs-GPU Comparison Table

Fill this in after running benchmarks on CPU and GPU:

| Configuration      | Pass Rate | Avg Tok/s | Peak VRAM | Cost/Run | Speedup |
|--------------------|-----------|-----------|-----------|----------|---------|
| CPU (baseline)     | ___%      | __ tok/s  | N/A       | ~$0.02   | 1.0x    |
| T4 (qwen2.5:7b)    | ___%      | __ tok/s  | __ GB     | ~$0.30   | __x     |
| L4 (qwen2.5:7b)    | ___%      | __ tok/s  | __ GB     | ~$0.35   | __x     |
| L4 (qwen2.5:14b)   | ___%      | __ tok/s  | __ GB     | ~$0.35   | __x     |

## Release Checks

After each run, the script automatically:

1. **Polls for completion** every 60 seconds
2. **Checks for remaining VMs**: deletes any instances labelled `purpose=gwaya-bench`
3. **Cleans up disks**: removes any orphaned disks with the same label
4. **Prints final status**: lists any remaining resources (should be empty)

## Comparing Runs

Use `scripts/compare_runs.py` to compare CPU and GPU results:

```bash
# Print deltas
python3 scripts/compare_runs.py cpu_results.json gpu_results.json

# With gate rule (fail if pass rate drops > 5%)
python3 scripts/compare_runs.py \
  --baseline cpu_results.json \
  --compare gpu_results.json \
  --gate "pass_rate_pct >= -5"
```

Output includes:
- Pass rate delta (percentage points)
- Tokens/sec delta
- VRAM usage delta
- Gate rule status (PASS/FAIL)

## Troubleshooting

### Spot Instance Preemption

If the instance is preempted before completion:
- Results up to preemption time are synced to the lake every 90s
- Resume the same run with the same tag: `--tag <previous-tag>`
- Script syncs output state from lake and continues

### Out-of-Capacity Errors

If all zones are out of capacity:
- Try again in 5–10 minutes
- Or use a different GPU: `--gpu t4` has higher availability than `--gpu l4`
- Or reduce `--max-hours` to request shorter durations (sometimes available)

### Cost Overruns

The script aborts if worst-case cost exceeds the cap:
- Lower `--max-hours` to reduce time estimate
- Lower `--n` to reduce expected completion time
- Use `--gpu t4` (cheaper) instead of `--gpu l4`

### Docker Registry Failures

If image pull fails, the startup script falls back to loading from tarball:
- Ensure `gs://<lake>/images/gwaya-bench.tar.gz` exists
- Or pre-pull image to avoid during startup

## Environment Variables

| Variable               | Effect                                                   |
|------------------------|----------------------------------------------------------|
| `GCP_PROJECT`          | GCP project ID (default: `gen-lang-client-0625573011`) |
| `GWAYA_CONFIRM_SPEND`  | Required set to `1` with `--yes-spend` to authorize     |

## Pricing Notes

Prices in `deploy/gpu_prices.json` are **unverified**. Always check current pricing:
- [GCP GPU Pricing](https://cloud.google.com/compute/gpus-pricing)
- Spot discounts typically 70–80% off on-demand
- Prices vary by region and zone

## Success Criteria

A successful run produces:
- `results.json`: Summary of pass rate, VRAM, tok/s
- `rows.jsonl`: Per-problem metrics (one JSON per line)
- `startup.log`: VM startup and container output
- `DONE` marker: Indicates completion status

All files are synced to `gs://<lake>/runs/<tag>/out/`

## Next Steps

After benchmarking:

1. **Analyze Results**
   ```bash
   python3 scripts/compare_runs.py cpu_results.json gpu_results.json
   ```

2. **Fill CPU-vs-GPU Table** in this runbook

3. **Document Findings**
   - Note any surprising performance gaps
   - Record cascade router token throughput
   - Estimate cost/accuracy tradeoff

4. **Archive Run**
   - Download results from lake
   - Tag in git or results database
   - Reference in paper/report
