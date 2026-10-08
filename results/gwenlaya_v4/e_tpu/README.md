# Full-E baselines on TPU v5e (analysis plan: docs/ANALYSIS_PLAN_E_TPU.md)

- `numbers_e_tpu.json` - every metric with point estimate and 95% cluster-bootstrap CI (10,000 resamples, seed 0).
- `data/rows_tiers.jsonl`, `data/rows_coder.jsonl` - scored rows, hash-chained, exactly as `run_study.py` wrote them.
- `data/gens_summary.jsonl` - per-generation metadata (tokens, attributed chip-seconds, mean log-prob, finish reason); no text.
- `overlay/rows_overlay_tiers.jsonl` - serial re-scoring of the two tasks whose parallel-run check flipped on a serial re-check.
- `audit/` - serial re-check reports: non-passing rows (1,536 re-checked) and passing rows (1,572 re-run, 0 unstable).
- `SHA256SUMS.txt` - hashes of these files and of the full generations/raw files, which are in the data lake
  (`gs://socrateai-datalake-gen-lang-client-0625573011/gwenlaya_v4/tpu_runs/`), not in git.

Reproduce: `scripts/import_remote_gens.py` -> `scripts/tpu/score_local.sh` / `run_study.py --mode score` ->
`scripts/tpu/recheck_failures.py` -> `scripts/tpu/serial_overlay.py` -> `scripts/analyze_e_tpu.py` (see the commit messages for flags).
