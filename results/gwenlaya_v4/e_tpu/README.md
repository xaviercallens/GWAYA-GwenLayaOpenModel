# Full-E baselines on TPU v5e (analysis plan: docs/ANALYSIS_PLAN_E_TPU.md)

- `numbers_e_tpu.json` - every metric with point estimate and 95% cluster-bootstrap CI (10,000 resamples, seed 0).
- `data/rows_tiers.jsonl`, `data/rows_coder.jsonl` - scored rows, hash-chained, exactly as `run_study.py` wrote them.
- `data/gens_summary.jsonl` - per-generation metadata (tokens, attributed chip-seconds, mean log-prob, finish reason); no text.
- `overlay/rows_overlay_tiers.jsonl` - serial re-scoring of the two tasks whose parallel-run check flipped on a serial re-check.
- `audit/` - serial re-check reports: non-passing rows (1,536 re-checked) and passing rows (1,572 re-run, 0 unstable).
- `SHA256SUMS_repo.txt` - check from the repository root: `sha256sum -c results/gwenlaya_v4/e_tpu/SHA256SUMS_repo.txt`.
- `SHA256SUMS_dataroot.txt` - hashes of the full generations, raw files and task file, which live on the data disk and in the
  data lake (`gs://socrateai-datalake-gen-lang-client-0625573011/gwenlaya_v4/`), not in git; check from `$GWAYA_DATA_ROOT`.

Reproduce: `scripts/import_remote_gens.py` -> `scripts/tpu/score_local.sh` / `run_study.py --mode score` ->
`scripts/tpu/recheck_failures.py` -> `scripts/tpu/serial_overlay.py` -> `scripts/analyze_e_tpu.py` (see the commit messages for flags).
