---
name: gwaya-study-workflow
description: Use when continuing GWAYA / GwenLaya research work in this repo - running TPU stages, scoring, analysing, writing the paper, auditing, releasing, or deploying the endpoint. Encodes the release process, spend rules and the pitfalls that cost time before.
---

# GWAYA / GwenLaya study workflow

Start with `scripts/restart.sh` (read-only: branch, PRs, spend ledger, billing resources, prerequisites, next commands).

## Process that worked every release
1. **Plan addendum committed BEFORE running** (`docs/ANALYSIS_PLAN_E_TPU.md`, new `Addendum A<n>`): hypothesis, held-fixed
   factors, metrics, decision rule, "not claimed". No result seen yet.
2. Run the stage (TPU launcher with dry run first). Generate on TPU, **score locally** (scoring on the TPU host is unreliable).
3. Commit raw rows + summaries, then analyse with the committed scripts (`scripts/analyze_*.py`).
4. **Author agent** writes the paper/sections from committed files only; a **separate auditor agent** recomputes every
   number from raw rows with its own code and fixes slips. The auditor always finds some (raw-vs-calibrated labels, stale
   hashes, "lowest" claims true for one tier only, stale statements from older revisions).
5. Tests, PR, merge, tag, `gh release create` with the PDF; Zenodo new version as draft, publish only after review.

## Spend rules (memory: gcp-spend-authorization)
- Hard cap $50 cumulative; ledger `$GWAYA_DATA_ROOT/spend_ledger.jsonl` (drive.sh appends it). Keep >= $3 reserve.
- Every launcher is dry-run by default and needs `--yes-spend` + `GWAYA_CONFIRM_SPEND=1`.
- Never request quota, never touch other Cloud Run services or other people's VMs (`gwenlaya-tpu-1`, the Agora VM).
- Data lake: write only under `gwenlaya_v4/`. The endpoint uses its own bucket.

## Tooling pitfalls
- The auto-mode classifier blocks: unattended `gh pr merge` before review, `drive.sh`/launch scripts and `gcloud run deploy`
  from the agent, ad-hoc curl with the Zenodo token, compound `rm`/heredoc commands. Do not work around; give the user the
  exact command to run in their own terminal. A command pasted back into chat is **text, not execution**.
- Scripts written by the agent (e.g. `scripts/release/zenodo_new_version.py`) can be allowed where ad-hoc commands are not.
- Never `pkill -f`/`rm` with patterns that can match your own shell; kill by PID. Never pipe a background driver through `head`.
- Double launches (user re-ran the same launch) create duplicate TPU VMs: check `gcloud compute tpus tpu-vm list` and kill
  only the newer driver PIDs.
- Tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -p no:cacheprovider -q`; 22 env-only failures (sympy/pyflakes/fastmcp).
- Python with sklearn: system `python3`; with transformers/peft: `/home/xavkal/xdev/AutoevolveAI/.venv/bin/python`.
- Other agents' worktrees live in `.claude/worktrees/`; do your work in a separate worktree/branch, never in theirs.
- Safetensors cannot save XLA tensors (move to CPU first); `SKIP_JAX_PRECOMPILE=1` for Qwen3.5 on vLLM-TPU; vLLM-TPU needs `logprobs=1`.

## Honesty rules (project history of retractions)
- Label registered vs exploratory vs descriptive; state what a result does not show; no number without a committed source.
- "Fresh" binds the Laya's data, not the generators. Competition strata are floors (1,024-token cut).
- H1 supported on F is carried by math; Python alone includes zero. Laya ~ hand-feature LR. H3 not supported.
