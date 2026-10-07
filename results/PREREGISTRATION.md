# Pre-registration: GWAYA low-tier optimizer on MBPP-sanitized

Written 2026-10-03 before any measured run of `scripts/run_benchmark.py`.

**History (disclosed).** An earlier 3-problem run on hand-written problems gave
A0 = A1 = A2 = A3 = A4 = 100 % (delta 0.0, gate FAIL): the problems were too easy and arms
A1/A4 were not genuinely different configurations. A hand-written results file with
invented numbers (A0 33 %, A3 100 %) was created during that session and used in a
published paper revision. It was fabricated, has been deleted, and is NOT evidence.
This benchmark replaces it.

- Model: `qwen2.5-coder:1.5b` (Ollama, CPU), seeds `1000*(problem_index+1) + call_index`.
- Problems: first N = 60 task_ids (ascending) of MBPP-sanitized `test`.
- Public test: `test_list[0]`; hidden tests: `test_list[1:]`.
- Outcome: final code passes ALL tests in the bwrap sandbox. Scoring refuses to run without bwrap.
- Arms: A0 raw sample; A2 best-of-3 by public test; A3 best-of-3 + <=3 repair rounds;
  A4 = A3 + one retrieved train-split exemplar (leakage-guarded).
- Fallback policy for A2-A4: verified code, else the A0 sample.
- **Gate:** A3 - A0 >= +8.0 percentage points. A failing gate is reported as a finding.
- Also reported: paired bootstrap 95 % CI, exact McNemar p, gate coverage and precision.
- With N = 60 the CI is wide; a PASS with a CI that includes 0 is stated as such.

## Addendum (2026-10-03, written before the N = 257 run starts): GPU replication

- **N = 60 on local CPU remains the pre-registered primary result** (Ollama 0.1.44). Its gate verdict is
  reported whatever it shows.
- A second run on the **full MBPP-sanitized test split (N = 257)** on a spot NVIDIA L4 (Ollama 0.5.7,
  container image `gwaya-bench/runner:v1`) is declared a **replication / extension**, not a rescue run.
  It uses a FRESH generation cache; no local rows are reused. Same seeds, arms, fallback policy and gate.
- Same-seed Ollama outputs differ between CPU/GPU and across Ollama versions (observed on 5 smoke
  problems: A4 differed on 2 of 5). The two runs are therefore different samples of the same
  procedure; both are published. If they disagree on the gate, both verdicts are stated and the
  larger-N CI is the better-powered estimate.
- Smoke run (n = 5, tasks 11,12,14,16,17) was an infrastructure check only; it is not evidence.
- Hardware/Ollama version are recorded in `results.json` from what Ollama actually used
  (`ollama ps`), not from `nvidia-smi` presence.
