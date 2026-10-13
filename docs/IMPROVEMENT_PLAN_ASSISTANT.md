# Improvement plan: GWAYA as a local coding assistant on TPU (2026-10-13)

Status: plan, not results. Evidence so far is one smoke run (`results/assist_smoke_2026-10-13.json`): Qwen3.5-4B bf16 on
one v5e chip behind the gateway, 8 easy Python tasks, 8/8 VERIFIED in round 1, cold start 427 s, warm request 1.8-4.2 s.
That run shows the plumbing works. It does **not** show the assistant is good: the tasks were easy, hand-written, few,
and never exercised the repair loop on a real failure.

## Measured constraints that drive the plan
- Cold start ~7 min per session; the VM costs ~$1.20/h, so a session is ~$0.2-0.6. Unusable as "always on" without a
  warm window; fine as a pre-warmed work session.
- Qwen3.5 runs on the PyTorch fallback (not JAX-native): single-stream ~50 tok/s, far below Qwen2.5-Coder JAX-native.
- The gate is weak when visible tests are few (known; H1 carried by math, Python alone includes zero). VERIFIED means
  "passed the tests you gave", not "correct".

## Workstreams (ordered by value per dollar)

### P0 - Make the assistant useful in a real editor (no TPU spend)
1. **Session mode**: `gwaya_assist.py serve` keeps one warmed endpoint for a work session, auto-`/v1/stop` after N idle
   minutes (the VM already self-deletes at 10 min idle; surface this in the CLI).
2. **Context in, patch out**: accept a file path + failing test output, return a unified diff instead of a whole file;
   extend the gate to run the project's own test command in the sandbox (read-only mount, no network).
3. **MCP tool** `gwaya_assist_remote` in `mcp_server.py` so Claude Code / Cursor can call the TPU-backed tier.
4. **Gate strength surfaced**: report `n_tests` with every verdict and mark VERIFIED with < k tests as "weak"
   (use the existing `min_visible_tests` policy; do not pick a default k without held-out data).
Done when: one real bug in this repo is fixed end to end via the CLI, with the diff reviewed by a human.

### P1 - Evaluate honestly (about $1-3 TPU)
Addendum A23 in `docs/ANALYSIS_PLAN_E_TPU.md` written **before** running. Hypothesis (exploratory, not registered):
the repair loop (<= 3 rounds) raises hidden-test pass rate over single-shot on the same 4B tier.
- Data: fresh set F Python tasks (committed), 100-150 tasks; visible tests only in the loop, hidden tests for scoring.
- Arms: single-shot, repair x3, and repair x3 + weak-gate demotion. Same prompts, greedy, thinking off.
- Metrics: hidden pass rate, confident-wrong rate among VERIFIED, rounds/task, wall-clock, $/solved task.
  Cluster bootstrap by source problem, as in A1-A6.
- Decision rule fixed in advance; "not claimed": generalisation beyond F-Python, any Laya benefit (Laya is not applied).
Risk to watch: repair loops can overfit visible tests (VERIFIED but hidden-fail); that rate is the headline cost.

### P2 - Latency and cost
1. Persist weights on the endpoint bucket (already) and test a **pre-built VM image / pip wheel cache** to cut the 427 s.
2. Try `--max-num-seqs 32` and speculative or prefix caching only if single-stream decode is the bottleneck.
3. Compare Qwen2.5-Coder-7B (JAX-native, ~2k tok/s batched) as an assistant tier: faster but weaker/not Qwen3.5;
   measure on the same P1 set before choosing.
4. Cold-start fallback: if the TPU is unavailable (capacity), route to the Cloud Run L4 endpoint or local llama.cpp.

### P3 - Quality (needs GPU/TPU training budget; defer until P1 reads out)
- Laya needs log-probs from the serving stack it was calibrated on; add `logprobs` passthrough check in the gateway and a
  quant/stack-shift probe for this endpoint before using P(correct).
- Longer-window Laya (512-token limit drops long prompts); only worthwhile if P1 shows routing headroom.
- 27B tier remains unloaded; out of scope for one v5e chip.

## Spend and governance
Ledger $22.18 of the $50 cap after the smoke run (estimate). P1 fits well under the cap; every TPU run is dry-run first,
needs an explicit go from the user, and ends with the VM deleted and the ledger line appended.

## Known limits of this plan
Small n, one language (Python), one model; hand-written smoke tasks; list-price cost estimates; no human-usability study.
