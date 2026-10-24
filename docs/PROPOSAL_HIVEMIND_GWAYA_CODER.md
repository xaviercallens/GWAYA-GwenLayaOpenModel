# Proposal: GWAYA as the low-cost local Python coder of HiveMind (Code expert + GWAYA-Lite for 6-8 GB GPUs)

Status: P0 implemented 2026-10-24 (`gwaya/lite.py`, `gwaya/hivemind_expert.py`, `tests/test_lite_hivemind.py`, 13 tests pass with assistant tests); P1 not run (see section 9). Owner decisions of 2026-10-24: Lite tiers are 3B and 7B Q4; adapter lives in this repo and is shared with HiveMind by `pip install` of this repo (no HiveMind source change yet). Written 2026-10-24 after reading `AutoevolveAI/sub_projects/hivemind` (README, `btm_moe.py`, `configs/rtx2070_dual_profile.yaml`, `mcp_hivemind_server.py`) and this repo's `gwaya/cascade_router.py`, `llama_tier.py`, `assistant.py`, `mcp_server.py`.

## 1. What each side already has

| | HiveMind | GWAYA |
|---|---|---|
| Role | Branch-Train-Mix sparse MoE: 8 domain branches (one is `Code`), Top-2 routing, 3-tier 128-expert gate, NF4 QLoRA experts, 8 GB envelope (RTX 2070, 6656 MiB usable) | Generate -> AST stub audit -> sandboxed tests -> repair; Qwen2.5-Coder ladder 0.5B/1.5B/7B; OpenAI-format client (`llama_tier`); MCP server |
| Missing | A verified Python-coding expert and a verifier that can reject its output | A trained-expert mixture; an IDE-facing endpoint |

They compose: HiveMind supplies the experts and the routing, GWAYA supplies the verification gate and the escalation policy.

## 2. Evidence boundary (read before committing to any claim)

- The only local small-model numbers are `results/gwaya_v3_multi_model/` (Qwen2.5-Coder-0.5B, **n = 20** tasks): pass@1 40% raw -> 45% with the GWAYA loop, repair recovered 1 of 12 failures, and of the candidates the gate called verified only **52.9%** passed hidden tests. At n = 20 that is not a significant gain, and the gate is weak at 0.5B.
- The 226.9 / 135 / 71 tok/s figures in `cascade_router.py` come from a different RTX machine. They have not been measured here.
- This machine has an **NVIDIA Quadro K3100M (Kepler, ~4 GB)** plus Intel iGPU. It cannot run current CUDA builds of llama.cpp/vLLM, so nothing in this proposal can be benchmarked locally. Benchmarks need the RTX 2070 HiveMind node or a rented 8 GB GPU (the $50 GCP cap is spent to $41.75 and the standing order is no further spend, so this needs an explicit new authorization).
- The larger GWAYA study (Qwen3.5, TPU) concluded its verifier goal was **not met** (phase 3, G sealed read). The gate strength problem is known; the proposal must keep verdicts honest (VERIFIED / FAILED / UNVERIFIED as in `assistant.py`) and never present a visible-test pass as correctness.

## 3. Tiers (VRAM budget, Q4_K_M GGUF, 4k context, KV cache included; sizes are estimates until measured)

| Tier | Model | VRAM | Target hardware | Use |
|---|---|---|---|---|
| Lite-S | Qwen2.5-Coder-1.5B-Instruct Q4 | ~1.5-2 GB | 4 GB GPUs, iGPU | autocomplete (FIM), docstrings, small edits |
| **Lite (default)** | Qwen2.5-Coder-7B-Instruct Q4_K_M | ~5-5.5 GB | **6-8 GB** | chat, function-level generation, repair loop |
| Lite+ | Lite (7B) + 1.5B draft for speculative decoding | ~7-7.5 GB | 8 GB | latency |
| Full | 14B Q4 (~9.5 GB) or the Qwen3.5 tiers from the study | 10+ GB | 12 GB+ / remote endpoint | hard escalation only |

Rule: never co-load two models above 8 GB total; the 8 GB HiveMind invariant (<= 6656 MiB usable on Windows/WDDM) applies, so on Windows the default is 7B Q4 with a short context or the 3B tier.

## 4. Integration along the HiveMind MoE approach

Two levels, do level A first.

**A. Expert-of-experts at the serving layer (no training, works now).**
GWAYA registers as the `Code` branch of `Season3DomainExpertRegistry` (and as a `Python` sub-expert). HiveMind's router (`route_domain_query`, keyword signatures, Top-2) picks branches per request; when `Code` wins, the request goes to the GWAYA endpoint:

1. Router stage (HiveMind): Top-2 macro-domain; if `Code` weight below threshold, answer elsewhere.
2. Tier stage (GWAYA cascade): 1.5B first for short tasks, 7B otherwise, escalate on gate failure with the diagnostic transferred (already what `GwayaCascadeRouter` does). This is a sparse-activation analogue: most requests activate one small expert.
3. Gate stage (GWAYA): AST audit and sandboxed visible tests; return verdict + code.
4. Escalation: after the local ladder fails, hand to the HiveMind swarm/Cloud Run La Ruche vault (remote) or Claude; this is the third "tier" and the only one that costs money or swarm credits.

**B. True BTM expert (later, needs GPU and data).**
Train a Python-coding QLoRA branch (r=16, as the other Season 3 experts) on top of Qwen2.5-Coder-1.5B/7B, plus a `Python_Tests` branch. Register through `Season3DomainExpertRegistry.register_expert` with a CAS location, hot-plug into the upcycled MoE. DiLoCo/Multi-Krum is HiveMind's own machinery; GWAYA's contribution is the *reward and spot-check oracle*: a worker's update is accepted when GWAYA's sandbox pass rate on a held-out task set does not regress. This is the one place GWAYA directly strengthens HiveMind's Byzantine story and it is the part worth a paper claim, but only after measurement.

## 5. IDE integration (Cursor, VS Code)

- Serve with `llama-server` (or Ollama) behind an **OpenAI-compatible** `/v1` endpoint on `127.0.0.1`; `gwaya/llama_tier.py` already speaks this protocol and supports a bearer key.
- Cursor: custom OpenAI base URL and model name. VS Code: Continue or Cline pointing at the same base URL; FIM completion with the 1.5B tier, chat with 7B.
- Add a thin `gwaya serve-lite` gateway: `/v1/chat/completions` runs the cascade and returns code plus the verdict in a response header and a trailing comment line; `/v1/completions` passes through to the 1.5B tier without the gate (latency).
- MCP: reuse `mcp_server.py` (stub audit, sandbox, repair) so Claude Code sessions on this machine can call the local model as a cheap first pass. HiveMind's `mcp_hivemind_server.py` is registered in `AutoevolveAI/.mcp.json`; the GWAYA server would be added next to it, not merged into it.

## 6. Plan and acceptance criteria

1. **P0 (no GPU, 1 day):** `gwaya-lite` config (`deploy/lite/`): model table, llama-server launch lines, VRAM estimator; a `HiveMindCodeExpert` adapter implementing the registry interface that calls the cascade over HTTP. Unit tests with a fake generator, as the existing cascade tests do.
2. **P1 (needs an 8 GB GPU, explicit authorization):** measure on one 8 GB card: VRAM at 4k/8k context, tok/s, and pass@1 on MBPP-sanitized (the repo's benchmark) for 1.5B and 7B, n >= 257, raw vs GWAYA loop. Accept the Lite tier only if 7B fits <= 6656 MiB and the loop does not lower pass@1.
3. **P2:** gate calibration. Report hidden-pass precision of VERIFIED verdicts; if below 0.8, ship the verdict labelled "tests passed" not "verified".
4. **P3:** IDE gateway and Cursor/VS Code recipes, tested by hand.
5. **P4 (optional):** BTM Python expert + oracle-gated update acceptance.

## 7. Risks

- The 0.5B/1.5B tiers fail often; the cascade then costs more latency than going straight to 7B. P1 must measure escalation rate before enabling the small-first policy by default.
- Keyword routing in HiveMind (`route_domain_query`) misroutes mixed prompts (code about biology); prefer the learned `HierarchicalMoERouter` once available, keyword as fallback.
- HiveMind's charter forbids payment content and `eval/exec/shell=True` in contributed code; GWAYA's sandbox runs candidate code under bwrap but the adapter itself must not use `exec`.
- Two repos, two release processes: the adapter lives in HiveMind (`src/hivemind/...`, via PR with the 5-layer scanner), the lite configs and gateway live here. No cross-repo change is made without a PR in each.

## 8. Decisions needed from the owner

1. Authorize a short rented/borrowed 8 GB GPU run for P1 (the cap is exhausted), or point to the RTX 2070 node.
2. Default Lite model: 7B Q4 (best quality that fits) or 3B (safer headroom on Windows).
3. Where the adapter lives: HiveMind `src/hivemind/core` (recommended) or this repo.

## 9. P1 status and blockers (2026-10-24)

- GCP `GPUS_ALL_REGIONS` limit is 1 and usage is 1: the user's own T4 VM `socreateai-agora-hermes-node1` (us-east4-b, RUNNING) holds the slot and must not be stopped. A second T4 spot VM cannot be created, and quota requests are declined. TPU cannot measure Q4 GGUF VRAM, so it is not a substitute for P1.
- Options: (a) run P1 on the RTX 2070 node (address needed; no `rtx2070` SSH entry resolves), (b) run it on hermes-node1 itself with the user's say-so (no extra GPU cost), (c) the user stops hermes-node1 for the run.
- `deploy/run_on_gcp.sh --model-ladder a,b` puts a comma inside `--metadata=MODEL=a,b`, which gcloud splits into two keys; run one model per launch until fixed.
- Cost if a T4 spot slot is available: dry-run worst case $0.88 per launch (ledger 41.75 -> at most 42.63 per launch, cap 50).
- Resolved 2026-10-24: P1 ran on `hermes-node1` once its GPU was idle (see section 10); no new GPU was created.

## 10. P1 results (2026-10-24, Tesla T4 on `socreateai-agora-hermes-node1`, llama.cpp b11854 commit 8e5a58d, Q4_K_M, greedy, 3 repair rounds, 512 new tokens)

Raw data: `results/gwaya_lite/t4_{3b,7b}/` (`rows.jsonl`, `summary.json`), driver `scripts/lite_bench.py`. MBPP-sanitized test split, first 257 problems; the model sees the problem and test_list[0]; an arm passes only if the final code passes ALL tests, run in the local bwrap sandbox (generation on the T4 over an ssh tunnel, no model-generated code ran on the VM). GPU spend: $0 new (existing VM); ledger unchanged at $41.75.

| Tier | Resident VRAM 4k / 8k ctx | llama-bench prompt / generation | A0 raw pass@1 (95% CI) | A3 with GWAYA loop | gained / lost, McNemar p | VERIFIED claimed -> pass all tests |
|---|---|---|---|---|---|---|
| 3B Q4 | 2191 / 2339 MiB | 2658 / 87.4 tok/s | 70.8% (65.0-76.0) | 71.6% (65.8-76.8) | +2 / 0, p = 0.5 | 199 -> 184 (92.5%) |
| 7B Q4 | 4663 / 4891 MiB | 1349 / 40.4 tok/s | 82.1% (77.0-86.3) | 82.9% (77.8-87.0) | +2 / 0, p = 0.5 | 222 -> 213 (95.9%) |

Findings (what the data supports):
- Both tiers fit the 6656 MiB HiveMind ceiling with margin; 7B at 8k context uses 4891 MiB. `gwaya/lite.py` now covers the measured values (a test enforces it).
- The repair loop did NOT significantly improve pass@1 on MBPP for either tier (+2 problems, p = 0.5). Do not claim an accuracy gain.
- The gate's VERIFIED label is informative: 92-96% of VERIFIED answers pass all tests. Caveats: the visible test is test_list[0] and the label is checked against the other tests of the same problem, not an independent set; MBPP is small and likely seen in pre-training.
- 3B-first cascade (3B answer kept when VERIFIED, else the 7B answer), computed from the two runs: 58 of 257 escalated, 83.3% pass@1 versus 82.9% for 7B alone (one problem, not significant), mean generation time 2.91 s versus 2.45 s per problem. Small-first does not pay off on this benchmark; the default stays 7B (`small_first=False`). Latencies are indicative only: the two runs were not under identical load (a LeanMaster Ollama process started on the same GPU during the 3B run).
- Not measured: 1.5B tier, FIM autocomplete quality, IDE latency, Windows/WDDM, more than 3 repair rounds, any benchmark other than MBPP-sanitized, edit/refactor tasks.

Sharing with HiveMind: HiveMind already ships GWAYA as a verifier (E15: `scripts/e15_gate.py`, vendored `third_party/gwaya/`, Qwen2.5-Coder-1.5B + LoRA under 8 GB). The expert route is to register `gwaya.hivemind_expert.HIVEMIND_SPEC` and call `GwayaCodeExpert`; the trained route (level B) is the E15 LoRA as the Python branch. No HiveMind source has been changed.
