# Proposal: an open, calibrated prompt-level router ("OpenLaya-R") for HiveMind expert selection

Status: study + proposal, 2026-10-24. Nothing here changes the HiveMind repo. "OpenLaya" is read as: take the Laya idea (a small router that outputs calibrated probabilities and may abstain, with a start-cheap-then-escalate policy) and build an open version trained on HiveMind's own domains. The frozen Laya itself is NOT proposed as a drop-in (see section 3).

## 1. What HiveMind routes today (read from `sub_projects/hivemind/HiveMind/src/hivemind/core/`)

| Level | Code | What it does |
|---|---|---|
| Token level, inside the MoE | `HierarchicalMoERouter`, `ThreeTierHierarchicalMoERouter`, `fomoe_sigma_routing.py` | learned gates, Top-2 over 8 macro-domains x experts; unaffected by this proposal |
| Prompt level, serving / hot-plug | `Season3DomainExpertRegistry.route_domain_query` (`btm_moe.py:286`) | score = 1.0 + 2.5 x (number of keywords that are *substrings* of the lowercased prompt); Top-k with normalised weights |

OpenLaya-R targets only the second level: deciding which domain LoRA experts to hot-plug and call for a request. That is where an 8 GB card makes routing matter, because only a few experts fit resident at once.

## 2. Verified defects of the current prompt router (probe run 2026-10-24, HiveMind venv, unmodified code)

| Prompt | Routed to (weights) |
|---|---|
| "Write a python function that merges overlapping intervals" | Rust_Systems_Security 0.5, Solidity_Smart_Contracts 0.5 |
| "Explain how the heart pumps blood" | Rust 0.5, Solidity 0.5 |
| "What is the capital of France?" | Rust 0.5, Solidity 0.5 |
| "Prove that the sum of two even numbers is even" | Rust 0.5, Solidity 0.5 |
| "fix this def with a traceback in pandas" | Rust 0.5, Solidity 0.5 |
| "Is the search nation public maybe a bypass?" | Rust 0.5, **Lean4 0.5** (substring hits "arc", "nat", "pub", "by") |
| "Explain the patient clinical trial design" | Biomedical 0.895, Rust 0.105 |

1. **No-hit ties go to the first dict entries.** With zero keyword hits every domain scores 1.0, the stable sort keeps dict order, so Rust + Solidity win with 0.5/0.5 for any off-domain prompt (5 of 7 probes). There is no abstain and no "send to the base model" outcome. Even a correct hit carries a second, arbitrary expert (Rust 0.105 in the last row).
2. **Substring matching** makes short keywords (`fn`, `pub`, `arc`, `mut`, `nat`, `by`, `atp`) fire inside ordinary words.
3. **No Python expert** exists in the four registered specs, so Python prompts cannot be routed correctly at all (this is what `gwaya.hivemind_expert.HIVEMIND_SPEC` adds).
4. **No calibration or utility signal:** the weights are normalised keyword counts, not a probability that the expert helps, so they cannot drive a cost decision (how many experts to load).

These are properties of the code as read, reproducible with the probe in section 6; they are not claims about trained-model quality.

## 3. What the Laya evidence does and does not transfer (this repo, paper revision 3.13.0)

- Frozen Laya (ModernBERT-large + LoRA) predicts P(correct) for three **Qwen3.5 size tiers** on python/math prompts. It knows nothing about HiveMind domains, so it cannot classify them. `docs/LAYA_RUNTIME.md`: tier mismatch makes it abstain.
- Its measured value is thin: router at tau = 0.5 vs always-9B 0.476 vs 0.472 accuracy at 0.89x cost (exploratory); gate-driven cascade vs always-9B: no saving (cost ratio 1.74); Laya vs logistic regression on the same hand features: AUROC -0.004, text adds nothing measurable; Laya vs the gate alone (H3) not supported; phase 3 sealed read: goal not met.
- What IS transferable is the discipline and the shape: calibrated output + abstention threshold chosen on a dev set, start-at-cheapest policy, sealed test read once, a cheap feature baseline that the heavy model must beat.

Consequence: start with the cheap baseline, and only escalate to a text transformer if the baseline fails a pre-registered bar.

## 4. Proposal

**Step A (no training, small PR in HiveMind): make the keyword router honest. PREPARED, see section 8.**
- whole-word matching for keywords (a keyword that starts or ends with punctuation, like `.py`, gets no boundary on that side);
- abstain when no keyword hits: return `([], {})` so the caller uses the base model; ties broken by domain name instead of dict order;
- optional `min_weight` (default 0.0, so the existing two-expert contract is unchanged) to drop weak secondary experts;
- one shared helper, `keyword_route`, used by both `route_domain_query` and `route_with_serving_experts` (HiveMind already adopted the Python expert in commit 3ca9466, and that function copies the same substring scoring);
- regression tests from the probe prompts.
Expected effect: off-domain prompts stop being sent to Rust + Solidity. This is a correctness fix, not an accuracy claim. **One contract change:** a no-hit prompt now returns `([], {})`, so the property test `test_routing_weights_are_a_distribution_over_known_domains` (asserts `1 <= len(domains)`) is relaxed to allow the empty result; callers must treat an empty list as "use the base model" (in the repo only tests call these functions).

**Step B: OpenLaya-R, a learned calibrated router (needs data, CPU is enough).**
- Features: hashed word + char n-grams (TF-IDF) -> multinomial logistic regression, one probability per domain expert plus `none`; isotonic calibration on a dev split; abstain when max p < tau.
- Labels: prompts from the registry's own `dataset_id`s (domain classification), plus held-out base-model prompts as the `none` class.
- Policy `start_at`: Top-1 if p1 >= tau1; Top-2 only if p2 >= tau2; else base model. Fewer experts hot-plugged means less VRAM and compute; this is the saving to measure, not assume.
- Utility check (verifiable domains only: Python via GWAYA sandbox, Lean via the Lean oracle, Rust via compile): does routing to the predicted expert beat the base model on pass@1? This is the real test; domain-classification accuracy alone is not sufficient.
- Escalation rule: only if Step B's baseline misses the bar below, try a small embedding model (e.g. a 100M-parameter encoder) on CPU; ModernBERT-large only after that.

**Pre-registered bar for Step B (set before any test read):** on a sealed held-out routing set, (i) macro top-1 accuracy of OpenLaya-R above the fixed keyword router with a Wilson 95% interval excluding it, (ii) abstain rate on off-domain prompts >= 90% at a tau chosen on dev, (iii) on Python tasks, routed-expert pass@1 not below base-model pass@1 (McNemar). Report as-is if missed.

**Step C (optional, later):** use the router's top-k to drive `predict_candidate_experts` / hot-plug planning in `ThreeTierHierarchicalMoERouter`, so the 8 GB resident set follows the predicted domains.

## 5. Division of work

| Owner | Work |
|---|---|
| HiveMind session (`autoevolveai-b8`) | Step A PR in the HiveMind repo; labelled routing set from its domain datasets; Step B training script next to `btm_moe.py` |
| GWAYA session (this one) | Python/Lean/Rust utility oracles for the verifiable-domain check; the sealed-read protocol and certificate arithmetic (`.claude/skills/gwaya-sealed-read`); `gwaya.hivemind_expert` |

Cost: Steps A and B run on CPU; no GCP spend is needed or requested.

## 6. Reproduce the probe

```
cd AutoevolveAI
PYTHONPATH=sub_projects/hivemind/HiveMind/src .venv/bin/python - <<'E'
from hivemind.core.btm_moe import Season3DomainExpertRegistry as R
for t in ["Write a python function that merges overlapping intervals", "What is the capital of France?",
          "Is the search nation public maybe a bypass?", "Explain the patient clinical trial design"]:
    print(R().route_domain_query(t), "|", t)
E
```

## 7. Risks

- A router that abstains too often sends everything to the base model and removes the MoE benefit; tau must be chosen on dev against the utility check, not on classification accuracy.
- Domain labels from dataset provenance are noisy (a "Rust" dataset prompt can be generic); expect a ceiling on top-1 accuracy and measure it.
- MBPP-style tasks are likely seen in pre-training; the Python utility check should use a fresh set (the repo's sealed-set protocol applies).

## 8. What is prepared on the GWAYA side (2026-10-24)

Everything below was built against a read-only `git archive` of HiveMind at `eb0cdfb`; the real HiveMind checkout was not modified.

- **Patch for step A:** `docs/patches/hivemind_route_part_a.patch` (205 lines; `src/hivemind/core/btm_moe.py`, `src/hivemind/core/serving_experts.py`, `tests/test_serving_experts.py`, new `tests/test_route_whole_word.py`). `git apply --check -p1` against the real checkout: applies cleanly. Apply with `git apply -p1 docs/patches/hivemind_route_part_a.patch` from the HiveMind root.
- **Test results on the patched scratch copy** (HiveMind venv): the new `test_route_whole_word.py` and `test_serving_experts.py` pass (16 passed, 2 skipped); with `test_cov_core_and_swarm.py` and `test_seti_swarm_v2.py` the unpatched and patched copies show the same 3 failures in `test_seti_swarm_v2.py` (they need repository files my scratch archive omitted), so no regression from the patch; this subset is not the full HiveMind suite, which the HiveMind owner should run.
- **Routing harness:** `scripts/routing/route_eval.py` computes the part B bar (top-1 on in-domain with Wilson 95%, abstain rate on off-domain, false-route rate, lost coverage, paired exact McNemar between two routers). Usage: `PYTHONPATH=<hivemind>/src python scripts/routing/route_eval.py --router hivemind.core.serving_experts:route_with_serving_experts --rows <jsonl>`.
- **Smoke run on `routing_fixture_synthetic.jsonl`** (27 prompts, hand-written by the author of the fix; NOT evidence): original router abstain on off-domain 0/10, fixed router 10/10 (+10/0, exact McNemar p = 0.002); in-domain top-1 17/17 for both. The in-domain prompts were written with the keywords in mind, so that 17/17 is a ceiling effect.
- **`gwaya.hivemind_expert.score`** now matches whole words as well, so the two scorers agree (HiveMind's `test_spec_keywords_match_gwaya_when_installed` still holds for its prompt).

### Why no accuracy baseline for step B yet
The local datasets cannot give an honest one: MBPP prompts literally say "python" and the MultiPL-E Rust set is the same tasks translated (even "rsthon"), so a keyword router would score near-perfect for artifact reasons; there is no local Solidity or biomedical corpus. Part B therefore needs (1) a labelled routing set whose prompts do not name the language (paraphrased or collected IDE-style prompts), including Solidity and biomedical sources, (2) a sealed split read once, with the bar registered first (section 4).
