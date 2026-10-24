# Proposal: a routing dataset for HiveMind and a different expert-selection strategy

Status: proposal, 2026-10-24, follows `docs/PROPOSAL_OPENLAYA_HIVEMIND_ROUTING.md` (PR #22, merged; part A patch accepted in principle by the user, including the "no keyword hit returns `([], {})`" contract). Nothing here changes HiveMind. No GCP spend.

## 1. Why a different strategy, not only a better classifier

Part B of the earlier proposal is a topic classifier (TF-IDF + logistic regression) that picks a domain. Three facts argue for changing the question it answers:

1. **Topic is a proxy.** What matters for a request is whether the expert beats the base model on it. A prompt can be "about Python" and still be served as well by the base model, or be off-topic and benefit from an expert. Our own data: the repair loop adds nothing on MBPP for a 7B (82.1% -> 82.9%, p = 0.5), so routing Python prompts to a GWAYA-gated expert is worth it for the *verdict*, not for accuracy on easy tasks.
2. **Experts are added by the community.** HiveMind hot-plugs experts (`Season3DomainExpertRegistry.register_expert`). A trained classifier has a fixed label set and must be retrained, and its calibration redone, for every new expert. An open-world mechanism is needed.
3. **Some domains are verifiable and some are not.** Python (sandboxed tests), Lean 4 (kernel), Rust (compile) have an executable check; biomedical QA and Solidity auditing do not. For verifiable domains the check can act as the router; for the others only a learned or retrieval-based signal exists.

## 2. Proposed strategy: a three-stage cascade (cheapest first, abstain by default)

| Stage | Mechanism | Covers | Cost |
|---|---|---|---|
| 0 | whole-word keyword shortcut (the merged part A patch) | prompts that name the domain explicitly | negligible |
| 1 | **exemplar retrieval** (nonparametric): embed the prompt, find nearest labelled exemplars per expert, score = calibrated similarity, abstain when below a threshold chosen on dev | all domains, open world: a new expert ships with 20-50 exemplar prompts and is routable at once, no retraining | one small-encoder forward pass on CPU |
| 2 | **verify-then-serve** for verifiable domains: send the request to the candidate expert(s), accept only a gate-passing answer (GWAYA stub audit + sandboxed tests; Lean kernel; Rust compile), else escalate to base or the next candidate | Python, Lean 4, Rust | the expert's generation; escalation costs a second one |
| fallback | base model | everything abstained | none |

Why this ordering: stage 0 is free and high precision; stage 1 handles paraphrases and unseen experts without retraining; stage 2 replaces prediction by execution where an oracle exists (GWAYA measured 92-96% of VERIFIED answers passing all tests), so router mistakes in those domains are caught by the gate instead of shipped.

Not proposed: using the frozen Laya as is (trained for Qwen3.5 size tiers; text added nothing over hand features in our data), and a ModernBERT-sized router unless stage 1 fails the bar below.

### Alternatives considered, with the reason each is not first
- **Perplexity / likelihood routing** (each expert scores the prompt): needs a forward pass per expert, which is too heavy on an 8 GB card; keep only as a rerank over stage-1 top-2. Hypothesis, no evidence here.
- **Learned softmax classifier over a fixed label set** (the earlier part B): good baseline, kept as the comparison arm, but closed-world.
- **Token-level learned gates** (HiveMind's existing hierarchical routers): different level, unaffected.

## 3. The dataset

### 3.1 Requirements that the earlier probes exposed
- Prompts must **not name the language or domain by artifact**: MBPP says "python", MultiPL-E Rust is the same tasks translated (even "rsthon"), so any router looks perfect on them.
- Labels must come from **provenance, not from the keywords the router uses**.
- Must contain **hard negatives** (look-alike words, adjacent domains) and a real **off-domain** class.
- Must cover the registry: Python, Rust, Solidity, Biomedical, Lean 4, `none`.

### 3.2 Sources (status checked 2026-10-24)

| Class | Local (this machine) | Gap / external source to obtain |
|---|---|---|
| Python | BigCodeBench (API-style prompts), LiveCodeBench, LeetCodeDataset, FullStackBench Python items (844 of 1,687) | rewrite or filter to remove the word "python"; MBPP/HumanEval+ only as contamination-checked extras |
| Rust | McEval Rust: 53 tasks and FullStackBench: 30 (83 in total, far below the 200 needed for the test split alone) | `anon/rust-code-instructions` (the registry's own `dataset_id`), Rust forum/issue-style questions |
| Solidity | none | `hivemind/solidity-audit-invariants` (registry `dataset_id`), public audit reports |
| Biomedical | none | `qiaojin/PubMedQA` (registry `dataset_id`) |
| Lean 4 | minif2f-lean4 statements (Lean syntax only) | natural-language requests about proofs ("how do I close this goal"), written or collected |
| none (off-domain) | gsm8k, MATH-500, FullStackBench non-Python/Rust/Solidity items (Go, Java, ... are code but belong to no expert) | generic chat/instruction prompts with a clear licence |

Local data alone cannot supply Solidity, Biomedical or enough Rust. Obtaining the external sources is the first work item; licences must be checked before any redistribution (nothing is committed that cannot be).

### 3.3 Construction
1. **Label from provenance** (which dataset a prompt came from), never from a keyword match.
2. **Keyword-blind stratum:** compute, with the merged whole-word router, which prompts have zero registry-keyword hits. The routers are judged on this stratum separately, because it is where a learned or retrieval router can add anything over stage 0. Report both strata.
3. **Hard negatives** (about 15% of `none`): ordinary words that contained keywords under substring matching ("search", "public", "nation", "maybe", "bypass", "important"), code in other languages, and cross-domain prompts. Cross-domain prompts get a **multi-label** (e.g. a Python script that talks to a Solidity contract), scored as correct when the top-1 is any of its labels.
4. **Dedupe** within and across splits by normalised 8-gram hashing (MBPP and MBPP+ overlap); split by source task, not by prompt, so paraphrases of one task do not straddle splits.
5. **Human audit:** the owner (or a second reviewer) re-labels a random 150-prompt sample blind; report agreement (Cohen's kappa) and fix systematic label errors before the splits are sealed.

### 3.4 Size (arithmetic, Wilson 95%)
- 200 test prompts per class gives a half-width of about 0.05 at p = 0.85: observed 170/200 -> [0.794, 0.893]; 190/200 -> [0.910, 0.973]. Smaller (100 per class) is too loose for the abstain bar: 95/100 -> [0.888, 0.978] but 60/100 -> [0.502, 0.691].
- Plan: 6 classes x 200 = **1,200 sealed test**, 1,200 dev, remainder train/exemplar pool; at least 250 test prompts in the keyword-blind stratum.
- Both routers compared on the same rows with an exact McNemar test (`scripts/routing/route_eval.py`).

### 3.5 Registered bar (written before any test read; all must hold on the sealed set)
1. Stage 0 + 1 top-1 on in-domain prompts is higher than the merged whole-word keyword router, paired exact McNemar p < 0.05, and the Wilson lower bound on the keyword-blind stratum is above the keyword router's observed value.
2. Abstain rate on `none` >= 0.90 (Wilson lower bound >= 0.85), at the threshold fixed on dev.
3. Open-world check: add one held-out expert (its 30 exemplars only, no retraining of anything) and show its routing top-1 is not below the closed-world classifier's on that class.
4. Verifiable domains: the full cascade's pass@1 on routed Python tasks is not below the base model's (McNemar), and the number of experts loaded per request is reported.
Missed bars are reported as missed. The sealed set is read once (`.claude/skills/gwaya-sealed-read`).

## 4. Experiments, in order
| # | Experiment | Needs |
|---|---|---|
| E1 | build and audit the dataset (3.2-3.4) | external sources, owner time for the 150-prompt audit |
| E2 | arms on dev: fixed keyword, TF-IDF+logistic, exemplar retrieval (small CPU encoder), retrieval + keyword shortcut | CPU only |
| E3 | threshold selection on dev, bar registered, sealed read | none |
| E4 | cascade with the verifiable-domain gate on fresh Python tasks | the T4 (`~/gwaya-lite`), the user's go-ahead for GPU time; no GCP spend |

## 5. Division of work
- HiveMind session: step A PR, dataset sources it owns (`anon/rust-code-instructions`, `hivemind/solidity-audit-invariants`, `qiaojin/PubMedQA` access), exemplar format per expert (`SEASON_3_DOMAIN_EXPERT_SPECS` entries gain an `exemplars` list).
- GWAYA session: dataset build/audit scripts, eval harness (`scripts/routing/route_eval.py`), gate oracles, sealed-read protocol, E4 on the T4.

## 6. Risks
- Exemplar retrieval depends on the encoder; a general encoder may separate "Rust" from "Solidity" prompts worse than on topic-distinct sets. Mitigation: that is exactly what the keyword-blind stratum measures.
- Provenance labels leak style (dataset-specific phrasing). Mitigation: paraphrase a sample and compare accuracy on original vs paraphrased text; a large drop flags style leakage.
- Verify-then-serve doubles generation cost on escalation; report the mean generations per request.
- An abstaining router can look good on `none` by abstaining on everything; bar 1 and the in-domain abstain rate (`abstain_in`) guard against it.
