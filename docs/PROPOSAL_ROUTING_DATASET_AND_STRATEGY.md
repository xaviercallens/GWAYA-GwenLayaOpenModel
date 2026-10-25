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

**Existing component to reuse for stage 2 (found by the "Genetic sub project" session, read and confirmed):** `AutoevolveAI/anse/v2/model_router.py` routes (domain, hardness tier) to the model that has *earned* it. It holds a capability matrix of measured attempts, passes and false-item acceptances (unmeasured cells are `None`), ranks by the Wilson lower bound of the pass rate (so 1/1 never outranks 8/10 and there are no ties to break by dict order), excludes any model that accepted a false item at that tier, allows one unmeasured exploration slot, and escalates to `top-tier` or `human` when the best lower bound is under the budget. It does not map a free-text prompt to a domain or tier, which is what stages 0-1 add. Our measured T4 results fit its `Cell` directly (Python, MBPP-sanitized: 7B Q4 213/257 and 3B Q4 184/257 pass all tests with the GWAYA loop; false-item acceptance not yet measured, so those cells would not yet count as sound until a negative-control set is run). Proposal: rank experts inside a routed domain with that matrix instead of with keyword counts, and feed `record_outcome` with gate verdicts only. Note from the same session: `anse/admin/intent_router.py` routes on `q.lower()` substring matching with an LLM fallback, the defect part A fixed in HiveMind.

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
| Rust | McEval Rust: 53 tasks and FullStackBench: 30 (83 in total, far below the 200 needed for the test split alone) | a public Rust instruction/question corpus with a clear licence; none verified yet. The registry's `anon/rust-code-instructions` returns HTTP 401 (absent or private), so it is NOT a usable source |
| Solidity | none | candidates seen on the Hugging Face API 2026-10-24: `Offlin33er/solidity-audit-instruct` (licence field: MIT), `Harish-as-harry/solidity_contract_sft` and `verify-ppt/marin-starcoderdata_solidity` (no licence in the metadata; the latter is derived from per-file-licensed code, so it is the riskiest). Content not inspected. The registry's `hivemind/solidity-audit-invariants` returns HTTP 401 |
| Biomedical | none | `qiaojin/PubMedQA` (exists, MIT, not gated; the only registry `dataset_id` that resolves) |
| Lean 4 | minif2f-lean4 statements (Lean syntax only) | natural-language requests about proofs ("how do I close this goal"), written or collected |
| none (off-domain) | gsm8k, MATH-500, FullStackBench non-Python/Rust/Solidity items (Go, Java, ... are code but belong to no expert) | generic chat/instruction prompts with a clear licence |

Local data alone cannot supply Solidity, Biomedical or enough Rust. Obtaining the external sources is the first work item; licences must be checked before any redistribution (nothing is committed that cannot be).

### 3.3 Construction
0. **Licence first:** no corpus is downloaded, labelled or committed until its licence has been read and recorded in a manifest (`name, URL, licence text/URL, date checked, rows used`). A missing licence field in the metadata is treated as "not allowed" until the dataset card says otherwise.
1. **Label from provenance** (which dataset a prompt came from), never from a keyword match.
2. **Keyword-blind stratum:** compute, with the merged whole-word router, which prompts have zero registry-keyword hits. The routers are judged on this stratum separately, because it is where a learned or retrieval router can add anything over stage 0. Report both strata.
3. **Hard negatives** (about 15% of `none`): ordinary words that contained keywords under substring matching ("search", "public", "nation", "maybe", "bypass", "important"); **ordinary English words that are themselves registry keywords and still hit under whole-word matching** (found by `autoevolveai-b8`, confirmed in `SEASON_3_DOMAIN_EXPERT_SPECS`): `by`, `nat`, `require`, `contract`, `trial`, `therapy`, `proof`, `lean`, `borrow`, `lifetime`, `unsafe`, `patient`, `arc`, `pub`, `mut` (e.g. "a legal contract", "a fair trial", "I need proof", "lean beef", "borrow a book", "written by"); code in other languages; and cross-domain prompts. The stratum report lists, per keyword, how many `none` prompts it falsely fires on, which also tells the owner which keywords to drop from stage 0. Cross-domain prompts get a **multi-label** (e.g. a Python script that talks to a Solidity contract), scored as correct when the top-1 is any of its labels.
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
5. **Discrimination band, fixed now and applied to every stratum:** if the pooled top-1 over all arms on a stratum falls outside [0.3, 0.7], that stratum cannot rank the arms (all near-perfect or all near-zero) and is reported as non-discriminating; arms are then ranked only on strata inside the band. Difficulty is tuned on dev (more or fewer ambiguous prompts), never on the sealed set, and the band is not changed after seeing results. (Lesson from the "Genetic sub project": its first crossover experiment had both arms at 0/24, an instrument that could not separate them.) Before optimising against any ceiling `route_eval.py` reports, compare it once with the raw router output (`--dump-preds` writes the per-prompt predictions); the part A smoke run was checked this way against the manual probe (all five off-domain probes went to Rust + Solidity before the patch).
6. **Complexity tier (section 7):** on the verifiable sealed subset the tier cascade's pass@1 is not below always-top-tier (paired McNemar) and its mean cost (paired bootstrap 95% CI, cost unit fixed beforehand) is below always-top-tier; reported next to the oracle ceiling and the fixed 3B-first cascade; a saving above the oracle ceiling is treated as a bug.
7. **Computed band check:** the report contains `discrimination(...)` output for every stratum (section 7.5).
Missed bars are reported as missed. The sealed set is read once (`.claude/skills/gwaya-sealed-read`).

## 4. Experiments, in order
| # | Experiment | Needs |
|---|---|---|
| E1 | build and audit the dataset (3.2-3.4) | external sources, owner time for the 150-prompt audit |
| E2 | arms on dev: fixed keyword, TF-IDF+logistic, exemplar retrieval (small CPU encoder), retrieval + keyword shortcut | CPU only |
| E3 | threshold selection on dev, bar registered, sealed read | none |
| E4 | cascade with the verifiable-domain gate on fresh Python tasks | the T4 (`~/gwaya-lite`), the user's go-ahead for GPU time; no GCP spend |

### 4.1 Python first (user decision, 2026-10-24)
Every experiment above is run for the Python expert before any other domain. Python is the only domain with all of: a local verifier (GWAYA sandbox), a measured tier ceiling (section 7.2), a cleared fresh source (BigCodeBench, section 8) and an expert already adopted by HiveMind (3ca9466).
- **Scope of the first pass:** the router decides `Python_Code_GWAYA` vs everything else, and the tier (3B / 7B / 20B) for the prompts it keeps. Rust, Solidity, Biomedical and Lean prompts enter only as confusers labelled `none-for-Python` (a Python router must not claim them). Their own experts and the licence work in section 8 for them are deferred until the Python bars hold.
- **Keyword-blind Python stratum is mandatory:** Python prompts that do not contain `python`, `def`, `import`, `.py` or any other KEYWORDS entry (IDE-style requests such as "merge these intervals", a bare stack trace, a pasted function). Without it the keyword router scores near-perfect for artifact reasons (MBPP says "python").
- **First E1 measurement (BigCodeBench v0.1.4, Apache-2.0, n = 1140, `scripts/routing/build_python_stratum.py`, counts in `results/routing/python_stratum_counts.json`):** with the full instruction (it embeds the `def`/`import` scaffold) the keyword scorer hits on 1140/1140 (Wilson 0.9966-1.0), a scaffold artifact. With only the natural-language description it hits on 399/1140 (35.0%, Wilson 0.3229-0.3782); 741 tasks (65.0%) have no keyword hit, so the current keyword router abstains on them. These 741 ids are the Python keyword-blind stratum; the 35% hit rate is mostly library names (numpy, pandas). It measures what the keyword arm misses on Python descriptions, not how a better arm performs. Off-domain negatives, GSM8K test (MIT, n = 1319, plain-language math word problems): the keyword scorer fires on 39 (3.0%, Wilson 0.0217-0.0402), 38 on `class ` (a school class) and 1 on `typing`, so the keyword arm's false-route rate on that set is 3.0%. Together: on these two sets the keyword arm misses 65% of keyword-blind Python and falsely routes 3% of math word problems. Other-domain confusers (Rust, Solidity, Lean, biomedical) are still to be collected.
- **First E2 run, dev only (`scripts/routing/e2_python_dev.py`, `results/routing/e2_python_dev.json`; hash-split halves, train 1222, dev 1237):** keyword arm Python recall 193/570 (33.9%), keyword-blind recall 0/377, negatives correct 647/667 (97.0%); TF-IDF + logistic regression 570/570, 377/377 and 667/667 (paired: 397 vs 0 discordant). **This 100% is a dataset artifact, not a routing result:** BigCodeBench descriptions and GSM8K word problems differ in style and vocabulary, so any text classifier separates them. It shows that the keyword arm cannot see blind Python and that the harness runs; it does NOT show the learned arm is good. The pooled-band check (3.5) will be run only on a set with hard confusers (non-Python code, other-domain prompts), which still has to be collected. Not the sealed read.
- **E2 with hard confusers, dev only (`scripts/routing/e2_confusers.py`, `results/routing/e2_confusers.json`):** adds Python as code (BigCodeBench `complete_prompt`, 558 dev), Rust (Strandset-Rust-v1 test, 115) and Solidity (Offlin33er eval, 126) to the negatives. Keyword arm: blind-description recall 193/570, code-form recall 558/558 (scaffold artifact: `def`/`import`), rejects GSM8K 647/667, Rust 113/115, Solidity 124/126. TF-IDF+LR trained on Python vs GSM8K only: 570/570 and 558/558 recall, but it rejects only 32/115 Rust and 36/126 Solidity (28% and 29%): it learned "code vs prose", which confirms the style-artifact reading of the previous run. Trained with the code confusers it scores 100% on every part, but Python-vs-Rust/Solidity is lexically easy and the code snippets are raw while the Python descriptions are prose, so this still does not rank the arms. What it does support: the keyword arm is a good rejector of other-language code and a poor finder of keyword-blind Python; a learned arm must be trained with code confusers or it over-claims. Prompts that mix a language with an IDE-style request (the real target) remain uncollected.
- **Keyword change measured on the same data (BigCodeBench descriptions n = 1140, GSM8K n = 1319):** dropping `class ` takes GSM8K false routes from 39 to 1 and costs 8 of the 399 blind-description hits (399 to 391); code-form Python still hits 1140/1140. Patch for HiveMind: `docs/patches/hivemind_drop_class_keyword.patch` (applies cleanly to eb0cdfb; the HiveMind parity test requires `gwaya.hivemind_expert.KEYWORDS` to match, so the GWAYA side changes in the same step).
- **Held-out-source check, dev only (`scripts/routing/e2_heldout_source.py`, `results/routing/e2_heldout_source.json`):** TF-IDF+LR trained on all of BigCodeBench (description and code form), GSM8K, Rust and Solidity (4072 rows), tested on HumanEval+ (Apache-2.0, 164 tasks, a source it never saw). On the docstring text alone the keyword arm recalls 2/164 (1.2%, Wilson 0.0034-0.0434) and the LR arm 116/164 (70.7%, Wilson 0.6336-0.7716); on the full code stub keyword 164/164 (the `def` artifact) and LR 133/164 (81.1%, Wilson 0.7442-0.8635). So the 100% in the hash-split runs was a within-source artifact: on a new Python source the learned arm loses about 19-29% of recall, while still finding 58 times as many docstring-only Python tasks (116 vs 2) than the keyword arm. Still no IDE-style mixed prompts and one test source, so the intervals describe HumanEval+ only. Not the sealed read.
- **Exemplar-retrieval arm, dev only (`scripts/routing/e2_exemplar.py`, `results/routing/e2_exemplar.json`):** same 4072 training exemplars and the same HumanEval+ test as the previous bullet, encoder BAAI/bge-small-en-v1.5 (MIT, CPU), top-k cosine vote. Docstring text, recall of 164: top-1 140 (85.4%, Wilson 0.7915-0.8996), top-5 151 (92.1%, 0.8691-0.9531), top-15 157 (95.7%, 0.9145-0.9792); LR on the same embeddings 127 (77.4%, 0.7046-0.8317); TF-IDF LR 116; keyword 2. Full code stub: top-1 136, top-5 144, top-15 150 (91.5%, 0.8618-0.9485), embedding LR 92, TF-IDF LR 133, keyword 164 (`def` artifact). False routes on the negatives are leave-one-out within the training sources only, so optimistic (near-duplicate neighbours inside a source): top-15 7 of 1792 (0.4%), top-1 2, top-5 6. k was NOT tuned (all three reported); no k is selected here. Not paired-tested and not the sealed read. Reading: retrieval over exemplars beats both the keyword and the TF-IDF arms on a new Python source, the exemplar list is the thing HiveMind would need to carry (an `exemplars` key per expert), and the off-domain side is not yet tested on a held-out source.
- **Fresh tasks, not MBPP, for the utility check:** BigCodeBench (cleared) for E4; MBPP stays the tier-ceiling reference only.
- **Bars are the registered ones in 3.5 and 7.3, applied to the Python-vs-rest pair first.** No new bar is added; no domain beyond Python is read from the sealed set until the Python result is reported as-is.
- **GPU:** E4 on the T4 still needs the user's go-ahead; E1-E3 are CPU only.

## 5. Division of work
- HiveMind session: step A PR, licence check of each candidate source (the registry's own `anon/rust-code-instructions`, `hivemind/solidity-audit-invariants` and `hivemind/lean4-mathlib-proofs` return HTTP 401, so only `qiaojin/PubMedQA` is a confirmed source), exemplar format per expert (`SEASON_3_DOMAIN_EXPERT_SPECS` entries gain an `exemplars` list).
- GWAYA session: dataset build/audit scripts, eval harness (`scripts/routing/route_eval.py`), gate oracles, sealed-read protocol, E4 on the T4.

## 6. Risks
- Exemplar retrieval depends on the encoder; a general encoder may separate "Rust" from "Solidity" prompts worse than on topic-distinct sets. Mitigation: that is exactly what the keyword-blind stratum measures.
- Provenance labels leak style (dataset-specific phrasing). Mitigation: paraphrase a sample and compare accuracy on original vs paraphrased text; a large drop flags style leakage.
- Verify-then-serve doubles generation cost on escalation; report the mean generations per request.
- An abstaining router can look good on `none` by abstaining on everything; bar 1 and the in-domain abstain rate (`abstain_in`) guard against it.

## 7. Complexity tier as a routed output (HiveMind owner's addition, 2026-10-24)

The router's output becomes a pair **(expert, tier)**, where the tiers are coder-lite 3B (tier 0), coder 7B (tier 1) and moe-general gpt-oss 20B (tier 2). It picks the cheapest tier likely to be good enough and escalates one tier at a time only when a check fails. Escalation signals: retrieval confidence below its threshold (stage 1) and a failed gate (stage 2, verify-then-serve). Only 3B and 7B were measured here; tier 2 is unmeasured, and gpt-oss 20B is not expected to fit the 6656 MiB ceiling (to be confirmed by the owner), so where it runs and what one attempt costs there must be fixed before any cost number is quoted.

### 7.1 The tier label
- **Judged from solve rates, never from prompt length.** For each prompt and tier, the tier's answer is scored by an oracle (hidden tests for Python; compile + tests for Rust; the kernel for Lean 4; gold label for PubMedQA). Use k = 4 samples per tier (temperature 0.2, fixed seeds) so a single lucky or unlucky greedy sample does not set the label. `p_t` = share of samples that pass.
- **Label = the lowest tier with `p_t >= 0.5`.** A prompt no tier solves gets `tier = unsolved`: routed to the top tier, reported separately, never counted as a tier-2 success. The 0.5 cut is fixed now and is not tuned on the sealed set.
- **Domains without an executable oracle** (Solidity audit text) get no tier label; they are excluded from the tier bar and served at the owner's default tier.

### 7.2 What the data already says (MBPP-sanitized test split, n = 257, greedy, GWAYA loop, T4)
| Outcome | Items |
|---|---|
| solved by both 3B and 7B (tier 0 suffices) | 178 (69%) |
| solved only by 7B (tier 1) | 35 (14%) |
| solved only by 3B (noise or tier-0 luck: single greedy samples) | 6 (2%) |
| solved by neither (candidate for tier 2, or flawed items) | 38 (15%) |

- The 6 "only 3B" items show why a single greedy sample is too noisy for the label (hence k = 4).
- **Ceiling for any 3B/7B tier router on these rows:** a perfect oracle that sends each item to the cheapest passing tier cuts mean generated tokens from 61.6 (always 7B) to 57.5, **6.6%**, and mean generation time from 2.45 s to 1.88 s, 23% (indicative only: the 3B and 7B runs were not under the same GPU load; the 3B emits more tokens per answer, 80 against 62). The measured 3B-first cascade (83.3% against 82.9%) was slower, 2.91 s against 2.45 s. So on easy function-level tasks tier routing can save little; the registered claim must not exceed this ceiling, and larger savings can only come from harder prompts where the 7B and 20B tiers differ widely.
- Whether the 38 unsolved items are solved by the 20B tier, or are MBPP items with flawed tests, is unknown.

### 7.3 Metrics and the registered bar for tiers
`scripts/routing/route_eval.py` now has `evaluate_tiers` (exact, **under** = routed too cheap and paying the failed attempts on the way up, **over** = routed too high, mean cost, oracle cost, always-top cost, always-lowest-with-escalation cost). Cost unit per attempt (tokens, generations or seconds) is chosen by the owner and fixed before the sealed read. Bar 6 (added to section 3.5): on the verifiable sealed subset, the tier cascade's pass@1 is not below always-top-tier (paired McNemar), its mean cost (paired bootstrap 95% CI) is below always-top-tier, and it is reported next to the oracle ceiling and the fixed 3B-first cascade. A saving above the oracle ceiling is a bug, not a result.

### 7.4 Non-destructiveness of the verify-and-repair loop (a claim the data supports)
Recounted independently by the "Genetic sub project" session from `rows.jsonl`: over the two models the loop changed 4 verdicts to passing and **broke none**. A regression could only occur on items that already passed raw (211 for the 7B and 182 for the 3B, 393 in all), so the rate at which the loop breaks a passing answer has a one-sided 95% upper bound of about 0.76% (0 in 393; `1 - 0.05^(1/393) = 0.0076`). This is a non-destructiveness claim, not an accuracy claim: the pass@1 difference stays not significant at this n. Caveat: the policy returns the raw sample unless the loop's answer is VERIFIED against the visible test, which makes a regression less likely by construction.

### 7.5 The band check is computed, not read by eye
`route_eval.discrimination` returns, per stratum, the pooled in-domain top-1 over all arms and an `in_band` boolean against the band fixed at [0.3, 0.7]; the sealed-run report must contain these fields, so the check is asserted by code (suggested by the "Genetic sub project" session; tested in `tests/test_lite_hivemind.py`).

### 7.6 Tier 2 (gpt-oss 20B) as measured by `autoevolveai-b8` on the T4, 2026-10-24 (reported by the peer, not re-run by me)
- **Memory:** 12,331 MiB resident (Ollama 0.34.1, flash attention, 8-bit KV, one slot, num_ctx 4096, think=low). It does NOT fit the 6,656 MiB ceiling; it needs most of the 15,360 MiB card. Tier 2 is therefore a T4-class option, not an 8 GB-card option.
- **Co-residency:** beside a 3B (num_ctx 2048) it loads at 13,873 MiB total but Ollama put the 3B 36% on CPU; beside the 7B (5.6 GB) it does not fit. A 7B + 20B cascade on one T4 means swapping models.
- **Speed:** 19-45 tok/s on single short prompts (indicative only); cold load 285 s, warm 0.2 s. The cold load makes "escalate only on a failed check" expensive unless the 20B stays resident.
- **Quality:** 6 harder hand-picked Python tasks with hidden tests: 7B 3/6, 20B 6/6 (first try). n = 6 gives direction only and is not a tier-label source; labels still come from solve rates on the real set (7.1).
- **Cost unit:** GPU-seconds on equal load, card otherwise empty. The 7B timings in that run shared the GPU and must not be used for a ratio. Running it on MBPP-sanitized needs the T4 to be given to it; E17 training needs the card next, so this waits for the owner's decision.
- **Cross-check:** base 7B via llama-server on the same 257 MBPP-sanitized tasks: 210 first-attempt passes (81.7%), 215 with repair; 0 of 30 repeated requests differed (greedy is deterministic).

## 8. Licence manifest (read from dataset cards by `autoevolveai-b8` on 2026-10-24; metadata and card text only, no data downloaded; not independently re-read by me except the HF metadata fields I fetched earlier)

Rule (section 3.3, step 0): allowed use is what the card states and nothing more; no source is cleared until its card body has been read in full and recorded here.

| Source | Class | Card licence | Status and caveat |
|---|---|---|---|
| qiaojin/PubMedQA | Biomedical | MIT | Cleared for evaluation and routing labels only; questions derive from PubMed abstracts whose copyright is not covered by the field: no redistribution of abstract text |
| Offlin33er/solidity-audit-instruct | Solidity | MIT | Cleared; derived from msc-smart-contract-auditing/audits-with-reasons (also MIT, checked). Versions v2, v3 by the same author unchecked |
| Harish-as-harry/solidity_contract_sft | Solidity | none | EXCLUDED |
| verify-ppt/marin-starcoderdata_solidity | Solidity | none | EXCLUDED (297k rows of per-file-licensed StarCoderData) |
| Fortytwo-Network/Strandset-Rust-v1 | Rust | apache-2.0 | CLEARED for routing prompts (card body read 2026-10-24: Apache-2.0, unrestricted research and commercial use with attribution; small open models in a peer-ranked swarm, code validated by rustc; Claude Sonnet 4 and GPT-4o are named only as scorers for non-code tasks). Caveat to record: derived from 2,300 crates.io crates whose licences the card does not state; keep attribution |
| Tesslate/Rust_Dataset | Rust | apache-2.0 | NOT cleared: the whole card is the licence line, no description or provenance |
| WrittenWithRust/Magicoder-OSS-Instruct-Rust-cleaned-3.9K | Rust | MIT | Allowed only with the caveat that it is LLM-derived from GitHub projects (Magicoder); caveat recorded |
| gaianet/learn-rust | Rust | apache-2.0 | UNSUITABLE as routing prompts (a vector knowledge base from the Rust Book and Rust by Example, not instruction prompts); not a licence problem |
| AI-MO/NuminaMath-LEAN | Lean 4 | apache-2.0 | Competition problems with Lean formalisations; original problem sources keep their licences via NuminaMath 1.5: record per source before use |
| internlm/Lean-Workbook | Lean 4 | apache-2.0 | NOT cleared: per the arXiv 2406.03847 abstract (the paper body was not read) the problems are contest questions translated to Lean 4 by a synthetic pipeline, with proofs searched from a math contest forum plus 21 new IMO questions; the forum is not named and its licence is unrecorded, so forum-derived text stays not cleared. Also math statements, so only a "formalise this problem" stratum, not Lean help requests |
| cat-searcher/minif2f-lean4 | Lean 4 | MIT (tags) | Statements only; usable as Lean-syntax exemplars |
| natural-language Lean help requests | Lean 4 | n/a | no dataset with a clear licence found: write them (owner/author, original text) |
| `anon/rust-code-instructions`, `hivemind/solidity-audit-invariants`, `hivemind/lean4-mathlib-proofs` | registry ids | n/a | do not resolve (HTTP 401); not sources |
| bigcode/bigcodebench (local) | Python | apache-2.0 | cleared on the card field (read from the cached card 2026-10-24) |
| newfacade/LeetCodeDataset (local) | Python | apache-2.0 | conditional: the card licence does not address the copyright of the LeetCode problem texts; use for routing labels only, no redistribution of prompts |
| openai/gsm8k (local) | none | MIT | cleared (front matter and the card's Licensing Information section both state MIT) |
| livecodebench/code_generation_lite (local) | Python | "cc" | NOT cleared: the variant of the Creative Commons licence is not stated |
| HuggingFaceH4/MATH-500 (local) | none | none in card | NOT cleared (no licence in the card) |
| ByteDance/FullStackBench (local; mirror luzimu/FullStack-Bench) | Python / Rust / other code | conflicting | NOT cleared as an unconditional source (card read by `autoevolveai-b8`): front matter says apache-2.0, but the card body badges give Code License Apache 2.0 and Data License CC-BY-SA-4.0. Treat the data as CC-BY-SA-4.0 (attribution + share-alike) until the repo says otherwise; usable for labels-only evaluation with the conflict recorded; no redistribution of prompts |

Remaining licence work (before any labelling): read the Lean-Workbook paper body for the forum name and licence (only the abstract has been read), find out what Tesslate/Rust_Dataset is, pin the Creative Commons variant for LiveCodeBench, find a licence statement for MATH-500, and write the manifest as a file next to the dataset. Rust now has one cleared source (Strandset-Rust-v1), so the 200-per-class Rust test split is no longer blocked by licensing; Solidity has one (Offlin33er); Biomedical has one (PubMedQA, labels only); Lean 4 has none cleared for natural-language help requests.
