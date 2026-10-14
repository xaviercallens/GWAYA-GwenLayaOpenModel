# GWAYA: Fail-Closed Gate & Real Toolchain Verification for LLM Code Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23123923.svg)](https://doi.org/10.5281/zenodo.23123923)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/callensxavier/gwaya-v3-verified-report)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](papers/gwaya_v3_verified_report.pdf)

> **GWAYA (Gwen-Laya Open Model)** is a zero-trust, fail-closed verification layer that sits between code-generating language models (such as the Qwen / Gwen series) and execution environments. 
> 
> Unlike traditional code checkers that yield false positives when toolchains are missing or models emit placeholder stubs, **GWAYA is strictly fail-closed**: a candidate is accepted **only** if real, deterministic compilers and sandboxed execution positively succeed. Every situation where checking cannot run (missing compiler, offline fallback, lack of generator) yields `UNVERIFIED` with maximum penalty energy ($E = 10^6$).

---

## 📌 Links to Official Releases & Reports

- **Zenodo Permanent Archival (v3.1.1):** [https://doi.org/10.5281/zenodo.23123923](https://doi.org/10.5281/zenodo.23123923)
- **Paper revision 3.12.0** is in [`papers/gwenlaya_v4.pdf`](papers/gwenlaya_v4.pdf); no Zenodo record is claimed for it here, and the Zenodo record below is revision 3.10.0.
- **GwenLaya v4 paper, revision 3.10.0 (Zenodo):** [https://doi.org/10.5281/zenodo.23300156](https://doi.org/10.5281/zenodo.23300156) (concept DOI of the record family: 10.5281/zenodo.23121787)
- **Serverless TPU endpoint (OpenAI format):** tooling and user guide in [`docs/TPU_ENDPOINT.md`](docs/TPU_ENDPOINT.md). **Not deployed or measured yet**; the cold start (minutes, a TPU VM is created on demand) is an estimate.
- **Hugging Face Verified Artifacts & Raw Telemetry:** [callensxavier/gwaya-v3-verified-report](https://huggingface.co/datasets/callensxavier/gwaya-v3-verified-report)
- **Verified Research Report (PDF):** [`papers/gwaya_v3_verified_report.pdf`](papers/gwaya_v3_verified_report.pdf) (7-page comprehensive audited report)
- **Pre-Registration & Audit Trail:** [`results/PREREGISTRATION.md`](results/PREREGISTRATION.md)

---

## 🔍 Verified Findings vs. Unverified Research Directions

In compliance with zero-trust empirical principles, this repository strictly separates **mathematically & experimentally verified claims** from **ongoing, unverified research directions**.

```
========================================================================================
                          GWAYA VERIFICATION LANDSCAPE
========================================================================================
  [VERIFIED CLAIMS]                                 [ACTIVE RESEARCH ROADMAP]
  -----------------                                 -------------------------
  ✓ AST Zero-Stub Audit (rejection of               ? Full Qwen Series Scaling
    pass, ..., mock_*, fake_*, sorry, etc.)           (0.5B, 1.5B, 3B, 7B, 14B, 32B)
  ✓ Fail-Closed Python Sandbox                      ? Multi-turn Execution Trace Repair
    (Bubblewrap isolation, time/memory bounds)        with Rich Exception Feedback
  ✓ Real Rustc Type & Borrow Checking               ? Lean 4 Mathlib Formal Proof
    (metadata emission, placeholder rejection)        Search at Scale (>10k theorems)
  ✓ Lean 4 Kernel Soundness Verification            ? Exemplar Retrieval Optimization
    (with #print axioms closure audit)                (Resolving A4 vs A3 degradation)
  ✓ Low-Tier Sample-and-Repair Loop                 ? Direct Preference Optimization (DPO)
    (+7.78 pt pass@1 on Qwen2.5-Coder:1.5B)           on Deterministic Fail-Closed Pairs
========================================================================================
```

### 1. Retraction of Earlier Unverified Claims
The first revision of this report (version 3.1.0, Zenodo DOI 10.5281/zenodo.23121788) included claims that the low-tier optimizer improved pass rates by at least +8.0 percentage points based on a hand-written results file. **That claim was retracted**. Version 3.1.1 and this open-source release report only the real, measured benchmarks described below.

---

## 📊 Measured Empirical Benchmarks (MBPP-Sanitized)

All benchmarks evaluate **`qwen2.5-coder:1.5b`** on the **MBPP-sanitized** test split. 
- **Setup:** Public test is `test_list[0]`; hidden tests are `test_list[1:]`.
- **Outcome criterion:** A problem is solved **only if the final candidate passes all asserts** inside an isolated Linux Bubblewrap (`bwrap`) sandbox. The optimizer's internal flags are never taken as proof.
- **Arms:**
  - **A0:** Raw first sample (zero-shot baseline).
  - **A2:** Best-of-3 candidate sampling scored against the public test.
  - **A3:** Best-of-3 plus up to 3 iterative self-repair rounds on public test failures.
  - **A4:** A3 plus one retrieved exemplar from MBPP train/validation splits (leakage-guarded).
- **Pre-registered Gate:** $A3 - A0 \ge +8.0$ percentage points.

### Benchmark Results Table

| Evaluation Run | Arm | Configuration | pass@1 (%) | $\Delta$ vs A0 | 95% Bootstrap CI | Gained / Lost | Exact McNemar $p$ | Pre-registered Gate |
|---|---|---|---|---|---|---|---|:---:|
| **Primary Run**<br>($n=60$, Local CPU,<br>Ollama 0.1.44) | **A0**<br>**A2**<br>**A3**<br>**A4** | Raw baseline<br>Best-of-3<br>Best-of-3 + 3 repairs<br>A3 + Exemplar retrieval | 68.3%<br>71.7%<br>**75.0%**<br>75.0% | —<br>+3.33<br>**+6.67**<br>+6.67 | —<br>[0.0, +8.3]<br>**[+1.7, +13.3]**<br>[-1.7, +15.0] | —<br>2 / 0<br>**4 / 0**<br>6 / 2 | —<br>0.50<br>**0.125**<br>0.289 | <br><br>**FAIL**<br>(+6.67 vs +8.0) |
| **Full Replication**<br>($n=257$, Spot L4 GPU,<br>Ollama 0.5.7) | **A0**<br>**A2**<br>**A3**<br>**A4** | Raw baseline<br>Best-of-3<br>Best-of-3 + 3 repairs<br>A3 + Exemplar retrieval | 64.2%<br>69.3%<br>**72.0%**<br>71.2% | —<br>+5.06<br>**+7.78**<br>+7.00 | —<br>[+2.7, +7.8]<br>**[+4.7, +11.3]**<br>[+3.1, +10.9] | —<br>13 / 0<br>**20 / 0**<br>23 / 5 | —<br>0.0002<br>**< 0.001**<br>0.0009 | <br><br>**FAIL**<br>(+7.78 vs +8.0) |

### Key Benchmark Takeaways
1. **Statistically Clear Gain on Full Scale ($n=257$):** In the full replication, A3 gained **20 problems and lost 0** compared to raw baseline A0 (exact McNemar $p < 0.001$), demonstrating that test-driven self-repair reliably recovers broken solutions.
2. **Honest Gate Reporting:** The pre-registered gate required $\Delta \ge +8.0$ points. The measured gain was **+7.78 points** (0.22 points below threshold). In strict accordance with the measured-improvement contract, the gate verdict is reported as **FAIL** without adjusting the threshold post-hoc.
3. **Exemplar Retrieval Underperformed A3:** Arm A4 (adding in-context retrieved exemplars) yielded +7.00 points but lost 5 previously-solved problems. Indiscriminate exemplar injection can bias small models away from straightforward solutions.

---

## 🧪 GwenLaya v4 (work in progress, exploratory results only)

GwenLaya asks whether a small open model (Qwen3.5) can answer only when an executed check or a calibrated
score supports the answer, and otherwise abstain or escalate. **No hypothesis test has run in its registered form**:
the two primary hypotheses were tested only in deviated forms (a trained Laya calibrator and a separate calibration split
exist since 3.6.0; see the status paragraph below). The three bullets that follow are the 3.4.0 baselines.
What has been measured (full 1,536-task E set, one greedy sample, one TPU v5e chip; see
[`papers/gwenlaya_v4.pdf`](papers/gwenlaya_v4.pdf) and [`results/gwenlaya_v4/e_tpu/`](results/gwenlaya_v4/e_tpu/)):

- On the 4B, answering only when the executed gate verifies cut confident-wrong answers from 32.5% to 13.8% of tasks
  (Python) and from 47.1% to 6.7% (Rust), and did better than a log-probability threshold at the same coverage,
  even with that threshold tuned on the test set. The gate answers nothing on math (no visible check).
- Negative results: raw log-probability confidence is badly calibrated (about 0.93 mean confidence against 53-68% accuracy),
  and a gate-driven 2B-to-4B cascade is **not** cheaper than always using the 4B (cost ratio 1.36) for the same accuracy.
- All of this is on public benchmarks the models have probably seen. Every number is traceable to a committed file;
  an independent claim audit of the paper is in [`docs/CLAIM_AUDIT.md`](docs/CLAIM_AUDIT.md).

**Status of the paper, revision 3.12.0 (14 October 2026; 3.11.x were tooling-only).** Phase 2 (section "Phase 2" of the paper;
independent audit: [`docs/CLAIM_AUDIT_3120.md`](docs/CLAIM_AUDIT_3120.md); the author's register it started from: [`docs/CLAIM_AUDIT_3120_author.md`](docs/CLAIM_AUDIT_3120_author.md)) is exploratory Python-only development work with
rules written before the numbers; nothing registered changed. A prompt-only accept/refuse decision (U) does not reach the 0.80 working target; a
frozen stacked accept policy (D3) was read once on the sealed, previously seen set E and met both of its rules on the 9B
(coverage 0.73, precision 0.886 [0.850, 0.914]), on an easy set (9B base rate 0.726; the gate there also rests on one visible test); on the development table (F Python plus E' MBPP) the goal holds
only at coverage 0.24 to 0.40 (4B, 9B), and on its F Python rows alone only for the 9B at coverage 0.17. A feedback-retrofitted Laya v2, best-of-8 sampling with the visible gate (2B, 4B, 9B) and a learned selector over the sampled pools are
null results; the Qwen3.5 LoRA probe on TPU timed out (inconclusive). The sealed E table is spent. The goal redefinition of A30 was made post hoc.
The 3.10.0 paragraph below is the previous status; its statement that the endpoint tooling was "tested against a fake TPU API only" is superseded
by one 8-task endpoint smoke run on 2026-10-13 (a plumbing check, not an evaluation).

**Status of the paper, revision 3.10.0 (11 October 2026).** The bullets above are the 3.4.0 baselines; the paper
([`papers/gwenlaya_v4.pdf`](papers/gwenlaya_v4.pdf)) has since added a trained Laya, fresh evaluation sets and Lean 4 results.
All of it is deviated or exploratory and every number is read from a committed file (claim map for this revision:
[`docs/CLAIM_AUDIT_3100.md`](docs/CLAIM_AUDIT_3100.md); the audit of the previous one is
[`docs/CLAIM_AUDIT_390.md`](docs/CLAIM_AUDIT_390.md); deviations D1-D77 in [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md)).
In 3.10.0 the frozen Laya was run unchanged on fresh Python and math sets F (908 tasks): H1 (calibrated score vs raw
log-prob, AURC) is supported in a deviated test on Python and math pooled (within Python alone the 9B interval includes
zero; it was not supported on the fresh Rust set of 3.7.0), H3 (vs the gate alone at the gate's coverage) is not, and Laya is
not distinguishable from a logistic regression on the same features. First Lean 4 results are descriptive (greedy proof rate on miniF2F 0.10 to 0.33
by tier, hardened gate with known limits). The Laya runtime and a llama.cpp tier exist as opt-in system changes and have
**not** been validated end to end (no Q4_K_M calibration, accuracy or threshold evaluation). Python and math are the
primary domains, Lean 4 secondary, Rust gate-only.

### Latest measured results (paper revision 3.10.0)

All rows are read from committed files and were recomputed by an independent audit
([`docs/CLAIM_AUDIT_3100.md`](docs/CLAIM_AUDIT_3100.md)). Fresh sets F: Python 407, math 501, Lean 4 229 tasks; greedy,
bf16, Qwen3.5 2B / 4B / 9B on TPU v5e. "Exploratory" means not a registered confirmatory test.

| Question | Result | Status |
|---|---|---|
| Accuracy on F, Python + math pooled (2B / 4B / 9B) | 0.313 / 0.427 / 0.472 | descriptive |
| Executed gate vs log-prob threshold at equal coverage, confident-wrong rate (9B) | Python 0.147 vs 0.224; math 0.004 vs 0.078 | exploratory |
| Frozen Laya, calibrated score vs raw log-prob (H1, 9B, 908 tasks) | AURC difference -0.104, 95% CI [-0.132, -0.078], Holm p = 0.0002 | **supported** (deviated; carried by math, Python alone includes zero) |
| Frozen Laya vs the gate alone at the gate's coverage (H3) | -0.002, 95% CI [-0.022, +0.016] | **not supported** |
| Laya vs logistic regression on the same hand features | AUROC -0.004 [-0.015, +0.006]; AURC +0.010 [+0.003, +0.019] (regression slightly better) | text adds nothing measurable |
| Router at tau = 0.5 vs always-9B | accuracy 0.476 vs 0.472, cost 0.89x | exploratory |
| Gate-driven 2B-4B-9B cascade vs always-9B (pooled) | cost ratio 1.74, accuracy -0.007 [-0.026, +0.012] | **no saving** |
| Lean 4 proofs, miniF2F test, 223 (0.8B / 2B / 4B / 9B) | 22 / 41 / 59 / 69 proved, all exact statement matches | descriptive |
| Lean 4 proofs, fresh miniF2F-valid set, 229 (2B / 4B / 9B) | 33 / 65 / 76 proved | descriptive |
| Laya candidate-first text layout (A22) | not better (ΔAUROC -0.010 on E', +0.004 on R', CI spans 0) | negative |

What this does **not** show: that the generators have not seen these public benchmarks, anything about a 27B model
(never loaded on the available TPU slice), end-to-end behaviour of the opt-in Laya runtime or the llama.cpp tier, or a
deployed serverless endpoint (the tooling in [`docs/TPU_ENDPOINT.md`](docs/TPU_ENDPOINT.md) is written and tested
against a fake TPU API only).

## 🛠️ Architecture: The GWAYA Fail-Closed Gate

For any candidate $c$ in language $\ell \in \{\text{python}, \text{rust}, \text{lean4}\}$, the gate accepts if and only if:

$$\mathrm{accept}(c) = \mathrm{Avail}_\ell \wedge \mathrm{StubAudit}_\ell(c) \wedge \mathrm{Oracle}_\ell(c) \wedge \bigl(\ell \neq \text{lean} \vee \mathrm{Ax}(c) \subseteq A_{\text{std}}\bigr)$$

where:
1. **$\mathrm{Avail}_\ell$ (Toolchain Availability):** Confirms compiler binaries are present on the host. Missing compilers immediately yield `UNVERIFIED` (never pass).
2. **$\mathrm{StubAudit}_\ell(c)$ (AST Zero-Stub Law):** Analyzes the AST to reject `pass`, `...`, `mock_*`, `fake_*`, `dummy_*`, `sorry`, `admit`, `unimplemented!()`.
3. **$\mathrm{Oracle}_\ell(c)$ (Compiler/Execution Oracles):**
   - **Python:** Syntactic non-triviality check + isolated Linux Bubblewrap execution with memory bounds, temporary directory sandbox, and strict timeout.
   - **Rust:** Verifies syntax, types, borrow checker, and lifetime rules via `rustc --emit=metadata`.
   - **Lean 4:** Kernel check ensuring no non-standard axioms were introduced.
4. **$\mathrm{Ax}(c) \subseteq A_{\text{std}}$ (Axiom Soundness):** The Lean 4 oracle executes `#print axioms` on every named declaration. It strictly limits dependencies to standard Lean axioms: $A_{\text{std}} = \{\texttt{propext}, \texttt{Classical.choice}, \texttt{Quot.sound}\}$. Bypasses such as `axiom bad : False` or `sorryAx` are caught and rejected even if `lean` exits 0.

---

## 🤖 Model Context Protocol (MCP) Server for AI Agents

GWAYA includes a production-ready **Model Context Protocol (FastMCP)** server, enabling autonomous AI coding agents (**Claude Code**, **Antigravity**, **Cursor**, **Windsurf**, **OpenHands**) to inspect, verify, and repair code before executing or presenting it.

### MCP Tools Provided

| Tool Name | Parameters | Purpose |
|---|---|---|
| `gwaya_system_status` | *None* | Discovers availability of Python, `rustc`, `lean`, `bwrap`, and local Ollama GPU models. |
| `gwaya_audit_stubs` | `language`, `code` | AST audit catching placeholder stubs, mock identifiers, and proof holes. |
| `gwaya_verify_code` | `language`, `code`, `test_spec` | Full fail-closed verification (Bubblewrap test sandbox, rustc, Lean 4 kernel). |
| `gwaya_repair_code` | `prompt`, `failing_code`, `test_spec`, `model` | Invokes the Qwen sample-and-repair loop with real test feedback. |

### Connecting to Claude Code (`.mcp.json`)

Add the following to your project's `.mcp.json` or `claude.json`:

```json
{
  "mcpServers": {
    "gwaya": {
      "command": "python",
      "args": ["/path/to/GWAYA-GwenLayaOpenModel/mcp_server.py"],
      "env": {
        "PYTHONPATH": "/path/to/GWAYA-GwenLayaOpenModel"
      }
    }
  }
}
```

### Connecting to Antigravity CLI / IDE

Add to your `.antigravity/mcp_servers.json`:

```json
{
  "gwaya-verification": {
    "command": "python",
    "args": ["-m", "mcp_server"],
    "cwd": "/path/to/GWAYA-GwenLayaOpenModel"
  }
}
```

---

## ⚡ Deployment: Consumer RTX GPUs & Ollama

GWAYA is optimized to run locally on consumer NVIDIA GeForce RTX graphics cards using Ollama.

### 1. Hardware VRAM Sizing Guide

| Qwen (Gwen) Model | Parameters | Required VRAM | Recommended Consumer GPUs |
|---|---|---|---|
| `qwen2.5-coder:0.5b` | 0.5 Billion | ~0.5 GB | Any Laptop GPU, Intel/Apple iGPU |
| `qwen2.5-coder:1.5b` | 1.5 Billion | ~1.5 GB | GTX 1660, RTX 3050, RTX 2060 |
| `qwen2.5-coder:3b` | 3.0 Billion | ~2.5 GB | RTX 3060 Laptop, RTX 4050 |
| `qwen2.5-coder:7b` | 7.0 Billion | ~5.0 GB | RTX 3060 (12GB), RTX 4060, RTX 3070 |
| `qwen2.5-coder:14b` | 14.0 Billion | ~9.5 GB | RTX 3060 (12GB), RTX 4070 (12GB), RTX 3080 |
| `qwen2.5-coder:32b` | 32.0 Billion | ~20.0 GB | RTX 3090 (24GB), RTX 4090 (24GB), RTX 5090 |

### 2. One-Click Setup Script

We provide an automated setup script that detects your RTX card, checks VRAM, pulls the optimal model, and validates 100% GPU layer offloading:

```bash
# Clone the repository
git clone https://github.com/xaviercallens/GWAYA-GwenLayaOpenModel.git
cd GWAYA-GwenLayaOpenModel

# Auto-detect GPU and pull optimal model:
./scripts/deploy_rtx_ollama.sh

# Or pull a specific model:
./scripts/deploy_rtx_ollama.sh 7b
```

### 3. Running the FastMCP Server

```bash
uv pip install -e .
fastmcp run mcp_server.py
```

### 4. Running the MBPP Benchmark Locally

```bash
# Run 10 problems on your local RTX GPU
python scripts/run_benchmark.py --n 10 --model qwen2.5-coder:1.5b

# Run full MBPP replication (n=257)
python scripts/run_benchmark.py --n 257 --model qwen2.5-coder:7b --out-dir results/my_rtx4090_run
```

---

## ⚡ Local Agent Deployment: `gwaya-agent` CLI & Docker Compose

For developer workstations with NVIDIA RTX GPUs (RTX 3060, 3080, 4060, 4080, 4090) or CPU fallback:

### 1. Zero-Config GPU Doctor & CLI
```bash
# Check GPU VRAM, detect installed models, and get automated model recommendations:
gwaya-agent doctor

# Run fail-closed verified generation:
gwaya-agent ask "def is_prime(n): check if n is prime" --test "assert is_prime(7) and not is_prime(8)"
```

### 2. Full Local Stack with Docker Compose
Run both the Ollama GPU backend and GWAYA verification agent with a single command:

```bash
# Launch Ollama with GPU pass-through + GWAYA agent:
docker compose up -d

# Verify agent status
docker compose run gwaya-agent doctor
```

### 3. Anti-Hallucination Grounding & Consensus Self-Testing
Small models frequently suffer from:
- **Symbol hallucinations:** Inventing functions (`numpyx.fast_solve`), non-existent stdlib attributes, or unbound variables.
- **Over-fitting to a single public test:** Emitting hardcoded checks that fail hidden edge cases.

GWAYA resolves this through:
1. **`gwaya.grounding`:** Static AST and pyflakes inspection rejects candidates referencing unresolvable imports or unbound local names *before execution*.
2. **`gwaya.consensus_agent`:** Generates candidate unit tests from multiple independent generations and computes consensus ($k \ge 2$) assert agreement, rejecting candidates that fail consensus tests even if they pass the public prompt test.
3. **Full-Context Trace Feedback:** Replaces naive error truncations with sandbox-evaluated LHS/RHS runtime values for targeted self-repair.

---

## ☁️ Cloud Spot Deployment (Google Cloud / RunPod)

For larger models or massive multi-benchmark runs, we provide a self-deleting Google Cloud spot VM launcher with strict cost caps:

```bash
# Launches spot L4 GPU (g2-standard-8, ~$0.45/h), syncs results to bucket, and force-deletes on exit:
./deploy/run_on_gcp.sh --tag gcp_l4_n257 --n 257 --cap-usd 5.0
```

---

## 🧪 Open Call for Community Contributions (Humans & AI Agents)

GWAYA and GwenLaya are open research code with a habit of publishing negative results and retracting claims that did not
survive an audit. **You are welcome to contribute, to reproduce, and above all to try to break it.** Everything here is
MIT-licensed; the paper revision 3.10.0 is archived at
[doi:10.5281/zenodo.23300156](https://doi.org/10.5281/zenodo.23300156). Start with [`CONTRIBUTING.md`](CONTRIBUTING.md).

### Where help matters most
1. **Break the gate.** Try to get a wrong or hollow candidate `VERIFIED` (Python sandbox escapes, Rust stubs that compile,
   Lean statement-matching bypasses). The Lean gate's known limits (binder info not compared, defeq accepted, red team
   incomplete) are listed in the paper; a reproducible bypass is a first-class contribution.
2. **Replicate or refute.** Re-run the audit scripts on the committed rows, or run the protocol on another model family
   (Qwen3, Llama, Gemma, DeepSeek) and tell us whether H1 and the gate's confident-wrong reduction transfer.
3. **A bigger tier.** Qwen3.5-27B (or any 20B+ model) never loaded on one v5litepod-4. Multi-host TPU, int8, or a GPU run
   would close the "tier gap" question left open by the cascade result.
4. **Make Laya's text matter.** Laya is not distinguishable from a logistic regression on gate and log-prob features, and
   its 512-token window drops the candidate on long prompts. Variable-length inputs, better pooling or fine-tuning
   recipes are open; a layout change alone did not help (A22).
5. **Fresh, decontaminated evaluation sets** (post-training-cutoff problems, Rust, Lean, other languages) built with
   `gwaya.decontaminate` and sanity checks as in `scripts/data/build_eval_sets_F.py`.
6. **Run and measure the serverless TPU endpoint** (`deploy/tpu_endpoint`): real cold-start time, throughput, and fixes
   for the points listed as unverified in `docs/TPU_ENDPOINT.md`.
7. **New language oracles** (C++, Go, Julia) with fail-closed semantics and a stub-audit.

### Ground rules
- Plan before you run: write the analysis plan (as an addendum in `docs/ANALYSIS_PLAN_E_TPU.md` or in the PR) *before*
  looking at results; record every deviation in `docs/DEVIATIONS.md`.
- Every number must be traceable to a committed rows/JSON file. No hand-typed results. Negative and null results are
  welcome and will be merged.
- Open an issue or PR with the exact command, the commit hash, the hardware, and the raw rows.
- **AI coding agents** (Claude Code, Codex, OpenHands, Antigravity, ...) are welcome: fork, implement, run
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q`, check `git diff`, and open a PR with the full execution log. State in the
  PR that an agent wrote it.

---

## 📜 Citation

If you use GWAYA in your research or applications, please cite the permanent Zenodo record:

```bibtex
@software{callens_2026_gwaya,
  author       = {Callens, Xavier},
  title        = {GWAYA v3.1.1: Fail-Closed Gate \& Real Toolchain Verification for LLM Code Generation},
  month        = oct,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {3.1.1},
  doi          = {10.5281/zenodo.23123923},
  url          = {https://doi.org/10.5281/zenodo.23123923}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
