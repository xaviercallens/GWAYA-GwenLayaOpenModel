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
score supports the answer, and otherwise abstain or escalate. **No pre-registered hypothesis test has been
completed**: both primary hypotheses need the Laya calibrator and a separate calibration split, which do not exist yet.
What has been measured (full 1,536-task E set, one greedy sample, one TPU v5e chip; see
[`papers/gwenlaya_v4.pdf`](papers/gwenlaya_v4.pdf) and [`results/gwenlaya_v4/e_tpu/`](results/gwenlaya_v4/e_tpu/)):

- On the 4B, answering only when the executed gate verifies cut confident-wrong answers from 32.5% to 13.8% of tasks
  (Python) and from 47.1% to 6.7% (Rust), and did better than a log-probability threshold at the same coverage,
  even with that threshold tuned on the test set. The gate answers nothing on math (no visible check).
- Negative results: raw log-probability confidence is badly calibrated (about 0.93 mean confidence against 53-68% accuracy),
  and a gate-driven 2B-to-4B cascade is **not** cheaper than always using the 4B (cost ratio 1.36) for the same accuracy.
- All of this is on public benchmarks the models have probably seen. Every number is traceable to a committed file;
  an independent claim audit of the paper is in [`docs/CLAIM_AUDIT.md`](docs/CLAIM_AUDIT.md).

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

We actively welcome contributions from **human researchers, software engineers, and autonomous AI agents**!

### Top Priority Research Tracks:
1. **Qwen (Gwen) Model Series Scaling:**
   - Run `scripts/run_benchmark.py` on `0.5b`, `3b`, `7b`, `14b`, and `32b` models and submit raw `rows.jsonl` and `results.json` pull requests.
2. **Multi-Turn Exception Trace Self-Repair:**
   - Enhance `low_tier_engine.py` to parse traceback line numbers and exception types into the repair prompt.
3. **Formal Verification (Lean 4 Mathlib):**
   - Extend the Lean oracle to automatically discover `lake_packages` and evaluate auto-formalization datasets (MiniF2F, ProofNet).
4. **New Language Oracles:**
   - Implement fail-closed compiler oracles for C++ (`clang++`), Go (`go vet`), and Julia.
5. **AI Agent Pull Request Protocol:**
   - If you are an AI coding agent (Claude Code, Antigravity, OpenHands), you are invited to fork this repo, implement an improvement, execute `pytest tests/`, verify `git diff`, and open an automated PR with full execution telemetry.

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
