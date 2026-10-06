# GWAYA Roadmap

> This roadmap extends the `[ACTIVE RESEARCH ROADMAP]` section of the README with concrete, prioritized
> evolution tracks. In keeping with the zero-trust principles of this project, every track below is
> explicitly marked as **UNVERIFIED / PLANNED** until a pre-registered evaluation demonstrates otherwise.
> No claim on this page may be treated as measured until it appears in `results/` with a pre-registered gate.

---

## Track A — Numerical / Scientific-Code Oracle (priority: high)

**Goal.** Extend the fail-closed gate to AI-generated scientific computing code, where unit tests
alone are insufficient (a wrong solver can still converge and pass superficial tests).

**Planned verification criteria (fail-closed semantics):**

- Absolute/relative error tolerance against a deterministic reference solver on standard problems
  (stiff, oscillatory, chaotic regimes).
- Observed convergence order must be consistent with the declared integration method.
- Stability bounds (time step limits, energy drift) must hold within pre-registered tolerances.
- Any non-reproducible result against the deterministic reference yields `UNVERIFIED` with maximal
  penalty energy.

**Reference infrastructure.** Real benchmark corpora (e.g., Robertson ODE family, CI-measured
reference runs) with measured, non-simulated telemetry only. All reference data must come from
real CI runs, in line with the project's retraction discipline.

---

## Track B — Formal Escalation via Lean 4 (priority: high)

**Goal.** Introduce a second, independent verification tier: execution (tier 1, existing) plus
mathematical proof (tier 2, new) for critical functions.

**Planned semantics:**

- Automatic generation of a Lean 4 specification from the declared contract
  (preconditions, postconditions, invariants).
- Proof of correctness against kernel-checked libraries; zero-`sorry` audit and
  `#print axioms` closure audit remain mandatory.
- Failure to prove does not imply rejection of the executable tier, but the candidate is never
  marked PROVEN; silent acceptance is forbidden.

**Relation to existing roadmap.** Implements and supersedes the `Lean 4 Mathlib formal proof search
at scale (>10k theorems)` research direction with a two-tier acceptance semantics.

---

## Track C — Kernel-Level Verification Oracle (priority: medium)

**Goal.** Extend the gate from application code to system code (drivers, kernel modules,
syscall paths), for which no standard safe execution environment exists.

**Planned verification criteria:**

- A candidate patch is accepted only if a kernel rebuilt with the patch boots on all supported
  target architectures (x86_64, RISC-V, AArch64) under QEMU.
- Formal verification of the patched kernel must still pass.
- A syscall-firewall sandbox (eBPF-style, default-deny) may serve as a second-generation execution
  sandbox, replacing or complementing Bubblewrap for system-level candidates.

**Caveat.** Boot-success is a necessary but not sufficient condition for kernel correctness; the
gate must report it as exactly that.

---

## Track D — OS-Integrated Verification Service (priority: low, exploratory)

**Goal.** Long-term: move the gate from an application-level service to a system-level component,
executing close to (or inside) the kernel with hardware isolation, such that any code generated
on the machine passes through the gate before local deployment.

**Status.** Exploratory. Blocked on Tracks A-C. No claims.

---

## Track E — Industrial Evaluation Program (priority: high, orthogonal)

Evaluations required for industrial adoption, in priority order:

1. **False-UNVERIFIED rate** — proportion of genuinely correct code rejected due to missing
   toolchains, over-tight timeouts, or sandbox misconfiguration. Determines operational acceptability.
2. **Cost per verified decision** — compute/energy cost of one verification versus an equivalent
   human review, expressed in time and money.
3. **Adversarial robustness** — behavior against candidates engineered to evade verification
   (undeclared dependencies, masked network calls, runtime code generation, obfuscation).
4. **Traceability** — completeness and auditability of the emitted verification trace against
   safety-relevant frameworks (IEC 62304, ISO 26262, DO-178C, EU AI Act). Target is gap measurement,
   not immediate certification.
5. **Generalization** — replication of the pre-registered protocol across the full model series
   (0.5B-32B), other model families, and datasets beyond MBPP-sanitized (HumanEval+, SWE-bench,
   ClassEval). Single-cell results do not constitute evidence of generalization.
6. **Comparative benchmarking** — GWAYA versus the alternatives industry already uses (unit tests,
   static analyzers, LLM-assisted review), because the industrial question is never "does it work"
   but "does it work better than what we already do, at acceptable cost".

---

## Governance

- Every track must define its pre-registered gate **before** any benchmark run.
- Any claim found to be based on hand-written or non-reproducible data must be publicly retracted,
  following the precedent of v3.1.0 -> v3.1.1.
- Verified results migrate from this page to the README's `[VERIFIED CLAIMS]` column only with
  archived artifacts (Zenodo DOI / Hugging Face).
