# Track A — Pre-Registered Evaluation Protocol A1
# Domain: AI-Generated Quantum-Fluid / Superfluid Simulation Code

> **STATUS: PRE-REGISTERED / NOT YET RUN.** This protocol is registered before any benchmark run.
> It may not be amended after data collection begins; deviations must be logged as amendments with reasons.
> No result in this file is measured. All outcome language is conditional.
> Domain anchor: the SocrateAI-Scientific-QuantumFluids corpus (superfluid dynamics: Josephson
> relation, dispersion, pair statistics, vortex screening, kinetic benchmarks).

---

## 1. Objective

Evaluate whether the GWAYA fail-closed gate can distinguish **physically correct** from
**physically wrong** AI-generated simulation code in the quantum-fluids domain, where unit tests
are insufficient: a wrong integrator can converge, a wrong spectrum can look smooth.

**Primary question (Q1).** Does the numerical oracle reduce the acceptance of physically
incorrect candidates relative to the execution-only gate, without materially increasing the
false-rejection rate on physically correct candidates?

---

## 2. Candidate classes (the gate must separate all of them)

| Class | Description | Expected correct verdict |
|-------|-------------|--------------------------|
| C1 | Correct solver, correct physics | ACCEPTED |
| C2 | Convergent but wrong physics (e.g., classical-viscosity term applied where the superfluid component requires none; classical Maxwell-Boltzmann statistics used where Bose-Einstein is required) | REJECTED |
| C3 | Correct physics, wrong discretization order (declared 4th-order, actually 1st) | REJECTED |
| C4 | Stability violation (time step beyond the declared stability bound; energy drift) | REJECTED |
| C5 | Stub / mock / fabricated result (AST zero-stub audit domain) | REJECTED (existing audit) |
| C6 | Non-reproducible result (run-to-run variance above tolerance on deterministic input) | UNVERIFIED |

C2 and C3 are the classes the current execution-only gate cannot catch; they are the reason this
track exists.

## 3. Reference problems (pre-registered, fixed before generation)

1. **P1 — Kinetic / dispersion benchmark.** Excitation spectrum against the published Godfrin-type
   dispersion table; acceptance on absolute/relative error on the roton minimum position and depth.
2. **P2 — Josephson relation.** Phase-current relation on equilibrated snapshots; acceptance on the
   relation constant within pre-registered tolerance.
3. **P3 — Transport.** Pair statistics / mobility relations (Einstein-type relation, transverse
   friction) at fixed temperature; acceptance on the computed coefficient against a pre-registered
   reference value and its CI.
4. **P4 — Vortex screening.** Vortex-charge-density certification criterion (matching-screening
   style) on a seeded configuration; acceptance on classification agreement.

Reference values for P1-P4 are frozen BEFORE any LLM candidate is generated, in the form
`results/track_a/reference_values.json`, each with: value, tolerance, provenance (which run,
which commit, which dataset), and the deterministic re-run command that reproduces it.

## 4. Pre-registered gates

- **G1 (primary).** Physical-false-acceptance rate on C2+C3 must be strictly below 10% (one-sided
  95% bootstrap CI upper bound), evaluated on >= 50 candidates per class.
- **G2 (safety).** False-UNVERIFIED rate on C1 must not exceed 5% (upper bound of one-sided 95%
  bootstrap CI). A gate that rejects everything trivially passes G1; G2 prevents that.
- **G3 (non-regression).** Verdicts on C5 must be unchanged versus the existing execution-only gate.
- **G4 (cost).** Median added verification cost per candidate must not exceed 5x the existing
  execution-only verification cost.

The gate passes Track A only if G1, G2, G3 and G4 all hold. Partial results must be reported as
partial; no cherry-picking of gates.

## 5. Controls against known failure modes

- **Convergence-is-not-correctness:** every C2 candidate must pass its unit tests by construction,
  so that any rejection is attributable to the numerical oracle, not to test failure.
- **Lucky-seed rejection:** each candidate is run on 3 fixed seeds; verdict is the worst verdict.
- **Reference circularity:** reference values are produced by the existing deterministic reference
  pipeline, never by a candidate or by an LLM.
- **Amendment discipline:** any post-hoc tolerance change is logged in the amendments section with
  a public reason; the pre/post distinction is what the v3.1.0 -> v3.1.1 retraction taught.

## 6. Reporting obligations

For each run: candidate id, class, model and version that generated it, seed triple, per-problem
verdicts (P1-P4), per-gate outcome, compute cost, and the full raw telemetry archived. Results
land in `results/track_a/` with the same provenance discipline as the existing PREREGISTRATION.md
trail. A claim enters the README verified column only with archived artifacts.

## 7. Explicit non-goals

This protocol does not evaluate: generalization beyond quantum-fluids code (Track E), formal
proof of physical models (Track B), kernel-level code (Track C). A pass on A1 claims nothing
outside its pre-registered scope.

---

## Amendments

| # | Date | Change | Reason |
|---|------|--------|--------|
| - | - | (none) | - |
