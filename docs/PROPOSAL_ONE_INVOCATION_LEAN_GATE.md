# Proposal: verify a batch of Lean candidates in one invocation, without weakening the gate

**Status:** proposal, not implemented here. **Source of the measurements:** Kernel-Gated
Evolution, AutoevolveAI `anse/formal/lean_runner.py` (`verify_file_per_declaration`,
`verify_batch_against_spec`) and Paper J ([concept DOI 10.5281/zenodo.23635678](https://doi.org/10.5281/zenodo.23635678)).

## The observation

`Lean4CompilerOracle._interpret_run` decides `success = res.returncode == 0` and only then
reads the axiom closure. That is correct for one snippet per invocation, which is how
`verify_snippet` is called today. It becomes wrong the moment several candidates share an
invocation: **a Lean compile containing one false declaration exits nonzero, yet still prints
an axioms line for every declaration it elaborated**, the failing ones carrying `sorryAx`.
Gating on the exit code first discards those per-candidate verdicts and makes the batch look
atomic. We measured this directly:

```
lake env lean <file with 8 candidates, 2 false>   -> exit 1, 8 axiom lines printed
'At0.At0_m' depends on axioms: [propext, Classical.choice, Quot.sound]
'At2.At2_m' depends on axioms: [propext, sorryAx, Classical.choice, Quot.sound]
```

## Why it matters for GWAYA

About 97% of a Lean invocation is fixed overhead (toolchain and Mathlib load) on the T4 host;
a candidate's own elaboration is ~0.14 s of ~5.4 s. Reading the per-declaration report gives
every verdict from **one** invocation at **any** failure rate:

| false of 8 | invocations | wall clock | false accepted |
|---:|---:|---:|---:|
| 0 | 1 | 10.38 s | 0 |
| 1 | 1 | 10.21 s | 0 |
| 2 | 1 | 10.88 s | 0 |
| 4 | 1 | 12.73 s | 0 |
| 8 | 1 | 14.19 s | 0 |

against 8 individual invocations at a 6.36 s median (50.88 s). Two isolation strategies built
before this was understood — bisection, and a flat fall-back — were both worse and unnecessary.
GWAYA's repair loop and pass@k evaluation produce several Lean candidates per task, which is
exactly where this pays.

## Why it does not weaken the gate

Acceptance must never rest on the exit code; it rests on three conditions, all of which GWAYA
already enforces and a batch path must keep:

1. **The source scan** (`_lexical_flaws`, `_ESCAPE_RE`): no `sorry`/`admit`/`axiom`/
   `native_decide`/`implemented_by`/`extern`, no `#exit` truncation, no `#print` string spoofing.
2. **Every declared theorem has its own axioms line** (`_audit_count` ≥ expected): a declaration
   that never elaborated prints nothing and must fail as unchecked.
3. **That line is inside `STANDARD_AXIOMS`** and free of `sorryAx`.

A nonzero exit then only says *some* candidate failed, which conditions 2 and 3 catch by name.
Nothing is accepted that the one-at-a-time path would have accepted.

## Proposed shape

```python
def verify_snippets_batch(self, snippets: list[tuple[str, str]]) -> dict[str, OracleResult]:
    """Verify several named candidates in one `lean` invocation.

    Each snippet is wrapped in its own namespace (names must be globally unique: a
    duplicate is refused up front rather than surfacing as an ambiguous axiom line).
    The lexical scan runs per snippet BEFORE compilation; the per-declaration axiom
    report is read regardless of the exit code; a candidate passes only if every one of
    its theorems has a clean, whitelisted axiom line.
    """
```

Required tests, each of which must be able to fail:
- a batch with one false candidate names exactly that candidate and accepts the rest;
- a batch where every candidate is false accepts none, in one invocation;
- a planted `axiom cheat : False` candidate is rejected by the lexical scan *and* would be by the axiom line;
- a candidate whose declaration never elaborates (missing axiom line) fails as unchecked;
- duplicate theorem names across candidates are refused before compiling.

## Caveat

The numbers above are from AutoevolveAI's pinned Lean/Mathlib on this host; GWAYA's sandboxed
toolchain should be measured the same way (single vs batched, 0–8 failures of 8) before relying
on the speed-up.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
