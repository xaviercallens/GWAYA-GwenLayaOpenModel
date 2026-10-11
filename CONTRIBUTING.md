# Contributing to GWAYA / GwenLaya

Thank you for helping. This project publishes negative results and retracts claims that fail an audit, so the bar for
a contribution is **traceability**, not good news.

## Quick start
```bash
git clone https://github.com/xaviercallens/GWAYA-GwenLayaOpenModel && cd GWAYA-GwenLayaOpenModel
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -p no:cacheprovider -q
```
A few tests need `sympy`, `pyflakes` or `fastmcp`; the 22 failures in a bare environment are that missing set.

## What we would like
See the README section "Open Call for Community Contributions". Highest value first: break the gate, replicate or
refute a result, a larger generator tier, a better Laya, fresh decontaminated sets, measuring the TPU endpoint.

## Rules
1. **Plan before you run.** Describe the analysis (data, metric, decision rule) in the PR or as an addendum in
   `docs/ANALYSIS_PLAN_E_TPU.md` before you look at the results. Record changes of plan in `docs/DEVIATIONS.md`.
2. **Every number comes from a committed file.** Include the rows or JSON, the exact command, the commit hash and the
   hardware. No hand-typed results. If a number cannot be regenerated, it does not go in the paper or README.
3. **Report nulls.** A result that does not replicate is a contribution. Say what was run and what it does not show.
4. **Fail closed.** A gate change must keep "cannot verify" as `UNVERIFIED`, never as a pass; add a test that would fail
   on the old behaviour.
5. **No secrets or weights** in commits (API keys, tokens, model files). Data stays under `GWAYA_DATA_ROOT`.
6. **Spend.** Anything that costs money (TPU/GPU) needs a dry run and a stated cap; scripts here default to dry run.
7. **AI agents are welcome.** State in the PR description that an agent wrote it and attach the execution log.

## Reporting a gate bypass
Open an issue with a minimal candidate, the language, the expected verdict and the observed one. If it is a sandbox
escape rather than a logic bypass, please send it privately to the maintainer first (see the GitHub profile).

## Licence
Contributions are accepted under the repository's MIT licence.
