---
name: gwaya-release-and-audit
description: Use when writing a paper revision section from results, auditing it, and releasing (PR, merge, tag, GitHub release, Zenodo draft) in the GWAYA repo, including the gh and Zenodo quirks hit in phase 3.
---

# Paper section, audit, release

Overall process is in `gwaya-study-workflow`; pitfalls in `LL.md` items 19 to 24. The user's "commit, push, merge, release" means PR -> merge -> tag -> `gh release create` with the PDF, then memory and docs updated.

## Write
- Generate tables and macros from committed JSON with a script (`scripts/paper/tables_phase3.py` is the pattern: reads results + ledger, writes `papers/tables_*.tex`, `papers/numbers*.tex`, `results/.../spend_summary_*.json`). Add `\input{numbersNNNN.tex}` near the other `\input` lines. Prose numbers are copied from the generated tables.
- Insert the section before `\section{Hypothesis status}`; add a Next steps item; update README status. Build: `cd papers && latexmk -pdf -interaction=nonstopmode -halt-on-error gwenlaya_v4.tex`, then check the LAST pass for `Undefined` (earlier passes always warn) and render the new pages (`pdftoppm -r 70`) to look at them.
- Label registered vs exploratory vs descriptive; state what a result does not show; list order of events with commit hashes and times.

## Audit (never skip, even when told to release)
Spawn one independent agent (general-purpose), no spend/push/tag, prompt = files + "recompute every number from raw files with your own code, adversarially, write docs/CLAIM_AUDIT_<ver>.md". Give it the paths of raw inputs and the rules, not your conclusions. Fix every slip, add an "Author follow-up" section to the audit file, rebuild the PDF, commit.

## Release
1. Tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -p no:cacheprovider -q --ignore=tests/test_gwenlaya_endpoints.py --ignore=tests/test_mcp_server.py` (22 env-only failures are known, and the two ignored files need `fastmcp`, which is not installed).
2. `git push origin <branch>`; `gh pr create`; `gh pr ready N`; `gh pr merge N --merge` (it may print nothing: check `gh pr view N --json state`). `gh pr edit` can fail on Projects-classic: use `gh api -X PATCH repos/OWNER/REPO/pulls/N -f title=... -f body=...`.
3. `git checkout main && git pull`; `git tag -a vX.Y.Z -m ... <merge commit>`; `git push origin vX.Y.Z`; `gh release create vX.Y.Z <pdf> --title ... --notes ...` (tooling/docs-only revisions get a patch tag).
4. Zenodo: edit `VERSION`, `RELEASE_URL`, `DESCRIPTION` in `scripts/release/zenodo_new_version.py`, run it (creates a DRAFT; `--draft ID` repairs an existing draft). Show the draft URL. Publish (`--publish ID`, irreversible DOI) ONLY after an explicit "publish" from the user.
5. Update memory (`gwenlaya-restart-state`, lessons) and `scripts/restart.sh` hints if they changed.
