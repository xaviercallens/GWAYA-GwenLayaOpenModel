# Lessons learnt (LL)

Written 2026-10-17 after phase 3 (A33 to A33.4, release v3.13.0). Read this and `scripts/restart.sh` before you launch, score or release
anything. Each lesson says what happened, what it cost, and the rule that would have prevented it. The skills in `.claude/skills/` turn
these into checklists: `gwaya-study-workflow` (overall), `gwaya-tpu-stage-launch`, `gwaya-sealed-read`, `gwaya-release-and-audit`.

## Money and cloud

1. **Memory and notes go stale; the ledger does not.** The restart note said "$26.33 / nothing running"; two days later the ledger
   read $40.74. Always run `scripts/restart.sh` first and trust the ledger and `gcloud compute tpus tpu-vm list` over any note.
2. **Two identical T5 VMs ran in parallel** (`p3-t5-200529` and `p3-t5-200546`), about $5 for no extra result. Cause not established
   (a launch command that was re-run, or a launcher chained twice in one shell line, are the suspects). Before any launch: list TPU VMs
   and `pgrep -af "launch_|drive.sh"`; launch once, from one shell line; the launcher should refuse to start if a driver for the same stage is alive.
3. **A hung SSH session held an idle VM for about 1.5 hours.** The remote script had finished (`SMOKE_DONE` in the status file) but the
   local `gcloud ... ssh` never returned. Poll the remote status file for the end marker; use SSH keepalives (`drive.sh` has them since
   `805bfc5`); when a stage has printed its end marker and the VM is idle, copy results and delete it.
4. **The ledger books `now - timestamp-in-VM-name`.** A VM that existed for two minutes after a retry was booked for 153 minutes.
   Ledger lines are list-price estimates and can over-attribute; read them, do not assume them.
5. **Check your headroom before proposing a rerun.** After A33 the ceiling was $47 and a rerun's worst case was $5.04 against $6.26 left.
   State the worst case and the margin when asking for spend authorisation, and ask for it explicitly; "continue" is not authorisation.
6. **Fail-fast ladders are cheap insurance.** Six failing 35B steps cost under $1.20 because each step is a separate process with its own
   short timeout. Keep that structure.

## Launching and long jobs

7. **Verify the inputs of a launcher are non-empty.** The T6 launcher ran with an empty `tasks_sample.jsonl` (0 lines), so no G samples
   exist for the 2B, 4B and 9B and the sealed readout could cover the 27B only. A dry run must print row counts and abort on 0.
8. **Smoke-test the exact checkpoint, dtype and tensor-parallel size** before a stage that depends on them. The 35B-A3B AWQ checkpoint
   failed twice: first float16 requested where `auto_awq` supports bfloat16 only (launcher defect, fixed in `4f94b88`), then
   "input size is not aligned with the quantized weight shape" at TP=4 (a real incompatibility). Plan a different layout (TP=2
   replicas, another quantization) instead of a third retry; the plan's stop rule forbade one anyway.
9. **A plain background job is not safe for work that matters.** The shell tool reported the first sealed run as finished after a few
   seconds while it kept running; a second run was started and both wrote the same file, so 1,089 of 1,090 G candidates were scored twice
   (2,179 rows). Use `setsid nohup ... < /dev/null &`, write a pid file, check `pgrep` before relaunching, and make writers append
   under a lock or dedupe by key. The outcome did not change, but the paper had to disclose it (found by the audit).
10. **Do not wait with `sleep`.** Chained sleeps are blocked; use Monitor with an `until` loop, or `run_in_background`. Monitors expire
    after their timeout: re-arm them.
11. **`python3 -I` hides user site-packages.** It is right for sandboxed scoring scripts and wrong for anything importing numpy/sklearn
    (`ModuleNotFoundError`). Use the system `python3` for analysis.

## Method and honesty

12. **Compute what the certificate can actually certify before you set a certified goal.** At risk 0.10 and delta 0.10, a policy with true
    precision 0.92 certifies only 16 to 47 percent of the time at 88 to 400 accepted tasks; the goal needs about 0.95 precision at
    coverage 0.4 or a far larger sealed set (`docs/NEXT_STEPS_PHASE4.md`, Section 2). A goal that cannot be met by a good policy is a
    design error, not a null result.
13. **Amend before the number, and withhold numbers until the pipeline is fixed.** A33.2 forbade any V2 number until the test pool was
    settled (35B run or drop). It cost a day and kept the readout clean. Write the amendment, commit it, then compute.
14. **Implement every registered rule before the read.** Rule (b) and the `sealed` command did not exist when A33 was registered;
    they were added and committed (A33.4) before G was scored, and the paper says so. Do this at registration time next time.
15. **Freeze, commit, then read G once.** The order was: code and frozen policy committed (`a4202ab`), then label scoring, then the
    result committed (`ce47e16`). Sealed means one analysis; accidental duplicate labelling is a process failure to disclose.
16. **Labels can be non-deterministic.** One sampled candidate (`py/BigCodeBench/720`, sample 2) was VERIFIED in one scoring pass and
    FAILED in the other. If a set is labelled twice for any reason, diff the copies and report the discrepancy.
17. **Five learned selectors over the same signals were all null.** New evidence per task (tests from another model, probe-input
    agreement, a stronger visible gate) is the only untested lever; another scorer over the same features is not.
18. **The coupled-writer problem is structural.** With the 35B dropped, the 27B wrote its own tests; decoupling needs a different
    model for every tier. Say which condition of the goal fails by construction instead of hiding it.

## Tooling and process

19. **The auto-mode classifier blocks compound destructive commands** (`pkill -f` chained with `gcloud ... delete`); a single plain
    `gcloud compute tpus tpu-vm delete <our VM>` after copying results went through. `gh pr merge` and `git push` of the branch went
    through when the user had asked for them. Do not work around a denial; split the command, or give the user the exact line.
20. **`gh pr edit` can fail** with a Projects-classic GraphQL deprecation error and print nothing useful; `gh pr merge` may print nothing
    on success. Check `gh pr view N --json state` and fix titles with `gh api -X PATCH repos/OWNER/REPO/pulls/N -f title=... -f body=...`.
21. **Release scripts rot.** `zenodo_new_version.py` still had 3.10.0 hard-coded, so the first draft carried wrong metadata. Set the
    version, release URL, description and file name first, use `--draft ID` to repair a draft, and never publish without an explicit "publish".
22. **Audit before you tag, even when told to release.** The audit took two minutes and found a real slip (the "G read once" claim) plus
    a wrong interval and a ledger wording; ~2 slips per release is normal. The author never audits their own numbers.
23. **Generate tables and macros from committed JSON.** `scripts/paper/tables_phase3.py` writes the phase-3 tables and spend macros; prose
    numbers were then checked against them. No number was typed from memory, which made the audit fast.
24. **Keep a null a null.** Rule (a) was "not met" even where one paired interval was above zero (the 2B); the section says so in
    both directions. Report positive descriptive findings as descriptive.
