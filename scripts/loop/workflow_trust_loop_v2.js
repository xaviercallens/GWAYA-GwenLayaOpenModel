export const meta = {
  name: 'gwaya-python-trust-loop-v2',
  description: 'Closed loop toward 0.80 (stretch 0.90) on prompt-only accept (U) and answer-trust (D) for Python, phase 2 = RL accept policy and generator-RL launch packet; dev only, one sealed E confirmation, independent audit. Offline, no GCP spend.',
  phases: [
    { title: 'Table', detail: 'reuse the built dev and sealed test tables, verify and baseline' },
    { title: 'Levers', detail: 'phase 1: four independent supervised levers on dev only' },
    { title: 'Synthesis', detail: 'combine levers into one frozen supervised policy' },
    { title: 'RL phase', detail: 'phase 2: offline policy-gradient accept policy and generator-RL packet (A27)' },
    { title: 'Sealed test', detail: 'apply the frozen policy once to E (HumanEval+, MBPP+)' },
    { title: 'Audit', detail: 'independent recomputation, provenance and leakage check' },
  ],
}

const REPO = '/home/xavkal/xdev/GWAYA-GwenLayaOpenModel'
const DATA = '/mnt/data/home/xavkal/gwaya-data'
const WORK = DATA + '/trust_loop'
const PY = '/home/xavkal/xdev/AutoevolveAI/.venv/bin/python'

const CTX = `
PROJECT: GWAYA/GwenLaya at ${REPO} (branch feat/assistant-cli). Read docs/ANALYSIS_PLAN_E_TPU.md Addenda A26 (WITH its two dated amendment/correction blocks at the end) and A27 FIRST: they fix the goal and the data discipline. Working target = 0.80 on U, D and E2E (stretch 0.90: always report both); "met with confidence" = point >= 0.80 and Wilson lower bound >= 0.75. Also skim .claude/skills/gwaya-study-workflow/SKILL.md for pitfalls.
RULES (hard):
- Offline only. NO gcloud, no TPU, no spend, no launches, no git commit/push/merge. Do not touch .claude/worktrees or others' files.
- Data root ${DATA}; working outputs under ${WORK}/. New code in ${REPO}/scripts/loop/ with a unique filename per lever (never overwrite another agent's file; existing: build_table.py, baseline_dev.py and earlier lever_* drafts which you may read and reuse).
- Python: system python3 has numpy/sklearn; ${PY} has transformers/peft/datasets/torch.
- DEV table = ${WORK}/dev_table.jsonl (F python + E' python). The SEALED test table ${WORK}/test_table.jsonl (E: HumanEval+, MBPP+; tiers Coder-1.5B, 2B, 4B, 9B) must not be read, opened or scored by anyone except the 'Sealed test' and 'Audit' stages. Choose every threshold/model by cross-validation GROUPED by source problem on dev.
- No feature may be derived from the hidden-check outcome ('correct'/'score' = LABELS only). Allowed: prompt text, candidate text, mean/min log-prob, completion tokens, finish reason, the gate's verdict on the VISIBLE first assert, AST/stub audit, label-free Laya router/calibrator probabilities, cross-tier agreement of candidates (2B/4B/9B exist on both dev and test).
- Honesty: Wilson 95% CIs; say what each number does not show; report negatives plainly; if a target is unreachable say so with evidence.
- Metrics (A26): U = accuracy and balanced accuracy of the PROMPT-ONLY accept decision vs 'tier solves the task'; D = precision of accepted answers at coverage >= 0.50 (report coverage and the precision curve); E2E = precision of tasks passing U and D.
`

const TABLE_SCHEMA = {
  type: 'object',
  properties: {
    dev_table: { type: 'string' }, test_table: { type: 'string' }, builder_script: { type: 'string' },
    columns: { type: 'array', items: { type: 'string' } },
    n_dev_rows: { type: 'number' }, n_test_rows: { type: 'number' },
    tiers_dev: { type: 'array', items: { type: 'string' } }, tiers_test: { type: 'array', items: { type: 'string' } },
    baseline: { type: 'object', additionalProperties: true },
    notes: { type: 'string' },
  },
  required: ['dev_table', 'test_table', 'builder_script', 'columns', 'n_dev_rows', 'n_test_rows', 'tiers_dev', 'tiers_test', 'baseline', 'notes'],
}
const LEVER_SCHEMA = {
  type: 'object',
  properties: {
    lever: { type: 'string' }, script: { type: 'string' }, result_file: { type: 'string' },
    U: { type: 'object', additionalProperties: true },
    D: { type: 'object', additionalProperties: true },
    finding: { type: 'string' },
    reusable_features: { type: 'array', items: { type: 'string' } },
    caveats: { type: 'string' },
  },
  required: ['lever', 'script', 'result_file', 'U', 'D', 'finding', 'caveats'],
}
const SYNTH_SCHEMA = {
  type: 'object',
  properties: {
    policy_script: { type: 'string' }, policy_spec: { type: 'string' },
    dev_cv: { type: 'object', additionalProperties: true },
    targets_met_dev: { type: 'boolean' },
    improvement_points_vs_prev: { type: 'number' },
    dominant_failure_modes: { type: 'array', items: { type: 'string' } },
    next_levers: { type: 'array', items: { type: 'string' } },
    caveats: { type: 'string' },
  },
  required: ['policy_script', 'policy_spec', 'dev_cv', 'targets_met_dev', 'improvement_points_vs_prev', 'dominant_failure_modes', 'caveats'],
}
const RL_SCHEMA = {
  type: 'object',
  properties: {
    script: { type: 'string' }, result_file: { type: 'string' },
    dev_cv: { type: 'object', additionalProperties: true },
    vs_supervised: { type: 'object', additionalProperties: true },
    improves: { type: 'boolean' },
    finding: { type: 'string' },
    change_to_apply: { type: 'string' },
    caveats: { type: 'string' },
  },
  required: ['script', 'result_file', 'dev_cv', 'vs_supervised', 'improves', 'finding', 'caveats'],
}
const PACKET_SCHEMA = {
  type: 'object',
  properties: {
    files: { type: 'array', items: { type: 'string' } },
    dry_run_output: { type: 'string' },
    user_commands: { type: 'array', items: { type: 'string' } },
    est_cost_usd: { type: 'number' },
    risks: { type: 'array', items: { type: 'string' } },
  },
  required: ['files', 'dry_run_output', 'user_commands', 'est_cost_usd', 'risks'],
}
const TEST_SCHEMA = {
  type: 'object',
  properties: {
    result_file: { type: 'string' },
    per_benchmark: { type: 'object', additionalProperties: true },
    U: { type: 'object', additionalProperties: true },
    D: { type: 'object', additionalProperties: true },
    E2E: { type: 'object', additionalProperties: true },
    verdict: { type: 'string' },
    notes: { type: 'string' },
  },
  required: ['result_file', 'U', 'D', 'E2E', 'verdict', 'notes'],
}
const AUDIT_SCHEMA = {
  type: 'object',
  properties: {
    numbers_match: { type: 'boolean' },
    recomputed: { type: 'object', additionalProperties: true },
    leakage_found: { type: 'boolean' },
    leakage_details: { type: 'string' },
    sealed_discipline_ok: { type: 'boolean' },
    provenance_ok: { type: 'boolean' },
    corrections: { type: 'array', items: { type: 'string' } },
    honest_verdict: { type: 'string' },
  },
  required: ['numbers_match', 'leakage_found', 'sealed_discipline_ok', 'provenance_ok', 'corrections', 'honest_verdict'],
}

// ---- Table: reuse --------------------------------------------------------------------------
phase('Table')
const table = await agent(`${CTX}
TASK: the dev and sealed test tables ALREADY EXIST (built earlier by ${REPO}/scripts/loop/build_table.py): ${WORK}/dev_table.jsonl, ${WORK}/test_table.jsonl, summaries ${WORK}/build_summary.json and ${WORK}/baseline_dev.json (script scripts/loop/baseline_dev.py). Do NOT rebuild and do NOT compute any metric on test_table. Verify only structure: read build_summary.json and the first row of dev_table.jsonl (dev is allowed); list the columns, row counts, tiers on dev and test (names only, from build_summary.json), and return the baseline numbers already in baseline_dev.json. In 'notes' flag anything suspicious you see in the summary (missing gate fields, null columns, tier name mismatches).`,
  { label: 'reuse tables + baseline', phase: 'Table', schema: TABLE_SCHEMA })
if (!table) { return { error: 'table stage failed' } }
log('dev rows ' + table.n_dev_rows + ', test rows ' + table.n_test_rows + ' (sealed, not read)')

// ---- Levers (phase 1, supervised) -------------------------------------------------------------
phase('Levers')
const LEVERS = [
  { key: 'D_gate_strength_stub', prompt: `LEVER D1 (downstream, hollow/weak implementations): use gwaya/ast_audit.py (ZeroStubAudit) and gwaya/test_harness.py to detect stubs/hardcoded returns/trivial bodies, count visible asserts, and demote gate passes backed by too few/too weak visible tests (gwaya/gwenlaya.py min_visible_tests, A15.2/A17). Add differential execution in the bwrap sandbox on type-preserving mutations of the visible assert's inputs and flag candidates that crash or return a constant. Report D precision vs coverage with each add-on, grouped CV. A draft exists at scripts/loop/lever_D_gate_strength_stub.py: continue/finish it rather than starting over.` },
  { key: 'D_consensus_agreement', prompt: `LEVER D2 (downstream, agreement): candidates of 2B/4B/9B exist for every task on dev AND test. Execute them on shared inputs (visible assert arguments plus mutations) in the sandbox and compute agreement features (n_agree, majority output equals this candidate, pairwise equality, agreement with the 9B). Test whether agreement beats log-prob or the gate alone as an accept signal. A draft and an execution cache exist (scripts/loop/lever_D_consensus_agreement.py, ${WORK}/lever_D_exec_cache.json): reuse them.` },
  { key: 'U_prompt_only', prompt: `LEVER U1 (upstream, prompt only): predict 'tier T solves the task' for T in {9B, 4B, 2B} from the prompt only: length, structure (docstring examples, I/O spec, constraints), imports, identifiers, TF-IDF + logistic regression / gradient boosting, optional frozen embeddings only if cached offline. Compare with Laya router_p. Report accuracy AND balanced accuracy under grouped CV, the operating point maximising balanced accuracy and the one guaranteeing accepted-precision >= 0.80 (and >= 0.90). State plainly whether 0.80/0.90 balanced accuracy is reachable from the prompt alone. A draft exists at scripts/loop/lever_U_prompt_only.py.` },
  { key: 'D_stacked_calibrated', prompt: `LEVER D3 (downstream, stacked calibrated scorer): regularised logistic regression (optionally shallow boosting) over the ALLOWED features, grouped CV by cluster, isotonic/Platt calibration within folds; threshold by Wilson lower bound so that precision >= 0.80 holds in held-out folds (also report 0.90); coverage at that threshold; full precision-coverage curve; compare with 'gate then log-prob'. This defines the supervised baseline scorer. A draft exists at scripts/loop/lever_D_stacked_calibrated.py.` },
]
const round1 = (await parallel(LEVERS.map(l => () =>
  agent(`${CTX}
Tables: dev ${table.dev_table}; builder ${table.builder_script}; columns ${table.columns.join(', ')}. Dev tiers ${table.tiers_dev.join(', ')}; test tiers ${table.tiers_test.join(', ')} (sealed).
${l.prompt}
Script: ${REPO}/scripts/loop/lever_${l.key}.py (reads dev_table only; deterministic; seed 0). Results: ${WORK}/lever_${l.key}.json. In 'finding' state the number that matters vs 0.80 and 0.90, with CI.`,
    { label: l.key, phase: 'Levers', schema: LEVER_SCHEMA })
))).filter(Boolean)
log('levers returned: ' + round1.map(r => r.lever).join(', '))

// ---- Synthesis (supervised policy) ------------------------------------------------------------
phase('Synthesis')
let synth = await agent(`${CTX}
Lever results (JSON): ${JSON.stringify(round1)}
TASK: combine the levers that helped under grouped CV (ONLY those) into ONE frozen supervised two-stage policy: U (accept/refuse before generation, prompt-only inputs) and D (accept/reject after generation). Implement ${REPO}/scripts/loop/policy.py with decide_prompt(row) and decide_answer(row, peers) and thresholds ${WORK}/policy_thresholds.json chosen on dev by grouped CV only. It must run on the test-table schema (tiers 2B/4B/9B available there). Report dev CV: U accuracy/balanced accuracy, D precision/coverage with Wilson lower bounds, E2E, per tier, each against 0.80 and 0.90. targets_met_dev = all three >= 0.80 with CV margin. Name dominant failure modes (confident-wrong accepted tasks: benchmark/stratum/pattern). Do NOT read the test table. improvement_points_vs_prev = gain over the best single baseline from the Table stage (percentage points).`,
  { label: 'synthesize supervised policy', phase: 'Synthesis', schema: SYNTH_SCHEMA })
if (!synth) { return { error: 'synthesis failed', round1 } }
log('supervised policy: targets_met_dev=' + synth.targets_met_dev + ', gain ' + synth.improvement_points_vs_prev + ' pts')

// ---- RL phase (A27) --------------------------------------------------------------------------
phase('RL phase')
const [rl, packet] = await parallel([
  () => agent(`${CTX}
A27.1: implement the OFFLINE policy-gradient accept policy exactly as in Addendum A27.1 on the dev table: REINFORCE with baseline and entropy bonus, small linear/MLP policy (torch or numpy), reward +1 accepted-correct, -lambda accepted-wrong, 0 refuse with lambda = 4 (sensitivity at 2 and 9), seeds 0-4, grouped CV by cluster, same feature sets as the supervised stage: (a) U = prompt-only features, (b) D = allowed answer-side features. Compare at EQUAL COVERAGE (>= 0.50) against the supervised policy ${synth.policy_script} (dev CV ${JSON.stringify(synth.dev_cv)}) using the same Wilson-bound thresholding, with a paired grouped bootstrap of the precision difference. Script ${REPO}/scripts/loop/rl_accept_policy.py, results ${WORK}/rl_accept_policy.json. If it improves by >= 1 point (CI above zero), give the exact change to apply; do not edit policy.py yourself. Be strict about leakage: lambda and hyperparameters chosen inside folds only.`,
    { label: 'RL accept policy (REINFORCE)', phase: 'RL phase', schema: RL_SCHEMA }),
  () => agent(`${CTX}
A27.2 PACKET (no execution on GPU/TPU, no spend): prepare the generator-RL launch packet for the USER to run. Deliver, under ${REPO}/scripts/rl/ : (1) build_rl_prompts.py that turns pool O (${DATA}/pool_O/tasks_O_train.jsonl, being built; if absent use the P_train python rows ${DATA}/laya_data/tasks_P_train.jsonl) into GRPO prompts with VISIBLE-gate tests only; (2) reward.py: verifiable reward = +1 passes visible gate AND passes stub audit, -lambda wrong, 0 abstain ('CANNOT_SOLVE'), with a unit test file tests/test_rl_reward.py proving that a hard-coded-return stub and a test-special-casing candidate get no positive reward; (3) a launch script scripts/rl/launch_grpo.sh that is dry-run by default and needs --yes-spend and GWAYA_CONFIRM_SPEND=1, targets Qwen3.5-0.8B or 2B QLoRA with TRL GRPO on one L4, appends the spend ledger like scripts/tpu/launch_*.sh, and syncs only under gwenlaya_v4/. Look at scripts/train_lora.py, deploy/stage_launch.py and scripts/tpu/launch_f_gen.sh for the house style. Run the unit tests and the launcher's dry run on the CPU and paste the dry-run output. Do NOT install packages that need the network beyond what exists, do NOT run gcloud. Estimate cost at list price (L4 0.35 USD/h) and list the risks (TRL support for Qwen3.5 hybrid, reward hacking, abstention collapse).`,
    { label: 'generator-RL launch packet', phase: 'RL phase', schema: PACKET_SCHEMA }),
])
if (rl) { log('RL accept policy improves=' + rl.improves + ' :: ' + rl.finding) }
if (packet) { log('RL packet files: ' + packet.files.join(', ') + ' est $' + packet.est_cost_usd) }

if (rl && rl.improves) {
  const s2 = await agent(`${CTX}
Current supervised policy ${synth.policy_script}, dev CV ${JSON.stringify(synth.dev_cv)}.
RL result: ${JSON.stringify(rl)}
TASK: apply the RL accept policy ONLY where it beat the supervised scorer by >= 1 point with CI above zero, re-choose thresholds on dev by grouped CV, rewrite ${REPO}/scripts/loop/policy.py and ${WORK}/policy_thresholds.json, report dev CV as before. After this the policy is FROZEN. Do NOT read the test table.`,
    { label: 'merge RL into policy', phase: 'RL phase', schema: SYNTH_SCHEMA })
  if (s2) { synth = s2 }
} else {
  log('RL accept policy did not improve on the supervised policy: policy stays supervised')
}

// ---- Sealed test --------------------------------------------------------------------------------
phase('Sealed test')
const test = await agent(`${CTX}
The policy is FROZEN: ${synth.policy_script} with thresholds ${WORK}/policy_thresholds.json. You may now read ${table.test_table} EXACTLY ONCE for scoring. Do not modify policy.py, thresholds or any lever script after seeing a result; if something crashes for a reason unrelated to metrics (schema/key error) fix only the loader and say so.
Compute on E for tier 9B (primary) and 4B (secondary), also 2B, per benchmark (HumanEval+, MBPP+) and pooled: U accuracy and balanced accuracy (prompt-only decision vs tier-solves), D precision and coverage with Wilson 95% intervals, E2E precision and coverage, plus the always-accept and gate-only baselines for context. Write ${WORK}/sealed_test_results.json. State against A26 (working 0.80, stretch 0.90): met / met with confidence / not met per metric in plain words. Include n and the base rate of solvable tasks.`,
  { label: 'sealed test scoring', phase: 'Sealed test', schema: TEST_SCHEMA })

// ---- Audit -------------------------------------------------------------------------------------
phase('Audit')
const audit = await agent(`${CTX}
You are the INDEPENDENT AUDITOR (allowed to read the sealed table). Do not trust previous stages.
1) Recompute every headline number of ${WORK}/sealed_test_results.json from RAW rows (${DATA}/etpu/** rows.jsonl for hidden outcomes, gens.jsonl for candidates, tasks ${DATA}/night/results/tasks_E_primary.d25.jsonl) with your own code, applying the frozen ${synth.policy_script} exactly as shipped.
2) Provenance: confirm the 9B rows in the sealed table really are E tasks (HumanEval+/MBPP+) and not E' rows relabelled; find where the E 9B generations came from and how their prompts/protocol compare with the other tiers.
3) Leakage: grep policy and lever scripts for any use of correct/score/hidden tests as a feature, any read of the test table before the Sealed stage (compare file mtimes/atime in ${WORK}), any threshold tuned on E, any dev/test overlap (cluster ids, prompt hashes).
4) Confirm U is truly prompt-only (no candidate/log-prob/gate input).
5) Honest one-paragraph verdict a skeptical reviewer would sign: which targets (0.80 working, 0.90 stretch) are met, which not, the single most important caveat.
Sealed-stage result to audit: ${JSON.stringify(test)}`,
  { label: 'independent audit', phase: 'Audit', schema: AUDIT_SCHEMA })

return { table: { dev_rows: table.n_dev_rows, test_rows: table.n_test_rows, baseline: table.baseline }, round1, synth, rl, packet, test, audit }
