"""Verified SFT / DPO data building for GwenLaya v4 (python, rust, lean4, math).

Shared engine behind scripts/data/build_sft_dataset.py and build_dpo_pairs.py:

* generator ladder: rungs tried in order (cheap first); each rung is an Ollama or an
  OpenAI-compatible endpoint (vLLM / llama.cpp server), or the offline FakeGenerator;
* rejection sampling with fixed seeds 1000*(task_index+1)+call_index (experiments/plan.json);
* every candidate is judged by gwaya.domains.check (VERIFIED / FAILED / UNVERIFIED). Only VERIFIED
  becomes an SFT target. DPO pairs are chosen=VERIFIED vs rejected=FAILED: UNVERIFIED and
  infrastructure failures (missing toolchain, checker error, sandbox/timeout ambiguity) never
  appear as `rejected`;
* decontamination (gwaya.decontaminate) runs on the tasks before sampling and on the final rows
  (prompt + solution) before anything is written; the written files are re-read and re-checked;
* provenance per row: model, seed, checker version, source, license, sha256.

Nothing here downloads data or starts a model: endpoints are only contacted when a non-fake
generator is configured. No measured number is hard-coded; every statistic in the build report
and dataset card is computed from the rows of that run.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import statistics
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from gwaya import decontaminate as dc
from gwaya.domains.task import DOMAINS, CheckResult, Task

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "experiments" / "plan.json"
MANIFEST_PATH = ROOT / "experiments" / "DATA_MANIFEST.json"
DEFAULT_DATA_ROOT = os.path.join(os.environ.get("GWAYA_DATA_ROOT", os.path.expanduser("~/gwaya-data")), "builders")

SFT_SCHEMA = "gwaya.sft/v1"
DPO_SCHEMA = "gwaya.dpo/v1"
CAND_SCHEMA = "gwaya.candidate/v1"

# Pre-registered (docs/GWENLAYA_PREREGISTRATION.md section 5): k=4 (code, math) / 8 (Lean),
# temperature 0.8, top_p 0.95, at most 2 accepted traces per prompt, planning caps (upper bounds).
DEFAULT_K = {"python": 4, "rust": 4, "math": 4, "lean4": 8}
DEFAULT_CAPS = {"python": 6000, "rust": 3000, "lean4": 2000, "math": 6000}
TEMPERATURE = 0.8
TOP_P = 0.95
KEEP_PER_PROMPT = 2
DEFAULT_MAX_TOKENS = {"python": 1024, "rust": 1024, "math": 1024, "lean4": 1024}
RESERVED_BUCKETS = (0, 1, 2)          # sha256(source||id) mod 20: 0=calib, 1-2=router_train
LEAN_MIN_ACCEPTED = 300               # pre-registered Lean rule
MAX_SHALLOW_PAIR_FRACTION = 0.25      # compile/stub failures <= 25 % of DPO pairs

EXCLUDED_SOURCE_MARKERS = ("kodcode", "magpie", "vezora", "goedel-pset", "primeintellect",
                           "gwaya_v2_verifier_curriculum")
BAD_LICENSE_MARKERS = ("-nc", "noncommercial", "non-commercial", "unknown", "other", "none")

# FAILED verdicts that are really infrastructure, not a refuted candidate.
INFRA_REASONS = {"rust_linker_missing", "lean_environment_missing_dependency", "checker_error",
                 "no_tests_in_payload", "zero_tests_executed", "no_assertions_in_tests",
                 "no_reference_answer"}
INFRA_TEXT_MARKERS = ("linker `cc` not found", "no such file or directory", "toolchain",
                      "sandbox", "permission denied", "out of memory", "oomkilled", "cannot allocate",
                      "command not found", "unknown package", "unknown module prefix",
                      "timeout", "timed out")

_FENCE = {"python": "python", "rust": "rust", "lean4": "lean4", "math": ""}
_INSTRUCTION = {
    "python": "Respond with one ```python code block containing the complete solution.",
    "rust": "Respond with one ```rust code block containing the complete solution.",
    "lean4": "Respond with one ```lean4 code block containing the full theorem and its proof. "
             "Do not change the statement and do not use sorry.",
    "math": "Reason briefly and put the final answer in \\boxed{}.",
}


def sha256_text(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def split_bucket(source: str, task_id: str) -> int:
    """sha256(source||id) mod 20 (experiments/plan.json seeds.split_hash)."""
    return int(hashlib.sha256((source + task_id).encode("utf-8")).hexdigest(), 16) % 20


def checker_version(fake: bool = False) -> str:
    """Content hash of the checker code (domains + oracles). Not a release number."""
    if fake:
        return "fake-dryrun-checker"
    files = [ROOT / "gwaya" / "domains" / n for n in ("checkers.py", "math_check.py", "task.py")]
    files.append(ROOT / "gwaya" / "oracles.py")
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes() if f.exists() else b"missing")
    return "gwaya.domains+oracles@" + h.hexdigest()[:16]


# ---------------------------------------------------------------------------------------------
# Generators


class GeneratorUnavailable(RuntimeError):
    """The endpoint could not be reached / returned an error: an infra failure, never a FAILED
    candidate."""


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None, timeout: float
               ) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310 - operator URL
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise GeneratorUnavailable(f"{url}: {type(exc).__name__}: {exc}") from exc


@dataclass
class Generator:
    """Base: `generate` returns the assistant text for one (prompt, seed)."""
    model: str
    kind: str = "base"

    @property
    def rung(self) -> str:
        return f"{self.kind}:{self.model}"

    def generate(self, prompt: str, *, seed: int, temperature: float, top_p: float,
                 max_tokens: int) -> str:
        raise NotImplementedError


@dataclass
class OllamaGenerator(Generator):
    base_url: str = "http://localhost:11434"
    timeout_s: float = 600.0
    kind: str = "ollama"

    def generate(self, prompt, *, seed, temperature, top_p, max_tokens):
        out = _post_json(f"{self.base_url.rstrip('/')}/api/chat", {
            "model": self.model, "stream": False, "think": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": temperature, "top_p": top_p, "seed": seed,
                        "num_predict": max_tokens}}, None, self.timeout_s)
        try:
            return str(out["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise GeneratorUnavailable(f"unexpected Ollama reply keys: {list(out)[:5]}") from exc


@dataclass
class OpenAICompatGenerator(Generator):
    """vLLM / llama.cpp server / any /v1/chat/completions endpoint. `base_url` is the API root
    (with or without a trailing /v1). Key (if any) is read from GWAYA_OPENAI_API_KEY."""
    base_url: str = "http://localhost:8000/v1"
    timeout_s: float = 600.0
    kind: str = "openai"

    def generate(self, prompt, *, seed, temperature, top_p, max_tokens):
        root = self.base_url.rstrip("/")
        if not root.endswith("/v1"):
            root += "/v1"
        key = os.environ.get("GWAYA_OPENAI_API_KEY")
        out = _post_json(f"{root}/chat/completions", {
            "model": self.model, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "top_p": top_p, "seed": seed, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}},
            {"Authorization": f"Bearer {key}"} if key else None, self.timeout_s)
        try:
            return str(out["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise GeneratorUnavailable(f"unexpected OpenAI-compatible reply keys: {list(out)[:5]}") from exc


@dataclass
class FakeGenerator(Generator):
    """Offline generator for --dry-run and tests. `responses` maps a key (a substring of the
    prompt, longest match wins) to a list indexed by call_index = seed % 1000 (clamped)."""
    responses: dict[str, list[str]] = field(default_factory=dict)
    default: str = "no answer"
    kind: str = "fake"

    def generate(self, prompt, *, seed, temperature, top_p, max_tokens):
        keys = [k for k in self.responses if k in prompt]
        if any(k.startswith("Source task id:") for k in keys):    # translation prompts embed the task
            keys = [k for k in keys if k.startswith("Source task id:")]
        if not keys:
            return self.default
        lst = self.responses[max(keys, key=len)]
        return lst[min(seed % 1000, len(lst) - 1)]


def parse_rung(spec: str) -> Generator:
    """`ollama@http://host:11434#model` or `openai@http://host:8000/v1#model` (model may contain
    colons/slashes). `fake@#name` gives an empty FakeGenerator."""
    kind, sep, rest = spec.partition("@")
    url, sep2, model = rest.rpartition("#")
    if not sep or not sep2 or not model:
        raise ValueError(f"bad rung {spec!r}; expected kind@url#model")
    if kind == "ollama":
        return OllamaGenerator(model=model, base_url=url or "http://localhost:11434")
    if kind == "openai":
        return OpenAICompatGenerator(model=model, base_url=url or "http://localhost:8000/v1")
    if kind == "fake":
        return FakeGenerator(model=model)
    raise ValueError(f"unknown rung kind {kind!r} in {spec!r}")


# ---------------------------------------------------------------------------------------------
# Tasks


@dataclass
class BuildTask:
    task: Task
    source: str
    license: str
    split: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)   # translated_from, fake_responses, ...

    @property
    def key(self) -> str:
        return f"{self.source}::{self.task.task_id}"


def plan_licenses(plan_path: Path = PLAN_PATH) -> dict[str, str]:
    if not plan_path.exists():
        return {}
    return {d["hf_id"]: d.get("license", "") for d in json.loads(plan_path.read_text()).get("datasets", [])}


def load_build_tasks(path: str | Path, licenses: dict[str, str] | None = None) -> list[BuildTask]:
    """JSONL rows {domain, task_id, prompt, checker_payload, source, license?, split?, ...}."""
    licenses = licenses or {}
    out: list[BuildTask] = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        source = r.get("source") or r["task_id"].split("/", 1)[0]
        out.append(BuildTask(
            Task(r["domain"], r["task_id"], r["prompt"], r.get("checker_payload") or {}),
            source=source, license=r.get("license") or licenses.get(source, ""), split=r.get("split"),
            extra={k: v for k, v in r.items() if k not in
                   {"domain", "task_id", "prompt", "checker_payload", "source", "license", "split"}}))
    return out


def license_problem(bt: BuildTask) -> str | None:
    s = bt.source.lower()
    if any(m in s for m in EXCLUDED_SOURCE_MARKERS):
        return "excluded_source"
    lic = (bt.license or "").strip().lower()
    if not lic:
        return "no_license"
    if lic != "derived" and any(m in lic for m in BAD_LICENSE_MARKERS):
        return "license_not_permitted"
    return None


def generation_prompt(task: Task) -> str:
    return f"{task.prompt.rstrip()}\n\n{_INSTRUCTION[task.domain]}"


def extract_code(response: str, domain: str) -> str:
    if domain == "math":
        return response.strip()
    from gwaya.low_tier_engine import extract_code_block
    return extract_code_block(response, _FENCE[domain] or "python")


# ---------------------------------------------------------------------------------------------
# Decontamination gate


class DecontamAborted(RuntimeError):
    pass


def _record(bt_or_none: BuildTask | None, domain: str, source: str, task_id: str, prompt: str,
            payload: dict[str, Any], solution: str = "") -> dict[str, Any]:
    fields: dict[str, str] = {"prompt": prompt}
    if payload.get("tests"):
        fields["tests"] = str(payload["tests"])
    if payload.get("formal_statement"):
        fields["statement"] = str(payload["formal_statement"])
    if solution:
        fields["solution"] = solution
    return {"dataset": source, "item_id": task_id, "domain": domain, "fields": fields}


class DecontamGate:
    """Streams build items through gwaya.decontaminate against the eval index. Fails closed."""

    def __init__(self, dec: dc.Decontaminator, cfg: dc.Config, report: dict[str, Any]) -> None:
        self.dec, self.cfg, self.report = dec, cfg, report
        self.flagged: list[dict[str, Any]] = []

    @classmethod
    def from_index(cls, index_path: str | Path, *, allow_missing: bool = False,
                   check_plan: bool = True, cfg: dc.Config | None = None,
                   manifest_path: Path | None = MANIFEST_PATH, plan_path: Path = PLAN_PATH
                   ) -> "DecontamGate":
        cfg = cfg or dc.Config()
        manifest = json.loads(manifest_path.read_text()) if manifest_path and manifest_path.exists() else None
        plan_ds = json.loads(plan_path.read_text()).get("datasets", []) if plan_path.exists() else []
        sources = dc.sources_from_jsonl(index_path, manifest, {d["hf_id"]: d["domain"] for d in plan_ds})
        items, status = dc.load_eval(sources)
        if check_plan:
            indexed = {s.hf_id for s in sources}
            for hf in sorted({d["hf_id"] for d in plan_ds if d.get("role") == "eval"}):
                if hf not in indexed:
                    status[hf] = {"hf_id": hf, "status": "not_in_index", "n_items": 0}
        missing = sorted(n for n, st in status.items() if st["status"] != "loaded")
        report = {"index_path": str(index_path), "index_sha256": dc.files_sha256([Path(index_path)]),
                  "eval_status": status, "missing_eval_sets": missing, "complete": not missing,
                  "plan_eval_check": bool(check_plan), "config": cfg.public()}
        if missing and not allow_missing:
            raise DecontamAborted(f"eval sets not loaded: {missing} (pass allow_missing to proceed "
                                  "with an INCOMPLETE decontamination, recorded in the report)")
        return cls(dc.Decontaminator(items, cfg), cfg, report)

    def _hits(self, rec: dict[str, Any], hf_id: str, dedup: bool) -> list[dc.Hit]:
        item = dc.item_from_record(rec)
        hits = self.dec.check(item, hf_id)
        if not hits and dedup and hf_id in self.cfg.dedup_group:
            hits = self.dec.dedup_check(item)
        return hits

    def check_task(self, bt: BuildTask) -> list[dict[str, str]]:
        rec = _record(bt, bt.task.domain, bt.source, bt.task.task_id, bt.task.prompt, bt.task.checker_payload)
        hits = dc.hits_to_json(self._hits(rec, bt.source, dedup=True))
        if hits:
            self.flagged.append({"stage": "task", "key": bt.key, "hits": hits})
        return hits

    def check_solution(self, bt: BuildTask, response: str) -> list[dict[str, str]]:
        rec = _record(bt, bt.task.domain, bt.source, bt.task.task_id, bt.task.prompt,
                      bt.task.checker_payload, extract_code(response, bt.task.domain))
        hits = dc.hits_to_json(self._hits(rec, bt.source, dedup=False))
        if hits:
            self.flagged.append({"stage": "solution", "key": bt.key, "hits": hits})
        return hits

    def verify_rows(self, rows: Iterable[tuple[str, str, str, str, str]]) -> dict[str, Any]:
        """Independent pass over what was WRITTEN: rows are (source, domain, task_id, prompt,
        response) re-read from disk; any hit => ok=False."""
        n, bad = 0, []
        for source, domain, tid, prompt, resp in rows:
            n += 1
            rec = _record(None, domain, source, tid, prompt, {}, extract_code(resp, domain))
            hits = self.dec.check(dc.item_from_record(rec), source)
            if hits:
                bad.append({"key": f"{source}::{tid}", "hits": dc.hits_to_json(hits)[:3]})
        return {"ok": not bad, "n_rechecked": n, "n_hits": len(bad), "examples": bad[:5]}


# ---------------------------------------------------------------------------------------------
# Checking, auditing, failure classification


def audit_flags(domain: str, code: str, prompt: str = "") -> list[str]:
    """AST stub audit (+ grounding for python). Empty list = clean."""
    from gwaya.ast_audit import ZeroStubAudit
    flags: list[str] = []
    if domain == "python":
        res = ZeroStubAudit.audit_python_code(code)
        flags += list(res.violations)
        from gwaya.grounding import grounding_flags
        flags += [f"grounding:{f}" for f in grounding_flags(code, prompt)]
    elif domain == "rust":
        flags += list(ZeroStubAudit.audit_rust_code(code).violations)
    elif domain == "lean4":
        flags += list(ZeroStubAudit.audit_lean_code(code).violations)
    return flags


def is_infra(res: CheckResult) -> bool:
    """True when the verdict says nothing reliable about the candidate itself."""
    if res.status == "UNVERIFIED":
        return True
    ev = res.evidence or {}
    if ev.get("reason") in INFRA_REASONS:
        return True
    text = f"{ev.get('error', '')} {json.dumps(ev.get('details', {}), default=str)}".lower()
    return any(m in text for m in INFRA_TEXT_MARKERS)


def failure_category(domain: str, code: str, res: CheckResult, flags: Sequence[str]) -> str:
    """empty/compile/stub are 'shallow'; others 'wrong' (full check refuted the candidate)."""
    ev = res.evidence or {}
    if ev.get("reason") == "empty_response" or not code.strip():
        return "empty"
    if ev.get("reason") == "statement_not_preserved":
        return "wrong"
    err = str(ev.get("error", ""))
    if domain == "python":
        try:
            ast.parse(code)
        except SyntaxError:
            return "compile"
    if domain == "rust" and ("error[E" in err or "could not compile" in err or "aborting due to" in err):
        return "compile"
    if domain == "lean4" and re.search(r"error:\s*(unknown|unexpected|expected)", err):
        return "compile"
    if flags:
        return "stub"
    return "wrong"


SHALLOW = {"empty", "compile", "stub"}


def baseline_response(task: Task) -> str:
    """A trivial answer the checker must NOT verify, or the task's tests prove nothing."""
    if task.domain == "python":
        ep = task.checker_payload.get("entry_point")
        return (f"```python\ndef {ep}(*a, **k):\n    return None\n```" if ep
                else "```python\nx = None\n```")
    if task.domain == "rust":
        return "```rust\nfn _gwaya_baseline() {}\n```"
    if task.domain == "lean4":
        stmt = task.checker_payload.get("formal_statement", "theorem t : True := by")
        return f"```lean4\n{stmt}\n  sorry\n```"
    return "\\boxed{0}"


# ---------------------------------------------------------------------------------------------
# Sampling


@dataclass
class Candidate:
    task_key: str
    task_id: str
    domain: str
    source: str
    license: str
    prompt: str
    response: str
    rung: str
    model: str
    endpoint_kind: str
    seed: int
    call_index: int
    round: int
    status: str                      # VERIFIED | FAILED | UNVERIFIED
    infra: bool
    reason: str
    error: str
    audit: list[str]
    failure_category: str | None
    gate_pass: bool | None
    norm_hash: str
    schema: str = CAND_SCHEMA


def _norm_hash(domain: str, code: str) -> str:
    norm = None
    if domain == "python":
        norm = dc.python_normalized_source(code)
    if norm is None:
        norm = re.sub(r"\s+", " ", code).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


@dataclass
class SampleConfig:
    k: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_K))
    keep: int = KEEP_PER_PROMPT
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    max_tokens: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_MAX_TOKENS))
    need_failure: bool = False       # DPO: keep sampling until a FAILED sample exists too
    workers: int = 1


class TaskState:
    def __init__(self, index: int, bt: BuildTask) -> None:
        self.index, self.bt = index, bt
        self.cands: list[Candidate] = []
        self.calls = 0
        self.gen_errors: list[str] = []

    @property
    def verified(self) -> list[Candidate]:
        seen: set[str] = set()      # distinct normalized solutions only (duplicates do not count)
        out = []
        for c in self.cands:
            if c.status == "VERIFIED" and not c.audit and c.norm_hash not in seen:
                seen.add(c.norm_hash)
                out.append(c)
        return out

    def done(self, cfg: SampleConfig) -> bool:
        if len(self.verified) < cfg.keep:
            return False
        if cfg.need_failure and not any(c.status == "FAILED" and not c.infra for c in self.cands):
            return False
        return True


CheckFn = Callable[[Task, str], CheckResult]


def _judge(bt: BuildTask, gen: Generator, response: str, seed: int, call_index: int, rnd: int,
           check_fn: CheckFn) -> Candidate:
    task = bt.task
    code = extract_code(response, task.domain)
    res = check_fn(task, response)
    flags = audit_flags(task.domain, code, task.prompt) if code.strip() else []
    infra = is_infra(res) if res.status != "VERIFIED" else False
    cat = gate = None
    if res.status == "FAILED" and not infra:
        cat = failure_category(task.domain, code, res, flags)
        pub = task.checker_payload.get("public_tests")
        if pub and cat not in SHALLOW:
            gate = check_fn(Task(task.domain, task.task_id, task.prompt, {**task.checker_payload, "tests": pub}),
                            response).status == "VERIFIED"
            if gate:
                cat = "confident_wrong"
    ev = res.evidence or {}
    return Candidate(task_key=bt.key, task_id=task.task_id, domain=task.domain, source=bt.source,
                     license=bt.license, prompt=task.prompt, response=response, rung=gen.rung,
                     model=gen.model, endpoint_kind=gen.kind, seed=seed, call_index=call_index,
                     round=rnd, status=res.status, infra=infra, reason=str(ev.get("reason", "")),
                     error=str(ev.get("error", ""))[:300], audit=flags, failure_category=cat,
                     gate_pass=gate, norm_hash=_norm_hash(task.domain, code))


def sample_stage(states: Sequence[TaskState], ladder: Sequence[Generator], cfg: SampleConfig,
                 check_fn: CheckFn, rnd: int = 1) -> None:
    """Walk the ladder for every state that is not done; escalate to the next rung only for tasks
    still short of `keep` verified samples. Unreachable endpoints are recorded, not turned into
    candidates (infra failures are never negative training signal)."""

    def work(st: TaskState) -> None:
        task = st.bt.task
        for gen in ladder:
            for _ in range(cfg.k[task.domain]):
                if st.done(cfg):
                    return
                seed = 1000 * (st.index + 1) + st.calls
                call = st.calls
                st.calls += 1
                try:
                    resp = gen.generate(generation_prompt(task), seed=seed, temperature=cfg.temperature,
                                        top_p=cfg.top_p, max_tokens=cfg.max_tokens[task.domain])
                except GeneratorUnavailable as exc:
                    st.gen_errors.append(f"{gen.rung}: {exc}"[:300])
                    break          # this rung is down for this task; try the next rung
                st.cands.append(_judge(st.bt, gen, resp, seed, call, rnd, check_fn))

    todo = [s for s in states if not s.done(cfg)]
    if cfg.workers > 1:
        with ThreadPoolExecutor(cfg.workers) as ex:
            list(ex.map(work, todo))
    else:
        for s in todo:
            work(s)


def fake_check(task: Task, response: str) -> CheckResult:
    """Deterministic stand-in checker for --dry-run / tests: verdict from fixture markers."""
    if response.strip() == baseline_response(task).strip():
        return CheckResult("VERIFIED" if task.checker_payload.get("fixture_trivial") else "FAILED",
                           {"reason": "fixture_baseline"})
    if "FIXTURE_OK" in response:
        return CheckResult("VERIFIED", {"reason": "fixture"})
    if "FIXTURE_UNVERIFIED" in response:
        return CheckResult("UNVERIFIED", {"reason": "fixture"})
    if "FIXTURE_INFRA" in response:
        return CheckResult("FAILED", {"reason": "rust_linker_missing", "error": "linker `cc` not found"})
    if "FIXTURE_TIMEOUT" in response:
        return CheckResult("FAILED", {"error": "timeout after 5s"})
    return CheckResult("FAILED", {"reason": "fixture_wrong", "error": "assertion failed"})


# ---------------------------------------------------------------------------------------------
# Rust via verified translation of Python tasks

_RUST_FN = re.compile(r"^\s*(?:pub\s+)?fn\s+\w+[^{;]*", re.M)
_LIT = re.compile(r"-?\d+(?:\.\d+)?|\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'")


def translation_prompt(row: dict[str, Any]) -> str:
    return (f"Source task id: {row['task_id']}\n"
            "Translate this verified Python task into Rust. Give (1) a ```rust block with the "
            "function(s) only (no main) and (2) a ```rust_tests block of plain assert_eq!/assert! "
            "statements that call the function, covering the same cases as the Python tests.\n\n"
            f"Task:\n{row['prompt']}\n\nPython solution:\n```python\n{row['solution']}\n```\n\n"
            f"Python tests:\n```python\n{row['tests']}\n```")


def parse_translation(text: str) -> tuple[str, str] | None:
    blocks = dict((tag, body) for tag, body in
                  re.findall(r"```(rust_tests|rust)\s*\n(.*?)\n```", text, flags=re.S))
    if "rust" in blocks and "rust_tests" in blocks:
        return blocks["rust"].strip(), blocks["rust_tests"].strip()
    return None


def literal_overlap(py_tests: str, rust_tests: str) -> float:
    """Fraction of literals in the Rust asserts that also occur in the Python tests. A heuristic
    guard against hallucinated expectations; it does not prove semantic equivalence."""
    rl = [x.strip("'\"") for x in _LIT.findall(rust_tests)]
    pl = {x.strip("'\"") for x in _LIT.findall(py_tests)}
    return 1.0 if not rl else sum(1 for x in rl if x in pl) / len(rl)


def translate_python_rows(py_rows: Sequence[dict[str, Any]], translator: Generator,
                          check_fn: CheckFn, *, min_overlap: float = 0.8, seed_base: int = 0
                          ) -> tuple[list[BuildTask], dict[str, int]]:
    """py_rows: {task_id, source, license, prompt, solution, tests} from VERIFIED python SFT data.
    A translation becomes a Rust task only if (1) the translated reference solution is VERIFIED by
    the Rust checker on the translated tests, (2) those tests reject a trivial baseline, (3) the
    Rust test literals overlap the Python test literals (heuristic) and (4) the Rust tests contain
    an assertion. The Python parent was already VERIFIED against its own tests."""
    stats = {"attempted": 0, "unparseable": 0, "no_signature": 0, "low_literal_overlap": 0,
             "rust_not_verified": 0, "baseline_not_rejected": 0, "accepted": 0, "generator_down": 0}
    out: list[BuildTask] = []
    for i, row in enumerate(py_rows):
        stats["attempted"] += 1
        try:
            text = translator.generate(translation_prompt(row), seed=seed_base + 1000 * (i + 1),
                                       temperature=0.2, top_p=TOP_P, max_tokens=DEFAULT_MAX_TOKENS["rust"])
        except GeneratorUnavailable:
            stats["generator_down"] += 1
            continue
        parsed = parse_translation(text)
        if not parsed:
            stats["unparseable"] += 1
            continue
        sol, tests = parsed
        sig = _RUST_FN.search(sol)
        if not sig:
            stats["no_signature"] += 1
            continue
        if literal_overlap(row["tests"], tests) < min_overlap:
            stats["low_literal_overlap"] += 1
            continue
        tid = f"translated/{row['task_id']}"
        prompt = ("Write a Rust function that solves the following task (translated from Python). "
                  "The tests call it directly.\n\n"
                  f"{row['prompt'].rstrip()}\n\nUse exactly this signature: `{sig.group(0).strip()}`")
        task = Task("rust", tid, prompt, {"tests": tests})
        if check_fn(task, f"```rust\n{sol}\n```").status != "VERIFIED":
            stats["rust_not_verified"] += 1
            continue
        if check_fn(task, baseline_response(task)).status == "VERIFIED":
            stats["baseline_not_rejected"] += 1
            continue
        stats["accepted"] += 1
        out.append(BuildTask(task, source=f"translated:{row['source']}", license=row.get("license") or "",
                             extra={"translated_from": row["task_id"], "translator": translator.rung, "reference_solution": sol,
                                    "literal_overlap": round(literal_overlap(row["tests"], tests), 4)}))
    return out, stats


# ---------------------------------------------------------------------------------------------
# Pipeline


@dataclass
class BuildConfig:
    sample: SampleConfig = field(default_factory=SampleConfig)
    ladder: Sequence[Generator] = ()
    lean_rounds: Sequence[Sequence[Generator]] = ()    # per expert-iteration round; empty => [ladder]
    caps: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_CAPS))
    domains: Sequence[str] = DOMAINS
    check_fn: CheckFn | None = None
    checker_ver: str = ""
    gate: DecontamGate | None = None
    dry_run: bool = False


def prefilter(tasks: Sequence[BuildTask], cfg: BuildConfig, drops: dict[str, list[str]]
              ) -> list[BuildTask]:
    """License, split, reserved-slice, decontamination and trivial-baseline filters (no GPU)."""
    kept: list[BuildTask] = []
    for bt in sorted(tasks, key=lambda t: sha256_text(t.source, t.task.task_id)):
        why = None
        if bt.task.domain not in cfg.domains:
            continue
        elif (p := license_problem(bt)):
            why = p
        elif bt.split and "test" in bt.split.lower():
            why = "test_split"
        elif split_bucket(bt.source, bt.task.task_id) in RESERVED_BUCKETS:
            why = "reserved_calib_or_router_slice"
        elif cfg.gate is None:
            raise DecontamAborted("no decontamination gate configured")
        elif cfg.gate.check_task(bt):
            why = "decontam_task"
        elif cfg.check_fn(bt.task, baseline_response(bt.task)).status == "VERIFIED":
            why = "tests_accept_trivial_baseline"
        if why:
            drops.setdefault(why, []).append(bt.key)
        else:
            kept.append(bt)
    return kept


def run_sampling(tasks: Sequence[BuildTask], cfg: BuildConfig) -> tuple[list[TaskState], dict[str, Any]]:
    """Rejection sampling with the cap per domain; Lean via expert-iteration rounds."""
    states = [TaskState(i, bt) for i, bt in enumerate(tasks)]
    check_fn = cfg.check_fn
    per_domain = {d: [s for s in states if s.bt.task.domain == d] for d in cfg.domains}
    info: dict[str, Any] = {"rounds": {}}
    batch = max(1, cfg.sample.workers) * 4
    for d, sts in per_domain.items():
        rounds = [list(r) for r in cfg.lean_rounds] if d == "lean4" and cfg.lean_rounds else [list(cfg.ladder)]
        for r, ladder in enumerate(rounds, 1):
            todo = [s for s in sts if not s.verified]          # expert iteration: unsolved only
            for i in range(0, len(todo), batch):
                if sum(1 for s in sts if s.verified) >= cfg.caps.get(d, 10**9):
                    break
                sample_stage(todo[i:i + batch], ladder, cfg.sample, check_fn, rnd=r)
            info["rounds"][f"{d}:{r}"] = {"solved_total": sum(1 for s in sts if s.verified),
                                          "tasks": len(sts), "rungs": [g.rung for g in ladder]}
    return states, info


def _dedup_pick(cands: Sequence[Candidate], n: int) -> list[Candidate]:
    seen, out = set(), []
    for c in sorted(cands, key=lambda c: sha256_text(c.norm_hash, str(c.seed))):  # no length bias
        if c.norm_hash not in seen:
            seen.add(c.norm_hash)
            out.append(c)
        if len(out) >= n:
            break
    return out


def _prov(c: Candidate, bt: BuildTask, ver: str, dry: bool) -> dict[str, Any]:
    p = {"model": c.model, "rung": c.rung, "endpoint_kind": c.endpoint_kind, "seed": c.seed,
         "call_index": c.call_index, "round": c.round, "temperature": TEMPERATURE, "top_p": TOP_P,
         "checker_version": ver, "checker_status": c.status, "source": bt.source, "license": bt.license,
         "sha256": sha256_text(c.prompt, c.response), "prompt_sha256": sha256_text(c.prompt),
         "dry_run": dry}
    if c.status == "FAILED":
        p["failure_category"] = c.failure_category
    p.update({k: v for k, v in bt.extra.items() if k in ("translated_from", "translator", "literal_overlap")})
    return p


def select_sft(states: Sequence[TaskState], cfg: BuildConfig, drops: dict[str, list[str]]
               ) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_domain: dict[str, int] = {}
    for st in states:
        d = st.bt.task.domain
        if per_domain.get(d, 0) >= cfg.caps.get(d, 10**9):
            drops.setdefault("over_domain_cap", []).append(st.bt.key)
            continue
        picks = _dedup_pick(st.verified, cfg.sample.keep)
        if not picks:
            drops.setdefault("no_verified_candidate", []).append(st.bt.key)
            continue
        for c in picks:
            if cfg.gate.check_solution(st.bt, c.response):
                drops.setdefault("decontam_solution", []).append(st.bt.key)
                continue
            if per_domain.get(d, 0) >= cfg.caps.get(d, 10**9):
                break
            per_domain[d] = per_domain.get(d, 0) + 1
            gp = generation_prompt(st.bt.task)
            rows.append({"schema": SFT_SCHEMA, "id": sha256_text(st.bt.key, c.response)[:32],
                         "domain": d, "task_id": st.bt.task.task_id, "source": st.bt.source,
                         "license": st.bt.license,
                         "messages": [{"role": "user", "content": gp},
                                      {"role": "assistant", "content": c.response}],
                         "provenance": _prov(c, st.bt, cfg.checker_ver, cfg.dry_run)})
    return rows


def length_stats(ratios: Sequence[float]) -> dict[str, Any]:
    if not ratios:
        return {"n": 0}
    s = sorted(ratios)
    return {"n": len(s), "definition": "len(chosen)/len(rejected) in characters (no tokenizer)",
            "mean": round(statistics.fmean(s), 4), "median": round(statistics.median(s), 4),
            "min": round(s[0], 4), "p90": round(s[min(len(s) - 1, int(0.9 * len(s)))], 4),
            "max": round(s[-1], 4), "frac_chosen_longer": round(sum(1 for x in s if x > 1) / len(s), 4)}


def build_pairs(cands_by_task: dict[str, list[Candidate]], tasks: dict[str, BuildTask],
                cfg: BuildConfig, drops: dict[str, list[str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One pair per prompt: chosen VERIFIED+audit-clean, rejected FAILED (never UNVERIFIED/infra),
    preferring confident_wrong > wrong > shallow; shallow pairs capped at 25 %."""
    pairs: list[dict[str, Any]] = []
    excl = {"unverified_or_infra_ignored": 0}
    for key, cs in cands_by_task.items():
        bt = tasks.get(key)
        excl["unverified_or_infra_ignored"] += sum(1 for c in cs if c.status != "VERIFIED" and (c.infra or c.status == "UNVERIFIED"))
        good = [c for c in cs if c.status == "VERIFIED" and not c.audit]
        bad = [c for c in cs if c.status == "FAILED" and not c.infra]
        if not good or not bad:
            drops.setdefault("no_pair_possible", []).append(key)
            continue
        rank = {"confident_wrong": 0, "wrong": 1}
        chosen = min(good, key=lambda c: sha256_text("c", c.norm_hash))
        rejected = min(bad, key=lambda c: (rank.get(c.failure_category or "", 2), sha256_text("r", c.norm_hash, str(c.seed))))
        if bt is None:     # candidates cache without the task file: rebuild what provenance needs
            bt = BuildTask(Task(chosen.domain, chosen.task_id, chosen.prompt), chosen.source, chosen.license)
        if cfg.gate.check_solution(bt, chosen.response) or cfg.gate.check_solution(bt, rejected.response):
            drops.setdefault("decontam_solution", []).append(key)
            continue
        gp = generation_prompt(Task(chosen.domain, chosen.task_id, chosen.prompt))
        ratio = len(chosen.response) / max(1, len(rejected.response))
        pairs.append({"schema": DPO_SCHEMA, "id": sha256_text(key, chosen.response, rejected.response)[:32],
                      "domain": chosen.domain, "task_id": chosen.task_id, "source": chosen.source,
                      "license": chosen.license,
                      "prompt": [{"role": "user", "content": gp}],
                      "chosen": [{"role": "assistant", "content": chosen.response}],
                      "rejected": [{"role": "assistant", "content": rejected.response}],
                      "length": {"chosen_chars": len(chosen.response), "rejected_chars": len(rejected.response),
                                 "ratio": round(ratio, 4)},
                      "provenance": {"chosen": _prov(chosen, bt, cfg.checker_ver, cfg.dry_run),
                                     "rejected": _prov(rejected, bt, cfg.checker_ver, cfg.dry_run),
                                     "rejected_category": rejected.failure_category}})
    shallow = [p for p in pairs if p["provenance"]["rejected_category"] in SHALLOW]
    deep = [p for p in pairs if p["provenance"]["rejected_category"] not in SHALLOW]
    allowed = int(len(deep) * MAX_SHALLOW_PAIR_FRACTION / (1 - MAX_SHALLOW_PAIR_FRACTION))
    shallow.sort(key=lambda p: p["id"])
    for p in shallow[allowed:]:
        drops.setdefault("shallow_pair_over_25pct_cap", []).append(f"{p['source']}::{p['task_id']}")
    pairs = deep + shallow[:allowed]
    pairs.sort(key=lambda p: p["id"])
    return pairs, excl


# ---------------------------------------------------------------------------------------------
# IO


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_written_sft(gate: DecontamGate, path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    return gate.verify_rows((r["source"], r["domain"], r["task_id"], r["messages"][0]["content"],
                             r["messages"][1]["content"]) for r in rows)


def verify_written_dpo(gate: DecontamGate, path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    return gate.verify_rows((r["source"], r["domain"], r["task_id"], r["prompt"][0]["content"],
                             r["chosen"][0]["content"] + "\n" + r["rejected"][0]["content"]) for r in rows)


def provenance_complete(row_prov: dict[str, Any]) -> bool:
    need = ("model", "seed", "checker_version", "source", "license", "sha256")
    return all(row_prov.get(k) not in (None, "") for k in need)


# ---------------------------------------------------------------------------------------------
# Dataset card


def _j(x: Any, **kw: Any) -> str:
    return json.dumps(x, sort_keys=True, **kw)


def render_card(kind: str, report: dict[str, Any]) -> str:
    """Markdown dataset card from a build report. Every number comes from `report`."""
    dry = report.get("dry_run")
    dcm = report.get("decontamination", {})
    lines = [f"# GwenLaya v4 {kind.upper()} dataset", ""]
    if dry:
        lines += ["> **DRY RUN.** Built from toy fixtures with a fake generator and a fake checker. "
                  "These rows are not real model outputs and must not be trained on.", ""]
    lines += ["## Composition", "", "| Domain | Tasks considered | Tasks with output | Rows |", "|---|---|---|---|"]
    for d, c in sorted(report.get("per_domain", {}).items()):
        lines.append(f"| {d} | {c.get('tasks_sampled', 'TBD')} | {c.get('tasks_with_output', 'TBD')} | {c.get('rows', 'TBD')} |")
    lines += ["", f"Total rows: {report.get('n_rows', 'TBD')}  ", f"Output sha256: `{report.get('output_sha256', 'TBD')}`", "",
              "## Sources and licenses", ""]
    for s, lic in sorted(report.get("sources", {}).items()):
        lines.append(f"- `{s}`: {lic or 'MISSING'}")
    g = report.get("generation", {})
    lines += ["", "## Generation", "",
              f"- Ladder (cheap to strong): {', '.join(g.get('rungs', [])) or 'TBD'}",
              f"- k per domain: {_j(g.get('k', {}))}; temperature {g.get('temperature')}, top_p {g.get('top_p')}; "
              "seeds 1000*(task_index+1)+call_index",
              f"- Thinking disabled; max new tokens {_j(g.get('max_tokens', {}))}",
              "", "## Verification", "",
              f"- Checker: `{report.get('checker_version')}` (content hash of the checker code, not a release)",
              "- Keep rule: checker VERIFIED, AST stub audit clean, grounding clean (python), deduplicated by "
              f"normalized hash, at most {KEEP_PER_PROMPT} per prompt, task tests reject a trivial baseline.",
              "- UNVERIFIED and infrastructure failures are never labels."]
    if kind == "dpo":
        lines += ["- Pair = same prompt, chosen VERIFIED vs rejected FAILED; one pair per prompt; "
                  "compile/stub/empty rejections capped at 25 % of pairs.",
                  f"- Rejected categories: {_j(report.get('rejected_categories', {}))}",
                  f"- Length ratio: {_j(report.get('length_ratio', {}))}"]
    rb = report.get("rust_translation")
    if rb:
        lines += ["", "## Rust translation", "", f"Stats: {_j(rb)}",
                  "Translated tests are model-written; the guard is Rust-checker VERIFIED + baseline rejection + "
                  "a literal-overlap heuristic against the Python tests. Semantic equivalence is NOT proven."]
    lean = report.get("lean")
    if lean:
        lines += ["", "## Lean expert iteration", "", _j(lean, indent=2),
                  f"Pre-registered rule: fewer than {LEAN_MIN_ACCEPTED} accepted proofs => Lean is eval-only."]
    lines += ["", "## Decontamination", "",
              f"- Gate: gwaya.decontaminate; index `{dcm.get('index_path')}` sha256 `{dcm.get('index_sha256')}`",
              f"- Eval sets complete: {dcm.get('complete')}; missing: {dcm.get('missing_eval_sets')}; "
              f"plan eval-set check: {dcm.get('plan_eval_check')}",
              f"- Tasks flagged and removed: {report.get('decontam_task_removed', 'TBD')}; "
              f"solutions flagged and removed: {report.get('decontam_solution_removed', 'TBD')}",
              f"- Re-check of the written file: {_j(report.get('decontam_verification', {}))}",
              "", "## Dropped", ""]
    for why, n in sorted(report.get("drops", {}).items()):
        lines.append(f"- {why}: {n}")
    lines += ["", "## Not checked / limitations", ""]
    lines += [f"- {x}" for x in report.get("limitations", [])]
    lines.append("")
    return "\n".join(lines)
