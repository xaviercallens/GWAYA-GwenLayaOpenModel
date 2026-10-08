#!/usr/bin/env python3
"""GwenLaya v4 study runner (work item T8).

Executes one stage of experiments/plan.json: domains x models x arms x quant levels, against an
Ollama server or an OpenAI-compatible server (vLLM / llama.cpp --server).

One-cache design (docs/GWENLAYA_PREREGISTRATION.md section 4): every (served model, quant, item,
seed, call) generation is produced once, appended to ``gens.jsonl`` and replayed by every arm.
``rows.jsonl`` holds one scored row per (arm, model, quant, item, seed). Both files are append-only
hash chains (each record carries prev + sha256 over prev and its canonical JSON), fsynced per
record, so a killed run resumes where it stopped and tampering or truncation in the middle is
detected. A torn final line (kill during write) is dropped.

Arms (aliases in parentheses; plan.json arm names that this runner cannot execute are listed in
results.json ``arms_skipped``, never silently dropped):
  base (A0)                  greedy answer of each model at each quant
  gate_only (A3, B5)         base + gate verdict; answered iff the gate VERIFIES
  lora / lora_gate           same, with the tuned served name from --lora-map BASE=TUNED
  gwenlaya (GL)              gwaya.gwenlaya.GwenLaya over the tier ladder, replaying the cache
  always_smallest (B1)       first model of the tier order answers everything
  always_largest (B2)        last model of the tier order answers everything
  raw_confidence (B3)        largest tier + mean token logprob >= --tau-b3 (needs logprobs from the
                             backend; otherwise answered is null and the row says why)
  self_consistency (B4)      largest tier, k samples, majority share >= --tau-b4

Scoring (domain checkers, hidden/scoring payload) executes untrusted model output, so it is
REFUSED unless the bwrap sandbox works (exit 2). ``--mode generate`` needs no sandbox and writes
only gens.jsonl (the plan keeps verification off the GPU VM); ``--mode score`` replays the cache.
Gate checks use the task's separate ``gate_payload`` (gate-visible info only). Tasks without one
get an empty payload, so the gate returns UNVERIFIED for them: this is stated, not hidden.

Not claimed anywhere in this file: any accuracy, speed or cost number. All reported numbers are
computed at run time from the rows/gens logs; unknown values are null.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import shutil
import subprocess  # nosec B404
import sys
import threading
import types
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gwaya.domains.task import DOMAINS, Task  # noqa: E402

LAKE_PREFIX = "gs://socrateai-datalake-gen-lang-client-0625573011/gwenlaya_v4/"
STOP = ["```", "<|im_end|>", "<|endoftext|>"]
ARM_ALIASES = {
    "base": "base", "A0": "base",
    "gate_only": "gate_only", "A3": "gate_only", "B5": "gate_only",
    "lora": "lora", "lora_gate": "lora_gate",
    "gwenlaya": "gwenlaya", "GL": "gwenlaya",
    "always_smallest": "always_smallest", "B1": "always_smallest",
    "always_largest": "always_largest", "B2": "always_largest",
    "raw_confidence": "raw_confidence", "B3": "raw_confidence",
    "self_consistency": "self_consistency", "B4": "self_consistency",
}
KNOWN_QUANTS = ("bf16", "q8_0", "q4_K_M", "nf4")


class ChainError(RuntimeError):
    """A hash-chained log failed verification."""


class CacheMiss(RuntimeError):
    """--mode score needed a generation that is not in gens.jsonl."""


class SandboxRequired(RuntimeError):
    """Scoring was requested without a working sandbox."""


# ── hash-chained append-only log ────────────────────────────────────────────

def canon(rec: dict[str, Any]) -> str:
    return json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def chain_hash(prev: str, rec: dict[str, Any]) -> str:
    body = {k: v for k, v in rec.items() if k != "hash"}
    return hashlib.sha256((prev + canon(body)).encode("utf-8")).hexdigest()


class ChainedLog:
    GENESIS = "0" * 64

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.head = self.GENESIS
        self.records: list[dict[str, Any]] = []
        self.torn_tail_dropped = False
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = self.path.read_bytes()
        good_end = 0
        pos = 0
        lines = raw.split(b"\n")
        for i, line in enumerate(lines):
            end = pos + len(line) + 1
            if not line.strip():
                pos = end
                continue
            last = all(not x.strip() for x in lines[i + 1:])
            try:
                rec = json.loads(line.decode("utf-8"))
                ok = isinstance(rec, dict)
            except (ValueError, UnicodeDecodeError):
                ok = False
            if not ok:
                if last:  # torn tail from a kill mid-write
                    self.torn_tail_dropped = True
                    break
                raise ChainError(f"{self.path}: unparseable record at line {i + 1}")
            if rec.get("prev") != self.head or rec.get("hash") != chain_hash(self.head, rec):
                raise ChainError(f"{self.path}: hash chain broken at line {i + 1}")
            self.head = rec["hash"]
            self.records.append(rec)
            good_end = end
            pos = end
        if self.torn_tail_dropped:
            with open(self.path, "r+b") as fh:
                fh.truncate(good_end)

    def append(self, rec: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            rec = dict(rec)
            rec["prev"] = self.head
            rec["hash"] = chain_hash(self.head, rec)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "ab") as fh:
                fh.write((canon(rec) + "\n").encode("utf-8"))
                fh.flush()
                os.fsync(fh.fileno())
            self.head = rec["hash"]
            self.records.append(rec)
            return rec


# ── backends ────────────────────────────────────────────────────────────────

def _http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 600.0) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310 - operator-supplied backend URL
        return json.loads(resp.read().decode("utf-8"))


def _is_loopback(url: str) -> bool:
    m = re.match(r"^\w+://([^/:]+)", url)
    return bool(m) and m.group(1) in ("localhost", "127.0.0.1", "::1", "[::1]")


class Backend:
    kind = "base"
    url = ""

    def generate(self, model: str, prompt: str, domain: str, temperature: float, max_tokens: int,
                 seed: int) -> dict[str, Any]:
        raise NotImplementedError

    def info(self) -> dict[str, Any]:
        raise NotImplementedError

    def _probe(self, paths: list[str]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for p in paths:
            try:
                out[p] = _http_json(self.url + p, timeout=10.0)
            except Exception as exc:  # noqa: BLE001 - best-effort provenance
                out[p] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        return out


class OllamaBackend(Backend):
    kind = "ollama"

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        self._gens: dict[str, Any] = {}

    def generate(self, model, prompt, domain, temperature, max_tokens, seed):
        from gwaya.generators import OllamaGenerator
        gen = self._gens.get(model)
        if gen is None:
            gen = self._gens[model] = OllamaGenerator(model=model, host=self.url)
        text = gen(prompt, temperature=temperature, max_tokens=max_tokens, seed=seed, domain=domain)
        st = gen.last_stats
        return {"text": text, "prompt_tokens": st.prompt_tokens, "completion_tokens": st.completion_tokens,
                "eval_s": st.eval_ms / 1000.0, "gpu_s": st.total_ms / 1000.0,  # server-reported timing
                "mean_logprob": None, "done_reason": st.done_reason}

    def info(self):
        probes = self._probe(["/api/version", "/api/ps"])
        return {"kind": self.kind, "url": self.url, "version": probes["/api/version"], "hardware": probes["/api/ps"]}


def proc_cpu_seconds(pid: int | None) -> float | None:
    """utime+stime of a process in seconds from /proc (None if unavailable)."""
    if not pid:
        return None
    try:
        f = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


class OpenAICompatBackend(Backend):
    """/v1/completions with the same raw Qwen prompt as the Ollama path (vLLM, llama.cpp server)."""
    kind = "openai"

    def __init__(self, url: str, api_key: str | None = None, cpu_pid: int | None = None) -> None:
        self.url = url.rstrip("/")
        self.root = re.sub(r"/v1$", "", self.url)
        self.api_key = api_key
        self.cpu_pid = cpu_pid  # local server pid: per-call CPU-seconds = delta of its utime+stime

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return _http_json(self.url + path, payload)

    def generate(self, model, prompt, domain, temperature, max_tokens, seed):
        from gwaya.generators import OllamaGenerator
        raw = OllamaGenerator(model=model).build_raw_prompt(prompt, domain)
        tag = OllamaGenerator(model=model).fence_tag(domain)
        cpu0 = proc_cpu_seconds(self.cpu_pid)
        t0 = time.perf_counter()
        data = self._post("/completions", {"model": model, "prompt": raw, "temperature": temperature,
                                           "max_tokens": int(max_tokens), "seed": int(seed),
                                           "stop": STOP, "logprobs": 1})
        wall = time.perf_counter() - t0  # client wall time: no server timing in the OpenAI schema
        cpu1 = proc_cpu_seconds(self.cpu_pid)
        ch = (data.get("choices") or [{}])[0]
        if "text" not in ch:
            raise RuntimeError(f"OpenAI-compatible server returned no text: {str(data)[:200]}")
        lpo = ch.get("logprobs") or {}
        lps = [x for x in (lpo.get("token_logprobs") or []) if isinstance(x, (int, float))]
        if not lps:  # llama.cpp server: logprobs.content[].logprob
            lps = [c["logprob"] for c in (lpo.get("content") or [])
                   if isinstance(c, dict) and isinstance(c.get("logprob"), (int, float))]
        usage = data.get("usage") or {}
        return {"text": f"```{tag}\n{str(ch['text']).rstrip()}\n```",
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "eval_s": wall, "gpu_s": wall,
                "mean_logprob": (sum(lps) / len(lps)) if lps else None,
                "token_logprobs": lps,
                "cpu_seconds": (round(cpu1 - cpu0, 4) if cpu0 is not None and cpu1 is not None else None),
                "done_reason": str(ch.get("finish_reason") or "")}

    def info(self):
        saved = self.url
        self.url = self.root
        try:
            probes = self._probe(["/version", "/props", "/health"])
        finally:
            self.url = saved
        try:
            models = _http_json(self.url + "/models", timeout=10.0)
        except Exception as exc:  # noqa: BLE001
            models = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        return {"kind": self.kind, "url": self.url, "version": probes["/version"],
                "hardware": {"/props": probes["/props"], "/health": probes["/health"]}, "models": models}


# ── nvidia-smi sampling ─────────────────────────────────────────────────────

def nvidia_smi_used_mib(runner: Callable[..., Any] = subprocess.run) -> list[float] | None:
    """MiB used per GPU, or None when nvidia-smi is absent or fails."""
    if runner is subprocess.run and not shutil.which("nvidia-smi"):
        return None
    try:
        res = runner(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                     capture_output=True, text=True, timeout=10, check=False)  # nosec B603 B607
        if res.returncode != 0:
            return None
        vals = [float(x) for x in res.stdout.split() if x.strip()]
        return vals or None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


class VramSampler:
    """Background peak sampler, labelled per (model, quant). Peaks stay None if nvidia-smi is unusable."""

    def __init__(self, reader: Callable[[], list[float] | None] = nvidia_smi_used_mib, interval: float = 1.0,
                 enabled: bool = True) -> None:
        self.reader, self.interval, self.enabled = reader, interval, enabled
        self.label = "_"
        self.peak: dict[str, float] = {}
        self.overall: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self) -> None:
        vals = self.reader()
        if not vals:
            return
        v = max(vals)
        self.peak[self.label] = max(self.peak.get(self.label, 0.0), v)
        self.overall = v if self.overall is None else max(self.overall, v)

    def start(self) -> None:
        if not self.enabled:
            return
        self.sample()

        def loop() -> None:
            while not self._stop.wait(self.interval):
                self.sample()

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread:
            self._stop.set()
            self._thread.join(timeout=5)
            self.sample()


# ── git / sandbox / lake helpers ────────────────────────────────────────────

def git_info(root: Path = ROOT) -> dict[str, Any]:
    def run(*a: str) -> str | None:
        try:
            r = subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, timeout=20, check=False)  # nosec
            return r.stdout.strip() if r.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None
    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": (bool(status) if status is not None else None)}


def require_sandbox() -> None:
    """Refuse to score without bwrap isolation (no override flag, by design)."""
    from gwaya import sandbox
    if os.environ.get(sandbox.ALLOW_UNISOLATED_ENV) == "1":
        raise SandboxRequired(f"{sandbox.ALLOW_UNISOLATED_ENV}=1 is set; scoring refuses unisolated execution")
    if not sandbox.is_sandbox_available():
        raise SandboxRequired("bwrap sandbox unavailable; refusing to score model output "
                              "(use --mode generate on a host without it, and score where bwrap works)")


_STAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SYNC_FILES = ("rows.jsonl", "gens.jsonl", "results.json")


def lake_dest(stage: str, name: str, prefix: str = LAKE_PREFIX) -> str:
    if not _STAGE_RE.match(stage):
        raise ValueError(f"unsafe stage id {stage!r}")
    dest = f"{prefix}runs/{stage}/{name}"
    if not dest.startswith(LAKE_PREFIX):
        raise ValueError(f"refusing to write outside {LAKE_PREFIX}: {dest}")
    return dest


def lake_sync(out_dir: Path, stage: str, runner: Callable[..., Any] = subprocess.run,
              prefix: str = LAKE_PREFIX) -> dict[str, Any]:
    """Upload run files with `gcloud storage cp`. Only writes under <prefix>runs/<stage>/."""
    res: dict[str, Any] = {}
    for name in SYNC_FILES:
        src = Path(out_dir) / name
        if not src.exists():
            continue
        dest = lake_dest(stage, name, prefix)
        try:
            r = runner(["gcloud", "storage", "cp", str(src), dest], capture_output=True, text=True,
                       timeout=600, check=False)  # nosec B603 B607
            res[name] = {"dest": dest, "ok": r.returncode == 0, "err": (r.stderr or "")[-200:] if r.returncode else ""}
        except (OSError, subprocess.TimeoutExpired) as exc:
            res[name] = {"dest": dest, "ok": False, "err": f"{type(exc).__name__}: {exc}"[:200]}
    return res


def lake_restore(out_dir: Path, stage: str, runner: Callable[..., Any] = subprocess.run) -> list[str]:
    """Fetch rows/gens from the lake for a fresh local dir (resume after preemption). Best effort."""
    got = []
    for name in ("rows.jsonl", "gens.jsonl"):
        dst = Path(out_dir) / name
        if dst.exists():
            continue
        try:
            r = runner(["gcloud", "storage", "cp", lake_dest(stage, name), str(dst)], capture_output=True,
                       text=True, timeout=600, check=False)  # nosec B603 B607
            if r.returncode == 0 and dst.exists():
                got.append(name)
        except (OSError, subprocess.TimeoutExpired):
            pass
    return got


# ── study ───────────────────────────────────────────────────────────────────

def resolve_arms(requested: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    arms, skipped = [], []
    for a in requested:
        canon_name = ARM_ALIASES.get(a)
        if canon_name is None:
            skipped.append({"arm": a, "reason": "not implemented by run_study.py (needs trained Laya/LR/oracle/continuity pipeline)"})
        elif canon_name not in arms:
            arms.append(canon_name)
    return arms, skipped


def served_name(model: str, quant: str, quant_map: dict[str, str]) -> tuple[str, str]:
    """(name sent to the backend, how the quant level is bound)."""
    key = f"{model}@{quant}"
    if key in quant_map:
        return quant_map[key], "map"
    if re.search(r"-(bf16|q8_0|q4_K_M|nf4)$", model):
        base = re.sub(r"-(bf16|q8_0|q4_K_M|nf4)$", "", model)
        if model.endswith("-" + quant):
            return model, "tag"
        return f"{base}-{quant}", "tag_rewritten"
    return model, "label_only"  # the backend was started with one fixed quant; the label is not verified


def norm_code(text: str) -> str:
    from gwaya.low_tier_engine import extract_code_block
    return re.sub(r"\s+", " ", extract_code_block(text, "python") or text).strip()


def consensus_key(domain: str, text: str) -> str:
    if domain == "math":
        from gwaya.domains.math_check import extract_final_answer, normalize_answer
        fa = extract_final_answer(text)
        if fa:
            return "ans:" + normalize_answer(fa[1])
    return "txt:" + hashlib.sha256(norm_code(text).encode()).hexdigest()[:16]


class Study:
    def __init__(self, *, stage: dict[str, Any], plan: dict[str, Any], tasks: list[Task], models: list[str],
                 quants: list[str], arms: list[str], backend: Backend, out_dir: Path, mode: str = "all",
                 lora_map: dict[str, str] | None = None, quant_map: dict[str, str] | None = None,
                 tau_b3: float | None = None, tau_b4: float = 0.6, sc_k: int = 5, sc_temperature: float = 0.8,
                 calibrations: dict[str, Any] | None = None, checker: Callable[..., Any] | None = None,
                 sampler: VramSampler | None = None, sync: Callable[[], Any] | None = None,
                 sync_interval_s: float = 900.0, log: Callable[[str], None] = lambda m: print(m, file=sys.stderr)) -> None:
        self.stage, self.plan, self.models, self.quants, self.arms = stage, plan, list(models), list(quants), list(arms)
        self.backend, self.out_dir, self.mode = backend, Path(out_dir), mode
        self.lora_map, self.quant_map = lora_map or {}, quant_map or {}
        self.tau_b3, self.tau_b4, self.sc_k, self.sc_temperature = tau_b3, tau_b4, sc_k, sc_temperature
        self.calibrations = calibrations
        self.sampler, self.sync, self.sync_interval_s, self.log = sampler or VramSampler(enabled=False), sync, sync_interval_s, log
        if checker is None:
            from gwaya.domains.checkers import check as checker
        self.checker = checker
        self.max_tokens = dict(plan.get("generation_settings", {}).get("max_new_tokens", {}))
        # item_index = position in the task_id-sorted list, so seeds do not depend on eval order
        ordered = sorted(tasks, key=lambda t: (t.domain, t.task_id))
        self.item_index = {(t.domain, t.task_id): i for i, t in enumerate(ordered)}
        order_seed = plan.get("seeds", {}).get("eval_order", 0)
        self.tasks = list(ordered)
        random.Random(order_seed).shuffle(self.tasks)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.gens = ChainedLog(self.out_dir / "gens.jsonl")
        self.rows = ChainedLog(self.out_dir / "rows.jsonl")
        self.gen_index = {r["key"]: r for r in self.gens.records}
        self.row_keys = {r["key"] for r in self.rows.records}
        self.new_gens = self.new_rows = 0
        self._last_sync = time.monotonic()

    # generation (cached) ----------------------------------------------------
    def seed_for(self, task: Task, call_index: int) -> int:
        return 1000 * (self.item_index[(task.domain, task.task_id)] + 1) + call_index

    def gen(self, served: str, quant: str, task: Task, call_index: int, temperature: float) -> dict[str, Any]:
        seed = self.seed_for(task, call_index)
        key = f"{served}|{quant}|{task.domain}|{task.task_id}|{seed}|t{temperature}"
        hit = self.gen_index.get(key)
        if hit:
            return hit
        if self.mode == "score":
            raise CacheMiss(key)
        mt = int(self.max_tokens.get(task.domain, 1024))
        self.sampler.label = f"{served}|{quant}"
        self.sampler.sample()
        t0 = time.perf_counter()
        out = self.backend.generate(served, task.prompt, task.domain, temperature, mt, seed)
        wall = time.perf_counter() - t0
        self.sampler.sample()
        rec = self.gens.append({"key": key, "model": served, "quant": quant, "domain": task.domain,
                                "task_id": task.task_id, "seed": seed, "call_index": call_index,
                                "temperature": temperature, "max_tokens": mt, "wall_s": round(wall, 6),
                                **{k: out.get(k) for k in ("text", "prompt_tokens", "completion_tokens", "eval_s",
                                                           "gpu_s", "mean_logprob", "done_reason",
                                                           "token_logprobs", "cpu_seconds")}})
        self.gen_index[key] = rec
        self.new_gens += 1
        return rec

    # scoring ----------------------------------------------------------------
    def _score(self, task: Task, text: str | None) -> str | None:
        if self.mode == "generate" or text is None:
            return None
        return self.checker(task, text).status

    def _gate(self, task: Task, text: str) -> str:
        if self.mode == "generate":
            return "UNVERIFIED"  # placeholder: keeps escalation paths walking so every tier gets generated
        gt = Task(task.domain, task.task_id, task.prompt, dict(task.checker_payload.get("__gate__", {})))
        return self.checker(gt, text).status

    def _emit(self, key: str, row: dict[str, Any]) -> None:
        if self.mode == "generate":
            return
        self.row_keys.add(key)
        self.rows.append({"key": key, **row})
        self.new_rows += 1

    def _row_key(self, arm: str, model: str, quant: str, task: Task) -> str:
        return f"{arm}|{model}|{quant}|{task.domain}|{task.task_id}|{self.seed_for(task, 0)}"

    def _done(self, key: str) -> bool:
        return self.mode != "generate" and key in self.row_keys

    # arms -------------------------------------------------------------------
    def _greedy(self, model: str, quant: str, task: Task) -> dict[str, Any]:
        return self.gen(model, quant, task, 0, 0.0)

    def _single(self, arm: str, model: str, quant: str, task: Task, gated: bool, extra: dict | None = None) -> None:
        key = self._row_key(arm, model, quant, task)
        if self._done(key):
            return
        g = self._greedy(model, quant, task)
        verdict = self._gate(task, g["text"]) if gated else None
        self._emit(key, {"arm": arm, "domain": task.domain, "task_id": task.task_id, "model": model, "quant": quant,
                         "seed": self.seed_for(task, 0), "gate": verdict,
                         "answered": (verdict == "VERIFIED") if gated else True,
                         "score": self._score(task, g["text"]), "gpu_s": g["gpu_s"],
                         "completion_tokens": g["completion_tokens"], **(extra or {})})

    def _ladder(self, quant: str) -> list[str]:
        return [served_name(m, quant, self.quant_map)[0] for m in self.models]

    def _run_gwenlaya(self, quant: str, task: Task) -> None:
        from gwaya.gwenlaya import GwenLaya, Tier
        ladder = self._ladder(quant)
        key = self._row_key("gwenlaya", ladder[-1], quant, task)
        if self._done(key):
            return
        used: list[dict[str, Any]] = []

        def mk(name: str) -> Tier:
            def _g(_prompt: str, _domain: str) -> str:
                g = self._greedy(name, quant, task)
                used.append(g)
                return g["text"]
            return Tier(name, _g)

        gate_checker = (lambda t, r: types.SimpleNamespace(status="UNVERIFIED", evidence={})) \
            if self.mode == "generate" else self.checker
        system = GwenLaya([mk(n) for n in ladder], calibrations=self.calibrations, checker=gate_checker)
        out = system.answer(task.prompt, task.domain, task.checker_payload.get("__gate__", {}), task.task_id)
        last = used[-1]["text"] if used else None
        self._emit(key, {"arm": "gwenlaya", "domain": task.domain, "task_id": task.task_id, "model": out.get("tier"),
                         "quant": quant, "seed": self.seed_for(task, 0), "gate": out["verdict"],
                         "answered": bool(out["answered"]), "p_correct": out["p_correct"],
                         "tiers_invoked": [u["model"] for u in used],
                         "score": self._score(task, last), "gpu_s": round(sum(u["gpu_s"] or 0.0 for u in used), 6),
                         "completion_tokens": sum(u["completion_tokens"] or 0 for u in used)})

    def _run_b3(self, model: str, quant: str, task: Task) -> None:
        key = self._row_key("raw_confidence", model, quant, task)
        if self._done(key):
            return
        g = self._greedy(model, quant, task)
        lp = g.get("mean_logprob")
        if lp is None or self.tau_b3 is None:
            answered, why = None, ("backend returned no logprobs" if lp is None else "--tau-b3 not set")
        else:
            answered, why = lp >= self.tau_b3, None
        self._emit(key, {"arm": "raw_confidence", "domain": task.domain, "task_id": task.task_id, "model": model,
                         "quant": quant, "seed": self.seed_for(task, 0), "confidence": lp, "tau": self.tau_b3,
                         "answered": answered, "unavailable_reason": why, "score": self._score(task, g["text"]),
                         "gpu_s": g["gpu_s"], "completion_tokens": g["completion_tokens"]})

    def _run_b4(self, model: str, quant: str, task: Task) -> None:
        key = self._row_key("self_consistency", model, quant, task)
        if self._done(key):
            return
        samples = [self.gen(model, quant, task, i, self.sc_temperature) for i in range(1, self.sc_k + 1)]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for s in samples:
            groups[consensus_key(task.domain, s["text"])].append(s)
        top = max(groups.values(), key=len)  # ties: first group by insertion order
        share = len(top) / len(samples)
        self._emit(key, {"arm": "self_consistency", "domain": task.domain, "task_id": task.task_id, "model": model,
                         "quant": quant, "seed": self.seed_for(task, 0), "k": self.sc_k, "confidence": share,
                         "tau": self.tau_b4, "answered": share >= self.tau_b4,
                         "score": self._score(task, top[0]["text"]),
                         "gpu_s": round(sum(s["gpu_s"] or 0.0 for s in samples), 6),
                         "completion_tokens": sum(s["completion_tokens"] or 0 for s in samples)})

    def run_task(self, task: Task) -> None:
        ladders = {q: self._ladder(q) for q in self.quants}
        for q in self.quants:
            lad = ladders[q]
            for arm in self.arms:
                if arm == "base":
                    for m in lad:
                        self._single("base", m, q, task, gated=False)
                elif arm == "gate_only":
                    for m in lad:
                        self._single("gate_only", m, q, task, gated=True)
                elif arm in ("lora", "lora_gate"):
                    for base, tuned in self.lora_map.items():
                        self._single(arm, served_name(tuned, q, self.quant_map)[0], q, task, gated=arm == "lora_gate",
                                     extra={"base_model": base})
                elif arm == "always_smallest":
                    self._single("always_smallest", lad[0], q, task, gated=False)
                elif arm == "always_largest":
                    self._single("always_largest", lad[-1], q, task, gated=False)
                elif arm == "gwenlaya":
                    self._run_gwenlaya(q, task)
                elif arm == "raw_confidence":
                    self._run_b3(lad[-1], q, task)
                elif arm == "self_consistency":
                    self._run_b4(lad[-1], q, task)

    def run(self) -> None:
        self.sampler.start()
        try:
            for i, task in enumerate(self.tasks):
                self.run_task(task)
                if self.sync and time.monotonic() - self._last_sync >= self.sync_interval_s:
                    self.sync()
                    self._last_sync = time.monotonic()
                if (i + 1) % 25 == 0:
                    self.log(f"[study] {i + 1}/{len(self.tasks)} items; new gens {self.new_gens}, new rows {self.new_rows}")
        finally:
            self.sampler.stop()


# ── results ─────────────────────────────────────────────────────────────────

def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[tuple, dict[str, Any]] = {}
    for r in rows:
        c = cells.setdefault((r["arm"], r.get("model"), r["quant"], r["domain"]),
                             {"n": 0, "answered": 0, "answered_unknown": 0, "final_verified": 0,
                              "final_failed": 0, "final_unverified": 0, "answered_and_verified": 0, "gpu_s": 0.0})
        c["n"] += 1
        c["gpu_s"] += r.get("gpu_s") or 0.0
        if r.get("answered") is None:
            c["answered_unknown"] += 1
        elif r["answered"]:
            c["answered"] += 1
        sc = r.get("score")
        c["final_verified"] += sc == "VERIFIED"
        c["final_failed"] += sc == "FAILED"
        c["final_unverified"] += sc == "UNVERIFIED"
        c["answered_and_verified"] += bool(r.get("answered")) and sc == "VERIFIED"
    return {"|".join(str(x) for x in k): {**v, "gpu_s": round(v["gpu_s"], 6)} for k, v in sorted(cells.items(), key=lambda kv: tuple(map(str, kv[0])))}


def throughput(gens: list[dict[str, Any]]) -> dict[str, Any]:
    acc: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for g in gens:
        a = acc[f"{g['model']}|{g['quant']}"]
        a[0] += g.get("completion_tokens") or 0
        a[1] += g.get("eval_s") or 0.0
        a[2] += g.get("gpu_s") or 0.0
        a[3] += 1
    return {k: {"calls": int(a[3]), "completion_tokens": int(a[0]), "gpu_seconds": round(a[2], 6),
                "tokens_per_s": (a[0] / a[1]) if a[1] > 0 else None} for k, a in sorted(acc.items())}


def build_results(study: Study, *, started: datetime, t0: float, backend_info: dict[str, Any],
                  skipped: list[dict[str, str]], args_echo: dict[str, Any], plan_path: Path, status: str,
                  prior: dict[str, Any] | None, lake: Any = None) -> dict[str, Any]:
    gens, rows = study.gens.records, study.rows.records
    wall = time.perf_counter() - t0
    prior_wall = float((prior or {}).get("wall_clock", {}).get("seconds_total", 0.0) or 0.0)
    vs = study.sampler
    local = _is_loopback(getattr(study.backend, "url", ""))
    return {
        "schema": "gwenlaya_v4.study_results/1",
        "stage": study.stage["id"], "status": status, "mode": study.mode,
        "plan": {"path": str(plan_path), "sha256": hashlib.sha256(Path(plan_path).read_bytes()).hexdigest()},
        "git": git_info(), "python": platform.python_version(), "host": platform.node(),
        "backend": backend_info,
        "local_nvidia_smi_present": bool(shutil.which("nvidia-smi")),
        "seeds": {"generation": study.plan.get("seeds", {}).get("generation"), "eval_order": study.plan.get("seeds", {}).get("eval_order"),
                  "scheme": "1000*(item_index+1)+call_index; item_index = rank in sorted (domain, task_id)"},
        "models": study.models, "tier_order": study.models, "quants": study.quants, "arms": study.arms,
        "arms_skipped": skipped,
        "quant_binding": {f"{m}@{q}": served_name(m, q, study.quant_map) for m in study.models for q in study.quants},
        "domains": sorted({t.domain for t in study.tasks}), "n_items": len(study.tasks),
        "wall_clock": {"started_utc": started.isoformat(), "ended_utc": datetime.now(timezone.utc).isoformat(),
                       "seconds_this_invocation": round(wall, 3), "seconds_total": round(prior_wall + wall, 3)},
        "gpu_seconds_total": round(sum(g.get("gpu_s") or 0.0 for g in gens), 6),
        "gpu_seconds_note": "ollama: server-reported total_duration; openai-compat: client wall time per call (sequential)",
        "throughput": throughput(gens),
        "peak_vram_mib": {"overall": vs.overall if (vs.enabled and local) else None,
                          "by_model_quant": dict(vs.peak) if (vs.enabled and local) else None,
                          "source": "local nvidia-smi" if (vs.enabled and local and vs.overall is not None) else None,
                          "note": "null when nvidia-smi is absent, failed, or the backend is not on this host"},
        "counts": {"gens": len(gens), "rows": len(rows), "new_gens_this_invocation": study.new_gens,
                   "new_rows_this_invocation": study.new_rows},
        "chain_heads": {"gens": study.gens.head, "rows": study.rows.head},
        "torn_tail_dropped": {"gens": study.gens.torn_tail_dropped, "rows": study.rows.torn_tail_dropped},
        "summary": summarize_rows(rows),
        "lake_sync": lake, "args": args_echo,
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

def load_tasks_jsonl(path: Path) -> list[Task]:
    out = []
    for ln, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        d = json.loads(line)
        payload = dict(d.get("checker_payload") or {})
        if d.get("gate_payload"):
            payload["__gate__"] = dict(d["gate_payload"])  # kept apart: scoring payload never reaches the gate
        out.append(Task(d["domain"], str(d["task_id"]), d["prompt"], payload))
    return out


def parse_kv(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"expected KEY=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def main(argv: list[str] | None = None, *, backend: Backend | None = None, checker=None,
         sampler: VramSampler | None = None, runner: Callable[..., Any] = subprocess.run) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", default=str(ROOT / "experiments" / "plan.json"))
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", help="output dir (default $GWAYA_DATA_DIR/runs/<stage> or ./runs/<stage>)")
    ap.add_argument("--tasks", action="append", default=[], help="JSONL: domain, task_id, prompt, checker_payload, [gate_payload]")
    ap.add_argument("--dataset", action="append", default=[], help="DATA_MANIFEST id/hf_id (local path, never downloads)")
    ap.add_argument("--domains", nargs="*")
    ap.add_argument("--limit", type=int, help="max items per domain")
    ap.add_argument("--models", nargs="*", help="tier order, cheapest first (default: plan stage models)")
    ap.add_argument("--quants", nargs="*")
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--mode", choices=("all", "generate", "score"), default="all")
    ap.add_argument("--backend", choices=("ollama", "openai"), default="ollama")
    ap.add_argument("--backend-url", help="Ollama host or OpenAI-compatible base URL (e.g. http://h:8000/v1)")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--cpu-pid", type=int, help="local server pid; records per-call cpu_seconds (sequential calls only)")
    ap.add_argument("--lora-map", action="append", default=[], metavar="BASE=TUNED")
    ap.add_argument("--quant-map", action="append", default=[], metavar="MODEL@QUANT=SERVED")
    ap.add_argument("--calibrations", help="JSON from gwaya.gwenlaya.save_calibrations (GL arm)")
    ap.add_argument("--tau-b3", type=float, help="B3 mean-logprob threshold (set on C, not E)")
    ap.add_argument("--tau-b4", type=float, default=0.6)
    ap.add_argument("--sc-k", type=int, default=5)
    ap.add_argument("--sc-temperature", type=float, default=0.8)
    ap.add_argument("--lake-sync", action="store_true", help=f"copy run files to {LAKE_PREFIX}runs/<stage>/")
    ap.add_argument("--sync-interval-min", type=float, default=15.0)
    ap.add_argument("--no-vram", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    plan_path = Path(a.plan)
    plan = json.loads(plan_path.read_text())
    stage = next((s for s in plan["stages"] if s["id"] == a.stage), None)
    if stage is None:
        print(f"unknown stage {a.stage!r}; plan has {[s['id'] for s in plan['stages']]}", file=sys.stderr)
        return 2
    models = a.models or stage.get("models") or []
    if not models:
        print(f"stage {a.stage} has no models (where={stage.get('where')}); nothing for run_study to generate", file=sys.stderr)
        return 2
    quants = a.quants or stage.get("quant") or []
    if not quants:
        print(f"stage {a.stage} lists no quant levels; pass --quants", file=sys.stderr)
        return 2
    requested = a.arms if a.arms is not None else (stage.get("arms") or ["base"])
    arms, skipped = resolve_arms(requested)
    if "lora" in arms or "lora_gate" in arms:
        if not a.lora_map:
            skipped += [{"arm": x, "reason": "no --lora-map given"} for x in ("lora", "lora_gate") if x in arms]
            arms = [x for x in arms if x not in ("lora", "lora_gate")]
    if not arms:
        print("no runnable arms", file=sys.stderr)
        return 2
    domains = set(a.domains or stage.get("domains") or DOMAINS)

    tasks: list[Task] = []
    for p in a.tasks:
        tasks += load_tasks_jsonl(Path(p))
    if a.dataset:
        from gwaya.domains.loaders import load_tasks
        for ds in a.dataset:
            tasks += load_tasks(ds, limit=a.limit)
    tasks = [t for t in tasks if t.domain in domains]
    if a.limit:
        seen: dict[str, int] = defaultdict(int)
        kept = []
        for t in sorted(tasks, key=lambda t: (t.domain, t.task_id)):
            if seen[t.domain] < a.limit:
                seen[t.domain] += 1
                kept.append(t)
        tasks = kept
    if not tasks:
        print("no tasks (pass --tasks FILE.jsonl or --dataset ID)", file=sys.stderr)
        return 2
    ids = [(t.domain, t.task_id) for t in tasks]
    if len(set(ids)) != len(ids):
        print("duplicate (domain, task_id) in task set", file=sys.stderr)
        return 2

    data_dir = os.environ.get("GWAYA_DATA_DIR")
    out_dir = Path(a.out) if a.out else (Path(data_dir) / "runs" / a.stage if data_dir else Path("runs") / a.stage)
    print(f"[study] stage {a.stage}: {len(tasks)} items, models {models}, quants {quants}, arms {arms}, mode {a.mode}", file=sys.stderr)
    if skipped:
        print(f"[study] skipped arms: {skipped}", file=sys.stderr)
    if a.dry_run:
        print(json.dumps({"stage": a.stage, "items": len(tasks), "models": models, "quants": quants, "arms": arms,
                          "arms_skipped": skipped, "out": str(out_dir)}, indent=1))
        return 0

    if a.mode in ("all", "score"):
        try:
            require_sandbox()
        except SandboxRequired as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 2

    if backend is None and a.mode != "score":
        url = a.backend_url or (os.environ.get("OLLAMA_HOST", "http://localhost:11434") if a.backend == "ollama" else None)
        if not url:
            print("--backend-url is required for the openai backend", file=sys.stderr)
            return 2
        if not url.startswith("http"):
            url = "http://" + url
        backend = OllamaBackend(url) if a.backend == "ollama" else OpenAICompatBackend(url, os.environ.get(a.api_key_env), cpu_pid=a.cpu_pid)
    if backend is None:  # score mode never touches a backend
        backend = Backend()

    cals = None
    if a.calibrations:
        from gwaya.gwenlaya import load_calibrations
        cals = load_calibrations(a.calibrations)

    out_dir.mkdir(parents=True, exist_ok=True)
    prior = None
    if a.lake_sync:
        restored = lake_restore(out_dir, a.stage, runner)
        if restored:
            print(f"[study] restored from lake: {restored}", file=sys.stderr)
    if (out_dir / "results.json").exists():
        try:
            prior = json.loads((out_dir / "results.json").read_text())
        except ValueError:
            prior = None

    sync = (lambda: lake_sync(out_dir, a.stage, runner)) if a.lake_sync else None
    sampler = sampler or VramSampler(enabled=not a.no_vram and a.mode != "score")
    try:
        study = Study(stage=stage, plan=plan, tasks=tasks, models=models, quants=quants, arms=arms, backend=backend,
                      out_dir=out_dir, mode=a.mode, lora_map=parse_kv(a.lora_map), quant_map=parse_kv(a.quant_map),
                      tau_b3=a.tau_b3, tau_b4=a.tau_b4, sc_k=a.sc_k, sc_temperature=a.sc_temperature,
                      calibrations=cals, checker=checker, sampler=sampler, sync=sync,
                      sync_interval_s=a.sync_interval_min * 60)
    except ChainError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    started, t0 = datetime.now(timezone.utc), time.perf_counter()
    status, rc = "COMPLETE", 0
    try:
        binfo = backend.info() if a.mode != "score" else {"kind": "none (score mode)"}
    except Exception as exc:  # noqa: BLE001
        binfo = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    try:
        study.run()
    except CacheMiss as exc:
        status, rc = f"PARTIAL (cache miss: {exc})", 3
    except KeyboardInterrupt:  # time limit (SIGINT from `timeout`): keep partial progress
        status, rc = "PARTIAL (interrupted: wall-clock limit)", 130
    except Exception as exc:  # noqa: BLE001 - keep partial progress, report honestly
        status, rc = f"PARTIAL ({type(exc).__name__}: {exc})"[:300], 3
    args_echo = {k: v for k, v in vars(a).items() if k not in ("api_key_env",)}
    res = build_results(study, started=started, t0=t0, backend_info=binfo, skipped=skipped, args_echo=args_echo,
                        plan_path=plan_path, status=status, prior=prior)
    (out_dir / "results.json").write_text(json.dumps(res, indent=1, sort_keys=True))
    if a.lake_sync:
        res["lake_sync"] = lake_sync(out_dir, a.stage, runner)
        (out_dir / "results.json").write_text(json.dumps(res, indent=1, sort_keys=True))
        lake_sync_final = lake_sync(out_dir, a.stage, runner)  # results.json now includes the sync record
        if not all(v["ok"] for v in lake_sync_final.values()):
            print(f"lake sync failed: {lake_sync_final}", file=sys.stderr)
            rc = rc or 4
    print(f"[study] {status}: {study.new_gens} new gens, {study.new_rows} new rows -> {out_dir}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
