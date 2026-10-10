"""
gwaya/lean_gate.py
==================
Hardened Lean 4 gate (A18 amendment): VERIFIED means "the task's exact theorem statement is proved, kernel-checked,
with only the standard axioms".

Pipeline (each Lean run is a separate bwrap sandbox; the pinned project is mounted read-only):
  1. Lean-aware lexical rules on the candidate (comments/strings stripped first): reject meta-programming, IO,
     notation/instances/variables/namespaces that change what an intact statement means, and escape hatches.
  2. The reference: task header + formal_statement + `by sorry`, compiled alone (no candidate text) to Ref.olean.
  3. The candidate compiled to Cand.olean (`lean -o`); any error fails it.
  4. A trusted Lean process whose source is the fixed TRUSTED_CHECK below. It reads both .olean files as data
     (it never imports the candidate module, so no candidate environment extension runs), imports only the
     candidate's declared imports, re-checks every candidate declaration through the kernel (Environment.replay),
     then requires that the target constant is a theorem of the candidate module whose type is the reference type
     (exact `Expr` equality, else kernel definitional equality; which one is recorded) and whose `collectAxioms`
     closure is within {propext, Classical.choice, Quot.sound}. Its lines carry a per-run random nonce and only
     they are parsed, so candidate output cannot spoof them. Anything unexpected fails closed.
"""
from __future__ import annotations

import hashlib
import re
import secrets
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gwaya import sandbox

STANDARD_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})

# ── Lexer ─────────────────────────────────────────────────────────────────────

_CHAR_LIT = re.compile(r"'(?:\\(?:u\{[0-9a-fA-F]+\}|x[0-9a-fA-F]{2}|.)|[^\\'\n])'")


def _ident_char(c: str) -> bool:
    return c.isalnum() or c in "_'!?.₀₁₂₃₄₅₆₇₈₉"


def strip_lean(src: str) -> tuple[str, list[str]]:
    """Blank out comments (nested `/- -/`, `--`) and the contents of string/char literals, keeping newlines and
    the string delimiters. Returns (stripped, flaws); flaws name unterminated literals and interpolated strings,
    whose `{...}` holes are code the stripped text would hide."""
    out: list[str] = []
    flaws: set[str] = set()
    i, n = 0, len(src)

    def blank(s: str) -> str:
        return "".join("\n" if ch == "\n" else " " for ch in s)

    while i < n:
        c = src[i]
        prev = src[i - 1] if i else ""
        if src.startswith("/-", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("/-", j):
                    depth, j = depth + 1, j + 2
                elif src.startswith("-/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            if depth:
                flaws.add("unterminated_comment")
            out.append(blank(src[i:j]))
            i = j
        elif src.startswith("--", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(blank(src[i:j]))
            i = j
        elif c == "r" and not _ident_char(prev) and re.match(r'r#*"', src[i:]):
            hashes = re.match(r"r(#*)\"", src[i:]).group(1)
            start = i + 2 + len(hashes)
            j = src.find('"' + hashes, start)
            if j < 0:
                flaws.add("unterminated_string")
                j = n
            out.append('""' + blank(src[start:j]))
            i = min(n, j + 1 + len(hashes))
        elif c == '"':
            if prev == "!":
                flaws.add("interpolated_string")
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            if j >= n:
                flaws.add("unterminated_string")
            out.append('"' + blank(src[i + 1:min(j, n)]) + '"')
            i = j + 1
        elif c == "'" and not _ident_char(prev) and _CHAR_LIT.match(src, i):
            m = _CHAR_LIT.match(src, i)
            out.append("'" + blank(m.group(0)[1:-1]) + "'")
            i = m.end()
        elif c == "«":  # «name»: an identifier; its contents are not lexed as strings/comments
            j = src.find("»", i)
            j = n - 1 if j < 0 else j
            out.append(src[i:j + 1])
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out), sorted(flaws)


# ── Lexical rules ─────────────────────────────────────────────────────────────

_B = r"(?<![\w.'!?])"  # Lean identifier boundary on the left
_E = r"(?![\w'!?])"    # ... and on the right
_BANNED_WORDS = (
    "sorry", "admit", "native_decide", "axiom", "opaque", "unsafe", "implemented_by", "extern", "prelude",
    "variable", "variables", "universe", "notation", "notation3", "binder_predicate", "infix", "infixl", "infixr", "prefix", "postfix",
    "macro", "macro_rules", "syntax", "elab", "elab_rules", "declare_syntax_cat", "run_cmd", "run_elab", "run_meta",
    "run_tac", "by_elab", "initialize", "builtin_initialize", "namespace", "section", "export", "local", "scoped",
    "attribute", "instance", "deriving", "simproc", "dsimproc", "renaming", "hiding",
)
_BANNED_RE = re.compile(_B + r"(" + "|".join(sorted(_BANNED_WORDS, key=len, reverse=True)) + r")" + _E)
# trust-the-compiler hooks, also when qualified (`Lean.ofReduceBool`)
_QUALIFIED_RE = re.compile(r"(?<![\w'!?])(?:[\w']+\.)*(ofReduceBool|ofReduceNat|reduceBool|reduceNat|sorryAx|"
                           r"native_decide|implemented_by)" + _E)
_BANNED_PATTERNS = (
    ("hash_command", re.compile(r"(?<![\w.])#(?!check\b|print\b|synth\b)[A-Za-z_]\w*")),
    ("unsafe_ident", re.compile(_B + r"(?:[\w.]*\.)?unsafe\w+")),
    ("register_command", re.compile(_B + r"register_\w+")),
    ("lean_meta", re.compile(_B + r"(?:open\s+Lean\b|Lean\.(?:Elab|Meta|Parser|Environment|Kernel|Compiler|IR|Core)\b)")),
    ("io", re.compile(_B + r"(?:IO|EIO|BaseIO)\.")),
    ("open_in", re.compile(_B + r"open[ \t]+(?:[^\s]+[ \t]+)+in" + _E)),
    ("syntax_quotation", re.compile(r"`\(")),
)
# set_option is allowed only for resource/linter/printing options, never ones that weaken checking.
_SET_OPTION_RE = re.compile(_B + r"set_option\s+([\w.]+)")
_SAFE_OPTIONS = re.compile(r"(?:maxHeartbeats|maxRecDepth|synthInstance\.maxHeartbeats|synthInstance\.maxSize|"
                           r"linter\.[\w.]+|pp\.[\w.]+|trace\.[\w.]+)$")


def lexical_flaws(code: str) -> list[str]:
    """Constructs the hardened gate rejects outright, scanned on the comment/string-stripped text."""
    stripped, flaws = strip_lean(code)
    found = set(flaws) | set(_BANNED_RE.findall(stripped)) | set(_QUALIFIED_RE.findall(stripped))
    found.update(name for name, rx in _BANNED_PATTERNS if rx.search(stripped))
    found.update(f"set_option {o}" for o in _SET_OPTION_RE.findall(stripped) if not _SAFE_OPTIONS.match(o))
    return sorted(found)


# ── Statement, name and import handling ───────────────────────────────────────

_TARGET_RE = re.compile(r"^\s*(?:@\[[^\]]*\]\s*)*(?:(?:protected|noncomputable|nonrec)\s+)*(?:theorem|lemma)\s+"
                        r"([^\s(:{\[«]+)(?=[\s(:{\[])")
_IMPORT_RE = re.compile(r"(?<![\w.])import\s+([^\s]+)")


def squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


def target_name(formal_statement: str) -> str | None:
    """`theorem foo.bar (x : T) : P :=` -> "foo.bar"; None when it is not a named theorem/lemma."""
    m = _TARGET_RE.match(strip_lean(formal_statement or "")[0])
    return m.group(1) if m else None


def normalize_statement(formal_statement: str) -> str:
    """The statement up to and including `:=` (a dataset's trailing `sorry` placeholder dropped)."""
    s = re.sub(r":=\s*(?:by\s+)?sorry\s*$", ":=", (formal_statement or "").rstrip())
    return s if s.endswith(":=") else s + " :="


def statement_preserved(formal_statement: str, code: str) -> bool:
    """Whitespace-insensitive containment, on comment/string-stripped text (a statement in a comment or string
    does not count)."""
    return squash(strip_lean(formal_statement)[0]) in squash(strip_lean(code)[0])


def has_imports(code: str) -> bool:
    return bool(_IMPORT_RE.search(strip_lean(code)[0]))


def imports_of(code: str) -> list[str]:
    return [m.replace("«", "").replace("»", "") for m in _IMPORT_RE.findall(strip_lean(code)[0])]


def reference_source(header: str, formal_statement: str) -> str:
    return f"{header}\n\n{normalize_statement(formal_statement)} by sorry\n" if header else \
        f"{normalize_statement(formal_statement)} by sorry\n"


def resolve_import(module: str, search_dirs: list[Path]) -> bool:
    rel = Path(*module.split(".")).with_suffix(".olean")
    return any((d / rel).is_file() for d in search_dirs)


# ── Trusted checker ───────────────────────────────────────────────────────────

# Fixed source; the task and candidate reach it only as arguments (nonce, target name, .olean paths).
TRUSTED_CHECK = r"""import Lean
import Lean.Replay
open Lean

def collectAx (env : Environment) (n : Name) : Array Name :=
  let ((), s) := ((Elab.Command.CollectAxioms.collect n).run env).run {}
  s.axioms

def main (args : List String) : IO UInt32 := do
  let [nonce, target, candPath, refPath] := args | return 2
  let out (k v : String) : IO Unit := IO.println s!"GWAYA-{nonce} {k} {v}"
  let tgt := target.toName
  let some bin := (← IO.appPath).parent | out "error" "no_sysroot"; return 1
  initSearchPath (bin.parent.getD bin)
  let (refMod, _) ← readModuleData refPath
  let (candMod, _) ← readModuleData candPath
  let some refCi := refMod.constants.find? (·.name == tgt) | out "error" "ref_missing"; return 1
  unless candMod.constNames.contains tgt do
    out "error" "target_missing"; return 1
  let env ← try importModules candMod.imports {} 0
    catch e => do out "error" "import"; IO.eprintln s!"{e}"; return (1 : UInt32)
  let mut m : HashMap Name ConstantInfo := {}
  for c in candMod.constants do m := m.insert c.name c
  let env ← try env.replay m
    catch e => do out "error" "replay"; IO.eprintln s!"{e}"; return (1 : UInt32)
  let some ci := env.find? tgt | out "error" "target_not_replayed"; return 1
  out "kind" (if ci matches .thmInfo _ then "theorem" else "other")
  out "levels" (toString (ci.levelParams == refCi.levelParams))
  let mtch := if ci.type == refCi.type then "exact" else
    match Kernel.isDefEq env {} refCi.type ci.type with
    | .ok true => "defeq"
    | _ => "none"
  out "match" mtch
  out "axioms" (",".intercalate ((collectAx env tgt).toList.map toString))
  out "done" "1"
  return 0
"""

_TRUSTED_KEYS = ("error", "kind", "levels", "match", "axioms", "done")
# trusted-process errors attributable to the candidate (FAILED); anything else is UNVERIFIED.
_CANDIDATE_ERRORS = {"target_missing": "target_theorem_missing", "replay": "kernel_replay_failed",
                     "target_not_replayed": "kernel_replay_failed"}


def parse_trusted(stdout: str, nonce: str) -> dict[str, str] | None:
    """Fields printed by the trusted process under this run's nonce; None when a key repeats or is unknown."""
    tag = f"GWAYA-{nonce} "
    fields: dict[str, str] = {}
    for line in stdout.splitlines():
        if not line.startswith(tag):
            continue
        key, _, val = line[len(tag):].partition(" ")
        if key not in _TRUSTED_KEYS or key in fields:
            return None
        fields[key] = val.strip()
    return fields


def interpret_trusted(fields: dict[str, str] | None) -> tuple[str, str, dict[str, Any]]:
    """(status, reason, evidence) from the trusted fields. VERIFIED only on a complete, consistent report."""
    if fields is None:
        return "UNVERIFIED", "trusted_check_malformed", {}
    if "error" in fields:
        err = fields["error"]
        reason = _CANDIDATE_ERRORS.get(err)
        return ("FAILED", reason, {"trusted_error": err}) if reason else \
            ("UNVERIFIED", "trusted_check_error", {"trusted_error": err})
    if fields.get("done") != "1" or not {"kind", "levels", "match", "axioms"} <= fields.keys():
        return "UNVERIFIED", "trusted_check_incomplete", {}
    axioms = sorted(a for a in fields["axioms"].split(",") if a)
    ev = {"match": fields["match"], "axioms": axioms}
    if fields["kind"] != "theorem":
        return "FAILED", "target_not_theorem", ev
    if fields["levels"] != "true" or fields["match"] not in ("exact", "defeq"):
        return "FAILED", "statement_mismatch", ev
    bad = sorted(set(axioms) - STANDARD_AXIOMS)
    if bad:
        return "FAILED", "nonstandard_axioms", {**ev, "disallowed_axioms": bad}
    return "VERIFIED", "verified", ev


# ── Pipeline ──────────────────────────────────────────────────────────────────

@dataclass
class GateVerdict:
    status: str  # VERIFIED | FAILED | UNVERIFIED
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    log: str = ""


_ENV_MARKERS = ("unknown package", "unknown module prefix", "object file", "no such file or directory")
_REF_CACHE: dict[str, bytes] = {}
_REF_LOCK = threading.Lock()
_MAX_OLEAN = 256 * 1024 * 1024


def _run(project: Any, cmd: list[str], work: Path, timeout_s: float, files: dict[str, str] | None = None
         ) -> tuple[bool, str, str, bool]:
    return sandbox.run_in_sandbox(cmd, timeout_s=timeout_s, cpu_s=int(2 * timeout_s) + 10, work_dir=work,
                                  files=files, ro_binds=list(project.ro_binds),
                                  env={"LEAN_PATH": project.lean_path_env})


def _compile(project: Any, name: str, source: str, timeout_s: float, mem_mb: int
             ) -> tuple[str, bytes | None, str]:
    """('ok'|'error'|'timeout', olean bytes, log). The olean is read only if it is a regular file."""
    work = Path(tempfile.mkdtemp(prefix="gwaya_lean_"))
    try:
        ok, out, err, timed_out = _run(project, [str(project.lean_bin), "-M", str(mem_mb), "-o",
                                                  f"/work/{name}.olean", f"/work/{name}.lean"],
                                       work, timeout_s, {f"{name}.lean": source})
        if timed_out:
            return "timeout", None, err
        olean = work / f"{name}.olean"
        if not ok or olean.is_symlink() or not olean.is_file() or olean.stat().st_size > _MAX_OLEAN:
            return "error", None, (out + "\n" + err).strip()
        return "ok", olean.read_bytes(), (out + "\n" + err).strip()
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _reference(project: Any, header: str, stmt: str, timeout_s: float, mem_mb: int) -> tuple[str, bytes | None, str]:
    src = reference_source(header, stmt)
    key = hashlib.sha256("\0".join((str(project.lean_bin), project.lean_path_env, src)).encode()).hexdigest()
    with _REF_LOCK:
        if key in _REF_CACHE:
            return "ok", _REF_CACHE[key], ""
    state, blob, log = _compile(project, "Ref", src, timeout_s, mem_mb)
    if state == "ok" and blob is not None:
        with _REF_LOCK:
            _REF_CACHE[key] = blob
    return state, blob, log


def _search_dirs(project: Any) -> list[Path]:
    return [Path(p) for p in project.lean_path] + [Path(project.lean_bin).parent.parent / "lib" / "lean"]


def verify_theorem(project: Any, code: str, formal_statement: str, header: str = "", timeout_s: float = 120.0,
                   mem_mb: int = 4096) -> GateVerdict:
    """Hardened check of `code` (already with the task header if it had no imports) against the formal statement."""
    t0 = time.perf_counter()

    def done(status: str, reason: str, log: str = "", **ev: Any) -> GateVerdict:
        ev["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        return GateVerdict(status, reason, ev, log[-2000:])

    if not (formal_statement or "").strip():
        return done("UNVERIFIED", "no_formal_statement")
    target = target_name(formal_statement)
    if target is None:
        return done("UNVERIFIED", "bad_formal_statement")
    if not code.strip():
        return done("FAILED", "empty_response")
    forbidden = lexical_flaws(code)
    if forbidden:
        return done("FAILED", "forbidden_construct", forbidden=forbidden)
    if not statement_preserved(normalize_statement(formal_statement), code):
        return done("FAILED", "statement_not_preserved")
    if project is None:
        return done("UNVERIFIED", "lean_project_missing")
    dirs = _search_dirs(project)
    header_mods = set(imports_of(header))
    missing = [m for m in imports_of(code) if not resolve_import(m, dirs)]
    if any(m in header_mods for m in missing) or [m for m in header_mods if not resolve_import(m, dirs)]:
        return done("UNVERIFIED", "lean_environment_missing_dependency", missing=missing)
    if missing:
        return done("FAILED", "unknown_import", missing=missing)
    return _trusted_pipeline(project, code, formal_statement, header, target, timeout_s, mem_mb, done)


def _trusted_pipeline(project: Any, code: str, formal_statement: str, header: str, target: str, timeout_s: float,
                      mem_mb: int, done: Any) -> GateVerdict:
    """Steps 2-4 (no lexical rules: tests call this directly to show the trusted check alone rejects exploits)."""
    state, ref, log = _reference(project, header, formal_statement, timeout_s, mem_mb)
    if state == "timeout":
        return done("UNVERIFIED", "timeout", stage="reference", timeout_s=timeout_s)
    if ref is None:
        return done("UNVERIFIED", "reference_not_elaborated", log)
    state, cand, log = _compile(project, "Cand", code, timeout_s, mem_mb)
    if state == "timeout":
        return done("UNVERIFIED", "timeout", stage="candidate", timeout_s=timeout_s)
    if cand is None:
        if any(m in log.lower() for m in _ENV_MARKERS):
            return done("UNVERIFIED", "lean_environment_missing_dependency", log)
        return done("FAILED", "lean_error", log)
    nonce = secrets.token_hex(16)
    work = Path(tempfile.mkdtemp(prefix="gwaya_lean_trusted_"))
    try:
        (work / "Check.lean").write_text(TRUSTED_CHECK, encoding="utf-8")
        (work / "Cand.olean").write_bytes(cand)
        (work / "Ref.olean").write_bytes(ref)
        _, out, err, timed_out = _run(project, [str(project.lean_bin), "--run", "/work/Check.lean", nonce, target,
                                                "/work/Cand.olean", "/work/Ref.olean"], work, timeout_s)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if timed_out:
        return done("UNVERIFIED", "timeout", stage="trusted", timeout_s=timeout_s)
    status, reason, ev = interpret_trusted(parse_trusted(out, nonce))
    return done(status, reason, "\n".join(x for x in (out + "\n" + err).splitlines() if nonce not in x),
                target=target, **ev)
