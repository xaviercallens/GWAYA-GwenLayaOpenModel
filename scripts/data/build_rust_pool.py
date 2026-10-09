#!/usr/bin/env python3
"""Fresh Rust evaluation tasks from Exercism's Rust track (MIT licence), for the harness' assertion format.

Source: https://github.com/exercism/rust (practice exercises, no external crates). Each exercise becomes one task:
  prompt          the exercise instructions + the starter code (todo!() stubs); the tests are NOT shown
  checker_payload tests = module-level `use`/helper items of the test file + every `#[test]` body as a plain block
  gate_payload    the same with only the FIRST test block (what the gate may see)
An exercise is kept only if its own reference solution (.meta/example.rs) is VERIFIED by the real checker on the hidden
tests AND the unmodified starter stub is not (so the tests actually test something); everything else is reported.

  python scripts/data/build_rust_pool.py --repo /path/to/exercism-rust --out tasks_Rust_fresh.jsonl --report report.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ATTR = r"(?:\s*#\[[^\]]*\])*"
_TEST_FN = re.compile(r"#\[test\]" + _ATTR + r"\s*fn\s+(\w+)\s*\(\s*\)\s*(->\s*[^{]+)?\{")
MAX_REF_MS = 4000.0  # a reference that needs more than 4 s of the 10 s budget would flip with machine load
PROMPT_HEAD = ("Implement the following Rust exercise. Reply with the complete Rust code (all types, functions and impls "
               "the tests need, using the starter code as the interface) and no main.\n\n")


def brace_end(src: str, open_idx: int) -> int | None:
    """Index just past the matching `}` for the `{` at open_idx; skips string/char literals and comments."""
    depth, i, n = 0, open_idx, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            i = src.find("\n", i)
            if i < 0:
                return None
            continue
        if src.startswith("/*", i):
            i = src.find("*/", i)
            if i < 0:
                return None
            i += 2
            continue
        if c == '"':
            i += 1
            while i < n and src[i] != '"':
                i += 2 if src[i] == "\\" else 1
        elif c == "'":
            m = re.match(r"'(?:\\.[^']*|[^\\'])'", src[i:])
            if m:
                i += m.end() - 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def convert_tests(src: str, crate: str) -> tuple[str, list[str], dict[str, Any]]:
    """-> (module-level items text, [test block, ...], info). Raises ValueError for shapes we do not convert."""
    if "should_panic" in src:
        raise ValueError("should_panic")
    src = re.sub(r"#\[ignore[^\]]*\]\s*", "", src)
    # the tests live in the same file as the candidate: drop `use crate::..;` (anywhere) and path-qualifiers `crate::f(..)`
    src = re.sub(rf"^[ \t]*use\s+{re.escape(crate)}::[^;]*;[ \t]*\n?", "", src, flags=re.M)
    src = re.sub(rf"\b{re.escape(crate)}::", "", src)
    items, blocks, pos = [], [], 0
    for m in _TEST_FN.finditer(src):
        if m.start() < pos:
            continue
        if m.group(2):
            raise ValueError("test_returns_value")
        end = brace_end(src, m.end() - 1)
        if end is None:
            raise ValueError("unbalanced_braces")
        items.append(src[pos:m.start()])
        blocks.append("{" + src[m.end():end - 1] + "}")
        pos = end
    items.append(src[pos:])
    head = "".join(items)
    head = re.sub(rf"^\s*extern\s+crate\s+{re.escape(crate)}\s*;\s*$", "", head, flags=re.M)
    head = re.sub(r"^\s*#!?\[[^\]]*\]\s*$", "", head, flags=re.M).strip()
    if "fn main" in head or "#[test]" in head:
        raise ValueError("leftover_main_or_test")
    if re.search(r"^\s*use\s+(?!std::|core::|alloc::)\w+", head, flags=re.M):
        raise ValueError("external_crate_in_tests")
    if not blocks:
        raise ValueError("no_test_functions")
    return head, blocks, {"n_tests": len(blocks)}


def assemble(head: str, blocks: list[str]) -> str:
    return (head + "\n" if head else "") + "\n".join(blocks) + "\n"


def build(repo: Path, check_fn) -> tuple[list[dict], dict[str, Any]]:
    base = repo / "exercises" / "practice"
    rows, report = [], {"kept": [], "dropped": {}}
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        slug = d.name
        try:
            cargo = tomllib.loads((d / "Cargo.toml").read_text())
            if cargo.get("dependencies"):
                raise ValueError("has_dependencies")
            crate = cargo["package"]["name"].replace("-", "_")
            tests_files = sorted((d / "tests").glob("*.rs"))
            if len(tests_files) != 1:
                raise ValueError("tests_files!=1")
            example = (d / ".meta" / "example.rs").read_text()
            stub = (d / "src" / "lib.rs").read_text()
            head, blocks, info = convert_tests(tests_files[0].read_text(), crate)
            docs = (d / ".docs" / "instructions.md").read_text()
            extra = d / ".docs" / "instructions.append.md"
            if extra.exists():
                docs += "\n\n" + extra.read_text()
        except (ValueError, KeyError, FileNotFoundError, OSError) as exc:
            report["dropped"][slug] = f"{type(exc).__name__}: {exc}"
            continue
        task = {
            "domain": "rust", "task_id": f"rs/exercism/{slug}", "cluster": f"exercism/{slug}",
            "prompt": PROMPT_HEAD + docs.strip() + "\n\nStarter code:\n```rust\n" + stub.strip() + "\n```",
            "checker_payload": {"tests": assemble(head, blocks)},
            "gate_payload": {"tests": assemble(head, blocks[:1]), "timeout_s": 10.0},
        }
        ref_ok, ref_ms = check_fn(task, "```rust\n" + example + "\n```")
        stub_ok, _ = check_fn(task, "```rust\n" + stub + "\n```")
        if ref_ok != "VERIFIED":
            report["dropped"][slug] = f"reference_not_verified:{ref_ok}"
            continue
        if ref_ms is not None and ref_ms > MAX_REF_MS:
            report["dropped"][slug] = f"reference_too_slow_for_a_stable_timeout:{ref_ms:.0f}ms"
            continue
        if stub_ok == "VERIFIED":
            report["dropped"][slug] = "starter_stub_passes_tests"
            continue
        rows.append(task)
        report["kept"].append({"slug": slug, "tests": info["n_tests"], "stub_status": stub_ok})
    report["n_kept"], report["n_dropped"] = len(rows), len(report["dropped"])
    return rows, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    a = ap.parse_args(argv)
    from gwaya.domains.checkers import check
    from gwaya.domains.task import Task

    def check_fn(t: dict, response: str) -> tuple[str, float | None]:
        r = check(Task(t["domain"], t["task_id"], t["prompt"], dict(t["checker_payload"])), response)
        return r.status, (r.evidence or {}).get("latency_ms")

    rows, report = build(a.repo, check_fn)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report["source"] = "https://github.com/exercism/rust (MIT)"
    a.report.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("n_kept", "n_dropped")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
