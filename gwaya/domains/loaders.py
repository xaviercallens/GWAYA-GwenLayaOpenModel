"""Dataset loaders: read rows from the local `path` recorded in experiments/DATA_MANIFEST.json
(never downloads) and normalize them to Task. Heavy readers (pyarrow, datasets) are imported
lazily. Field names follow the public dataset cards as recalled and were NOT checked against
downloaded data in this session (no dataset is materialized yet: manifest paths are empty);
unknown schemas raise KeyError rather than guessing."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterator

from gwaya.domains.task import Task

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "experiments" / "DATA_MANIFEST.json"


class DatasetNotMaterialized(RuntimeError):
    """The manifest entry has no local path (or it does not exist)."""


def load_manifest(path: Path | str = MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def manifest_entry(dataset_id: str, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = manifest or load_manifest()
    for group in manifest.get("datasets", {}).values():
        for e in group:
            if e.get("id") == dataset_id or e.get("hf_id") == dataset_id:
                return e
    raise KeyError(f"dataset {dataset_id!r} not in manifest")


def iter_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise DatasetNotMaterialized(f"{p} does not exist")
    if p.is_dir():
        from datasets import load_from_disk  # lazy
        ds = load_from_disk(str(p))
        if hasattr(ds, "keys"):  # DatasetDict: take the only split, else require a file path
            keys = list(ds.keys())
            if len(keys) != 1:
                raise ValueError(f"{p} has splits {keys}; point the manifest at one split")
            ds = ds[keys[0]]
        yield from (dict(r) for r in ds)
    elif p.suffix == ".jsonl":
        with p.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif p.suffix == ".json":
        data = json.loads(p.read_text())
        yield from (data if isinstance(data, list) else data["data"])
    elif p.suffix == ".parquet":
        import pyarrow.parquet as pq  # lazy
        yield from pq.read_table(str(p)).to_pylist()
    else:
        raise ValueError(f"unsupported dataset file type: {p}")


def _boxed_last(text: str) -> str | None:
    from gwaya.domains.math_check import _last_boxed
    return _last_boxed(text)


def _python_humaneval(r: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    ep = r["entry_point"]
    tests = f"{r['test']}\nassert check({ep}) is None\n"
    return r["prompt"], {"tests": tests, "entry_point": ep}


def _python_mbpp(r: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    imports = "\n".join(r.get("test_imports") or [])
    tests = (imports + "\n" if imports else "") + "\n".join(r["test_list"])
    return r["prompt"] if "prompt" in r else r["text"], {"tests": tests}


def _rust_multipl_e(r: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return r["prompt"], {"tests": r["tests"]}


_LEAN_PLACEHOLDER = re.compile(r":=\s*(?:by\s+)?sorry\s*$")


def _lean(r: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    # miniF2F / Lean-Workbook statements end in ":= sorry" (or ":= by sorry"). The gate requires the statement to be
    # preserved verbatim, so the placeholder must not be part of it: keep everything up to and including ":=".
    raw = (r.get("formal_statement") or r["statement"]).rstrip()
    stmt = _LEAN_PLACEHOLDER.sub(":=", raw)
    header = r.get("header", "").rstrip()
    body = stmt[:-2].rstrip() if stmt.endswith(":=") else stmt
    shown = f"{header}\n\n{body} := by\n  sorry" if header else f"{body} := by\n  sorry"
    prompt = ("Complete the following Lean 4 proof: replace `sorry` with a proof. Do not change the imports or the "
              f"theorem statement, and reply with the complete Lean 4 file.\n\n```lean4\n{shown}\n```")
    return prompt, {"formal_statement": stmt, "header": header}


def _gsm8k(r: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return r["question"], {"answer": r["answer"].rsplit("####", 1)[1].strip().replace(",", "")}


def _hendrycks(r: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    ans = _boxed_last(r["solution"])
    if ans is None:
        raise KeyError("no \\boxed answer in solution")
    return r["problem"], {"answer": ans}


def _answer_row(r: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return r.get("problem") or r["question"], {"answer": str(r["answer"])}


# hf_id -> (domain, row converter, id keys)
_ADAPTERS: dict[str, tuple[str, Callable, tuple[str, ...]]] = {
    "evalplus/humanevalplus": ("python", _python_humaneval, ("task_id",)),
    "evalplus/mbppplus": ("python", _python_mbpp, ("task_id",)),
    "google-research-datasets/mbpp": ("python", _python_mbpp, ("task_id",)),
    "nuprl/MultiPL-E": ("rust", _rust_multipl_e, ("name",)),
    "cat-searcher/minif2f-lean4": ("lean4", _lean, ("id", "name")),
    "internlm/Lean-Workbook": ("lean4", _lean, ("id", "name")),
    "openai/gsm8k": ("math", _gsm8k, ("id",)),
    "EleutherAI/hendrycks_math": ("math", _hendrycks, ("unique_id", "id")),
    "HuggingFaceH4/MATH-500": ("math", _answer_row, ("unique_id", "id")),
    "math-ai/aime25": ("math", _answer_row, ("id", "problem_id")),
}


def row_to_task(hf_id: str, row: dict[str, Any], index: int) -> Task:
    domain, conv, id_keys = _ADAPTERS[hf_id]
    prompt, payload = conv(row)
    rid = next((str(row[k]) for k in id_keys if row.get(k) not in (None, "")), str(index))
    return Task(domain=domain, task_id=f"{hf_id}/{rid}", prompt=prompt, checker_payload=payload)


def load_tasks(dataset_id: str, limit: int | None = None, manifest: dict[str, Any] | None = None) -> list[Task]:
    """Load tasks for a manifest dataset id (e.g. 'EVAL001') or hf_id from its local path."""
    entry = manifest_entry(dataset_id, manifest)
    hf_id = entry["hf_id"]
    if hf_id not in _ADAPTERS:
        raise KeyError(f"no loader for {hf_id}")
    if not entry.get("path"):
        raise DatasetNotMaterialized(f"{entry['id']} ({hf_id}) has no local path in the manifest")
    out: list[Task] = []
    for i, row in enumerate(iter_rows(entry["path"])):
        out.append(row_to_task(hf_id, row, i))
        if limit is not None and len(out) >= limit:
            break
    return out
