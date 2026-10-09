#!/usr/bin/env python3
"""A13 data side: build the pool P (disjoint from E), split it by cluster, write the files.

Sources (no model is run; the network is used only to download datasets):
  math    EleutherAI/hendrycks_math TRAIN split only (answer = last \\boxed{} of the solution),
          openai/gsm8k TRAIN split (answer after '####')
  python  google-research-datasets/mbpp (config full) minus every task in evalplus/mbppplus
          (matched by task id AND by normalised prompt hash); hidden tests = MBPP test_list,
          the gate sees only the first assert.
Rows use the schema of tasks_E_primary.d25.jsonl: domain, task_id, prompt, checker_payload,
gate_payload, cluster (payloads are dicts; math gate_payload is {}).
Split (A13, deviation D48): bucket = int(sha256(cluster), 16) % 10; 0-5 train, 6 calib, 7-9 eval (E').
E itself goes to TRAIN only and is not rewritten here.

usage: build_pool_p.py --e-tasks .../tasks_E_primary.d25.jsonl --out .../laya_data [--no-sanity]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "data"))
import build_eval_manifest as B  # noqa: E402

SEED = 20261009
MATH_N = 1500
GSM8K_N = 600
MATH_SUFFIX = B.MATH_SUFFIX
MBPP_MARK = "\nYour code should pass this test:"
HF_CACHE = "/mnt/data/home/xavkal/gwaya-data/hf-datasets"
MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus"]


def phash(s: str) -> str:
    return B.sha256_text(B.normalize(s))


def stmt_of(prompt: str) -> str:
    """Problem statement part of a prompt (MBPP prompts append the visible assert)."""
    return prompt.split(MBPP_MARK)[0]


# ---- math ---------------------------------------------------------------------------------

def math_row(subject: str, idx: int, r: dict[str, Any]) -> dict[str, Any] | None:
    from gwaya.domains.math_check import _last_boxed
    ans = _last_boxed(r.get("solution") or "")
    if not ans or not ans.strip():
        return None
    prompt = (r["problem"] or "").strip()
    if not prompt:
        return None
    return {"domain": "math", "task_id": f"ma/train/{subject}/{idx}", "prompt": prompt + MATH_SUFFIX,
            "checker_payload": {"answer": ans.strip()}, "gate_payload": {},
            "cluster": f"MATH/train/{subject}/{idx}",
            "_stratum": f"{subject}|{r.get('level', '')}"}


def gsm8k_answer(ans: str) -> str | None:
    if "####" not in (ans or ""):
        return None
    a = ans.rsplit("####", 1)[1].strip().replace(",", "")
    return a or None


def gsm8k_row(idx: int, r: dict[str, Any]) -> dict[str, Any] | None:
    a = gsm8k_answer(r.get("answer") or "")
    q = (r.get("question") or "").strip()
    if a is None or not q:
        return None
    return {"domain": "math", "task_id": f"ma/gsm8k/{idx}", "prompt": q + MATH_SUFFIX,
            "checker_payload": {"answer": a}, "gate_payload": {}, "cluster": f"GSM8K/train/{idx}"}


def stratified_sample(rows: list[dict], n: int, seed: int, key: Callable[[dict], str]) -> list[dict]:
    """Seeded, proportional (largest remainder) sample of up to n rows over strata `key`."""
    if n >= len(rows):
        return sorted(rows, key=lambda r: r["task_id"])
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    alloc = B.apportion(n, {k: len(v) for k, v in groups.items()})
    out: list[dict] = []
    for k in sorted(groups):
        g = sorted(groups[k], key=lambda r: r["task_id"])
        random.Random(f"{seed}:{k}").shuffle(g)
        out.extend(g[:alloc[k]])
    return sorted(out, key=lambda r: r["task_id"])


def sample_gsm8k(rows: list[dict], n: int, seed: int) -> list[dict]:
    g = sorted(rows, key=lambda r: r["task_id"])
    random.Random(f"{seed}:gsm8k").shuffle(g)
    return sorted(g[:n], key=lambda r: r["task_id"])


# ---- python -------------------------------------------------------------------------------

def mbpp_row(r: dict[str, Any]) -> dict[str, Any] | None:
    tl = list(r.get("test_list") or [])
    text = (r.get("text") or "").strip()
    if not tl or not text or not (r.get("code") or "").strip():
        return None
    setup = (r.get("test_setup_code") or "").strip()
    pre = setup + "\n" if setup else ""
    first = tl[0]
    prompt = f"{text}\nYour code should pass this test:\n{first}\nReply with the full Python function."
    tid = int(r["task_id"])
    return {"domain": "python", "task_id": f"py/MBPP/{tid}", "prompt": prompt,
            "checker_payload": {"tests": pre + "\n".join(tl), "timeout_s": 30.0},
            "gate_payload": {"tests": pre + first, "timeout_s": 10.0},
            "cluster": f"MBPP/{tid}", "_reference": r["code"].replace("\r\n", "\n")}


def mbpp_overlap_with_plus(r: dict[str, Any], plus_ids: set[int], plus_hashes: set[str]) -> str | None:
    if int(r["task_id"]) in plus_ids:
        return "task_id_in_mbppplus"
    if phash(r.get("text") or "") in plus_hashes:
        return "prompt_hash_in_mbppplus"
    return None


# ---- decontamination / split --------------------------------------------------------------

def decontaminate(rows: list[dict], e_rows: list[dict]) -> tuple[list[dict], dict]:
    e_full = {phash(r["prompt"]) for r in e_rows}
    e_stmt = {phash(stmt_of(r["prompt"])) for r in e_rows}
    e_cl = {r["cluster"] for r in e_rows}
    kept, drops = [], []
    for r in rows:
        why = None
        if r["cluster"] in e_cl:
            why = "cluster_in_E"
        elif phash(r["prompt"]) in e_full:
            why = "prompt_hash_in_E"
        elif phash(stmt_of(r["prompt"])) in e_stmt:
            why = "statement_hash_in_E"
        if why:
            drops.append({"task_id": r["task_id"], "reason": why})
        else:
            kept.append(r)
    rep = {"n_in": len(rows), "n_kept": len(kept), "n_dropped": len(drops),
           "by_reason": dict(Counter(d["reason"] for d in drops)),
           "by_domain": dict(Counter(d["task_id"].split("/")[0] for d in drops)), "dropped": drops}
    return kept, rep


def bucket_of(cluster: str) -> int:
    return int(hashlib.sha256(cluster.encode("utf-8")).hexdigest(), 16) % 10


def split_of(cluster: str) -> str:
    b = bucket_of(cluster)
    return "train" if b <= 5 else ("calib" if b == 6 else "eval")


# ---- harness sanity -----------------------------------------------------------------------

def sanity_python(row: dict) -> str | None:
    """Reference solution must pass the hidden AND the gate tests in the repo's python oracle."""
    from gwaya.domains.checkers import check_python
    from gwaya.domains.task import Task
    fenced = "```python\n" + row["_reference"] + "\n```"
    for label, p in (("hidden", row["checker_payload"]), ("gate", row["gate_payload"])):
        res = check_python(Task("python", row["task_id"], "", dict(p)), fenced)
        if res.status != "VERIFIED":
            return f"reference_fails_{label}_checks:{res.status}"
    return None


def sanity_math(row: dict) -> str | None:
    from gwaya.domains.math_check import check_math
    gold = row["checker_payload"]["answer"]
    if not gold.strip():
        return "empty_answer"
    if check_math("\\boxed{" + gold + "}", gold).status != "VERIFIED":
        return "gold_not_self_verifying"
    return None


def run_sanity(rows: list[dict], workers: int = 8) -> tuple[list[dict], list[dict]]:
    from concurrent.futures import ThreadPoolExecutor
    def one(r):
        return sanity_python(r) if r["domain"] == "python" else sanity_math(r)
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, rows))
    kept = [r for r, w in zip(rows, res) if w is None]
    drops = [{"task_id": r["task_id"], "reason": w} for r, w in zip(rows, res) if w is not None]
    return kept, drops


def clean(r: dict) -> dict:
    return {k: v for k, v in r.items() if not k.startswith("_")}


def dump(path: Path, rows: list[dict]) -> str:
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---- driver -------------------------------------------------------------------------------

def load_sources(cache: str) -> dict[str, Any]:
    from datasets import load_dataset
    from huggingface_hub import HfApi
    api = HfApi()
    src: dict[str, Any] = {"revisions": {}, "problems": []}
    def rev(repo):
        try:
            src["revisions"][repo] = api.dataset_info(repo).sha
        except Exception as e:  # noqa: BLE001
            src["revisions"][repo] = f"unknown ({type(e).__name__})"
    math_all: list[dict] = []
    try:
        rev("EleutherAI/hendrycks_math")
        for sub in MATH_SUBJECTS:
            for i, r in enumerate(load_dataset("EleutherAI/hendrycks_math", sub, split="train", cache_dir=cache)):
                m = math_row(sub, i, dict(r))
                if m is None:
                    src["problems"].append(f"MATH train {sub}/{i}: no boxed answer, dropped")
                else:
                    math_all.append(m)
    except Exception as e:  # noqa: BLE001
        src["problems"].append(f"EleutherAI/hendrycks_math failed: {e!r}")
    src["math_all"] = math_all
    gsm: list[dict] = []
    try:
        rev("openai/gsm8k")
        for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="train", cache_dir=cache)):
            g = gsm8k_row(i, dict(r))
            if g is None:
                src["problems"].append(f"GSM8K train {i}: no '####' answer, dropped")
            else:
                gsm.append(g)
    except Exception as e:  # noqa: BLE001
        src["problems"].append(f"openai/gsm8k failed: {e!r}")
    src["gsm_all"] = gsm
    mbpp: list[dict] = []
    plus_ids: set[int] = set()
    plus_hashes: set[str] = set()
    try:
        rev("google-research-datasets/mbpp")
        rev("evalplus/mbppplus")
        for r in load_dataset("evalplus/mbppplus", split="test", cache_dir=cache):
            plus_ids.add(int(r["task_id"]))
            plus_hashes.add(phash(r["prompt"]))
        d = load_dataset("google-research-datasets/mbpp", "full", cache_dir=cache)
        for sp in d:
            for r in d[sp]:
                mbpp.append(dict(r))
    except Exception as e:  # noqa: BLE001
        src["problems"].append(f"mbpp / mbppplus failed: {e!r}")
    src["mbpp_raw"], src["plus_ids"], src["plus_hashes"] = mbpp, plus_ids, plus_hashes
    return src


def build(src: dict[str, Any], e_rows: list[dict], do_sanity: bool = True, workers: int = 8,
          math_n: int = MATH_N, gsm_n: int = GSM8K_N, seed: int = SEED) -> tuple[list[dict], dict]:
    rep: dict[str, Any] = {"seed": seed, "problems": list(src.get("problems", []))}
    math_s = stratified_sample(src.get("math_all", []), math_n, seed, lambda r: r["_stratum"])
    gsm_s = sample_gsm8k(src.get("gsm_all", []), gsm_n, seed)
    py_rows, plus_drop = [], Counter()
    for r in src.get("mbpp_raw", []):
        w = mbpp_overlap_with_plus(r, src["plus_ids"], src["plus_hashes"])
        if w:
            plus_drop[w] += 1
            continue
        row = mbpp_row(r)
        if row is None:
            plus_drop["unusable_row"] += 1
        else:
            py_rows.append(row)
    rep["mbpp"] = {"n_raw": len(src.get("mbpp_raw", [])), "dropped_vs_mbppplus": dict(plus_drop),
                   "n_candidates": len(py_rows)}
    rep["math"] = {"n_math_train_available": len(src.get("math_all", [])), "n_math_sampled": len(math_s),
                   "n_gsm8k_available": len(src.get("gsm_all", [])), "n_gsm8k_sampled": len(gsm_s)}
    rows = math_s + gsm_s + py_rows
    rows, dec = decontaminate(rows, e_rows)
    rep["decontamination"] = dec
    if do_sanity:
        rows, sdrops = run_sanity(rows, workers)
    else:
        sdrops = []
    rep["sanity"] = {"ran": do_sanity, "n_dropped": len(sdrops),
                     "by_domain": dict(Counter(d["task_id"].split("/")[0] for d in sdrops)), "dropped": sdrops}
    ids = [r["task_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    out = []
    for r in sorted(rows, key=lambda r: r["task_id"]):
        c = clean(r)
        c["split"] = split_of(c["cluster"])
        out.append(c)
    return out, rep


def counts(rows: list[dict]) -> dict:
    out: dict[str, Any] = {}
    for dom in sorted({r["domain"] for r in rows}):
        out[dom] = dict(Counter(r["split"] for r in rows if r["domain"] == dom))
    out["all"] = dict(Counter(r["split"] for r in rows))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e-tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=HF_CACHE)
    ap.add_argument("--no-sanity", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import os
    os.environ.setdefault("HF_HOME", a.cache)
    e_path = Path(a.e_tasks)
    e_rows = [json.loads(line) for line in e_path.read_text().splitlines() if line.strip()]
    src = load_sources(a.cache)
    rows, rep = build(src, e_rows, not a.no_sanity, a.workers)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "tasks_P_all.jsonl": rows,
        "tasks_P_train.jsonl": [r for r in rows if r["split"] == "train"],
        "tasks_P_calib.jsonl": [r for r in rows if r["split"] == "calib"],
        "tasks_Eprime.jsonl": [r for r in rows if r["split"] == "eval"],
    }
    sha = {n: dump(out / n, rs) for n, rs in files.items()}
    manifest = {
        "addendum": "A13", "deviation": "D48", "seed": rep["seed"],
        "split_rule": "int(sha256(cluster).hexdigest(),16) % 10: 0-5 train, 6 calib, 7-9 eval (E')",
        "e_note": "E (all tasks of tasks_E_primary.d25.jsonl) goes to TRAIN only; not rewritten here",
        "e_tasks_file": str(e_path), "e_tasks_sha256": hashlib.sha256(e_path.read_bytes()).hexdigest(),
        "e_rows": len(e_rows),
        "source_revisions": src["revisions"],
        "files": {n: {"rows": len(rs), "sha256": sha[n]} for n, rs in files.items()},
        "counts_per_domain_split": counts(rows),
        "counts_per_source": dict(Counter("/".join(r["task_id"].split("/")[:2]) for r in rows)),
        "sampling": rep["math"], "mbpp": rep["mbpp"],
        "decontamination": {k: v for k, v in rep["decontamination"].items() if k != "dropped"},
        "sanity": {k: v for k, v in rep["sanity"].items() if k != "dropped"},
        "source_problems": rep["problems"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "decontamination_report.json").write_text(json.dumps(
        {"decontamination": rep["decontamination"], "sanity_dropped": rep["sanity"]["dropped"],
         "mbpp_vs_mbppplus": rep["mbpp"]}, indent=1))
    print(json.dumps({k: manifest[k] for k in ("files", "counts_per_domain_split", "sampling", "mbpp",
                                                "decontamination", "sanity", "source_problems",
                                                "source_revisions")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
