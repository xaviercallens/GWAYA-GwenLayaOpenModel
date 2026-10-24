#!/usr/bin/env python3
"""P1 of docs/PROPOSAL_HIVEMIND_GWAYA_CODER.md: MBPP-sanitized pass@1 of a GWAYA-Lite tier served by llama-server.

Generation is remote (any OpenAI-format endpoint, e.g. an ssh tunnel to the T4); candidate code and tests run HERE in the
bwrap sandbox. One pass of ``gwaya.assistant.Assistant`` gives both arms on the same problems:
  A0  the first greedy sample (no verifier, no repair)
  A3  the assistant's answer: the code of the round that passed the PUBLIC test (test_list[0]) after up to ``--rounds`` rounds,
      otherwise the A0 sample (a "verified answer, else best effort" policy)
Outcome for both arms = the final code passes ALL tests of the problem (hidden tests are never shown to the model); the
assistant's own VERIFIED flag is never the outcome. Rows are appended to ``<out>/rows.jsonl`` and the run resumes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gwaya.assistant import Assistant  # noqa: E402
from gwaya.llama_tier import LlamaServerClient  # noqa: E402
from gwaya.test_harness import run_candidate_tests  # noqa: E402
from run_benchmark import load_mbpp, problem_from_row  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p of discordant counts b (A3 only) and c (A0 only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    a0 = sum(r["a0_pass"] for r in rows)
    a3 = sum(r["a3_pass"] for r in rows)
    gained = sum(1 for r in rows if r["a3_pass"] and not r["a0_pass"])
    lost = sum(1 for r in rows if r["a0_pass"] and not r["a3_pass"])
    ver = [r for r in rows if r["verdict"] == "VERIFIED"]
    return {
        "n": n, "a0_pass": a0, "a3_pass": a3,
        "a0_pct": round(100 * a0 / n, 2) if n else 0, "a3_pct": round(100 * a3 / n, 2) if n else 0,
        "a0_wilson95": [round(100 * x, 1) for x in wilson(a0, n)], "a3_wilson95": [round(100 * x, 1) for x in wilson(a3, n)],
        "gained": gained, "lost": lost, "mcnemar_p": round(mcnemar_exact(gained, lost), 4),
        "verified_claimed": len(ver), "verified_hidden_pass": sum(r["a3_pass"] for r in ver),
        "verified_hidden_precision": round(sum(r["a3_pass"] for r in ver) / len(ver), 4) if ver else None,
        "mean_gen_s": round(sum(r["gen_s"] for r in rows) / n, 2) if n else 0,
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", required=True, help="model name, only a label for the server")
    ap.add_argument("--n", type=int, default=257)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--mbpp-dir", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_p = out / "rows.jsonl"
    rows = [json.loads(x) for x in rows_p.read_text().splitlines()] if rows_p.exists() else []
    done = {r["task_id"] for r in rows}
    df = load_mbpp(Path(a.mbpp_dir) if a.mbpp_dir else None)["test"].sort_values("task_id").head(a.n)
    asst = Assistant(LlamaServerClient(a.model, a.base_url, max_new_tokens=a.max_new, timeout_s=600), max_rounds=a.rounds)
    for _, row in df.iterrows():
        p = problem_from_row(row)
        if p["task_id"] in done:
            continue
        task = f"{p['goal']}\nYour code should pass this test:\n{p['public']}"
        t0 = time.perf_counter()
        res = asst.solve(task, p["public"])
        gen_s = time.perf_counter() - t0
        if not res.rounds:
            print("generator unavailable:", res.note, file=sys.stderr)
            return 2
        a0_code = res.rounds[0].code
        a3_code = res.code if res.verdict == "VERIFIED" else a0_code
        a0 = run_candidate_tests(a0_code, p["all_tests"], timeout_s=10.0).success if a0_code else False
        a3 = a0 if a3_code == a0_code else (run_candidate_tests(a3_code, p["all_tests"], timeout_s=10.0).success if a3_code else False)
        rec = {"task_id": p["task_id"], "a0_pass": bool(a0), "a3_pass": bool(a3), "verdict": res.verdict,
               "rounds": len(res.rounds), "gen_s": round(gen_s, 2),
               "completion_tokens": sum(r.completion_tokens for r in res.rounds)}
        rows.append(rec)
        with rows_p.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    s = summarize(rows)
    s.update({"model": a.model, "rounds": a.rounds, "max_new_tokens": a.max_new, "base_url": a.base_url})
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
