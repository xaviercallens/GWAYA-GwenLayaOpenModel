#!/usr/bin/env python3
"""GWAYA v3 Multi-Model Intense AI Coding Benchmark.

Evaluates the Qwen2.5-Coder model series (0.5b, 1.5b, 3b, 7b) on local RTX GPU hardware.
Compares:
  A0: Raw zero-shot baseline (T=0.2, single sample)
  A2: Best-of-3 candidate sampling with ANSE energy selection
  A3: GWAYA v3 Best-of-3 + 3-turn compiler/harness bounded self-repair

Evaluates across:
  - Complex algorithmic tasks (DP, intervals, sliding window, backtracking, graph/grid, arrays)
  - Canonical MBPP-sanitized benchmark tasks
  - ZeroStubAudit fail-closed gate enforcement
  - Tokens/sec generation throughput and GPU VRAM profiling
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gwaya.generators import OllamaGenerator
from gwaya.low_tier_engine import LowTierModelOptimizer, ModelTier, extract_code_block
from gwaya.oracles import PythonCompilerOracle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("GwayaV3Benchmark")

# ─────────────────────────────────────────────────────────────────────────────
# AI Coding Tasks Suite: Algorithms, Systems, and MBPP-sanitized
# ─────────────────────────────────────────────────────────────────────────────

TASKS = [
    {
        "id": "task_01_merge_intervals",
        "category": "intervals",
        "goal": "Write a function merge_intervals(intervals: list[list[int]]) -> list[list[int]] that merges all overlapping intervals and returns an array of non-overlapping intervals sorted by start time.",
        "public": "assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]\n",
        "all_tests": (
            "assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]\n"
            "assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]\n"
            "assert merge_intervals([]) == []\n"
            "assert merge_intervals([[1, 4]]) == [[1, 4]]\n"
            "assert merge_intervals([[1, 4], [0, 4]]) == [[0, 4]]\n"
            "assert merge_intervals([[1, 4], [2, 3]]) == [[1, 4]]\n"
            "assert merge_intervals([[2, 3], [4, 5], [6, 7], [8, 9], [1, 10]]) == [[1, 10]]\n"
        ),
    },
    {
        "id": "task_02_longest_consecutive",
        "category": "hash_table",
        "goal": "Write a function longest_consecutive(nums: list[int]) -> int that returns the length of the longest consecutive elements sequence in O(n) time.",
        "public": "assert longest_consecutive([100, 4, 200, 1, 3, 2]) == 4\n",
        "all_tests": (
            "assert longest_consecutive([100, 4, 200, 1, 3, 2]) == 4\n"
            "assert longest_consecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]) == 9\n"
            "assert longest_consecutive([]) == 0\n"
            "assert longest_consecutive([7]) == 1\n"
            "assert longest_consecutive([1, 2, 0, 1]) == 3\n"
            "assert longest_consecutive([-1, -2, -3, 0, 1]) == 5\n"
        ),
    },
    {
        "id": "task_03_trap_rain_water",
        "category": "two_pointers",
        "goal": "Write a function trap_rain_water(height: list[int]) -> int that computes how much water it can trap after raining given n non-negative integers representing an elevation map.",
        "public": "assert trap_rain_water([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6\n",
        "all_tests": (
            "assert trap_rain_water([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6\n"
            "assert trap_rain_water([4, 2, 0, 3, 2, 5]) == 9\n"
            "assert trap_rain_water([]) == 0\n"
            "assert trap_rain_water([3]) == 0\n"
            "assert trap_rain_water([1, 2, 3, 4, 5]) == 0\n"
            "assert trap_rain_water([5, 4, 3, 2, 1]) == 0\n"
            "assert trap_rain_water([3, 0, 0, 2, 0, 4]) == 10\n"
        ),
    },
    {
        "id": "task_04_group_anagrams",
        "category": "hash_table",
        "goal": "Write a function group_anagrams(strs: list[str]) -> list[list[str]] that groups anagrams together. The inner lists and outer list can be in any order, but each anagram group must contain the exact elements.",
        "public": "res = group_anagrams(['eat', 'tea', 'tan', 'ate', 'nat', 'bat'])\nassert sorted([sorted(g) for g in res]) == sorted([sorted(['bat']), sorted(['nat', 'tan']), sorted(['ate', 'eat', 'tea'])])\n",
        "all_tests": (
            "res = group_anagrams(['eat', 'tea', 'tan', 'ate', 'nat', 'bat'])\n"
            "assert sorted([sorted(g) for g in res]) == sorted([sorted(['bat']), sorted(['nat', 'tan']), sorted(['ate', 'eat', 'tea'])])\n"
            "assert group_anagrams(['']) == [['']]\n"
            "assert group_anagrams(['a']) == [['a']]\n"
            "res2 = group_anagrams(['abc', 'bca', 'cab', 'xyz'])\n"
            "assert sorted([sorted(g) for g in res2]) == sorted([sorted(['xyz']), sorted(['abc', 'bca', 'cab'])])\n"
        ),
    },
    {
        "id": "task_05_search_rotated_array",
        "category": "binary_search",
        "goal": "Write a function search_rotated_array(nums: list[int], target: int) -> int that finds target in an ascending sorted array rotated at an unknown pivot in O(log n) time. Return index or -1 if not found.",
        "public": "assert search_rotated_array([4, 5, 6, 7, 0, 1, 2], 0) == 4\n",
        "all_tests": (
            "assert search_rotated_array([4, 5, 6, 7, 0, 1, 2], 0) == 4\n"
            "assert search_rotated_array([4, 5, 6, 7, 0, 1, 2], 3) == -1\n"
            "assert search_rotated_array([1], 0) == -1\n"
            "assert search_rotated_array([1], 1) == 0\n"
            "assert search_rotated_array([3, 1], 1) == 1\n"
            "assert search_rotated_array([5, 1, 3], 5) == 0\n"
            "assert search_rotated_array([5, 1, 3], 3) == 2\n"
        ),
    },
    {
        "id": "task_06_subarray_sum_k",
        "category": "prefix_sum",
        "goal": "Write a function subarray_sum_k(nums: list[int], k: int) -> int that returns the total number of continuous subarrays whose sum equals to k.",
        "public": "assert subarray_sum_k([1, 1, 1], 2) == 2\n",
        "all_tests": (
            "assert subarray_sum_k([1, 1, 1], 2) == 2\n"
            "assert subarray_sum_k([1, 2, 3], 3) == 2\n"
            "assert subarray_sum_k([1, -1, 0], 0) == 3\n"
            "assert subarray_sum_k([3, 4, 7, 2, -3, 1, 4, 2], 7) == 4\n"
            "assert subarray_sum_k([], 5) == 0\n"
            "assert subarray_sum_k([0, 0, 0], 0) == 6\n"
        ),
    },
    {
        "id": "task_07_longest_valid_parentheses",
        "category": "dp_stack",
        "goal": "Write a function longest_valid_parentheses(s: str) -> int that returns the length of the longest valid (well-formed) parentheses substring.",
        "public": "assert longest_valid_parentheses('(()') == 2\n",
        "all_tests": (
            "assert longest_valid_parentheses('(()') == 2\n"
            "assert longest_valid_parentheses(')()())') == 4\n"
            "assert longest_valid_parentheses('') == 0\n"
            "assert longest_valid_parentheses('()(()') == 2\n"
            "assert longest_valid_parentheses('()()') == 4\n"
            "assert longest_valid_parentheses('((()))') == 6\n"
            "assert longest_valid_parentheses('(') == 0\n"
        ),
    },
    {
        "id": "task_08_coin_change",
        "category": "dynamic_programming",
        "goal": "Write a function coin_change(coins: list[int], amount: int) -> int that returns the fewest number of coins needed to make up amount. If not possible, return -1.",
        "public": "assert coin_change([1, 2, 5], 11) == 3\n",
        "all_tests": (
            "assert coin_change([1, 2, 5], 11) == 3\n"
            "assert coin_change([2], 3) == -1\n"
            "assert coin_change([1], 0) == 0\n"
            "assert coin_change([1, 5, 10, 25], 30) == 2\n"
            "assert coin_change([2, 5, 10, 1], 27) == 4\n"
            "assert coin_change([186, 419, 83, 408], 6249) == 20\n"
        ),
    },
    {
        "id": "task_09_spiral_order",
        "category": "matrix",
        "goal": "Write a function spiral_order(matrix: list[list[int]]) -> list[int] that returns all elements of the matrix in spiral order.",
        "public": "assert spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [1, 2, 3, 6, 9, 8, 7, 4, 5]\n",
        "all_tests": (
            "assert spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [1, 2, 3, 6, 9, 8, 7, 4, 5]\n"
            "assert spiral_order([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]) == [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]\n"
            "assert spiral_order([]) == []\n"
            "assert spiral_order([[7]]) == [7]\n"
            "assert spiral_order([[1, 2, 3]]) == [1, 2, 3]\n"
            "assert spiral_order([[1], [2], [3]]) == [1, 2, 3]\n"
        ),
    },
    {
        "id": "task_10_word_break",
        "category": "dynamic_programming",
        "goal": "Write a function word_break(s: str, word_dict: list[str]) -> bool that returns True if s can be segmented into a space-separated sequence of dictionary words.",
        "public": "assert word_break('leetcode', ['leet', 'code']) is True\n",
        "all_tests": (
            "assert word_break('leetcode', ['leet', 'code']) is True\n"
            "assert word_break('applepenapple', ['apple', 'pen']) is True\n"
            "assert word_break('catsandog', ['cats', 'dog', 'sand', 'and', 'cat']) is False\n"
            "assert word_break('a', ['a']) is True\n"
            "assert word_break('cars', ['car', 'ca', 'rs']) is True\n"
            "assert word_break('bb', ['a', 'b', 'bbb', 'bbbb']) is True\n"
        ),
    },
    {
        "id": "task_11_mbpp_02",
        "category": "mbpp_sanitized",
        "goal": "Write a function similar_elements(test_tup1, test_tup2) to find the shared elements between two tuples, returning a tuple of unique shared values in ascending order.",
        "public": "assert similar_elements((3, 4, 5, 6), (5, 7, 4, 10)) == (4, 5)\n",
        "all_tests": (
            "assert similar_elements((3, 4, 5, 6), (5, 7, 4, 10)) == (4, 5)\n"
            "assert similar_elements((1, 2, 3, 4), (5, 4, 3, 7)) == (3, 4)\n"
            "assert similar_elements((11, 12, 14, 13), (17, 15, 14, 13)) == (13, 14)\n"
        ),
    },
    {
        "id": "task_12_mbpp_03",
        "category": "mbpp_sanitized",
        "goal": "Write a python function is_not_prime(n) to identify non-prime numbers. Return True if n is not prime (including <= 1), and False if n is prime.",
        "public": "assert is_not_prime(2) is False\n",
        "all_tests": (
            "assert is_not_prime(2) is False\n"
            "assert is_not_prime(10) is True\n"
            "assert is_not_prime(35) is True\n"
            "assert is_not_prime(37) is False\n"
            "assert is_not_prime(1) is True\n"
            "assert is_not_prime(0) is True\n"
        ),
    },
    {
        "id": "task_13_mbpp_04",
        "category": "mbpp_sanitized",
        "goal": "Write a function heap_queue_largest(nums, n) to find the n largest integers from a given list of numbers, returned in descending order.",
        "public": "assert heap_queue_largest([10, 20, 50, 70, 90, 20, 50, 40, 60, 80, 100], 2) == [100, 90]\n",
        "all_tests": (
            "assert heap_queue_largest([10, 20, 50, 70, 90, 20, 50, 40, 60, 80, 100], 2) == [100, 90]\n"
            "assert heap_queue_largest([10, 20, 50, 70, 90, 20, 50, 40, 60, 80, 100], 5) == [100, 90, 80, 70, 60]\n"
            "assert heap_queue_largest([10, 20, 50, 70, 90, 20, 50, 40, 60, 80, 100], 3) == [100, 90, 80]\n"
        ),
    },
    {
        "id": "task_14_mbpp_07",
        "category": "mbpp_sanitized",
        "goal": "Write a function find_char_long(text) to find all words that are at least 4 characters long in a string, returned as a list of strings.",
        "public": "assert set(find_char_long('Please anchor on real evidence rather than abstract narratives')) == {'Please', 'anchor', 'real', 'evidence', 'rather', 'than', 'abstract', 'narratives'}\n",
        "all_tests": (
            "assert set(find_char_long('Please anchor on real evidence rather than abstract narratives')) == {'Please', 'anchor', 'real', 'evidence', 'rather', 'than', 'abstract', 'narratives'}\n"
            "assert set(find_char_long('cat dog elephant wolf lion')) == {'elephant', 'wolf', 'lion'}\n"
            "assert find_char_long('a bb ccc') == []\n"
        ),
    },
    {
        "id": "task_15_mbpp_08",
        "category": "mbpp_sanitized",
        "goal": "Write a function square_nums(nums) to find squares of individual elements in a list, returning a new list.",
        "public": "assert square_nums([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]\n",
        "all_tests": (
            "assert square_nums([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]\n"
            "assert square_nums([10, 20, 30]) == [100, 400, 900]\n"
            "assert square_nums([]) == []\n"
            "assert square_nums([-2, -3]) == [4, 9]\n"
        ),
    },
    {
        "id": "task_16_mbpp_11",
        "category": "mbpp_sanitized",
        "goal": "Write a python function remove_Occ(s, ch) to remove the first and last occurrence of a given character from a string.",
        "public": "assert remove_Occ('hello', 'l') == 'heo'\n",
        "all_tests": (
            "assert remove_Occ('hello', 'l') == 'heo'\n"
            "assert remove_Occ('abcda', 'a') == 'bcd'\n"
            "assert remove_Occ('PHP', 'P') == 'H'\n"
            "assert remove_Occ('world', 'z') == 'world'\n"
            "assert remove_Occ('a', 'a') == ''\n"
        ),
    },
    {
        "id": "task_17_mbpp_14",
        "category": "mbpp_sanitized",
        "goal": "Write a python function find_Volume(l, b, h) to find the volume of a triangular prism with triangular base length l, base width b, and height h. Volume is 0.5 * l * b * h.",
        "public": "assert find_Volume(10, 8, 6) == 240\n",
        "all_tests": (
            "assert find_Volume(10, 8, 6) == 240\n"
            "assert find_Volume(3, 2, 2) == 6\n"
            "assert find_Volume(4, 4, 10) == 80\n"
        ),
    },
    {
        "id": "task_18_mbpp_17",
        "category": "mbpp_sanitized",
        "goal": "Write a function square_perimeter(a) to find the perimeter of a square with side length a.",
        "public": "assert square_perimeter(10) == 40\n",
        "all_tests": (
            "assert square_perimeter(10) == 40\n"
            "assert square_perimeter(5) == 20\n"
            "assert square_perimeter(25) == 100\n"
        ),
    },
    {
        "id": "task_19_mbpp_18",
        "category": "mbpp_sanitized",
        "goal": "Write a function remove_dirty_chars(string, second_string) to remove all characters from the first string that are present in the second string.",
        "public": "assert remove_dirty_chars('probasscurve', 'pros') == 'bacurve'\n",
        "all_tests": (
            "assert remove_dirty_chars('probasscurve', 'pros') == 'bacurve'\n"
            "assert remove_dirty_chars('digital indian', 'indi') == 'gta a'\n"
            "assert remove_dirty_chars('exotic berry', 'ex') == 'otic berry'\n"
        ),
    },
    {
        "id": "task_20_mbpp_20",
        "category": "mbpp_sanitized",
        "goal": "Write a function is_woodall(x) to check if the given number is a Woodall number. A Woodall number is of the form n * 2^n - 1 for some positive integer n >= 1.",
        "public": "assert is_woodall(383) is True\n",
        "all_tests": (
            "assert is_woodall(383) is True\n"
            "assert is_woodall(1) is True\n"
            "assert is_woodall(2) is False\n"
            "assert is_woodall(7) is True\n"
            "assert is_woodall(23) is True\n"
            "assert is_woodall(100) is False\n"
        ),
    },
]


class BenchmarkGenerator:
    """Cached generator wrapper that tracks token throughput and deterministic seeds."""

    def __init__(self, inner: OllamaGenerator, cache_path: Path, base_seed: int) -> None:
        self.inner = inner
        self.cache_path = cache_path
        self.base_seed = base_seed
        self.calls = 0
        self.live_tokens = 0
        self.live_eval_ms = 0.0
        self.cache: dict[str, str] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                self.cache[rec["k"]] = rec["v"]

    def __call__(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512, **_: Any) -> str:
        seed = self.base_seed + self.calls
        self.calls += 1
        key = hashlib.sha256(
            f"{self.inner.model}|{prompt}|{temperature}|{max_tokens}|{seed}".encode()
        ).hexdigest()
        if key in self.cache:
            return self.cache[key]

        out = self.inner(prompt, temperature=temperature, max_tokens=max_tokens, seed=seed)
        if self.inner.last_stats:
            self.live_tokens += self.inner.last_stats.completion_tokens
            self.live_eval_ms += self.inner.last_stats.eval_ms

        self.cache[key] = out
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"k": key, "v": out}) + "\n")
        return out


def score_snippet(oracle: PythonCompilerOracle, code: str, all_tests: str) -> dict[str, Any]:
    cleaned = extract_code_block(code, "python")
    res = oracle.verify_with_test(cleaned, all_tests, timeout_s=15.0)
    return {
        "solved": bool(res.success),
        "passed": res.details.get("passed", 0),
        "total": res.details.get("total", 0),
        "error": (res.error_message or "").splitlines()[-1][:120] if res.error_message else "",
    }


def run_benchmark_for_model(
    model_name: str,
    tasks: list[dict[str, Any]],
    out_dir: Path,
    oracle: PythonCompilerOracle,
) -> dict[str, Any]:
    log.info("=" * 60)
    log.info("STARTING BENCHMARK FOR MODEL: %s", model_name)
    log.info("=" * 60)

    model_slug = model_name.replace(":", "_").replace(".", "_")
    model_dir = out_dir / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_path = model_dir / "cache.jsonl"
    rows_path = model_dir / "rows.jsonl"

    done_rows: dict[str, dict[str, Any]] = {}
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done_rows[row["id"]] = row

    tier = ModelTier.LOW if ("0.5b" in model_name or "1.5b" in model_name) else ModelTier.MID
    inner_gen = OllamaGenerator(model=model_name)
    
    total_tokens = 0
    total_eval_ms = 0.0
    start_time = time.perf_counter()

    for idx, task in enumerate(tasks):
        tid = task["id"]
        if tid in done_rows:
            log.info("[%d/%d] Skipping already completed task: %s", idx + 1, len(tasks), tid)
            continue

        base_seed = 1000 * (idx + 1)
        row: dict[str, Any] = {"id": tid, "category": task["category"], "goal": task["goal"]}

        # ── Arm A0: Raw Zero-Shot (T=0.2, 1 sample) ──
        gen_a0 = BenchmarkGenerator(inner_gen, cache_path, base_seed)
        t0_a0 = time.perf_counter()
        opt_a0 = LowTierModelOptimizer(tier=tier, generator_fn=gen_a0)
        prompt_a0 = opt_a0._build_base_prompt(task["goal"], "python", "", task["public"])
        code_a0 = gen_a0(prompt_a0, temperature=0.2, max_tokens=opt_a0.config.max_tokens_per_leaf)
        score_a0 = score_snippet(oracle, code_a0, task["all_tests"])
        dur_a0 = round((time.perf_counter() - t0_a0) * 1000.0, 1)
        row["A0"] = {
            "solved": score_a0["solved"],
            "passed": score_a0["passed"],
            "total": score_a0["total"],
            "latency_ms": dur_a0,
            "error": score_a0["error"],
        }

        # ── Arm A2: Best-of-3 Candidate Sampling (No Repair) ──
        gen_a2 = BenchmarkGenerator(inner_gen, cache_path, base_seed)
        t0_a2 = time.perf_counter()
        opt_a2 = LowTierModelOptimizer(tier=tier, generator_fn=gen_a2)
        opt_a2.config.best_of_n_candidates = 3
        opt_a2.config.max_repair_attempts = 1
        res_a2 = opt_a2.optimize_and_solve(task["goal"], "python", test_spec=task["public"])
        code_a2 = res_a2.selected_code if res_a2.verified else code_a0
        score_a2 = score_snippet(oracle, code_a2, task["all_tests"])
        dur_a2 = round((time.perf_counter() - t0_a2) * 1000.0, 1)
        row["A2"] = {
            "solved": score_a2["solved"],
            "gate_verified": bool(res_a2.verified),
            "candidates": res_a2.candidates_evaluated,
            "energy": res_a2.energy,
            "latency_ms": dur_a2,
            "passed": score_a2["passed"],
            "total": score_a2["total"],
        }

        # ── Arm A3: GWAYA v3 (Best-of-3 + 3-turn Iterative Self-Repair) ──
        gen_a3 = BenchmarkGenerator(inner_gen, cache_path, base_seed)
        t0_a3 = time.perf_counter()
        opt_a3 = LowTierModelOptimizer(tier=tier, generator_fn=gen_a3)
        opt_a3.config.best_of_n_candidates = 3
        opt_a3.config.max_repair_attempts = 3
        res_a3 = opt_a3.optimize_and_solve(task["goal"], "python", test_spec=task["public"])
        code_a3 = res_a3.selected_code if res_a3.verified else code_a0
        score_a3 = score_snippet(oracle, code_a3, task["all_tests"])
        dur_a3 = round((time.perf_counter() - t0_a3) * 1000.0, 1)
        row["A3"] = {
            "solved": score_a3["solved"],
            "gate_verified": bool(res_a3.verified),
            "repair_attempts": res_a3.repair_attempts,
            "candidates": res_a3.candidates_evaluated,
            "violations_caught": len(res_a3.violations_caught),
            "energy": res_a3.energy,
            "latency_ms": dur_a3,
            "passed": score_a3["passed"],
            "total": score_a3["total"],
        }

        done_rows[tid] = row
        with rows_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        total_tokens += (gen_a0.live_tokens + gen_a2.live_tokens + gen_a3.live_tokens)
        total_eval_ms += (gen_a0.live_eval_ms + gen_a2.live_eval_ms + gen_a3.live_eval_ms)

        log.info(
            "[%d/%d] %-28s | A0: %s | A2: %s (v=%s) | A3: %s (v=%s, r=%d) | dt=%.1fs",
            idx + 1,
            len(tasks),
            tid,
            "PASS" if row["A0"]["solved"] else "FAIL",
            "PASS" if row["A2"]["solved"] else "FAIL",
            "Y" if row["A2"]["gate_verified"] else "N",
            "PASS" if row["A3"]["solved"] else "FAIL",
            "Y" if row["A3"]["gate_verified"] else "N",
            row["A3"]["repair_attempts"],
            (dur_a0 + dur_a2 + dur_a3) / 1000.0,
        )

    # ── Summary Statistics ──
    rows = [done_rows[t["id"]] for t in tasks]
    n = len(rows)
    pass_a0 = sum(r["A0"]["solved"] for r in rows)
    pass_a2 = sum(r["A2"]["solved"] for r in rows)
    pass_a3 = sum(r["A3"]["solved"] for r in rows)

    delta_a3_a0 = 100.0 * (pass_a3 - pass_a0) / n
    delta_a2_a0 = 100.0 * (pass_a2 - pass_a0) / n
    delta_a3_a2 = 100.0 * (pass_a3 - pass_a2) / n

    # Repairs: cases where A0 failed but A3 succeeded
    a0_failed = [r for r in rows if not r["A0"]["solved"]]
    repaired_by_a3 = [r for r in a0_failed if r["A3"]["solved"]]
    repair_rate = 100.0 * len(repaired_by_a3) / len(a0_failed) if a0_failed else 0.0

    # Gate verification precision: when verified=True, did hidden tests pass?
    ver_a3 = [r for r in rows if r["A3"]["gate_verified"]]
    precision_a3 = 100.0 * sum(r["A3"]["solved"] for r in ver_a3) / len(ver_a3) if ver_a3 else 0.0

    tok_per_s = total_tokens / (total_eval_ms / 1000.0) if total_eval_ms > 0 else 0.0

    summary = {
        "model": model_name,
        "n_tasks": n,
        "pass_at_1": {
            "A0_raw": round(100.0 * pass_a0 / n, 2),
            "A2_best_of_3": round(100.0 * pass_a2 / n, 2),
            "A3_gwaya_v3": round(100.0 * pass_a3 / n, 2),
        },
        "deltas_percentage_points": {
            "A3_vs_A0": round(delta_a3_a0, 2),
            "A2_vs_A0": round(delta_a2_a0, 2),
            "A3_vs_A2": round(delta_a3_a2, 2),
        },
        "self_repair_efficiency": {
            "a0_failed_count": len(a0_failed),
            "repaired_by_a3_count": len(repaired_by_a3),
            "repair_recovery_rate_pct": round(repair_rate, 2),
        },
        "gate_verification_precision": {
            "verified_claimed_pct": round(100.0 * len(ver_a3) / n, 2),
            "hidden_pass_given_verified_pct": round(precision_a3, 2),
        },
        "performance": {
            "tokens_per_s": round(tok_per_s, 1),
            "total_tokens_generated": total_tokens,
            "total_duration_s": round(time.perf_counter() - start_time, 2),
        },
        "rows": rows,
    }

    summary_file = model_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("COMPLETED %s: A0=%.1f%% -> A2=%.1f%% -> A3=%.1f%% (ΔA3-A0 = +%.1f pts, repaired=%d/%d)",
             model_name,
             summary["pass_at_1"]["A0_raw"],
             summary["pass_at_1"]["A2_best_of_3"],
             summary["pass_at_1"]["A3_gwaya_v3"],
             delta_a3_a0,
             len(repaired_by_a3),
             len(a0_failed))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "qwen2.5-coder:0.5b",
        "qwen2.5-coder:1.5b",
        "qwen2.5-coder:3b",
        "qwen2.5-coder:7b",
    ])
    ap.add_argument("--out-dir", default="results/gwaya_v3_multi_model")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    oracle = PythonCompilerOracle()

    # Pre-flight check
    log.info("Starting GWAYA v3 multi-model benchmark across: %s", args.models)
    all_summaries: dict[str, Any] = {}
    for m in args.models:
        summary = run_benchmark_for_model(m, TASKS, out_dir, oracle)
        all_summaries[m] = summary

    # Write aggregate comparison table
    aggregate_file = out_dir / "multi_model_comparison.json"
    aggregate_file.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    log.info("Aggregate comparison saved to %s", aggregate_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
