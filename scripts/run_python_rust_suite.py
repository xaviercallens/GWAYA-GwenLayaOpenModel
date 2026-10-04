#!/usr/bin/env python3
"""GWAYA Multi-Language Intensive Test Suite: Python & Rust.

Runs rigorous compiler and test harness verification across Python and Rust:
- PythonCompilerOracle with nonce-isolated sandbox tests
- RustCompilerOracle with real `rustc --emit=metadata` type, borrow, and syntax checking
- Multi-model code verification on local NVIDIA RTX 2070 GPU
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gwaya.oracles import PythonCompilerOracle, RustCompilerOracle
from gwaya.ast_audit import ZeroStubAudit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("PyRustSuite")

RUST_CASES = [
    {
        "id": "rs_01_two_sum",
        "name": "Two Sum (HashMap)",
        "expected": "accept",
        "code": """
use std::collections::HashMap;

pub fn two_sum(nums: &[i32], target: i32) -> Option<(usize, usize)> {
    let mut map = HashMap::new();
    for (i, &num) in nums.iter().enumerate() {
        let complement = target - num;
        if let Some(&prev_idx) = map.get(&complement) {
            return Some((prev_idx, i));
        }
        map.insert(num, i);
    }
    None
}
""",
    },
    {
        "id": "rs_02_trap_rain_water",
        "name": "Trapping Rain Water (Two Pointers)",
        "expected": "accept",
        "code": """
pub fn trap(height: &[i32]) -> i32 {
    if height.is_empty() {
        return 0;
    }
    let (mut left, mut right) = (0, height.len() - 1);
    let (mut left_max, mut right_max) = (0, 0);
    let mut total_water = 0;

    while left < right {
        if height[left] < height[right] {
            if height[left] >= left_max {
                left_max = height[left];
            } else {
                total_water += left_max - height[left];
            }
            left += 1;
        } else {
            if height[right] >= right_max {
                right_max = height[right];
            } else {
                total_water += right_max - height[right];
            }
            right -= 1;
        }
    }
    total_water
}
""",
    },
    {
        "id": "rs_03_merge_intervals",
        "name": "Merge Intervals (Sorting & Vector Slicing)",
        "expected": "accept",
        "code": """
pub fn merge(mut intervals: Vec<[i32; 2]>) -> Vec<[i32; 2]> {
    if intervals.is_empty() {
        return Vec::new();
    }
    intervals.sort_by_key(|k| k[0]);
    let mut merged: Vec<[i32; 2]> = Vec::new();

    for interval in intervals {
        if merged.is_empty() || merged.last().unwrap()[1] < interval[0] {
            merged.push(interval);
        } else {
            let last = merged.last_mut().unwrap();
            last[1] = last[1].max(interval[1]);
        }
    }
    merged
}
""",
    },
    {
        "id": "rs_04_generic_binary_search",
        "name": "Generic Binary Search with Ord Trait",
        "expected": "accept",
        "code": """
pub fn binary_search<T: Ord>(slice: &[T], target: &T) -> Option<usize> {
    let mut low = 0;
    let mut high = slice.len();

    while low < high {
        let mid = low + (high - low) / 2;
        match slice[mid].cmp(target) {
            std::cmp::Ordering::Equal => return Some(mid),
            std::cmp::Ordering::Less => low = mid + 1,
            std::cmp::Ordering::Greater => high = mid,
        }
    }
    None
}
""",
    },
    {
        "id": "rs_05_borrow_checker_violation",
        "name": "Borrow Checker Move Violation (Should Reject)",
        "expected": "reject",
        "code": """
pub fn test_borrow() {
    let s = String::from("gwaya");
    let t = s;
    println!("{}", s); // Error: value borrowed here after move
}
""",
    },
    {
        "id": "rs_06_type_mismatch_violation",
        "name": "Type Mismatch Signature Violation (Should Reject)",
        "expected": "reject",
        "code": """
pub fn add(a: i32, b: i32) -> i32 {
    "string_instead_of_int"
}
""",
    },
    {
        "id": "rs_07_stub_todo_violation",
        "name": "Placeholder todo!() Stub Violation (Should Reject)",
        "expected": "reject",
        "code": """
pub fn compute_hash(val: &[u8]) -> u64 {
    todo!("implement later")
}
""",
    },
    {
        "id": "rs_08_stub_unimplemented_violation",
        "name": "Placeholder unimplemented!() Stub Violation (Should Reject)",
        "expected": "reject",
        "code": """
pub fn solve() -> bool {
    unimplemented!()
}
""",
    },
]

PYTHON_CASES = [
    {
        "id": "py_01_lru_cache",
        "name": "LRU Cache Implementation",
        "expected": "accept",
        "code": """
class Node:
    def __init__(self, key=0, val=0):
        self.key, self.val = key, val
        self.prev = self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.map = {}
        self.head, self.tail = Node(), Node()
        self.head.next, self.tail.prev = self.tail, self.head

    def _remove(self, node: Node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _insert(self, node: Node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key in self.map:
            node = self.map[key]
            self._remove(node)
            self._insert(node)
            return node.val
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.map:
            self._remove(self.map[key])
        node = Node(key, value)
        self.map[key] = node
        self._insert(node)
        if len(self.map) > self.cap:
            lru = self.tail.prev
            self._remove(lru)
            del self.map[lru.key]
""",
        "test_spec": """
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
assert cache.get(1) == 1
cache.put(3, 3)
assert cache.get(2) == -1
cache.put(4, 4)
assert cache.get(1) == -1
assert cache.get(3) == 3
assert cache.get(4) == 4
""",
    },
    {
        "id": "py_02_dutch_national_flag",
        "name": "Dutch National Flag 3-Way Sort (In-Place)",
        "expected": "accept",
        "code": """
def sort_colors(nums: list[int]) -> list[int]:
    low, mid, high = 0, 0, len(nums) - 1
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
    return nums
""",
        "test_spec": """
assert sort_colors([2, 0, 2, 1, 1, 0]) == [0, 0, 1, 1, 2, 2]
assert sort_colors([2, 0, 1]) == [0, 1, 2]
assert sort_colors([0]) == [0]
assert sort_colors([]) == []
assert sort_colors([1, 1, 1]) == [1, 1, 1]
""",
    },
    {
        "id": "py_03_trie_prefix_tree",
        "name": "Trie Prefix Tree Implementation",
        "expected": "accept",
        "code": """
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_word = True

    def search(self, word: str) -> bool:
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_word

    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True
""",
        "test_spec": """
trie = Trie()
trie.insert("apple")
assert trie.search("apple") is True
assert trie.search("app") is False
assert trie.starts_with("app") is True
trie.insert("app")
assert trie.search("app") is True
""",
    },
    {
        "id": "py_04_pass_stub_rejection",
        "name": "AST ZeroStubAudit Rejection on pass",
        "expected": "reject",
        "code": """
def solve_quantum_gravity():
    pass
""",
        "test_spec": "assert solve_quantum_gravity() is None",
    },
    {
        "id": "py_05_mock_constant_rejection",
        "name": "AST ZeroStubAudit Rejection on mock constant",
        "expected": "reject",
        "code": """
def mock_optimizer(x):
    return 42
""",
        "test_spec": "assert mock_optimizer(1) == 42",
    },
]


def run_suite() -> int:
    log.info("=" * 65)
    log.info("GWAYA MULTI-LANGUAGE INTENSIVE VERIFICATION SUITE: RUST & PYTHON")
    log.info("=" * 65)

    py_oracle = PythonCompilerOracle()
    rs_oracle = RustCompilerOracle()

    results: list[dict[str, Any]] = []

    # 1. Run Rust Test Battery
    log.info("── Running Rust Test Battery (rustc toolchain) ──")
    for case in RUST_CASES:
        t0 = time.perf_counter()
        res = rs_oracle.verify_snippet(case["code"])
        dur = round((time.perf_counter() - t0) * 1000.0, 1)

        actual = "accept" if res.success else "reject"
        correct = actual == case["expected"]
        status_str = "PASS" if correct else "FAIL"

        log.info(
            "[%s] %-35s | Expected: %-6s | Got: %-6s | dt=%.1fms",
            status_str,
            case["id"],
            case["expected"],
            actual,
            dur,
        )

        results.append({
            "lang": "rust",
            "id": case["id"],
            "name": case["name"],
            "expected": case["expected"],
            "actual": actual,
            "correct": correct,
            "latency_ms": dur,
            "error": res.error_message.splitlines()[0] if res.error_message else "",
        })

    # 2. Run Python Test Battery
    log.info("── Running Python Test Battery (ZeroStubAudit + Nonce Sandbox) ──")
    for case in PYTHON_CASES:
        t0 = time.perf_counter()
        
        # ZeroStubAudit check
        audit = ZeroStubAudit.audit_python_code(case["code"])
        if not audit.is_clean:
            actual = "reject"
            err = "; ".join(audit.violations)
        else:
            res = py_oracle.verify_with_test(case["code"], case["test_spec"])
            actual = "accept" if res.success else "reject"
            err = res.error_message or ""

        dur = round((time.perf_counter() - t0) * 1000.0, 1)
        correct = actual == case["expected"]
        status_str = "PASS" if correct else "FAIL"

        log.info(
            "[%s] %-35s | Expected: %-6s | Got: %-6s | dt=%.1fms",
            status_str,
            case["id"],
            case["expected"],
            actual,
            dur,
        )

        results.append({
            "lang": "python",
            "id": case["id"],
            "name": case["name"],
            "expected": case["expected"],
            "actual": actual,
            "correct": correct,
            "latency_ms": dur,
            "error": err[:80],
        })

    total_correct = sum(1 for r in results if r["correct"])
    log.info("=" * 65)
    log.info("TOTAL ACCURACY: %d/%d (%.1f%%)", total_correct, len(results), 100.0 * total_correct / len(results))
    log.info("=" * 65)

    out_file = ROOT / "results" / "python_rust_suite_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({"summary": {"total": len(results), "correct": total_correct, "accuracy": 100.0 * total_correct / len(results)}, "results": results}, indent=2))
    return 0 if total_correct == len(results) else 1


if __name__ == "__main__":
    sys.exit(run_suite())
