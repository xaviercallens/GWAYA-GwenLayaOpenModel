"""Offline tests for gwaya.decontaminate: planted overlaps per method, negative controls, the
pre-registered cross-domain rules, and the CLI end to end (tmp files only, no network)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gwaya import decontaminate as dc
from gwaya.decontaminate import Config, Decontaminator, Source, row_to_item

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "data" / "decontaminate.py"

HE_PROMPT = '''from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    """
'''
HE_SOLUTION = '''    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False
'''
RENAMED_PY = '''def close_pair(xs, t):
    for a, u in enumerate(xs):
        for b, w in enumerate(xs):
            if a != b:
                d = abs(u - w)
                if d < t:
                    return True
    return False
'''
UNRELATED_PY = '''def word_count(text):
    counts = {}
    for word in text.split():
        counts[word] = counts.get(word, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
'''
RUST_TESTS = '''
fn main() {
    let candidate = has_close_elements;
    assert_eq!(candidate(vec![1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3), true);
    assert_eq!(candidate(vec![1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05), false);
}
'''
RUST_TESTS_RENAMED = '''
// translated copy
fn main() {
    let f = close_pair;
    assert_eq!(f(vec![1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3), true);
    assert_eq!(f(vec![1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05), false);
}
'''
LEAN_EVAL = ("theorem mathd_algebra_10 (x : ℝ) (h₀ : x ≠ 0) (h₁ : 2 * x + 3 = 7) :\n"
             "    x = 2 := by\n  sorry")
LEAN_RENAMED = ("-- workbook copy\ntheorem lean_workbook_plus_99 (a : ℝ) (ha : a ≠ 0) (hb : 2 * a + 3 = 7) :"
                " a = 2 := by\n  linarith")
LEAN_OTHER = "theorem foo_bar_1 (a b : ℕ) (h : a + b = 5) : b + a = 5 := by\n  omega"
MATH_EVAL = ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips "
             "in May. How many clips did Natalia sell altogether in April and May? It cost $1,000.50.")


def he_eval_item():
    return row_to_item("evalplus/humanevalplus", "python",
                       {"task_id": "HumanEval/0", "prompt": HE_PROMPT,
                        "canonical_solution": HE_SOLUTION, "entry_point": "has_close_elements",
                        "test": "def check(candidate):\n    assert candidate([1.0, 2.0], 0.5) == False\n"},
                       0, "evalplus/humanevalplus")


def eval_items():
    return [
        he_eval_item(),
        row_to_item("nuprl/MultiPL-E:humaneval-rs", "rust",
                    {"name": "HumanEval_0_has_close_elements",
                     "prompt": "/// Check closeness\nfn has_close_elements(numbers: Vec<f64>, threshold: f64) -> bool {\n",
                     "tests": RUST_TESTS}, 0, "nuprl/MultiPL-E"),
        row_to_item("cat-searcher/minif2f-lean4:test", "lean4",
                    {"name": "mathd_algebra_10", "formal_statement": LEAN_EVAL,
                     "header": "import Mathlib\nopen Real\n"}, 0, "cat-searcher/minif2f-lean4"),
        row_to_item("openai/gsm8k:test", "math", {"question": MATH_EVAL, "answer": "#### 72"}, 0,
                    "openai/gsm8k"),
        row_to_item("evalplus/mbppplus", "python",
                    {"task_id": "Mbpp/2", "prompt": "Write a function to find the shared elements "
                     "from the given two lists.", "code": "def similar_elements(a, b):\n"
                     "    return tuple(set(a) & set(b))\n", "test": "assert similar_elements((3, 4), (5, 4)) == (4,)"},
                    0, "evalplus/mbppplus"),
    ]


@pytest.fixture(scope="module")
def dec():
    return Decontaminator(eval_items(), Config())


def methods(hits):
    return {h.method for h in hits}


def train(domain, row, hf_id="train/x"):
    return row_to_item(hf_id, domain, row, 0, hf_id)


class TestPlantedOverlaps:
    def test_exact_copy(self, dec):
        hits = dec.check(train("python", {"prompt": HE_PROMPT}))
        assert "exact" in methods(hits)
        assert all(h.other.startswith("evalplus/humanevalplus::") for h in hits if h.method == "exact")

    def test_exact_ignores_case_and_whitespace(self, dec):
        hits = dec.check(train("math", {"question": "  " + MATH_EVAL.upper().replace(" ", "   \n")}))
        assert "exact" in methods(hits)

    def test_python_ast_identifier_renaming(self, dec):
        hits = dec.check(train("python", {"solution": RENAMED_PY, "prompt": "Find close pairs."}))
        assert "py_ast" in methods(hits)

    def test_python_function_inside_larger_file(self, dec):
        big = "import sys\n\n" + RENAMED_PY + "\n\nif __name__ == '__main__':\n    print(close_pair([1.0], 0.1))\n"
        assert "py_ast" in methods(dec.check(train("python", {"solution": big})))

    def test_python_code_in_markdown_fence(self, dec):
        txt = "Here is the answer:\n```python\n" + RENAMED_PY + "```\nDone."
        assert "py_ast" in methods(dec.check(train("python", {"response": txt})))

    def test_python_json_encoded_solutions_list(self, dec):
        hits = dec.check(train("python", {"solutions": json.dumps([UNRELATED_PY, RENAMED_PY])}))
        assert "py_ast" in methods(hits)

    def test_rust_token_normalized(self, dec):
        hits = dec.check(train("rust", {"tests": RUST_TESTS_RENAMED}))
        assert "rust_tok" in methods(hits)

    def test_lean_alpha_renamed_statement(self, dec):
        hits = dec.check(train("lean4", {"formal_statement": LEAN_RENAMED}))
        assert "lean_stmt" in methods(hits)

    def test_lean_header_not_indexed(self, dec):
        hits = dec.check(train("lean4", {"formal_statement": LEAN_OTHER, "header": "import Mathlib\nopen Real\n"}))
        assert hits == []

    def test_ngram13_embedded_in_longer_text(self, dec):
        txt = ("Background story about a shop. Natalia sold clips to 48 of her friends in April, and then "
               "she sold half as many clips in May. Then something completely different happens.")
        assert "ngram13" in methods(dec.check(train("math", {"problem": txt})))

    def test_ngram_cross_field(self, dec):
        # eval prompt text hidden in a train solution field
        sol = "We recall: write a function to find the shared elements from the given two lists. Easy."
        assert "ngram13" in methods(dec.check(train("python", {"solution": sol})))

    def test_math_normalized_numbers_and_latex(self, dec):
        txt = MATH_EVAL.replace("48", "$48$").replace("$1,000.50", "$1000.5")
        hits = dec.check(train("math", {"problem": txt}))
        assert "math_norm" in methods(hits)

    def test_math_template_renumbered(self, dec):
        txt = MATH_EVAL.replace("48", "52").replace("1,000.50", "7")
        assert "math_template" in methods(dec.check(train("math", {"problem": txt})))

    def test_jaccard_near_duplicate(self):
        # one word changed in a ~100-word problem: no exact/template hash, Jaccard of 5-word
        # shingles >= 0.8. (On ~30-word texts one edit already drops J below 0.8; ngram13 covers.)
        words = [f"w{i}x" for i in range(100)]
        d = Decontaminator([row_to_item("e", "math", {"problem": " ".join(words)}, 0)])
        words[50] = "changed"
        hits = d.check(train("math", {"problem": " ".join(words)}))
        assert "jaccard:prompt" in methods(hits) and "jaccard:prompt+tests" in methods(hits)
        assert not {"exact", "math_norm", "math_template"} & methods(hits)


    def test_jaccard_per_solution_entry(self):
        sol = "\n".join(f"    acc_{i} = data[{i}] * weight + offset_{i} - bias" for i in range(25))
        prog = "def compute(data, weight, bias):\n" + sol + "\n    return acc_0\n"
        d = Decontaminator([row_to_item("e", "python", {"canonical_solution": prog}, 0)])
        near = prog.replace("data[7]", "data[70]")
        many = [UNRELATED_PY.replace("word", f"tok{i}") for i in range(6)] + [near]
        hits = d.check(train("python", {"question": "Prose.", "solutions": json.dumps(many)}))
        assert "jaccard:solution" in methods(hits)
        assert "py_ast" not in methods(hits)


class TestIds:
    def test_mbppplus_ids_in_calib(self, dec):
        hits = dec.check(row_to_item("google-research-datasets/mbpp:train", "python",
                                     {"task_id": 2, "prompt": "Totally different text"}, 0,
                                     "google-research-datasets/mbpp"))
        assert ("id", "mbpp:2") in {(h.method, h.detail) for h in hits}

    def test_rust_translation_of_humaneval(self, dec):
        hits = dec.check(train("rust", {"name": "HumanEval_0_whatever", "prompt": "x"}))
        assert ("id", "humaneval:0") in {(h.method, h.detail) for h in hits}
        assert any(h.other.startswith("evalplus/humanevalplus::") for h in hits)

    def test_lean_theorem_name(self, dec):
        hits = dec.check(train("lean4", {"formal_statement": "theorem mathd_algebra_10 : True := trivial"}))
        assert "id" in methods(hits)

    def test_canonical_ids(self):
        assert dc.canonical_ids("evalplus/mbppplus", {"task_id": "Mbpp/11"}) == {"mbpp:11"}
        assert dc.canonical_ids("nuprl/MultiPL-E", {"name": "mbpp_11_remove_Occ"}) == {"mbpp:11"}
        assert dc.canonical_ids("HuggingFaceH4/MATH-500", {"unique_id": "test/algebra/1.json"}) == {"math:test/algebra/1"}
        assert dc.canonical_ids("other", {"task_id": 11}) == set()


BZ_TEST_PY = '''def check(candidate):
    assert candidate([]) == False
    assert candidate([1, 2, -3, 1, 2, -3]) == False
    assert candidate([1, 2, -4, 5, 6]) == True
    assert candidate([1, -1, 2, -2, 5, -5, 4, -4]) == False
    assert candidate([1, -2, 2, -2, 5, -5, 4, -4]) == True
'''
BZ_TEST_RS = '''fn main() {
    let candidate = neg_seen;
    assert_eq!(candidate(vec![]), false);
    assert_eq!(candidate(vec![1, 2, -3, 1, 2, -3]), false);
    assert_eq!(candidate(vec![1, 2, -4, 5, 6]), true);
    assert_eq!(candidate(vec![1, -1, 2, -2, 5, -5, 4, -4]), false);
}
'''
SQ_PROMPT_RS = "/// Sum of squares of the odd numbers.\nfn sum_odd_squares(v: Vec<i64>) -> i64 {\n"
SQ_BODY_RS = ("    let mut total: i64 = 0;\n    for x in v.iter() {\n        if x % 2 != 0 {\n"
              "            total += x * x;\n        }\n    }\n    total\n}\n")
GARDEN = ("A rectangular garden is 24 meters long and 15 meters wide. A path of uniform width is built "
          "around the outside of the garden, and the total area of the garden and the path together is "
          "540 square meters. What is the width of the path, in meters?")


@pytest.fixture(scope="module")
def dec2():
    return Decontaminator([
        row_to_item("evalplus/humanevalplus", "python",
                    {"task_id": "HumanEval/3", "prompt": "def below_zero(ops):\n", "test": BZ_TEST_PY}, 0),
        row_to_item("nuprl/MultiPL-E", "rust", {"name": "HumanEval_121_solution", "prompt": SQ_PROMPT_RS,
                                                 "canonical_solution": SQ_BODY_RS}, 0),
        row_to_item("HuggingFaceH4/MATH-500", "math", {"unique_id": "test/algebra/1.json", "problem": GARDEN}, 0),
    ])


class TestParaphraseAndTranslation:
    def test_rust_full_function_renamed_vs_prompt_plus_body(self, dec2):
        full = ("fn odd_sq(nums: Vec<i64>) -> i64 {\n    let mut acc: i64 = 0;\n    for n in nums.iter() {\n"
                "        if n % 2 != 0 {\n            acc += n * n;\n        }\n    }\n    acc\n}\n")
        assert "rust_tok" in methods(dec2.check(train("rust", {"prompt": "Compute.", "solution": full})))

    def test_translation_by_test_values_without_id(self, dec2):
        hits = dec2.check(train("rust", {"prompt": "Running balance below zero?", "tests": BZ_TEST_RS}))
        assert ("test_io", "evalplus/humanevalplus::HumanEval/3") in {(h.method, h.other) for h in hits}

    def test_assertion_values_are_language_agnostic(self):
        py = dc.assertion_value_signatures("assert f([1, 2.0, -4], 'ab') == True")
        rs = dc.assertion_value_signatures('assert_eq!(g(vec![1, 2.0, -4], "ab"), true);')
        assert py and py == rs

    def test_math_reworded_same_numbers(self, dec2):
        reworded = (GARDEN.replace("garden", "yard").replace("path", "walkway")
                    .replace("What is the width", "Find the width"))
        assert "math_nums" in methods(dec2.check(train("math", {"problem": reworded})))

    def test_controls_not_flagged(self, dec2):
        # Same numbers, different problem; only trivial (< 3 literal) shared assertions.
        other = ("Tickets cost 24 dollars for adults and 15 dollars for children. A school spent 540 dollars "
                 "on a trip for its students and teachers. How many children could have gone on it?")
        assert dec2.check(train("math", {"problem": other})) == []
        trivial = "fn main() {\n    assert_eq!(f(vec![]), false);\n    assert_eq!(f(vec![1]), true);\n}\n"
        assert dec2.check(train("rust", {"prompt": "Other.", "tests": trivial})) == []
        one_shared = BZ_TEST_RS.splitlines()[0] + "\n" + BZ_TEST_RS.splitlines()[3] + "\n}\n"
        assert "test_io" not in methods(dec2.check(train("rust", {"prompt": "Other.", "tests": one_shared})))


class TestNegativeControls:
    def test_unrelated_python(self, dec):
        assert dec.check(train("python", {"prompt": "Count words in a text.", "solution": UNRELATED_PY})) == []

    def test_same_signature_stub_does_not_collide(self, dec):
        stub = 'def other_fn(values: List[float], limit: float) -> bool:\n    """Different problem."""\n'
        assert "py_ast" not in methods(dec.check(train("python", {"prompt": stub})))

    def test_number_runs_not_ngram_hits(self):
        d = Decontaminator([row_to_item("e", "python", {"test": "assert f(" + str(list(range(40))) + ")"}, 0)])
        hits = d.check(train("python", {"tests": "assert g(" + str(list(range(40))) + ") == 1"}))
        assert "ngram13" not in methods(hits)

    def test_ngram_max_eval_df_drops_boilerplate_only_when_enabled(self):
        boiler = "the first line of the input contains a single integer t the number of test cases"
        ev = [row_to_item("e", "python", {"prompt": f"Problem {i} about topic{i}. {boiler}."}, i) for i in range(3)]
        tr = train("python", {"question": f"Unrelated story. {boiler}."})
        assert "ngram13" in methods(Decontaminator(ev).check(tr))
        assert "ngram13" not in methods(Decontaminator(ev, Config(ngram_max_eval_df=2)).check(tr))

    def test_unrelated_lean(self, dec):
        assert dec.check(train("lean4", {"formal_statement": LEAN_OTHER})) == []


class TestRules:
    def test_numina_source_denylist(self):
        d = Decontaminator([], Config())
        hf = "AI-MO/NuminaMath-1.5"
        assert methods(d.check(row_to_item(hf, "math", {"problem": "p", "source": "amc_aime"}, 0, hf), hf)) == {"source"}
        assert methods(d.check(row_to_item(hf, "math", {"problem": "p"}, 0, hf), hf)) == {"source_missing"}
        assert d.check(row_to_item(hf, "math", {"problem": "p", "source": "olympiads"}, 0, hf), hf) == []

    def test_test_split_refused_for_train(self):
        with pytest.raises(ValueError):
            dc.check_train_source(Source("m", "EleutherAI/hendrycks_math", "math", "train", split="test"))
        with pytest.raises(ValueError):
            dc.check_train_source(Source("m", "x", "math", "eval"))

    def test_cross_source_dedup(self):
        d = Decontaminator([], Config())
        a = row_to_item("codeparrot/apps", "python", {"question": "q1", "solutions": RENAMED_PY}, 0)
        b = row_to_item("likaixin/TACO-verified", "python", {"question": "q2", "solutions": RENAMED_PY}, 0)
        a2 = row_to_item("codeparrot/apps", "python", {"question": "q3", "solutions": RENAMED_PY}, 1)
        assert d.dedup_check(a) == []
        assert any(h.method == "dedup:py_ast" for h in d.dedup_check(b))
        assert d.dedup_check(a2) == []  # same-source duplicates are not cross-source leakage


class TestEmbedding:
    def test_off_by_default_and_pluggable(self):
        assert Config().embedder is None

        def emb(texts):
            return [[1.0, float("cat" in t.lower())] for t in texts]

        d = Decontaminator([row_to_item("e", "math", {"problem": "the cat sat"}, 0)],
                           Config(embedder=emb, embed_threshold=0.99))
        assert "embed" in methods(d.check(train("math", {"problem": "a cat is here"})))
        assert "embed" not in methods(d.check(train("math", {"problem": "a dog is here"})))


class TestJaccardIndex:
    def test_exact_threshold(self):
        idx = dc.JaccardIndex(0.8)
        idx.add("a", frozenset(range(100)))
        assert [k for k, _ in idx.query(frozenset(range(10, 110)))] == ["a"]   # 90/110 = 0.818
        assert idx.query(frozenset(range(20, 120))) == []                        # 80/120 = 0.667

    def test_matches_brute_force(self):
        import random
        rng = random.Random(0)
        idx = dc.JaccardIndex(0.8)
        sets = {}
        for i in range(60):
            base = set(rng.sample(range(300), rng.randint(5, 60)))
            sets[f"k{i}"] = frozenset(base)
            idx.add(f"k{i}", sets[f"k{i}"])
        for _ in range(200):
            k = rng.choice(list(sets))
            q = set(sets[k])
            for _ in range(rng.randint(0, 6)):
                if q and rng.random() < 0.5:
                    q.discard(rng.choice(sorted(q)))
                else:
                    q.add(rng.randrange(300))
            q = frozenset(q)
            got = {k for k, _ in idx.query(q)}
            want = {k for k, s in sets.items() if q and len(q & s) / len(q | s) >= 0.8}
            assert got == want


# ------------------------------------------------------------------------------------------------
# End to end


def _jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


@pytest.fixture
def e2e(tmp_path):
    he = _jsonl(tmp_path / "he.jsonl", [{"task_id": "HumanEval/0", "prompt": HE_PROMPT,
                                         "canonical_solution": HE_SOLUTION, "test": "x"}])
    gsm = _jsonl(tmp_path / "gsm.jsonl", [{"question": MATH_EVAL, "answer": "#### 72"}])
    index = _jsonl(tmp_path / "index.jsonl", [
        {"hf_id": "evalplus/humanevalplus", "domain": "python", "role": "eval", "path": str(he), "row_count": 1},
        {"hf_id": "openai/gsm8k", "domain": "math", "role": "eval", "split": "test", "path": str(gsm)},
        {"eval_set": "custom/lean", "item_id": "t1", "domain": "lean4",
         "fields": {"statement": LEAN_EVAL}},
    ])
    tr_rows = [{"id": "ok1", "question": "Count words.", "solutions": json.dumps([UNRELATED_PY])},
               {"id": "bad1", "question": "Close pairs", "solutions": json.dumps([RENAMED_PY])},
               {"id": "bad2", "question": MATH_EVAL.replace("48", "50")}]
    apps = _jsonl(tmp_path / "apps.jsonl", tr_rows)
    taco = _jsonl(tmp_path / "taco.jsonl", [
        {"id": "dup", "question": "Other wording", "solutions": json.dumps([UNRELATED_PY])},
        {"id": "ok2", "question": "Reverse a string s and print it."}])
    lw = _jsonl(tmp_path / "lw.jsonl", [{"id": "w1", "formal_statement": LEAN_RENAMED},
                                       {"id": "w2", "formal_statement": LEAN_OTHER}])
    train = _jsonl(tmp_path / "train.jsonl", [
        {"hf_id": "codeparrot/apps", "domain": "python", "role": "train", "path": str(apps)},
        {"hf_id": "likaixin/TACO-verified", "domain": "python", "role": "train", "path": str(taco)},
        {"hf_id": "internlm/Lean-Workbook", "domain": "lean4", "role": "train", "path": str(lw)},
    ])
    return tmp_path, index, train


def _run_cli(*args):
    return subprocess.run([sys.executable, str(CLI), *map(str, args)], capture_output=True,
                          text=True, cwd=ROOT, timeout=120)


class TestEndToEnd:
    def test_cli_writes_verified_clean_sets(self, e2e):
        tmp, index, train = e2e
        out = tmp / "out"
        r = _run_cli("--index", index, "--train", train, "--out-dir", out, "--no-manifest", "--no-plan-check")
        assert r.returncode == 0, r.stderr + r.stdout
        rep = json.loads((out / "decontam_report.json").read_text())
        assert rep["complete"] is True and rep["status"] == "done"
        ts = rep["train_sets"]
        assert ts["codeparrot/apps"]["n_kept"] == 1 and ts["codeparrot/apps"]["n_flagged"] == 2
        assert ts["likaixin/TACO-verified"]["n_flagged"] == 1  # cross-source dup of apps ok1
        assert ts["internlm/Lean-Workbook"]["n_kept"] == 1
        assert all(t["verified_clean"] for t in ts.values())
        kept_ids = [json.loads(l)["id"] for l in Path(ts["codeparrot/apps"]["clean_path"]).read_text().splitlines()]
        assert kept_ids == ["ok1"]
        assert rep["eval_sets"]["evalplus/humanevalplus"]["n_eval_items_matched"] == 1
        assert rep["eval_sets"]["custom/lean"]["n_eval_items_matched"] == 1
        pairs = {(c["train_source"], c["eval_set"]): c["n_removed"] for c in rep["removal_counts"]}
        assert pairs[("codeparrot/apps", "evalplus/humanevalplus")] == 1
        assert pairs[("codeparrot/apps", "openai/gsm8k:test")] == 1
        assert {"train_source": "likaixin/TACO-verified", "rule": "dedup", "n_removed": 1} in rep["rule_removal_counts"]
        flagged = {json.loads(l)["key"] for l in (out / "flagged.jsonl").read_text().splitlines()}
        assert len(flagged) == 4
        for name, t in ts.items():
            for line in Path(t["clean_path"]).read_text().splitlines():
                assert f"{name}::{json.loads(line)['id']}" not in flagged

    def test_cli_fails_closed_on_missing_eval(self, e2e, tmp_path):
        tmp, index, train = e2e
        with open(index, "a") as f:
            f.write(json.dumps({"hf_id": "math-ai/aime25", "domain": "math", "role": "eval"}) + "\n")
        out = tmp / "out2"
        r = _run_cli("--index", index, "--train", train, "--out-dir", out, "--no-manifest", "--no-plan-check")
        assert r.returncode == 2
        rep = json.loads((out / "decontam_report.json").read_text())
        assert rep["missing_eval_sets"] == ["math-ai/aime25"] and rep["status"].startswith("aborted")
        assert not any((out / "clean").glob("*.jsonl"))
        r = _run_cli("--index", index, "--train", train, "--out-dir", out, "--no-manifest", "--no-plan-check", "--allow-missing")
        assert r.returncode == 0, r.stderr
        rep = json.loads((out / "decontam_report.json").read_text())
        assert rep["complete"] is False and "INCOMPLETE" in rep["status"]

    def test_cli_eval_only_report(self, e2e):
        tmp, index, _ = e2e
        out = tmp / "out3"
        r = _run_cli("--index", index, "--out-dir", out, "--no-manifest", "--no-plan-check")
        assert r.returncode == 0, r.stderr
        rep = json.loads((out / "decontam_report.json").read_text())
        assert rep["eval_status"]["evalplus/humanevalplus"]["n_items"] == 1
        assert rep["train_sets"] == {}


class TestCoverageAndFiles:
    def test_plan_eval_sets_required(self, e2e):
        tmp, index, _ = e2e
        out = tmp / "out4"
        r = _run_cli("--index", index, "--out-dir", out, "--no-manifest")  # plan check on
        assert r.returncode == 2
        rep = json.loads((out / "decontam_report.json").read_text())
        plan = json.loads((ROOT / "experiments" / "plan.json").read_text())
        want = {d["hf_id"] for d in plan["datasets"] if d["role"] == "eval"} - {
            "evalplus/humanevalplus", "openai/gsm8k"}
        assert want <= set(rep["missing_eval_sets"])
        assert rep["eval_status"]["math-ai/aime25"]["status"] == "not_in_index"

    def test_arrow_split_selection(self, tmp_path):
        d = tmp_path / "mbpp" / "sanitized"
        d.mkdir(parents=True)
        for n in ("mbpp-test.arrow", "mbpp-train.arrow", "mbpp-train-00001-of-00002.arrow"):
            (d / n).write_bytes(b"")
        assert [p.name for p in dc.dataset_files(tmp_path / "mbpp", "test")] == ["mbpp-test.arrow"]
        assert len(dc.dataset_files(tmp_path / "mbpp", "train")) == 2
        with pytest.raises(ValueError):
            dc.dataset_files(tmp_path / "mbpp")
        with pytest.raises(dc.DatasetNotMaterialized):
            dc.dataset_files(tmp_path / "mbpp", "validation")

    def test_arrow_rows_roundtrip(self, tmp_path):
        pa = pytest.importorskip("pyarrow")
        t = pa.table({"task_id": [1, 2], "prompt": ["a", "b"]})
        p = tmp_path / "x-test.arrow"
        with pa.OSFile(str(p), "wb") as f, pa.ipc.new_stream(f, t.schema) as w:
            w.write_table(t)
        assert list(dc.iter_dataset_rows(tmp_path, "test")) == [{"task_id": 1, "prompt": "a"},
                                                                {"task_id": 2, "prompt": "b"}]
