"""Train/calib-vs-eval decontamination for GwenLaya v4, all domains, CPU only, stdlib only.

Pre-registered in docs/GWENLAYA_PREREGISTRATION.md section 10 (which names the module
`gwaya/data/decontam.py`; this file implements it). Every train/calib item is compared with every
eval item, across domains. ANY hit flags the item. Methods (`Hit.method`):

- ``exact``         sha256 of NFKC/casefolded/whitespace-collapsed field text (>= MIN_EXACT_CHARS).
- ``py_ast``        Python AST hash with identifiers renamed (builtins, attributes and keyword names
                    kept, docstrings dropped); whole program, prompt+solution, and each top-level def.
- ``rust_tok``      Rust token hash: comments stripped, local identifiers alpha-renamed.
- ``lean_stmt``     Lean 4 statement hash (each theorem/lemma/example up to ``:=``) with the decl
                    name and binder names alpha-renamed, comments stripped.
- ``ngram13``       any shared word n-gram (n=13) over all text fields; texts shorter than n but
                    with >= MIN_SHORT_TOKENS words use the whole sequence. Grams with fewer than
                    MIN_ALPHA_IN_GRAM alphabetic tokens (number runs in test data) are ignored.
- ``jaccard:<doc>`` Jaccard >= threshold (0.8) on 5-word shingles of normalized docs (code
                    AST/token-normalized): prompt+tests (the pre-registered doc), and separately
                    prompt, tests, solution, statement. Computed EXACTLY with prefix filtering, so it
                    has no MinHash approximation error (MinHash estimates this same quantity).
- ``math_norm``     problem text with LaTeX wrappers stripped, \\frac/\\dfrac unified, thousands
                    separators and trailing decimal zeros removed: exact hash.
- ``math_template`` same with every number masked (>= MIN_TEMPLATE_TOKENS words): catches
                    re-numbered copies of an eval problem.
- ``math_nums``     same multiset of numbers (>= MATH_NUMS_MIN numbers) AND word-set Jaccard >=
                    MATH_NUMS_JACCARD (>= MIN_TEMPLATE_TOKENS words): catches reworded copies that
                    keep the numbers (synonym swaps throughout, so no 13-gram survives).
- ``test_io``       language-agnostic test values: per assertion line, the sequence of literals
                    (numbers, strings, booleans, None/null), function names and syntax dropped.
                    Sharing >= TEST_IO_MIN_SHARED informative assertions (>= TEST_IO_MIN_LITERALS
                    literals each; fewer only if the eval item has fewer, minimum 2) flags the
                    item: catches translations (e.g. Python -> Rust) that carry no benchmark id.
- ``id``            canonical id overlap: ``mbpp:N`` (MBPP, MBPP+, MultiPL-E mbpp-rs),
                    ``humaneval:N`` (HumanEval+, MultiPL-E humaneval-rs), ``math:test/<subj>/<n>``,
                    ``leanthm:<name>`` (distinctive Lean theorem names, e.g. mathd_algebra_10).
- ``source``        source denylist rule (NuminaMath-1.5: sources matching MATH/AIME/AMC);
                    ``source_missing`` when the rule applies but the item has no source field.
- ``embed``         optional cosine >= threshold with a caller-supplied embedder (off by default).
- ``dedup``         (train-vs-train, cross-source) for the APPS/TACO/CodeContests/LeetCode group:
                    all methods above except ngram13/embed/source, against already-kept items of
                    OTHER sources in the group; the first source listed wins.

Thresholds are conventional and not validated on this data (pre-registration states this).
"""
from __future__ import annotations

import ast
import builtins
import functools
import hashlib
import json
import math
import re
import textwrap
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Sequence

NGRAM_N = 13
MIN_SHORT_TOKENS = 8
MIN_ALPHA_IN_GRAM = 4
MIN_EXACT_CHARS = 20
MIN_PY_NODES = 12
MIN_PY_DEF_NODES = 20
MIN_CODE_TOKENS = 20
MIN_LEAN_TOKENS = 5
MIN_TEMPLATE_TOKENS = 10
SHINGLE_K = 5
JACCARD_THRESHOLD = 0.8
EMBED_THRESHOLD = 0.95
MAX_FIELD_CHARS = 500_000
MATH_NUMS_MIN = 2
MATH_NUMS_JACCARD = 0.5
TEST_IO_MIN_LITERALS = 3
TEST_IO_MIN_SHARED = 3

ROLES = ("prompt", "solution", "tests", "statement", "answer", "other")
JACCARD_DOCS = ("prompt+tests", "prompt", "tests", "solution", "statement")

# Column name -> role. Unknown string columns become "other" (exact + ngram only) unless
# `known_fields_only`. SKIP columns are ids/boilerplate identical across items (e.g. Lean header).
_COLS = {
    "prompt": {"prompt", "question", "problem", "text", "description", "question_content",
               "problem_statement", "problem_description", "informal_statement", "informal_prefix",
               "nl_statement", "natural_language_statement", "query", "instruction", "input",
               "starter_code", "question_title", "title", "content", "informal_problem"},
    "solution": {"canonical_solution", "code", "solution", "solutions", "completion",
                 "formal_proof", "response", "output", "generation", "gold_standard_solution",
                 "reference_solution", "proof", "full_proof", "completions"},
    "tests": {"test", "tests", "test_list", "challenge_test_list", "test_setup_code",
              "input_output", "public_tests", "private_tests", "generated_tests",
              "public_test_cases", "private_test_cases", "test_cases", "unit_tests", "test_code",
              "plus_input", "base_input"},
    "statement": {"formal_statement", "statement", "lean4_code", "formal_theorem",
                  "lean_statement", "formal_code", "autoformalization"},
    "answer": {"answer", "final_answer", "expected_answer", "gold_answer"},
}
_SKIP_COLS = {"header", "id", "task_id", "name", "unique_id", "problem_id", "question_id",
              "entry_point", "license", "split", "source", "data_source", "origin", "subject",
              "level", "type", "difficulty", "language", "lang", "stop_tokens", "doctests",
              "prompt_terminology", "original", "test_imports", "platform", "contest_id",
              "contest_date", "date", "tags", "raw_tags", "skill_types", "time_limit",
              "memory_limit", "dataset", "_gwaya_key"}
_ROLE_OF = {c: r for r, cs in _COLS.items() for c in cs}
_ID_KEYS = ("task_id", "id", "unique_id", "name", "problem_id", "question_id")
_SOURCE_KEYS = ("source", "data_source", "origin")

MBPP_HF_IDS = {"google-research-datasets/mbpp", "mbpp", "evalplus/mbppplus"}

# Pre-registered cross-domain rules (section 10).
DEFAULT_SOURCE_RULES: dict[str, tuple[str, str]] = {
    # "No NuminaMath items whose source is MATH-test, AIME or AMC 2025." The `source` values were
    # not read from the data this session; the pattern is deliberately broad (drops every
    # MATH/AIME/AMC-sourced item, not only 2025/test ones) so that it errs towards removal.
    "AI-MO/NuminaMath-1.5": ("source", r"(?i)(^math$|aime|amc)"),
}
DEFAULT_DEDUP_GROUP = ("likaixin/TACO-verified", "codeparrot/apps", "deepmind/code_contests",
                       "newfacade/LeetCodeDataset")


# ---------------------------------------------------------------------------------------------
# Items


@dataclass
class Item:
    """One record to compare. `fields` is a list of (role, text) pairs."""
    dataset: str
    item_id: str
    domain: str
    fields: list[tuple[str, str]] = field(default_factory=list)
    ids: frozenset[str] = frozenset()
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.dataset}::{self.item_id}"

    def texts(self, *roles: str) -> list[str]:
        return [t for r, t in self.fields if r in roles]


@dataclass(frozen=True)
class Hit:
    method: str
    other: str          # key of the matched eval item (or kept train item, for dedup)
    detail: str = ""


def _flatten(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, (bytes, bytearray)):
        return [bytes(v).decode("utf-8", "replace")]
    if isinstance(v, (list, tuple)):
        return [s for x in v for s in _flatten(x)]
    if isinstance(v, dict):
        return [s for x in v.values() for s in _flatten(x)]
    return [str(v)]


def _expand_json(text: str) -> list[str]:
    """TACO/APPS store `solutions`/`input_output` as JSON strings: index their elements (the
    container string itself is their concatenation plus JSON escapes, so it adds nothing)."""
    s = text.lstrip()
    if s[:1] in "[{":
        try:
            parts = [p for p in _flatten(json.loads(text)) if p]
        except (ValueError, RecursionError):
            return [text]
        return parts or [text]
    return [text]


def _fenced_blocks(text: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r"```[\w+-]*\n(.*?)```", text, re.S)]


_HE_RE = re.compile(r"(?i)(?:^|[/_.])humaneval[/_](\d+)(?:\D|$)")
_MBPP_RE = re.compile(r"(?i)(?:^|[/_.])mbpp[/_](\d+)(?:\D|$)")
_MATH_ID_RE = re.compile(r"^(test|train)/([\w-]+)/(\d+)\.json$")
_LEAN_DECL_RE = re.compile(r"\b(theorem|lemma|example)\b")
_LEAN_NAME_RE = re.compile(r"\b(?:theorem|lemma)\s+([^\s(:{\[]+)")


def lean_theorem_names(text: str) -> list[str]:
    return _LEAN_NAME_RE.findall(_strip_lean_comments(text))


def canonical_ids(hf_id: str, row: dict[str, Any], fields: Sequence[tuple[str, str]] = ()) -> set[str]:
    """Dataset-independent ids that identify the same benchmark task across datasets/languages."""
    out: set[str] = set()
    for k in _ID_KEYS + ("original",):
        v = row.get(k)
        if v in (None, ""):
            continue
        s = str(v)
        m = _HE_RE.search("/" + s)
        if m:
            out.add(f"humaneval:{int(m.group(1))}")
        m = _MBPP_RE.search("/" + s)
        if m:
            out.add(f"mbpp:{int(m.group(1))}")
        if hf_id in MBPP_HF_IDS and k == "task_id" and s.isdigit():
            out.add(f"mbpp:{int(s)}")
        m = _MATH_ID_RE.match(s)
        if m:
            out.add(f"math:{m.group(1)}/{m.group(2)}/{int(m.group(3))}")
    for role, text in fields:
        if role in ("statement", "prompt", "solution"):
            for name in lean_theorem_names(text):
                # Only distinctive names (contain '_' and a digit): generic names such as `test`
                # or `foo` would collide across unrelated datasets.
                if "_" in name and any(c.isdigit() for c in name) and not name.startswith("lean_workbook"):
                    out.add(f"leanthm:{name}")
    return out


def row_to_item(dataset: str, domain: str, row: dict[str, Any], index: int, hf_id: str = "",
                known_fields_only: bool = False, max_field_chars: int = MAX_FIELD_CHARS) -> Item:
    """Normalize an arbitrary dataset row. Schema-agnostic on purpose: every string column with a
    known role is indexed, unknown string columns go to role "other"."""
    hf_id = hf_id or dataset
    rid = next((str(row[k]) for k in _ID_KEYS if row.get(k) not in (None, "")), str(index))
    fields: list[tuple[str, str]] = []
    truncated = 0
    for col, v in row.items():
        c = str(col).lower()
        if c in _SKIP_COLS:
            continue
        role = _ROLE_OF.get(c)
        if role is None:
            if known_fields_only or not isinstance(v, (str, list, dict)):
                continue
            role = "other"
        budget = max_field_chars
        for part in _flatten(v):
            for t in (_expand_json(part) if role in ("solution", "tests", "other") else [part]):
                if not t or not t.strip():
                    continue
                if len(t) > budget:
                    t = t[:budget]
                    truncated += 1
                budget -= len(t)
                fields.append((role, t))
                if budget <= 0:
                    break
            if budget <= 0:
                break
    meta = {k: str(row[k]) for k in _SOURCE_KEYS if row.get(k) not in (None, "")}
    if truncated:
        meta["_truncated_fields"] = truncated
    return Item(dataset=dataset, item_id=rid, domain=domain, fields=fields,
                ids=frozenset(canonical_ids(hf_id, row, fields)), meta=meta)


def item_from_record(rec: dict[str, Any], default_set: str = "") -> Item:
    """Item-level JSONL record: {"eval_set"|"dataset", "item_id", "domain", "fields": {role: text
    or [texts]} or [[role, text], ...], "ids": [...], "meta": {...}}."""
    raw = rec.get("fields") or {}
    pairs = raw.items() if isinstance(raw, dict) else [tuple(p) for p in raw]
    fields = [(r if r in ROLES else "other", t) for r, v in pairs for t in _flatten(v) if t]
    ds = rec.get("eval_set") or rec.get("dataset") or default_set
    ids = set(rec.get("ids") or []) | canonical_ids(ds, rec, fields)
    return Item(dataset=ds, item_id=str(rec["item_id"]), domain=rec.get("domain", "unknown"),
                fields=fields, ids=frozenset(ids), meta=dict(rec.get("meta") or {}))


# ---------------------------------------------------------------------------------------------
# Normalizers


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "surrogatepass")).hexdigest()


def normalize_text(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


_WORD_RE = re.compile(r"\w+")


def words(s: str) -> list[str]:
    return _WORD_RE.findall(unicodedata.normalize("NFKC", s).casefold())


# Python ------------------------------------------------------------------------------------

_PY_BUILTINS = frozenset(dir(builtins))


class _PyRenamer(ast.NodeTransformer):
    def __init__(self, strip_docstrings: bool) -> None:
        self.map: dict[str, str] = {}
        self.strip = strip_docstrings

    def _n(self, name: str) -> str:
        if name in _PY_BUILTINS:
            return name
        return self.map.setdefault(name, f"v{len(self.map)}")

    def _body(self, node: ast.AST) -> None:
        body = getattr(node, "body", None)
        if (self.strip and isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]

    def visit_Module(self, node: ast.Module) -> ast.AST:
        self._body(node)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: Any) -> ast.AST:
        node.name = self._n(node.name)
        if self.strip and hasattr(node, "returns"):
            node.returns = None  # type hints are cosmetic: a copy may drop/add them
        self._body(node)
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self._n(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self._n(node.arg)
        if self.strip:
            node.annotation = None
        self.generic_visit(node)
        return node

    def visit_alias(self, node: ast.alias) -> ast.AST:
        if node.asname:
            node.asname = self._n(node.asname)
        return node


# Cheap pre-filter: only texts with at least one line that looks like a Python statement are
# parsed (a failed ast.parse on prose costs ~0.5 ms; most train fields are prose).
_LOOKS_PY = re.compile(r"(?m)^[ \t]*(?:def |class |import |from [\w.]+ import|assert |print\(|return\b"
                       r"|for .+:|if .+:|while .+:|with .+:|try:|@\w|[\w.\[\], ]+[ \t]*[-+*/%|&^]?=[^=])")


@functools.lru_cache(maxsize=2048)
def _py_parse_dump(code: str) -> bytes | None:
    """Parse once per distinct text; later trees are rebuilt from the cached source (cheaper than
    deepcopy) because normalization mutates them."""
    if not _LOOKS_PY.search(code):
        return None
    for c in (code, textwrap.dedent(code)):
        try:
            ast.parse(c)
            return c.encode()
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            continue
    return None


def _py_parse(code: str) -> ast.Module | None:
    src = _py_parse_dump(code)
    return None if src is None else ast.parse(src.decode())


def _py_norm_tree(tree: ast.AST, strip_docstrings: bool = True) -> ast.AST:
    """Normalize IN PLACE (callers pass a fresh tree)."""
    return _PyRenamer(strip_docstrings).visit(tree)


def _n_nodes(tree: ast.AST) -> int:
    return sum(1 for _ in ast.walk(tree))


def _substantive_nodes(stmts: Sequence[ast.stmt]) -> int:
    """Nodes in bodies, not in signatures/imports, so that renamed stubs (`def f(a: int) -> int:`
    with the docstring stripped) of different problems do not collide."""
    n = 0
    for s in stmts:
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            n += _substantive_nodes(s.body)
        elif isinstance(s, (ast.Import, ast.ImportFrom, ast.Pass)):
            continue
        elif isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant):
            continue
        else:
            n += _n_nodes(s)
    return n


@functools.lru_cache(maxsize=2048)
def python_ast_hashes(code: str) -> frozenset[str]:
    """Hash of the identifier-renamed AST of the whole program and of each top-level def/class
    (renamed independently, so a copied function is found inside a larger file)."""
    tree = _py_parse(code)
    if tree is None:
        return frozenset()
    out: set[str] = set()
    try:
        norm = _py_norm_tree(tree)
        if _substantive_nodes(norm.body) >= MIN_PY_NODES:
            out.add(_sha(ast.dump(norm, annotate_fields=False)))
        fresh = _py_parse(code)
        for node in fresh.body if fresh is not None else ():
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                mod = _py_norm_tree(ast.Module(body=[node], type_ignores=[]))
                if _substantive_nodes(mod.body) >= MIN_PY_DEF_NODES:
                    out.add(_sha(ast.dump(mod, annotate_fields=False)))
    except RecursionError:
        pass
    return frozenset(out)


def python_normalized_source(code: str) -> str | None:
    """Identifier-renamed source with docstrings kept (they carry the problem text)."""
    tree = _py_parse(code)
    if tree is None or _n_nodes(tree) < MIN_PY_NODES:
        return None
    try:
        return ast.unparse(_py_norm_tree(tree, strip_docstrings=False))
    except (RecursionError, ValueError, AttributeError):
        return None


# Rust --------------------------------------------------------------------------------------

_RUST_KW = frozenset("""as async await break const continue crate dyn else enum extern false fn for
if impl in let loop match mod move mut pub ref return self Self static struct super trait true type
unsafe use where while i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize f32 f64 bool char str
String Vec Option Some None Result Ok Err Box HashMap HashSet BTreeMap BTreeSet main""".split())
_RUST_TOK_RE = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])\'|[A-Za-z_]\w*|\d[\w.]*'
                          r'|::|->|=>|==|!=|<=|>=|&&|\|\||[^\s\w]')


def rust_tokens(code: str) -> list[str]:
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.S)
    code = re.sub(r"//[^\n]*", " ", code)
    toks = _RUST_TOK_RE.findall(code)
    names: dict[str, str] = {}
    out: list[str] = []
    for i, t in enumerate(toks):
        prev = toks[i - 1] if i else ""
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if (re.match(r"[a-z_]\w*$", t) and t not in _RUST_KW and prev not in (".", "::")
                and nxt not in ("!", "::")):
            t = names.setdefault(t, f"id{len(names)}")
        out.append(t)
    return out


def rust_token_hash(code: str) -> str | None:
    toks = rust_tokens(code)
    return _sha(" ".join(toks)) if len(toks) >= MIN_CODE_TOKENS else None


# Lean 4 ------------------------------------------------------------------------------------

_LEAN_TOK_RE = re.compile(r":=|[^\W\d][\w.'!?]*|\d+(?:\.\d+)?|[^\s\w]")
_LEAN_BINDERS = {"∀", "∃", "∃!", "fun", "λ", "∑", "∏", "⋃", "⋂", "forall", "exists"}
_LEAN_OPEN = {"(", "{", "[", "⦃", "⟨"}
_LEAN_IDENT = re.compile(r"[^\W\d][\w'!?]*$")


def _strip_lean_comments(text: str) -> str:
    text = re.sub(r"/-.*?-/", " ", text, flags=re.S)
    return re.sub(r"--[^\n]*", " ", text)


def lean_statements(text: str) -> list[str]:
    """Each theorem/lemma/example statement, from the keyword up to `:=` (proof dropped)."""
    text = _strip_lean_comments(text)
    starts = [m.start() for m in _LEAN_DECL_RE.finditer(text)]
    out = []
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        chunk = text[s:end]
        cut = chunk.find(":=")
        out.append(chunk[:cut] if cut >= 0 else chunk)
    return out


def lean_normalized_tokens(stmt: str) -> list[str]:
    """Alpha-rename the declaration name and every bound variable (bracket binders before `:`,
    and names after ∀/∃/fun/λ/∑/∏). Library names (Real.sqrt, Nat.Prime, ...) are kept."""
    toks = _LEAN_TOK_RE.findall(stmt)
    bound: dict[str, str] = {}

    def bind(name: str) -> None:
        if _LEAN_IDENT.match(name) and name != "_":
            bound.setdefault(name, f"x{len(bound)}")

    i = 0
    if toks and toks[0] in ("theorem", "lemma") and len(toks) > 1:
        bound[toks[1]] = "THM"
        i = 2
    while i < len(toks):
        t = toks[i]
        if t in _LEAN_OPEN or t in _LEAN_BINDERS:
            j = i + 1
            names = []
            while j < len(toks) and _LEAN_IDENT.match(toks[j]) and toks[j] not in _LEAN_BINDERS:
                names.append(toks[j])
                j += 1
            if names and (t in _LEAN_BINDERS or (j < len(toks) and toks[j] == ":")):
                for n in names:
                    bind(n)
        i += 1
    out = []
    for t in toks:
        head, dot, rest = t.partition(".")
        if t in bound:
            out.append(bound[t])
        elif dot and head in bound and bound[head] != "THM":
            out.append(bound[head] + "." + rest)
        else:
            out.append(t)
    return out


def lean_statement_hashes(text: str) -> set[str]:
    out = set()
    for st in lean_statements(text):
        toks = lean_normalized_tokens(st)
        if len(toks) >= MIN_LEAN_TOKENS:
            out.add(_sha(" ".join(toks[1:])))  # drop theorem/lemma/example keyword
    return out


# Math --------------------------------------------------------------------------------------

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _canon_number(m: re.Match) -> str:
    s = m.group(0)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    s = s.lstrip("0") or "0"
    return "0" + s if s.startswith(".") else s


def normalize_math(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    for _ in range(3):
        s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\(?:text|mathrm|mbox|textbf|mathbf|operatorname)\s*\{([^{}]*)\}", r" \1 ", s)
    s = re.sub(r"\\(?:left|right|displaystyle|textstyle|quad|qquad|[,;:! ])", " ", s)
    s = s.replace("\\(", " ").replace("\\)", " ").replace("\\[", " ").replace("\\]", " ")
    s = s.replace("$", " ").replace("{", " ").replace("}", " ")
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    s = _NUM_RE.sub(_canon_number, s)
    toks = re.findall(r"\d+(?:\.\d+)?|\\?\w+|[+\-*/^=<>()|!]", s)
    return " ".join(toks)


def math_template(norm: str) -> str | None:
    toks = norm.split()
    if sum(1 for t in toks if t.isalpha()) < MIN_TEMPLATE_TOKENS or not any(_NUM_RE.fullmatch(t) for t in toks):
        return None
    return " ".join("<n>" if _NUM_RE.fullmatch(t) else t for t in toks)


def math_numbers_signature(norm: str) -> tuple[str, frozenset[str]] | None:
    """(hash of the sorted number multiset, word set) of a normalized problem, or None if it has
    too few numbers/words to be distinctive."""
    toks = norm.split()
    nums = sorted(t for t in toks if _NUM_RE.fullmatch(t))
    wordset = frozenset(t for t in toks if t.isalpha())
    if len(nums) < MATH_NUMS_MIN or sum(1 for t in toks if t.isalpha()) < MIN_TEMPLATE_TOKENS:
        return None
    return _sha(" ".join(nums)), wordset


# Test values, language-agnostic ------------------------------------------------------------

_ASSERT_LINE_RE = re.compile(r"(?i)\b(?:assert\w*|expect_\w+|check_\w+)\b!?(.*)")
_LITERAL_RE = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|(?<![\w.])-?\d+(?:\.\d+)?'
                         r"|\b(?:true|false|none|null|nil)\b", re.I)


def _canon_literal(t: str) -> str:
    if t[:1] in "\"'":
        return "s:" + t[1:-1]
    low = t.lower()
    if low in ("true", "false"):
        return low
    if low in ("none", "null", "nil"):
        return "none"
    neg = t.startswith("-")
    n = _NUM_RE.sub(_canon_number, t.lstrip("-"))
    return ("-" if neg and n != "0" else "") + n


def assertion_value_signatures(text: str) -> set[str]:
    """Hashes of the literal sequence of each informative assertion line in `text`."""
    out = set()
    for line in text.splitlines():
        m = _ASSERT_LINE_RE.search(line)
        if not m:
            continue
        lits = [_canon_literal(t) for t in _LITERAL_RE.findall(m.group(1))]
        if len(lits) >= TEST_IO_MIN_LITERALS:
            out.add(_sha("\x1f".join(lits)))
    return out


# ---------------------------------------------------------------------------------------------
# Features


def _ngrams(text: str, n: int) -> Iterator[tuple[str, ...]]:
    toks = words(text)
    if len(toks) < n:
        if len(toks) >= MIN_SHORT_TOKENS:
            yield tuple(toks)
        return
    alpha = [any(c.isalpha() for c in t) for t in toks]
    run = sum(alpha[:n])
    for i in range(len(toks) - n + 1):
        if i:
            run += alpha[i + n - 1] - alpha[i - 1]
        if run >= MIN_ALPHA_IN_GRAM:
            yield tuple(toks[i:i + n])


def _shingles(tokens: Sequence[str], k: int = SHINGLE_K) -> frozenset[int]:
    if len(tokens) < k:
        return frozenset({hash(tuple(tokens))}) if len(tokens) >= 3 else frozenset()
    return frozenset(hash(tuple(tokens[i:i + k])) for i in range(len(tokens) - k + 1))


def _looks_lean(text: str) -> bool:
    return bool(_LEAN_DECL_RE.search(text)) and (":=" in text or "sorry" in text or ":" in text)


def _doc_text(role: str, text: str, domain: str) -> str:
    if role in ("statement",) or (domain == "lean4" and _looks_lean(text)):
        sts = lean_statements(text)
        if sts:
            return " ".join(" ".join(lean_normalized_tokens(s)[1:]) for s in sts)
    if domain == "rust" and role in ("solution", "tests"):
        return " ".join(rust_tokens(text))
    if role in ("solution", "tests", "prompt"):
        py = python_normalized_source(text)
        if py is not None:
            return py
    return normalize_math(text)


@dataclass
class Features:
    hashes: set[tuple[str, str]]                      # (method, hash): exact/py_ast/rust_tok/lean_stmt/math_*
    grams: set[int]
    gram_text: dict[int, str]
    ids: frozenset[str]
    docs: dict[str, list[frozenset[int]]]
    embed_text: str
    test_io: frozenset[str] = frozenset()
    math_nums: tuple[tuple[str, frozenset[str]], ...] = ()


@dataclass
class Config:
    ngram_n: int = NGRAM_N
    jaccard_threshold: float = JACCARD_THRESHOLD
    ngram_max_eval_df: int | None = None   # ignore grams shared by > this many eval items (None: never)
    embedder: Callable[[list[str]], list[list[float]]] | None = None
    embed_threshold: float = EMBED_THRESHOLD
    source_rules: dict[str, tuple[str, str]] = field(default_factory=lambda: dict(DEFAULT_SOURCE_RULES))
    dedup_group: tuple[str, ...] = DEFAULT_DEDUP_GROUP

    def public(self) -> dict[str, Any]:
        return {"ngram_n": self.ngram_n, "min_short_tokens": MIN_SHORT_TOKENS,
                "min_alpha_in_gram": MIN_ALPHA_IN_GRAM, "min_exact_chars": MIN_EXACT_CHARS,
                "jaccard_threshold": self.jaccard_threshold, "jaccard_docs": list(JACCARD_DOCS),
                "shingle_k": SHINGLE_K, "jaccard_mode": "exact (prefix filtering)",
                "min_template_tokens": MIN_TEMPLATE_TOKENS, "math_nums_min": MATH_NUMS_MIN,
                "math_nums_jaccard": MATH_NUMS_JACCARD, "test_io_min_literals": TEST_IO_MIN_LITERALS,
                "test_io_min_shared": TEST_IO_MIN_SHARED, "ngram_max_eval_df": self.ngram_max_eval_df,
                "embedder": None if self.embedder is None else getattr(self.embedder, "__qualname__", "custom"),
                "embed_threshold": self.embed_threshold if self.embedder else None,
                "source_rules": {k: list(v) for k, v in self.source_rules.items()},
                "dedup_group": list(self.dedup_group), "max_field_chars": MAX_FIELD_CHARS}


def features(item: Item, cfg: Config, with_grams: bool = True) -> Features:
    hashes: set[tuple[str, str]] = set()
    grams: set[int] = set()
    gram_text: dict[int, str] = {}
    doc_tokens: dict[str, list[str]] = defaultdict(list)
    entry_docs: dict[str, list[frozenset[int]]] = defaultdict(list)
    prompts = item.texts("prompt")
    test_io: set[str] = set()
    math_nums: set[tuple[str, frozenset[str]]] = set()
    for role, text in item.fields:
        if role == "answer":
            continue
        if len(text.strip()) >= MIN_EXACT_CHARS:
            hashes.add(("exact", _sha(normalize_text(text))))
        if with_grams:
            for g in _ngrams(text, cfg.ngram_n):
                h = hash(g)
                if h not in grams:
                    grams.add(h)
                    gram_text[h] = " ".join(g)
        if role == "other":
            continue
        codes = [text] + _fenced_blocks(text)
        for c in codes:
            hashes.update(("py_ast", h) for h in python_ast_hashes(c))
            # Not on prompts: a Rust prompt is a signature stub, and stubs of different problems
            # with the same shape would collide after renaming.
            if role in ("solution", "tests") and (item.domain == "rust" or "fn " in c):
                h = rust_token_hash(c)
                if h:
                    hashes.add(("rust_tok", h))
            if role == "statement" or item.domain == "lean4" or "theorem" in c or "lemma" in c:
                hashes.update(("lean_stmt", h) for h in lean_statement_hashes(c))
        if role in ("prompt", "statement") or (role == "solution" and item.domain == "math"):
            nm = normalize_math(text)
            if len(nm) >= MIN_EXACT_CHARS:
                hashes.add(("math_norm", _sha(nm)))
            if role == "prompt":
                tpl = math_template(nm)
                if tpl:
                    hashes.add(("math_template", _sha(tpl)))
                sig = math_numbers_signature(nm)
                if sig:
                    math_nums.add(sig)
        if role in ("tests", "solution"):
            test_io.update(assertion_value_signatures(text))
        if role in ("prompt", "tests", "solution", "statement"):
            toks = words(_doc_text(role, text, item.domain))
            doc_tokens[role].extend(toks)
            if role in ("solution", "tests"):  # each solution / test entry also on its own
                entry_docs[role].append(_shingles(toks))
    # HumanEval-style: the prompt is the signature+docstring, the solution is the body.
    if item.domain == "python" and prompts and _LOOKS_PY.search(prompts[0]):
        for sol in item.texts("solution"):
            hashes.update(("py_ast", h) for h in python_ast_hashes(prompts[0] + sol))
    # Same for Rust (MultiPL-E): signature stub in the prompt, body in the solution. The joined
    # text is a whole function, so it matches a full-function copy with renamed identifiers.
    if prompts and re.search(r"\bfn\s+\w+", prompts[0]) and (item.domain == "rust" or "{" in prompts[0]):
        for sol in item.texts("solution"):
            h = rust_token_hash(prompts[0] + sol)
            if h:
                hashes.add(("rust_tok", h))
    doc_tokens["prompt+tests"] = doc_tokens.get("prompt", []) + doc_tokens.get("tests", [])
    docs: dict[str, list[frozenset[int]]] = {}
    for d in JACCARD_DOCS:
        sets = {_shingles(doc_tokens.get(d, []))} | set(entry_docs.get(d, ()))
        sets.discard(frozenset())
        if sets:
            docs[d] = list(sets)
    embed_text = "\n".join(prompts + item.texts("statement"))[:8000]
    return Features(hashes, grams, gram_text, item.ids, docs, embed_text, frozenset(test_io),
                    tuple(sorted(math_nums, key=lambda x: x[0])))


# ---------------------------------------------------------------------------------------------
# Indexes


def _add(d: dict, k: Any, v: str) -> None:
    cur = d.get(k)
    if cur is None:
        d[k] = v
    elif isinstance(cur, str):
        if cur != v:
            d[k] = {cur, v}
    else:
        cur.add(v)


def _get(d: dict, k: Any) -> tuple[str, ...]:
    cur = d.get(k)
    if cur is None:
        return ()
    return (cur,) if isinstance(cur, str) else tuple(cur)


class JaccardIndex:
    """Exact Jaccard >= t search. Prefix filtering: if J(x,y) >= t then |x&y| >= t*max(|x|,|y|),
    so the first |x| - ceil(t|x|) + 1 elements of x and of y (same global order) intersect."""

    def __init__(self, t: float) -> None:
        self.t = t
        self.post: dict[int, list[int]] = defaultdict(list)
        self.sets: list[tuple[str, frozenset[int]]] = []   # several sets may share one key

    def _prefix(self, s: frozenset[int]) -> list[int]:
        n = len(s)
        return sorted(s)[: n - math.ceil(self.t * n - 1e-9) + 1]

    def add(self, key: str, s: frozenset[int]) -> None:
        if not s:
            return
        sid = len(self.sets)
        self.sets.append((key, s))
        for h in self._prefix(s):
            self.post[h].append(sid)

    def query(self, s: frozenset[int]) -> list[tuple[str, float]]:
        if not s:
            return []
        cands: set[int] = set()
        for h in self._prefix(s):
            cands.update(self.post.get(h, ()))
        n, best = len(s), {}
        for sid in cands:
            k, y = self.sets[sid]
            if self.t * n - 1e-9 <= len(y) <= n / self.t + 1e-9:
                inter = len(s & y)
                j = inter / (n + len(y) - inter)
                if j >= self.t - 1e-12 and j > best.get(k, -1.0):
                    best[k] = j
        return list(best.items())


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return 0.0 if not na or not nb else sum(x * y for x, y in zip(a, b)) / (na * nb)


class Index:
    """Feature index over a set of items (eval items, or kept train items for dedup)."""

    def __init__(self, cfg: Config, use_grams: bool = True, use_embed: bool = True) -> None:
        self.cfg = cfg
        self.use_grams = use_grams
        self.use_embed = use_embed and cfg.embedder is not None
        self.hashes: dict[tuple[str, str], Any] = {}
        self.grams: dict[int, Any] = {}
        self.ids: dict[str, Any] = {}
        self.jacc = {d: JaccardIndex(cfg.jaccard_threshold) for d in JACCARD_DOCS}
        self.embeds: list[tuple[str, list[float]]] = []
        self.test_io: dict[str, Any] = {}
        self.test_io_n: dict[str, int] = {}
        self.math_nums: dict[str, list[tuple[str, frozenset[str]]]] = defaultdict(list)
        self.owner: dict[str, str] = {}   # item key -> dataset
        self.n = 0
        self._frozen_df: set[int] | None = None

    def add(self, item: Item, feats: Features | None = None) -> None:
        f = feats or features(item, self.cfg, with_grams=self.use_grams)
        k = item.key
        self.owner[k] = item.dataset
        self.n += 1
        for h in f.hashes:
            _add(self.hashes, h, k)
        for g in f.grams:
            _add(self.grams, g, k)
        for i in f.ids:
            _add(self.ids, i, k)
        for t in f.test_io:
            _add(self.test_io, t, k)
        if f.test_io:
            self.test_io_n[k] = self.test_io_n.get(k, 0) + len(f.test_io)
        for h, ws in f.math_nums:
            self.math_nums[h].append((k, ws))
        for d, sets in f.docs.items():
            for s_ in sets:
                self.jacc[d].add(k, s_)
        if self.use_embed and f.embed_text:
            self.embeds.append((k, list(self.cfg.embedder([f.embed_text])[0])))

    def finalize(self) -> None:
        """Apply ngram_max_eval_df: grams shared by too many indexed items are boilerplate."""
        lim = self.cfg.ngram_max_eval_df
        if lim is not None:
            self._frozen_df = {g for g, v in self.grams.items() if not isinstance(v, str) and len(v) > lim}

    def query(self, item: Item, f: Features, exclude_dataset: str | None = None) -> list[Hit]:
        hits: list[Hit] = []

        def keep(k: str) -> bool:
            return exclude_dataset is None or self.owner.get(k) != exclude_dataset

        for h in f.hashes:
            for k in _get(self.hashes, h):
                if keep(k):
                    hits.append(Hit(h[0], k))
        for i in f.ids:
            for k in _get(self.ids, i):
                if keep(k):
                    hits.append(Hit("id", k, i))
        shared: Counter[str] = Counter()
        for t in f.test_io:
            for k in _get(self.test_io, t):
                if keep(k):
                    shared[k] += 1
        for k, n in shared.items():
            if n >= min(TEST_IO_MIN_SHARED, max(2, self.test_io_n.get(k, 0))):
                hits.append(Hit("test_io", k, f"{n} shared assertions"))
        for h, ws in f.math_nums:
            for k, ews in self.math_nums.get(h, ()):
                j = len(ws & ews) / max(1, len(ws | ews))
                if j >= MATH_NUMS_JACCARD and keep(k):
                    hits.append(Hit("math_nums", k, f"{j:.3f}"))
        if self.use_grams:
            seen: set[str] = set()
            for g in f.grams:
                if self._frozen_df and g in self._frozen_df:
                    continue
                for k in _get(self.grams, g):
                    if k not in seen and keep(k):
                        seen.add(k)
                        hits.append(Hit(f"ngram{self.cfg.ngram_n}", k, f.gram_text.get(g, "")[:200]))
        for d, sets in f.docs.items():
            best: dict[str, float] = {}
            for s_ in sets:
                for k, j in self.jacc[d].query(s_):
                    if keep(k) and j > best.get(k, -1.0):
                        best[k] = j
            hits.extend(Hit(f"jaccard:{d}", k, f"{j:.3f}") for k, j in best.items())
        if self.use_embed and f.embed_text and self.embeds:
            v = self.cfg.embedder([f.embed_text])[0]
            for k, e in self.embeds:
                c = _cos(v, e)
                if c >= self.cfg.embed_threshold and keep(k):
                    hits.append(Hit("embed", k, f"{c:.3f}"))
        return hits


def source_rule_hits(item: Item, hf_id: str, cfg: Config) -> list[Hit]:
    rule = cfg.source_rules.get(hf_id)
    if not rule:
        return []
    fld, pat = rule
    val = item.meta.get(fld)
    if val is None:
        return [Hit("source_missing", f"rule:{hf_id}", fld)]
    return [Hit("source", f"rule:{hf_id}", val)] if re.search(pat, val) else []




# ---------------------------------------------------------------------------------------------
# Pipeline


def _eval_set_of(key: str) -> str:
    return key.split("::", 1)[0]


def hits_to_json(hits: Iterable[Hit]) -> list[dict[str, str]]:
    seen, out = set(), []
    for h in hits:
        k = (h.method, h.other)
        if k not in seen:
            seen.add(k)
            out.append({"method": h.method, "other": h.other, "detail": h.detail})
    return out


class Decontaminator:
    """Build once from eval items, then stream train items through `check` (and, for sources in
    the dedup group, `dedup_check`)."""

    def __init__(self, eval_items: Iterable[Item], cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self.index = Index(self.cfg)
        self.eval_sizes: Counter[str] = Counter()
        for it in eval_items:
            self.index.add(it)
            self.eval_sizes[it.dataset] += 1
        self.index.finalize()
        self.dedup = Index(self.cfg, use_grams=False, use_embed=False)

    def check(self, item: Item, hf_id: str = "") -> list[Hit]:
        f = features(item, self.cfg)
        return self.index.query(item, f) + source_rule_hits(item, hf_id or item.dataset, self.cfg)

    def dedup_check(self, item: Item, add_if_clean: bool = True) -> list[Hit]:
        f = features(item, self.cfg, with_grams=False)
        hits = [Hit("dedup:" + h.method, h.other, h.detail)
                for h in self.dedup.query(item, f, exclude_dataset=item.dataset)]
        if not hits and add_if_clean:
            self.dedup.add(item, f)
        return hits


class Summary:
    """Streaming aggregation of flagged train items into per-eval-set and per-pair counts."""

    def __init__(self, eval_sizes: dict[str, int]) -> None:
        self.eval_sizes = dict(eval_sizes)
        self.matched: dict[str, set[str]] = defaultdict(set)
        self.by_method: dict[str, Counter[str]] = defaultdict(Counter)
        self.by_source: dict[str, Counter[str]] = defaultdict(Counter)
        self.pair: Counter[tuple[str, str]] = Counter()
        self.rules: Counter[tuple[str, str]] = Counter()
        self.grams: Counter[str] = Counter()   # matched n-gram text -> train items (boilerplate audit)

    def add(self, train_set: str, hits: Iterable[Hit]) -> None:
        by_eval: dict[str, set[str]] = defaultdict(set)
        rules: set[str] = set()
        grams: set[str] = set()
        for h in hits:
            if h.method.startswith("dedup:") or h.other.startswith("rule:"):
                rules.add(h.method.split(":", 1)[0] if h.method.startswith("dedup:") else h.method)
                continue
            e = _eval_set_of(h.other)
            by_eval[e].add(h.method)
            if h.method.startswith("ngram"):
                grams.add(h.detail)
            self.matched[e].add(h.other)
        for e, methods in by_eval.items():
            self.pair[(train_set, e)] += 1
            self.by_source[e][train_set] += 1
            for m in methods:
                self.by_method[e][m] += 1
        for r in rules:
            self.rules[(train_set, r)] += 1
        self.grams.update(grams)

    def to_json(self) -> dict[str, Any]:
        sets = sorted(set(self.eval_sizes) | set(self.matched))
        return {
            "eval_sets": {e: {
                "n_items": self.eval_sizes.get(e, 0),
                "n_eval_items_matched": len(self.matched.get(e, ())),
                "eval_items_matched": sorted(self.matched.get(e, ())),
                "train_items_flagged_by_method": dict(sorted(self.by_method.get(e, {}).items())),
                "train_items_flagged_by_source": dict(sorted(self.by_source.get(e, {}).items())),
            } for e in sets},
            "removal_counts": [{"train_source": t, "eval_set": e, "n_removed": n}
                               for (t, e), n in sorted(self.pair.items())],
            "rule_removal_counts": [{"train_source": t, "rule": r, "n_removed": n}
                                    for (t, r), n in sorted(self.rules.items())],
            # If a few generic phrases dominate, consider --ngram-max-eval-df (pre-register it).
            "top_matched_ngrams": [{"ngram": g, "n_train_items": n} for g, n in self.grams.most_common(25)],
        }


# ---------------------------------------------------------------------------------------------
# Dataset IO (local files only; never downloads)


class DatasetNotMaterialized(RuntimeError):
    """No local path for a dataset descriptor, or the path does not exist."""


def _iter_arrow(path: Path) -> Iterator[dict[str, Any]]:
    import pyarrow as pa  # lazy: only needed for HF cache/save_to_disk .arrow files
    with pa.memory_map(str(path)) as src:
        try:
            reader: Any = pa.ipc.open_stream(src)
        except pa.ArrowInvalid:
            reader = pa.ipc.open_file(src)
            for i in range(reader.num_record_batches):
                yield from reader.get_batch(i).to_pylist()
            return
        for batch in reader:
            yield from batch.to_pylist()


_SHARD_RE = re.compile(r"(-\d{5}-of-\d{5})?\.arrow$")


def _arrow_split(p: Path) -> str:
    stem = _SHARD_RE.sub("", p.name)
    return p.parent.name if stem == "data" else stem.rsplit("-", 1)[-1]


def dataset_files(path: str | Path, split: str | None = None) -> list[Path]:
    p = Path(path)
    if not p.exists():
        raise DatasetNotMaterialized(f"{p} does not exist")
    if p.is_file():
        return [p]
    arrows = sorted(p.rglob("*.arrow"))
    if arrows:
        if split:
            arrows = [a for a in arrows if _arrow_split(a) == split]
            if not arrows:
                raise DatasetNotMaterialized(f"{p} has no .arrow files for split {split!r}")
        elif len({_arrow_split(a) for a in arrows}) > 1:
            raise ValueError(f"{p} holds splits {sorted({_arrow_split(a) for a in arrows})}; set `split`")
        return arrows
    files = sorted(f for ext in ("*.jsonl", "*.json", "*.parquet") for f in p.rglob(ext))
    if split:
        files = [f for f in files if split in f.stem]
    if not files:
        raise DatasetNotMaterialized(f"{p} has no readable data files (split={split!r})")
    return files


def iter_dataset_rows(path: str | Path, split: str | None = None) -> Iterator[dict[str, Any]]:
    from gwaya.domains.loaders import iter_rows
    for f in dataset_files(path, split):
        if f.suffix == ".arrow":
            yield from _iter_arrow(f)
        else:
            yield from iter_rows(f)


def files_sha256(files: Iterable[Path]) -> str:
    h = hashlib.sha256()
    for f in files:
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def descriptor_name(d: dict[str, Any]) -> str:
    return d.get("name") or ":".join(str(x) for x in (d["hf_id"], d.get("config"), d.get("split")) if x)


def resolve_path(d: dict[str, Any], manifest: dict[str, Any] | None) -> str | None:
    if d.get("path"):
        return d["path"]
    for group in (manifest or {}).get("datasets", {}).values():
        for e in group:
            if (e.get("hf_id") == d.get("hf_id") and e.get("path")
                    and e.get("config") in (None, d.get("config"))
                    and e.get("split") in (None, d.get("split"))):
                return e["path"]
    return None


@dataclass
class Source:
    """A loaded dataset: either rows from a local file (`path`) or item-level records."""
    name: str
    hf_id: str
    domain: str
    role: str
    path: str | None = None
    split: str | None = None
    records: list[dict[str, Any]] | None = None
    declared_rows: int | None = None

    def items(self, known_fields_only: bool = False) -> Iterator[tuple[Item, dict[str, Any]]]:
        if self.records is not None:
            for r in self.records:
                yield item_from_record(r, self.name), r
            return
        if not self.path:
            raise DatasetNotMaterialized(f"{self.name}: no local path")
        for i, row in enumerate(iter_dataset_rows(self.path, self.split)):
            yield row_to_item(self.name, self.domain, row, i, self.hf_id, known_fields_only), row


def sources_from_jsonl(path: str | Path, manifest: dict[str, Any] | None = None,
                       plan_domains: dict[str, str] | None = None) -> list[Source]:
    """Descriptor lines {hf_id, domain, role, path?, split?, config?, name?, row_count?} and/or
    item-level lines {eval_set|dataset, item_id, domain, fields, ids?} (grouped per set)."""
    out: list[Source] = []
    by_set: dict[str, Source] = {}
    for rec in read_jsonl(path):
        if "item_id" in rec:
            name = rec.get("eval_set") or rec.get("dataset") or "items"
            if name not in by_set:
                by_set[name] = Source(name, rec.get("hf_id", name), rec.get("domain", "unknown"),
                                      rec.get("role", "eval"), records=[])
                out.append(by_set[name])
            by_set[name].records.append(rec)
            continue
        hf_id = rec["hf_id"]
        domain = rec.get("domain") or (plan_domains or {}).get(hf_id, "unknown")
        out.append(Source(descriptor_name(rec), hf_id, domain, rec.get("role", "eval"),
                          path=resolve_path(rec, manifest), split=rec.get("split"),
                          declared_rows=rec.get("row_count")))
    return out


def check_train_source(src: Source) -> None:
    """Pre-registered: no test split in T/C (e.g. MATH test); eval-role entries are not train."""
    if src.role == "eval":
        raise ValueError(f"{src.name}: role 'eval' cannot be a train/calib source")
    if src.split and "test" in src.split.lower():
        raise ValueError(f"{src.name}: split {src.split!r} is a test split; not allowed in T/C")


# ---------------------------------------------------------------------------------------------
# End-to-end


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "__", name)


def _write_line(fh: Any, obj: dict[str, Any]) -> None:
    fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def load_eval(sources: Sequence[Source], known_fields_only: bool = False
              ) -> tuple[list[Item], dict[str, dict[str, Any]]]:
    items: list[Item] = []
    status: dict[str, dict[str, Any]] = {}
    for s in sources:
        st: dict[str, Any] = {"hf_id": s.hf_id, "domain": s.domain, "path": s.path, "split": s.split,
                              "declared_row_count": s.declared_rows}
        try:
            n0 = len(items)
            items.extend(it for it, _ in s.items(known_fields_only))
            st["n_items"] = len(items) - n0
            st["status"] = "loaded"
            if s.path:
                st["data_sha256"] = files_sha256(dataset_files(s.path, s.split))
            if s.declared_rows is not None and s.declared_rows != st["n_items"]:
                st["note"] = (f"declared row_count {s.declared_rows} != loaded {st['n_items']} "
                              "(check split selection)")
        except DatasetNotMaterialized as e:
            st.update(status="not_materialized", n_items=0, error=str(e))
        except (ValueError, KeyError, OSError, ImportError) as e:
            st.update(status="error", n_items=0, error=f"{type(e).__name__}: {e}")
        status[s.name] = st
    return items, status


def run(eval_sources: Sequence[Source], train_sources: Sequence[Source], out_dir: str | Path,
        cfg: Config | None = None, allow_missing: bool = False, known_fields_only: bool = False,
        index_sha256: str | None = None, expected_eval: Sequence[str] = ()) -> dict[str, Any]:
    """Flag every train/calib item that matches any eval item; write clean train sets, flagged
    records and `decontam_report.json` to out_dir; then re-read every written clean file and
    re-check it (must have zero hits) so the report can state `verified_clean` per set.

    Fails closed: if any eval set is not loaded, or an `expected_eval` hf_id (the plan's eval sets)
    has no entry in the index, and not `allow_missing`, only the report is written ("aborted")."""
    cfg = cfg or Config()
    out = Path(out_dir)
    (out / "clean").mkdir(parents=True, exist_ok=True)
    for s in train_sources:
        check_train_source(s)
    eval_items, eval_status = load_eval(eval_sources, known_fields_only)
    indexed = {s.hf_id for s in eval_sources}
    for hf in expected_eval:
        if hf not in indexed:
            eval_status[hf] = {"hf_id": hf, "status": "not_in_index", "n_items": 0,
                               "error": "eval set listed in the plan but absent from the index"}
    missing = sorted(n for n, st in eval_status.items() if st["status"] != "loaded")
    report: dict[str, Any] = {
        "schema": "gwaya.decontam/v1", "config": cfg.public(), "index_sha256": index_sha256,
        "eval_status": eval_status, "missing_eval_sets": missing, "complete": not missing,
        "notes": ["thresholds are conventional, not validated on this data",
                  "a train item is removed on ANY hit; counts below are train items, not pairs"],
    }
    if missing and not allow_missing:
        report["status"] = "aborted: eval sets not loaded (use allow_missing to proceed)"
        (out / "decontam_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        return report

    dec = Decontaminator(eval_items, cfg)
    summary = Summary(dec.eval_sizes)
    train_report: dict[str, dict[str, Any]] = {}
    flagged_path = out / "flagged.jsonl"
    with open(flagged_path, "w") as ff:
        for s in train_sources:
            in_group = s.hf_id in cfg.dedup_group
            clean_path = out / "clean" / f"{_safe(s.name)}.jsonl"
            tr = {"hf_id": s.hf_id, "domain": s.domain, "role": s.role, "path": s.path,
                  "split": s.split, "n_in": 0, "n_flagged": 0, "n_kept": 0,
                  "flagged_by_method": Counter(), "clean_path": str(clean_path)}
            try:
                with open(clean_path, "w") as cf:
                    for item, row in s.items(known_fields_only):
                        tr["n_in"] += 1
                        hits = dec.check(item, s.hf_id)
                        if not hits and in_group:
                            hits = dec.dedup_check(item)
                        if hits:
                            tr["n_flagged"] += 1
                            for m in {h.method for h in hits}:
                                tr["flagged_by_method"][m] += 1
                            summary.add(s.name, hits)
                            _write_line(ff, {"train_set": s.name, "key": item.key,
                                             "hits": hits_to_json(hits)})
                        else:
                            tr["n_kept"] += 1
                            _write_line(cf, row)
                tr["status"] = "done"
            except (DatasetNotMaterialized, ValueError, KeyError, OSError, ImportError) as e:
                tr["status"] = f"error: {type(e).__name__}: {e}"
                clean_path.unlink(missing_ok=True)
            tr["flagged_by_method"] = dict(sorted(tr["flagged_by_method"].items()))
            train_report[s.name] = tr

    report["verification"] = verify_clean(dec, train_sources, train_report, cfg, known_fields_only)
    for name, tr in train_report.items():
        p = Path(tr["clean_path"])
        if p.exists():
            tr["clean_sha256"] = files_sha256([p])
        v = report["verification"].get(name, {})
        tr["verified_clean"] = bool(v.get("ok"))
    report["train_sets"] = train_report
    report.update(summary.to_json())
    report["flagged_path"] = str(flagged_path)
    report["flagged_sha256"] = files_sha256([flagged_path])
    report["status"] = "done" if all(t.get("status") == "done" for t in train_report.values()) \
        else "partial: some train sets failed to load"
    if missing:
        report["status"] += f"; INCOMPLETE: {len(missing)} eval set(s) not loaded"
    (out / "decontam_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def verify_clean(dec: Decontaminator, train_sources: Sequence[Source],
                 train_report: dict[str, dict[str, Any]], cfg: Config,
                 known_fields_only: bool = False) -> dict[str, dict[str, Any]]:
    """Independent second pass over the WRITTEN files: rebuild items from the JSONL on disk,
    re-check against the eval index and source rules, and re-run cross-source dedup from scratch."""
    fresh_dedup = Decontaminator([], cfg)
    out: dict[str, dict[str, Any]] = {}
    for s in train_sources:
        tr = train_report.get(s.name, {})
        p = Path(tr.get("clean_path", ""))
        if tr.get("status") != "done" or not p.exists():
            out[s.name] = {"ok": False, "reason": "not written"}
            continue
        n, bad = 0, []
        with open(p) as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                item = (item_from_record(row, s.name) if s.records is not None
                        else row_to_item(s.name, s.domain, row, i, s.hf_id, known_fields_only))
                n += 1
                hits = dec.check(item, s.hf_id)
                if s.hf_id in cfg.dedup_group:
                    hits += fresh_dedup.dedup_check(item)
                if hits:
                    bad.append({"key": item.key, "hits": hits_to_json(hits)[:5]})
        ok = not bad and n == tr.get("n_kept")
        out[s.name] = {"ok": ok, "n_rechecked": n, "n_hits": len(bad), "examples": bad[:5]}
    return out
