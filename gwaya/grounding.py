"""
gwaya/grounding.py
==================
Static *grounding* checks: does the code refer only to names, modules and attributes that exist?

Small code models hallucinate APIs: an undefined helper, ``import numpyx``, ``math.cube_root``.
These are cheap to detect before any test runs, and they are the failure classes the
hallucination metric counts. Everything here is a *proxy*: it finds references that cannot resolve
in the current environment; it does not prove the code is correct.

Public API
----------
``grounding_flags(code)``    -> list[str]   static findings (empty = nothing flagged)
``runtime_hallucination(error_text)`` -> bool   runtime error class attributable to a missing name
``is_hallucinated(code, error_text)`` -> bool   either of the above
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import re
import sys

try:  # pyflakes is a declared dependency; degrade loudly (not silently) if it is missing
    from pyflakes import messages as _pf_messages
    from pyflakes.checker import Checker as _PyflakesChecker
except ImportError:  # pragma: no cover
    _PyflakesChecker = None  # type: ignore[assignment,misc]
    _pf_messages = None  # type: ignore[assignment]

# Importing these has side effects (opens a browser, prints, needs a display, runs tests).
_NEVER_IMPORT = frozenset(
    {
        "antigravity",
        "this",
        "tkinter",
        "turtle",
        "idlelib",
        "test",
        "__main__",
        "__hello__",
        "__phello__",
        "turtledemo",
        "lib2to3",
    }
)

_RUNTIME_PATTERNS = (
    re.compile(r"\bNameError\b"),
    re.compile(r"\bModuleNotFoundError\b"),
    re.compile(r"\bImportError\b"),
    re.compile(r"\bAttributeError\b"),
    re.compile(r"\bUnboundLocalError\b"),
)


def _stdlib(name: str) -> bool:
    return name in sys.stdlib_module_names


def _import_flags(tree: ast.AST) -> tuple[list[str], dict[str, str]]:
    """Unresolvable imports, plus alias -> stdlib module map used by the attribute check."""
    flags: list[str] = []
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if importlib.util.find_spec(top) is None:
                    flags.append(f"unresolvable import '{a.name}'")
                elif _stdlib(top) and top not in _NEVER_IMPORT:
                    aliases[(a.asname or a.name).split(".")[0]] = a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            top = node.module.split(".")[0]
            if importlib.util.find_spec(top) is None:
                flags.append(f"unresolvable import '{node.module}'")
            elif _stdlib(top) and top not in _NEVER_IMPORT:
                try:
                    mod = importlib.import_module(node.module)
                except Exception:  # noqa: BLE001 - an import that fails is itself a finding
                    flags.append(f"unresolvable import '{node.module}'")
                    continue
                for a in node.names:
                    if a.name != "*" and not hasattr(mod, a.name):
                        try:
                            importlib.import_module(f"{node.module}.{a.name}")
                        except Exception:  # noqa: BLE001
                            flags.append(f"'{a.name}' does not exist in module '{node.module}'")
    return flags, aliases


def _attribute_flags(tree: ast.AST, aliases: dict[str, str]) -> list[str]:
    flags: list[str] = []
    mods: dict[str, object] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
        ):
            modname = aliases[node.value.id]
            if modname not in mods:
                try:
                    mods[modname] = importlib.import_module(modname)
                except Exception:  # noqa: BLE001
                    mods[modname] = None
            mod = mods[modname]
            if mod is not None and not hasattr(mod, node.attr):
                flags.append(f"module '{modname}' has no attribute '{node.attr}'")
    return flags


def _undefined_name_flags(tree: ast.AST) -> list[str]:
    checker = _PyflakesChecker(tree)
    return [
        f"undefined name '{m.message_args[0]}'"
        for m in checker.messages
        if isinstance(m, _pf_messages.UndefinedName)
    ]


def grounding_flags(code: str, preamble: str = "") -> list[str]:
    """Static findings for ``code``. Unparseable code returns ``[]``: syntax is the oracle's job.

    ``preamble`` is source (typically the test's own ``import`` lines) that the harness executes in
    the same namespace as the candidate; names it binds are not reported as undefined.
    """
    try:
        tree = ast.parse(f"{preamble}\n{code}" if preamble else code)
    except SyntaxError:
        return []
    import_flags, aliases = _import_flags(tree)
    out = [*import_flags, *_attribute_flags(tree, aliases), *_undefined_name_flags(tree)]
    seen: set[str] = set()
    return [f for f in out if not (f in seen or seen.add(f))]


def runtime_hallucination(error_text: str) -> bool:
    """True if a runtime error names a missing name/module/attribute (hallucinated API proxy)."""
    return any(p.search(error_text or "") for p in _RUNTIME_PATTERNS)


def is_hallucinated(code: str, error_text: str = "", preamble: str = "") -> bool:
    return bool(grounding_flags(code, preamble)) or runtime_hallucination(error_text)
