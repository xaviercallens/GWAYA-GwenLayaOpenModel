"""Static grounding: references that cannot resolve are flagged; legitimate code is not."""
from gwaya.grounding import grounding_flags, is_hallucinated, runtime_hallucination


def test_undefined_name_is_flagged():
    assert any("helper" in f for f in grounding_flags("def f(x):\n    return helper(x)\n"))


def test_unresolvable_import_is_flagged():
    assert any("numpyx" in f for f in grounding_flags("import numpyx\ndef f():\n    return 1\n"))


def test_nonexistent_stdlib_attribute_is_flagged():
    flags = grounding_flags("import math\ndef f(x):\n    return math.cube_root(x)\n")
    assert any("cube_root" in f for f in flags)


def test_nonexistent_from_import_name_is_flagged():
    assert any("sqroot" in f for f in grounding_flags("from math import sqroot\n"))


def test_clean_code_is_not_flagged():
    code = "import math\nfrom collections import Counter\ndef f(xs):\n    return math.sqrt(len(Counter(xs)))\n"
    assert grounding_flags(code) == []


def test_preamble_names_are_known():
    # The harness runs the test's own imports in the candidate's namespace.
    assert grounding_flags("def f(x):\n    return math.sqrt(x)\n") != []
    assert grounding_flags("def f(x):\n    return math.sqrt(x)\n", preamble="import math") == []


def test_syntax_error_is_left_to_the_oracle():
    assert grounding_flags("def f(:\n") == []


def test_antigravity_is_never_imported():
    # Must not open a browser: stdlib module on the deny list is only checked for resolvability.
    assert grounding_flags("import antigravity\nx = antigravity.nothing\n") == []


def test_runtime_classification():
    assert runtime_hallucination("TEST_FAILED (0/3 passed): assert f(1) == 2 -> NameError: name 'g' is not defined")
    assert runtime_hallucination("IMPORT_FAILED: ModuleNotFoundError: No module named 'foo'")
    assert not runtime_hallucination("TEST_FAILED (1/3 passed): assert f(1) == 2 -> AssertionError")
    assert is_hallucinated("def f():\n    return g()\n", "")
