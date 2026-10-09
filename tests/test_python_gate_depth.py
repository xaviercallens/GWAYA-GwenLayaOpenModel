import pytest

from scripts import python_gate_depth as G

MODULE_LEVEL = """import math

def assertion(out, exp, atol):
    assert out == exp


inputs = [[1, 2], [3, 4], [5, 6], [7, 8]]
results = [3, 7, 11, 15]
for i, (inp, exp) in enumerate(zip(inputs, results)):
    assertion(f(*inp), exp, 0)
"""

INSIDE_CHECK = """def check(candidate):
    inputs = [['a'], ['bb', 'é'], ['{', '}']]
    results = [1, 2, 3]
    for i, (inp, exp) in enumerate(zip(inputs, results)):
        assert candidate(*inp) == exp
"""


def test_truncate_module_level_lists():
    out = G.truncate_pairs(MODULE_LEVEL, 2)
    assert "inputs = [[1, 2], [3, 4]]" in out and "results = [3, 7]" in out and "[5, 6]" not in out
    assert G.n_pairs(out) == 2 and "zip(inputs, results)" in out


def test_truncate_inside_check_with_unicode_and_braces():
    out = G.truncate_pairs(INSIDE_CHECK, 2)
    assert "['a'], ['bb', 'é']" in out and "'}'" not in out and "results = [1, 2]" in out
    assert G.n_pairs(out) == 2


@pytest.mark.parametrize("src,why", [
    ("inputs = [1]\nresults = [1]\ninputs2 = 3\nx = 1", "fewer_than_k_pairs"),
    ("def f(): pass", "no_pairs"),
    ("inputs = [1, 2]\nresults = [1]", "no_pairs"),
    ("inputs = [1, 2]\nresults = [1, 2]\ninputs = [3]", "no_pairs"),
])
def test_unsupported_shapes_raise(src, why):
    with pytest.raises(ValueError, match=why):
        G.truncate_pairs(src, 2)


def test_k_equals_all_is_unchanged_semantically():
    assert G.n_pairs(G.truncate_pairs(MODULE_LEVEL, 4)) == 4
