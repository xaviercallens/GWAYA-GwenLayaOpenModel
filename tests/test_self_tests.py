"""Self-generated tests: parsing is strict, consensus never trusts a lone vote."""
from gwaya.self_tests import consensus_asserts, parse_asserts

PUBLIC = "assert f(2) == 4\n"


def test_parse_keeps_only_safe_unique_asserts():
    text = "```python\nassert f(3) == 6\nassert f(3) == 6\nassert f(2) == 4\nimport os\nassert os.getcwd()\nassert f(1) ==\nassert f(0) == 0\n```"
    assert parse_asserts(text, PUBLIC) == ["assert f(3) == 6", "assert f(0) == 0"]


def test_parse_drops_prose_and_dunder():
    assert parse_asserts("Here are tests:\nassert __import__('os')\nassert f(1) == 2", None) == ["assert f(1) == 2"]


def test_consensus_requires_two_votes():
    asserts = ["assert a", "assert b", "assert c"]
    matrix = [[True, True, False], [True, False, False], [True, False, True]]
    assert consensus_asserts(asserts, matrix, min_agree=2) == ["assert a"]


def test_consensus_needs_enough_candidates():
    assert consensus_asserts(["assert a"], [[True]], min_agree=2) == []
