"""Regression tests for fail-closed semantics in GWAYA."""
from gwaya.low_tier_engine import LowTierModelOptimizer
from gwaya.oracles import PythonCompilerOracle


def test_python_oracle_rejects_empty_code():
    oracle = PythonCompilerOracle()
    for snippet in ("", "# only a comment", '"""docstring only"""'):
        assert not oracle.verify(snippet).success


def test_no_generator_is_unverified():
    opt = LowTierModelOptimizer()
    out = opt.optimize_and_solve("sort a list")
    text = str(out)
    assert "UNVERIFIED" in text or "NO_GENERATOR" in text
