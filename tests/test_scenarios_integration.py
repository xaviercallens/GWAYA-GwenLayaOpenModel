import os
os.environ["GWAYA_ALLOW_UNISOLATED"] = "1"

import pytest
from webgwaya.app import SCENARIOS
from gwaya.low_tier_engine import LowTierModelOptimizer, ModelTier

@pytest.mark.parametrize("scenario", [s for s in SCENARIOS if s["id"].startswith("scenario_")])
def test_scenario_starter_code_evaluates_correctly(scenario):
    """
    Ensures that for every coding scenario, the provided starter_code
    actually passes its own test_spec using the low_tier_engine.
    """
    # Create the optimizer without a generator, we just want to verify
    opt = LowTierModelOptimizer(tier=ModelTier.LOW, generator=None)
    
    # We will simulate a generation by overriding the optimization loop
    # or by just calling the underlying verification oracle directly.
    from gwaya.oracles import PythonCompilerOracle
    oracle = PythonCompilerOracle()
    
    code = scenario["starter_code"]
    spec = scenario["test_spec"]
    
    res = oracle.verify_with_test(code, spec)
    assert res.success is True, f"Scenario {scenario['id']} starter_code failed verification: {res.error_message}"
    
    # Check that tests were executed
    assert res.details.get("passed", 0) > 0, "No tests passed!"
    assert res.details.get("passed") == res.details.get("total"), "Not all tests passed!"
