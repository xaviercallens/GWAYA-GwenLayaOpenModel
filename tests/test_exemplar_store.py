import pytest
from gwaya.exemplar_store import ExemplarStore

def test_leakage_guard(tmp_path):
    store = ExemplarStore(store_path=str(tmp_path / "exemplars.json"))
    with pytest.raises(ValueError, match="Leakage guard: test-split problems cannot enter the exemplar store"):
        store.add_exemplar("hash1", "test", "Do something", "pass")
    
    # Should work for dev
    store.add_exemplar("hash2", "dev", "Do something else", "pass")
    assert len(store.exemplars) == 1
