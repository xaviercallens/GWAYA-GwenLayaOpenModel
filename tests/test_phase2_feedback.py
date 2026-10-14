import importlib.util
import pathlib

import pytest

spec = importlib.util.spec_from_file_location("bfs", pathlib.Path(__file__).parents[1] / "scripts/phase2/build_feedback_set.py")
bfs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfs)


def row(i, correct=1, gate="verified", tier="t", src="f"):
    return {"source": src, "id": str(i), "tier": tier, "prompt": f"p{src}{i}", "correct": correct, "signals": {"gate": gate}}


def test_hard_negative_definition():
    assert bfs.is_hard_negative(row(1, correct=0, gate="verified"))
    assert not bfs.is_hard_negative(row(1, correct=0, gate="failed"))
    assert not bfs.is_hard_negative(row(1, correct=1, gate="verified"))


def test_heldout_never_in_train_and_tasks_grouped():
    f = [row(i, tier=t) for i in range(200) for t in ("a", "b", "c")]
    sets, rep = bfs.build([row(i, src="b") for i in range(5)], [], f, [], [], 3)
    ho = {bfs.key(r) for r in sets["calibrator_F_heldout"]}
    tr = {bfs.key(r) for r in sets["calibrator_v2_train"] if r["source"] == "f"}
    assert ho and not (ho & tr)
    assert all(bfs.fold_of(r) == 0 for r in sets["calibrator_F_heldout"])


def test_hard_negatives_duplicated_only_in_train():
    f = [row(i, correct=0) for i in range(100)]
    sets, rep = bfs.build([], [], f, [], [], 3)
    n_tr = rep["F_train_calibrator_rows"]
    assert rep["hard_negatives_in_train"] == n_tr
    assert len(sets["calibrator_v2_train"]) == 3 * n_tr
    assert len(sets["calibrator_F_heldout"]) == 100 - n_tr


def test_overlap_with_base_aborts():
    with pytest.raises(SystemExit):
        bfs.build([row(1)], [], [row(1)], [], [], 1)


def test_human_feedback_matched_or_dropped():
    f = [row(i, correct=1) for i in range(50)]
    k = next(r for r in f if bfs.fold_of(r) != 0)
    human = [{"source": "f", "id": k["id"], "tier": "t", "verdict": "reject"},
             {"source": "f", "id": "nope", "tier": "t", "verdict": "accept"}]
    sets, rep = bfs.build([], [], f, [], human, 1)
    assert rep["human_feedback_rows_matched"] == 1
    assert rep["human_feedback_stats"]["dropped"] == 1
    assert rep["human_feedback_stats"]["disagrees_with_stored_label"] == 1
