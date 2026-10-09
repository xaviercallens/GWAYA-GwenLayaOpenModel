from scripts import train_laya as T

ROW = {"source": "ma", "id": "1", "domain": "math", "prompt": "What is 2+2?", "tier": "qwen3.5-9b-bf16",
       "candidate": "The answer is \\boxed{4}", "correct": 1,
       "signals": {"gate": "verified", "mean_logprob": -0.05, "min_logprob": -1.2}}


def test_text_only_text_has_no_signal_leak():
    full = T.render_calibrator_text(ROW)
    only = T.render_calibrator_text_only(ROW)
    assert "[signals]" in full and "verified" in full and "-0.05" in full
    assert "[signals]" not in only and "verified" not in only and "-0.05" not in only
    assert "What is 2+2?" in only and "boxed{4}" in only and "[tier] qwen3.5-9b-bf16" in only


def test_text_only_examples_carry_no_numeric_features_and_same_labels():
    full = T.make_examples([ROW], "calibrator", ["t"])[0]
    only = T.make_examples([ROW], "calibrator", ["t"], text_only=True)[0]
    assert len(full["feats"]) == T.N_FEATURES and only["feats"] == []
    assert only["labels"] == full["labels"] == [1.0] and only["tier"] == full["tier"]
