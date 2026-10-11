"""scripts/analyze_laya_layout.py on synthetic predictions (no model, no TPU)."""
import json
import re

import numpy as np
import pytest

from scripts import analyze_laya_layout as AL

TIERS = ["qwen3.5-2b-bf16", "qwen3.5-4b-bf16", "qwen3.5-9b-bf16"]


class WordTok:
    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, **kw):
        spans = [m.span() for m in re.finditer(r"\S+", text)]
        ids = list(range(len(spans)))
        if add_special_tokens:
            ids = [-1] + ids + [-2]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out


def make(n_prob=60, domains=("python", "math"), seed=0, v2_signal=1.5, long_prompt_every=2):
    rng = np.random.default_rng(seed)
    rows, v1, v2 = [], {}, {}
    for d in domains:
        for i in range(n_prob):
            for t in TIERS:
                y = int(rng.random() < 0.5)
                rows.append({"source": d[:2], "id": f"{d}/{i}", "domain": d, "tier": t, "correct": y,
                             "prompt": " ".join(f"p{k}" for k in range(900 if i % long_prompt_every == 0 else 20)),
                             "candidate": "c0 c1 c2 c3",
                             "signals": {"gate": "verified" if rng.random() < 0.5 else "failed",
                                         "mean_logprob": float(rng.normal(-0.5, 0.2)), "min_logprob": float(rng.normal(-3, 1))}})
                key = (d[:2], f"{d}/{i}", t)
                v1[key] = float(rng.random())                                    # v1: noise
                v2[key] = float(1 / (1 + np.exp(-(v2_signal * (y - 0.5) * 2 + rng.normal(0, 1)))))  # v2: informative
    return rows, v1, v2


def write_preds(path, preds):
    path.write_text("".join(json.dumps({"key": [k[0], k[1]], "tier": k[2], "probs": [p]}) + "\n" for k, p in preds.items()))


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_paired_differences_detect_an_informative_v2_and_are_deterministic():
    rows, v1, v2 = make()
    lr = np.random.default_rng(1).random(len(rows))
    res = AL.analyze_set(rows, v1, v2, lr, {}, None, TIERS, n_boot=300, seed=0)
    d = res["pooled"]["paired"]["v2_minus_v1"]["dAUROC"]
    assert d["point"] > 0.2 and d["lo"] > 0 and d["hi"] >= d["point"] >= d["lo"]
    assert res["pooled"]["metrics"]["v2"]["auroc"] > 0.7 and abs(res["pooled"]["metrics"]["v1"]["auroc"] - 0.5) < 0.12
    assert set(res["pooled"]["paired"]) == {"v2_minus_v1", "v2_minus_gate", "v2_minus_hand_lr", "v2_minus_raw_logprob"}
    assert set(res["per_tier"]) == set(TIERS) and res["pooled"]["n"] == 360 and res["pooled"]["n_clusters"] == 120
    again = AL.analyze_set(rows, v1, v2, lr, {}, None, TIERS, n_boot=300, seed=0)
    assert again["pooled"]["paired"] == res["pooled"]["paired"]
    for m in res["pooled"]["metrics"].values():
        assert set(m) == {"auroc", "aurc", "ece15", "brier"}


def test_null_difference_ci_covers_zero_and_rule_says_not_better():
    rows, v1, _ = make(seed=3)
    rng = np.random.default_rng(9)
    v2 = {k: float(rng.random()) for k in v1}
    res = AL.analyze_set(rows, v1, v2, np.zeros(len(rows)), {}, None, TIERS, n_boot=300, seed=0)
    d = res["pooled"]["paired"]["v2_minus_v1"]["dAUROC"]
    assert d["lo"] < 0 < d["hi"]
    assert AL.decision(res, res)["v2_better"] is False


def test_decision_rule_needs_both_sets():
    def fake(lo):
        return {"pooled": {"paired": {"v2_minus_v1": {"dAUROC": {"lo": lo}}}}}
    assert AL.decision(fake(0.02), fake(0.05))["v2_better"] is True
    assert AL.decision(fake(0.02), fake(-0.01))["v2_better"] is False
    assert AL.decision(fake(float("nan")), fake(0.1))["v2_better"] is False


def test_misaligned_keys_are_refused():
    rows, v1, v2 = make(n_prob=3)
    v2.pop(next(iter(v2)))
    with pytest.raises(KeyError, match="no prediction"):
        AL.analyze_set(rows, v1, v2, np.zeros(len(rows)), {}, None, TIERS, n_boot=5)


def test_cluster_is_the_problem_not_the_row():
    rows, _, _ = make(n_prob=4, domains=("python",))
    c = AL.clusters(rows, None)
    assert len(set(c.tolist())) == 4 and len(c) == 12
    assert set(AL.clusters(rows, {("python", "py/python/0"): "grp"}).tolist()) >= {"py/python/1"}


def test_window_fractions_under_each_layout():
    rows, _, _ = make(n_prob=10, domains=("rust",), long_prompt_every=2)
    w = AL.window_fractions(rows, WordTok(), ("prompt_first", "candidate_first"))
    assert w["n"] == 30
    assert w["prompt_first"]["entirely_outside"] == pytest.approx(0.5)      # the 900-word prompts push it out
    assert w["candidate_first"]["entirely_outside"] == 0.0 and w["candidate_first"]["fully_visible"] == 1.0


def test_isotonic_from_c_changes_calibrated_metrics_not_auroc():
    rows, v1, v2 = make(n_prob=30)
    c_rows, c1, c2 = make(n_prob=30, seed=5)
    maps = {"v1": (np.array([c1[(r["source"], r["id"], r["tier"])] for r in c_rows]), np.array([r["correct"] for r in c_rows])),
            "v2": (np.array([c2[(r["source"], r["id"], r["tier"])] for r in c_rows]), np.array([r["correct"] for r in c_rows]))}
    a = AL.analyze_set(rows, v1, v2, np.zeros(len(rows)), maps, None, TIERS, n_boot=20)
    b = AL.analyze_set(rows, v1, v2, np.zeros(len(rows)), {}, None, TIERS, n_boot=20)
    assert a["pooled"]["metrics"]["v2"]["auroc"] == pytest.approx(b["pooled"]["metrics"]["v2"]["auroc"])
    assert a["pooled"]["metrics"]["v2"]["brier"] != b["pooled"]["metrics"]["v2"]["brier"]


def test_cli_end_to_end(tmp_path):
    tr, _, _ = make(n_prob=40, seed=11)
    e, e1, e2 = make(n_prob=30, seed=1)
    r, r1, r2 = make(n_prob=20, domains=("rust",), seed=2)
    for name, rows in (("train", tr), ("e", e), ("r", r)):
        write_rows(tmp_path / f"{name}.jsonl", rows)
    for name, p in (("e1", e1), ("e2", e2), ("r1", r1), ("r2", r2)):
        write_preds(tmp_path / f"{name}.jsonl", p)
    out = tmp_path / "out.json"
    rc = AL.main(["--train-rows", str(tmp_path / "train.jsonl"), "--eprime-rows", str(tmp_path / "e.jsonl"),
                  "--rfresh-rows", str(tmp_path / "r.jsonl"), "--v1-eprime", str(tmp_path / "e1.jsonl"),
                  "--v2-eprime", str(tmp_path / "e2.jsonl"), "--v1-rfresh", str(tmp_path / "r1.jsonl"),
                  "--v2-rfresh", str(tmp_path / "r2.jsonl"), "--n-boot", "100", "--out", str(out)])
    assert rc == 0
    d = json.loads(out.read_text())
    assert d["meta"]["plan"] == "A22" and d["meta"]["n_boot"] == 100 and d["meta"]["seed"] == 0
    assert d["decision"]["v2_better"] is True
    assert d["eprime"]["pooled"]["n"] == 180 and d["rfresh"]["pooled"]["n"] == 60
