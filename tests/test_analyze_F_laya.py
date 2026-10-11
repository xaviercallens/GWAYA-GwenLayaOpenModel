"""scripts/analyze_F_laya.py on a synthetic prediction file (A21 H1/H3 plumbing, Holm, frozen-map hash check, router policy)."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_F_laya as F

TIERS = ["qwen3.5-2b-bf16", "qwen3.5-4b-bf16", "qwen3.5-9b-bf16"]


def _world(tmp: Path, n: int = 120, informative: bool = True):
    rng = np.random.default_rng(3)
    tasks, rows, preds, router, srows, sgens = [], [], [], [], [], []
    for i in range(n):
        dom = "python" if i % 2 else "math"
        src = "py" if dom == "python" else "ma"
        tid = f"{src}/item{i}"
        tasks.append({"domain": dom, "task_id": tid, "prompt": "p", "cluster": f"c{i}",
                      "stratum": "python/LeetCodeDataset" if dom == "python" else "math/GSM8K",
                      "gate_visible_in_prompt": False if dom == "python" else None})
        router.append({"key": [src, f"item{i}"], "tier": None, "probs": [0.6, 0.7, 0.8]})
        for t_i, tier in enumerate(TIERS):
            ok = bool(rng.random() < 0.4 + 0.2 * t_i)
            mlp = float(-rng.random() * 2)                            # uninformative log-prob
            gate = "verified" if rng.random() < 0.3 else "unverified"  # weak, uninformative gate
            rows.append({"source": src, "id": f"item{i}", "domain": dom, "prompt": "p", "tier": tier, "candidate": "c",
                         "correct": int(ok), "signals": {"gate": gate, "mean_logprob": mlp, "min_logprob": mlp - 1}})
            score = (0.5 + 0.4 * (1 if ok else -1) * rng.random()) if informative else rng.random()
            preds.append({"key": [src, f"item{i}"], "tier": tier, "probs": [float(score)]})
            srows += [{"arm": "base", "model": tier, "domain": dom, "task_id": tid, "score": "VERIFIED" if ok else "FAILED"},
                      {"arm": "gate_only", "model": tier, "domain": dom, "task_id": tid, "gate": gate.upper(), "answered": gate == "verified", "score": "x"}]
            sgens.append({"model": tier, "domain": dom, "task_id": tid, "temperature": 0.0, "mean_logprob": mlp, "done_reason": "stop",
                          "gpu_s": 1.0 + t_i, "completion_tokens": 10})
            if dom == "math":
                sgens.append({"model": tier, "domain": dom, "task_id": tid, "temperature": 0.0, "kind": "pot", "gpu_s": 0.5})
    w = lambda name, recs: (tmp / name).write_text("".join(json.dumps(r) + "\n" for r in recs))
    w("tasks.jsonl", tasks); w("cal_rows.jsonl", rows); w("preds.jsonl", preds); w("router_preds.jsonl", router)
    (tmp / "study").mkdir(); w("study/rows.jsonl", srows); w("study/gens.jsonl", sgens)
    (tmp / "iso.json").write_text(json.dumps({"x": [0.0, 0.5, 1.0], "y": [0.1, 0.5, 0.9]}))
    return tmp


def test_h1_h3_supported_for_informative_predictions(tmp_path):
    _world(tmp_path)
    out = tmp_path / "res.json"
    rc = F.main(["--cal-rows", str(tmp_path / "cal_rows.jsonl"), "--cal-preds", str(tmp_path / "preds.jsonl"),
                 "--tasks", str(tmp_path / "tasks.jsonl"), "--isotonic", str(tmp_path / "iso.json"), "--freeze", str(tmp_path / "nofreeze"),
                 "--router-preds", str(tmp_path / "router_preds.jsonl"), "--study-dirs", str(tmp_path / "study"),
                 "--n-boot", "200", "--out", str(out)])
    assert rc == 0
    r = json.loads(out.read_text())
    inf = r["primary"]["inference"]
    assert r["primary"]["n"] == 120
    assert inf["H1_supported"] and inf["H3_supported"]
    assert inf["H1_dAURC_laya_minus_logprob"]["point"] < 0 and inf["H3_dRisk_at_gate_coverage_laya_minus_gate"]["point"] < 0
    assert inf["H1_p_holm"] >= inf["H1_p_one_sided"] and "H1_supported" not in r["per_tier"][TIERS[0]]["inference"]
    assert r["secondary"]["python"][TIERS[2]]["n"] == 60 and r["secondary"]["math/competition"][TIERS[2]]["n"] == 0
    assert r["router_policy"]["tau"] == 0.5 and r["router_policy"]["ladder_acc"]["point"] is not None


def test_uninformative_predictions_are_not_supported(tmp_path):
    _world(tmp_path, informative=False)
    out = tmp_path / "res.json"
    F.main(["--cal-rows", str(tmp_path / "cal_rows.jsonl"), "--cal-preds", str(tmp_path / "preds.jsonl"),
            "--tasks", str(tmp_path / "tasks.jsonl"), "--isotonic", str(tmp_path / "iso.json"), "--freeze", str(tmp_path / "nofreeze"),
            "--n-boot", "200", "--out", str(out)])
    inf = json.loads(out.read_text())["primary"]["inference"]
    assert not (inf["H1_supported"] and inf["H3_supported"])


def test_frozen_map_hash_is_enforced(tmp_path):
    p = tmp_path / "laya_isotonic_C.json"
    p.write_text('{"x":[0,1],"y":[0,1]}')
    fr = tmp_path / "freeze.txt"
    fr.write_text("0" * 64 + "  laya_isotonic_C.json  (isotonic)\n")
    with pytest.raises(SystemExit):
        F.frozen_map(p, fr)
    import hashlib
    fr.write_text(hashlib.sha256(p.read_bytes()).hexdigest() + "  laya_isotonic_C.json  (isotonic)\n")
    assert F.frozen_map(p, fr)["x"] == [0, 1]


def test_real_frozen_map_matches_freeze_file():
    root = Path(__file__).resolve().parents[1]
    assert F.frozen_map(root / "docs" / "laya_isotonic_C.json", root / "docs" / "LAYA_FREEZE.txt")["x"]
