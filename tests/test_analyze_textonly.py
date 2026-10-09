import numpy as np

from scripts import analyze_textonly as T


def test_analyze_one_signs_on_synthetic_scores():
    rng = np.random.default_rng(0)
    n = 300
    y = rng.random(n) < 0.5
    dom = np.array(["math"] * 150 + ["python"] * 150)
    clu = np.array([f"c{i}" for i in range(n)])
    text = np.where(y, 0.6 + 0.4 * rng.random(n), 0.4 * rng.random(n))   # good ranking
    full = np.clip(text + 0.05, 0, 1)
    lp = rng.random(n)                                                   # uninformative
    lr = text * 0.9 + 0.05 * rng.random(n)
    r = T.analyze_one(y, dom, clu, text, full, lp, lr, n_boot=200, seed=0)
    assert r["auroc_text"]["point"] > 0.95 and r["d_auroc_text_minus_logprob"]["lo"] > 0
    assert "auroc_text.math" in r and "auroc_text.python" in r and r["n"] == n
