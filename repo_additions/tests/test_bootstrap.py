"""Checks for scripts/bootstrap_contrasts.py (CPU, synthetic data)."""
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from bootstrap_contrasts import WeightedAUC, holm, patient_weights, summarise  # noqa: E402


def test_weighted_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(50, 600))
        y = (rng.random(n) < rng.uniform(0.03, 0.5)).astype(float)
        if y.sum() in (0, n):
            continue
        s = np.round(rng.random(n) + 0.3 * y, int(rng.integers(1, 4))).astype(np.float16)  # coarse -> many ties
        calc = WeightedAUC(y, s.astype(np.float64))
        assert abs(calc(np.ones((1, n), np.float32))[0] - roc_auc_score(y, s)) < 1e-12
        W = next(patient_weights(rng.integers(0, max(2, n // 2), n), 30, rng, 30))
        for w, a in zip(W, calc(W)):
            if (w * y).sum() > 0 and (w * (1 - y)).sum() > 0:
                assert abs(a - roc_auc_score(y, s, sample_weight=w)) < 1e-12
            else:
                assert np.isnan(a)


def test_patient_weights_resample_whole_patients():
    rng = np.random.default_rng(1)
    pats = np.array(["a", "a", "b", "c", "c", "c", "d"])
    W = np.concatenate(list(patient_weights(pats, 500, rng, 64)))
    assert W.shape == (500, 7)
    codes, _ = pd.factorize(pats)
    for k in range(4):                       # all recordings of a patient share one weight
        assert (W[:, codes == k] == W[:, codes == k][:, :1]).all()
    per_patient = np.stack([W[:, codes == k][:, 0] for k in range(4)], 1)
    assert (per_patient.sum(1) == 4).all()   # n patients drawn per resample
    assert abs(per_patient.mean() - 1.0) < 1e-6


def test_summarise_and_holm():
    rng = np.random.default_rng(2)
    boot = rng.normal(0.01, 0.002, size=(4000, 5))           # clearly positive difference
    out = summarise(np.full(5, 0.01), boot)
    assert out["ci95_lo"] > 0 and out["p_two_sided"] < 0.01
    null = rng.normal(0.0, 0.01, size=(4000, 5))
    assert summarise(np.zeros(5), null)["p_two_sided"] > 0.05
    adj = holm(np.array([0.01, 0.04, 0.03, 0.005]))
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])
