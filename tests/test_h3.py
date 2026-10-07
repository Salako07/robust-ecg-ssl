"""Checks for the H3 helpers (CPU, synthetic data)."""
import os, sys
import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from robust_ecg import augment, h3  # noqa: E402
from robust_ecg.models import SimCLRNet, XResNet1d50  # noqa: E402


def test_select_level_highest_and_ties_to_lower():
    assert h3.select_level(dict(low=0.80, mid=0.85, high=0.84)) == "mid"
    assert h3.select_level(dict(low=0.80, mid=0.85, high=0.86)) == "high"
    assert h3.select_level(dict(low=0.8498, mid=0.85, high=0.84)) == "low"        # within the margin -> lower level
    assert h3.select_level(dict(low=0.84, mid=0.8497, high=0.85)) == "mid"
    assert h3.select_level(dict(low=0.8494, mid=0.85, high=0.84)) == "mid"        # 0.0006 apart: not a tie
    with pytest.raises(ValueError):
        h3.select_level(dict(low=0.8, mid=0.9))


def test_needed_runs_grid_at_seed0_and_selected_at_all_seeds():
    runs = h3.needed_runs(dict(ecg="mid", gen="high"))
    assert len(runs) == 6 + 2 * 4
    assert sum(sel for *_, sel in runs) == 10
    assert {(p, l) for p, l, s, _ in runs if s == 0} == {(p, l) for p in h3.POLICIES for l in h3.LEVELS}
    assert {(p, l) for p, l, s, _ in runs if s > 0} == {("ecg", "mid"), ("gen", "high")}


def test_run_names_and_core_reuse():
    assert h3.ft_run_id("gen", "high", 3) == "C1_b1_s3_lik0_h3-gen-high"
    assert h3.ssl_run_id("gen", "low", 0) == "SSL_gen-low_s0"
    assert h3.is_core("ecg", "mid") and not h3.is_core("ecg", "low") and not h3.is_core("gen", "mid")
    assert h3.run_dir("w", "ecg", "mid", 2) == os.path.join("w", "runs", "C1_b1_s2_lik0")
    assert h3.run_dir("w", "gen", "mid", 2) == os.path.join("w", "runs_h3", "C1_b1_s2_lik0_h3-gen-mid")
    assert (h3.FT_POLICY, h3.FT_LEVEL) == ("ecg", "mid") and h3.LEVELS == tuple(augment.STRENGTH)


def test_frozen_features_load_simclr_encoder_and_match_finetune_encoder(tmp_path):
    torch.manual_seed(0)
    net = SimCLRNet()
    path = tmp_path / "encoder.pt"
    torch.save({"encoder": net.encoder.state_dict()}, path)
    f = h3.load_encoder(str(path), torch.device("cpu"))
    x = torch.randn(3, 12, 250)
    with torch.no_grad():
        out = f(x)
        ref = net.eval().pool(net.encoder(x))
    assert out.shape == (3, 512) and torch.allclose(out, ref)
    model = XResNet1d50(71)                                    # the same weights load into the fine-tuning model
    model.encoder.load_state_dict(torch.load(path)["encoder"], strict=True)


def test_fit_probe_learns_a_linear_rule_and_is_deterministic():
    rng = np.random.default_rng(0)
    F = rng.standard_normal((600, 20)).astype(np.float32) * 5 + 3          # unstandardised on purpose
    Y = np.stack([(F[:, 0] > 3), (F[:, 1] + F[:, 2] > 6)], 1).astype(np.float32)
    P1, loss1 = h3.fit_probe(F[:400], Y[:400], F[400:])
    P2, loss2 = h3.fit_probe(F[:400], Y[:400], F[400:])
    assert np.array_equal(P1, P2) and loss1 == loss2
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(Y[400:, 0], P1[:, 0]) > 0.97 and roc_auc_score(Y[400:, 1], P1[:, 1]) > 0.97


def test_view_distortion_grows_with_level():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 12, 1000)).astype(np.float32)
    mean, std = np.zeros((12, 1), np.float32), np.ones((12, 1), np.float32)
    d = [h3.view_distortion(X, np.arange(64), mean, std, "ecg", l, n_batches=4, bs=64) for l in h3.LEVELS]
    assert 0 < d[0] < d[1] < d[2] < 0.2


def test_iut_is_the_larger_p():
    assert h3.iut_p(0.01, 0.2) == 0.2 and h3.iut_p(0.3, 0.02) == 0.3
