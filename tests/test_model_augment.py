import os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from robust_ecg import augment, metrics  # noqa: E402
from robust_ecg.models import XResNet1d50, count_params  # noqa: E402


def test_model_shapes_and_size():
    m = XResNet1d50(71)
    assert m(torch.randn(2, 12, 250)).shape == (2, 71)
    assert count_params(m.encoder) == 886_400          # identical to the PTB-XL benchmark encoder
    assert m.encoder.out_dim == 256


def test_each_transform_preserves_shape_and_is_per_sample():
    torch.manual_seed(0)
    x = torch.randn(8, 12, 250)
    for pol in augment.POLICIES.values():
        for fn in pol:
            y = fn(x.clone(), 1.0)
            assert y.shape == x.shape and torch.isfinite(y).all() and not torch.equal(y, x)
    # per-lead scaling differs across leads and samples; global scaling is constant across leads
    r = augment.per_lead_scaling(torch.ones(4, 12, 250), 1.0)[:, :, 0]
    assert r.std(1).min() > 0 and r[:, 0].std() > 0
    g = augment.global_scaling(torch.ones(4, 12, 250), 1.0)[:, :, 0]
    assert torch.allclose(g, g[:, :1].expand_as(g)) and g[:, 0].std() > 0


def test_emg_bursts_are_local():
    torch.manual_seed(2)
    d = augment.emg_bursts(torch.zeros(64, 12, 250), 1.0) != 0
    frac = d.float().mean(2)                        # fraction of each lead affected
    assert frac.max() <= 1.0 and frac.mean() < 0.8 and (frac > 0).float().mean() > 0.9


def test_strength_scales_magnitude():
    torch.manual_seed(1)
    x = torch.zeros(2000, 12, 250)
    lo = augment.baseline_wander(x, 0.5).abs().mean().item()
    hi = augment.baseline_wander(x, 1.5).abs().mean().item()
    assert 2.5 < hi / lo < 3.5                      # ~3x for a 3x strength ratio


def test_time_warp_is_small_and_length_preserving():
    t = torch.linspace(0, 1, 250).repeat(4, 12, 1)
    w = augment.time_warp(t, 1.0)
    assert w.shape == t.shape and (w[..., 1:] >= w[..., :-1] - 1e-6).all()
    assert (w - t).abs().max() < 0.1


def test_time_mask_zeroes_one_segment():
    torch.manual_seed(3)
    y = augment.time_mask(torch.ones(16, 12, 250), 1.0)
    z = (y == 0).all(1)                             # zeroed across all leads
    assert (z.sum(1) <= int(0.25 * 250) + 1).all()


def test_augment_policy_wrapper_batch_and_single():
    a = augment.Augment("ecg", "mid", p=1.0)
    assert a.name == "ecg-mid"
    assert a(torch.randn(5, 12, 250)).shape == (5, 12, 250) and a(torch.randn(12, 250)).shape == (12, 250)
    b = augment.Augment("gen", "low", p=0.0)
    x = torch.randn(5, 12, 250)
    assert torch.equal(b(x), x)


def test_macro_auroc_skips_single_class_labels():
    Y = np.array([[1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]])
    P = np.array([[.9, .1, .2], [.2, .3, .8], [.8, .2, .1], [.1, .1, .9]])
    auc, n = metrics.macro_auroc(Y, P)
    assert n == 2 and auc == 1.0


def test_predict_windows_float32_and_raises_on_nonfinite():
    from robust_ecg import data
    X = np.random.randn(3, 12, 1000).astype(np.float32)
    X[1] *= 1e4                                   # extreme amplitudes: must stay finite in float32
    mean, std = np.zeros((12, 1), np.float32), np.ones((12, 1), np.float32)
    m = XResNet1d50(5).eval()
    dl = torch.utils.data.DataLoader(data.EvalWindows(X, [0, 1, 2], mean, std), batch_size=2)
    P = data.predict_windows(m, dl, torch.device("cpu"))
    assert P.shape == (3, 5) and np.isfinite(P).all()
    X[2, 0, 10] = np.nan
    dl = torch.utils.data.DataLoader(data.EvalWindows(X, [0, 1, 2], mean, std), batch_size=2)
    import pytest
    with pytest.raises(FloatingPointError, match=r"\[2\]"):
        data.predict_windows(m, dl, torch.device("cpu"))


def test_nt_xent_behaviour_and_simclr_net():
    import math
    from robust_ecg.models import SimCLRNet, nt_xent
    torch.manual_seed(0)
    z = torch.randn(64, 128)
    loss, acc = nt_xent(z, z.clone(), tau=0.1)                   # identical views: positive always retrieved
    assert acc == 1.0 and loss < 0.1
    loss, _ = nt_xent(torch.randn(64, 128), torch.randn(64, 128), tau=1e3)   # flat similarities
    assert abs(loss.item() - math.log(2 * 64 - 1)) < 1e-2
    net = SimCLRNet()
    assert net(torch.randn(4, 12, 250)).shape == (4, 128)
    assert count_params(net.encoder) == 886_400                  # same encoder that train.py loads
    XResNet1d50(71).encoder.load_state_dict(net.encoder.state_dict(), strict=True)


def test_pair_crops_are_two_views_of_one_record():
    from robust_ecg import data
    X = np.random.randn(5, 12, 1000).astype(np.float32)
    ds = data.PairCrops(X, np.array([1, 3]), np.zeros((12, 1), np.float32), np.ones((12, 1), np.float32))
    v1, v2 = ds[1]
    assert v1.shape == v2.shape == (12, 250) and len(ds) == 2
    s = np.lib.stride_tricks.sliding_window_view(X[3], 250, axis=1)          # every crop of record 3
    for v in (v1.numpy(), v2.numpy()):
        assert any(np.allclose(s[:, k], v) for k in range(s.shape[1]))
