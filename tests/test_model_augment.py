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


def test_each_transform_preserves_shape_and_changes_signal():
    torch.manual_seed(0)
    x = torch.randn(12, 250)
    for pol in augment.POLICIES.values():
        for fn in pol:
            y = fn(x.clone(), 1.0)
            assert y.shape == x.shape and torch.isfinite(y).all()
    # per-lead scaling differs across leads, global scaling does not
    r = augment.per_lead_scaling(torch.ones(12, 250), 1.0)[:, 0]
    assert r.std() > 0
    g = augment.global_scaling(torch.ones(12, 250), 1.0)[:, 0]
    assert torch.allclose(g, g[0].expand_as(g))


def test_strength_scales_magnitude():
    torch.manual_seed(1)
    x = torch.zeros(12, 250)
    lo = np.mean([augment.baseline_wander(x, 0.5).abs().mean().item() for _ in range(200)])
    hi = np.mean([augment.baseline_wander(x, 1.5).abs().mean().item() for _ in range(200)])
    assert 2.0 < hi / lo < 4.0                          # ~3x for a 3x strength ratio


def test_time_warp_is_small_and_length_preserving():
    t = torch.linspace(0, 1, 250).repeat(12, 1)
    w = augment.time_warp(t, 1.0)
    assert w.shape == t.shape and (w[:, 1:] >= w[:, :-1] - 1e-6).all()   # monotone
    assert (w - t).abs().max() < 0.1


def test_augment_policy_wrapper():
    a = augment.Augment("ecg", "mid", p=1.0)
    assert a.name == "ecg-mid" and a(torch.randn(12, 250)).shape == (12, 250)


def test_macro_auroc_skips_single_class_labels():
    Y = np.array([[1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]])
    P = np.array([[.9, .1, .2], [.2, .3, .8], [.8, .2, .1], [.1, .1, .9]])
    auc, n = metrics.macro_auroc(Y, P)
    assert n == 2 and auc == 1.0
