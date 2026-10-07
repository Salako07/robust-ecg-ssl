"""Augmentation policies P_ecg and P_gen (research_question_v2.md §5, D17, D20, D23), batched on the GPU.

Every transform takes a batch x of shape (B, 12, T): normalised 2.5 s windows (~unit per-lead variance) after
preprocessing (0.5–40 Hz band-pass, 100 Hz). Random parameters are drawn independently per sample (and per lead
where stated). Random temporal cropping without resizing is shared by both policies and happens in the dataset;
resizing is excluded because it changes apparent heart rate (D23).

Strength levels (H3 grid) scale every transform's magnitude. The core factorial uses P_ecg at "mid".
"""
import math
import torch

FS = 100
STRENGTH = {"low": 0.5, "mid": 1.0, "high": 1.5}


def _u(shape, like, lo=0.0, hi=1.0):
    return torch.rand(shape, device=like.device, dtype=like.dtype) * (hi - lo) + lo


# ---- P_ecg: physiologically motivated -------------------------------------------------------
def baseline_wander(x, s):
    """Sum of 3 sinusoids, 0.05–0.5 Hz, random phase per sample; amplitude up to 0.5*s, drawn per lead."""
    B, C, T = x.shape
    t = torch.arange(T, device=x.device, dtype=x.dtype) / FS
    f = _u((B, 3, 1), x, 0.05, 0.5)
    ph = _u((B, 3, 1), x, 0, 2 * math.pi)
    wave = torch.sin(2 * math.pi * f * t + ph).mean(1, keepdim=True)          # (B, 1, T)
    return x + _u((B, C, 1), x, 0, 0.5 * s) * wave


def emg_bursts(x, s, max_bursts=3):
    """Muscle artefact: Gaussian noise confined to 1–3 bursts of 0.2–1.0 s, independent per lead,
    noise sd up to 0.3*s drawn per lead."""
    B, C, T = x.shape
    n = torch.randint(1, max_bursts + 1, (B, C, 1), device=x.device)
    L = torch.randint(int(0.2 * FS), int(1.0 * FS) + 1, (B, C, max_bursts), device=x.device)
    a = (torch.rand(B, C, max_bursts, device=x.device) * (T - L).clamp(min=1)).long()
    t = torch.arange(T, device=x.device)
    active = torch.arange(max_bursts, device=x.device) < n                      # (B, C, K)
    inside = (t >= a[..., None]) & (t < (a + L)[..., None]) & active[..., None]  # (B, C, K, T)
    mask = inside.any(2).to(x.dtype)
    return x + torch.randn_like(x) * mask * _u((B, C, 1), x, 0, 0.3 * s)


def per_lead_scaling(x, s):
    """Electrode-contact variation: each lead scaled independently by U(1 - 0.3s, 1 + 0.3s)."""
    B, C, _ = x.shape
    return x * _u((B, C, 1), x, 1 - 0.3 * s, 1 + 0.3 * s)


# ---- P_gen: generic signal augmentations ----------------------------------------------------
def gaussian_noise(x, s):
    """Stationary white noise on all leads, sd up to 0.3*s per sample (matched to emg_bursts' peak sd)."""
    return x + torch.randn_like(x) * _u((x.shape[0], 1, 1), x, 0, 0.3 * s)


def global_scaling(x, s):
    return x * _u((x.shape[0], 1, 1), x, 1 - 0.3 * s, 1 + 0.3 * s)


def time_mask(x, s):
    """Zero one random segment across all leads, up to 25%*s of the window, per sample."""
    B, _, T = x.shape
    L = (_u((B, 1, 1), x) * 0.25 * s * T).long()
    a = (_u((B, 1, 1), x) * (T - L + 1)).long()
    t = torch.arange(T, device=x.device)
    return x * ~((t >= a) & (t < a + L))


def time_warp(x, s, n_knots=4):
    """Smooth random time warp per sample, local speed change up to ±10%*s, length preserved."""
    B, C, T = x.shape
    speed = _u((B, 1, n_knots), x, 1 - 0.10 * s, 1 + 0.10 * s)
    speed = torch.nn.functional.interpolate(speed, size=T, mode="linear", align_corners=True)[:, 0]  # (B, T)
    pos = torch.cumsum(speed, 1)
    pos = (pos - pos[:, :1]) / (pos[:, -1:] - pos[:, :1]) * (T - 1)
    i0 = pos.floor().long().clamp(0, T - 2)
    w = (pos - i0).unsqueeze(1)
    g0 = x.gather(2, i0.unsqueeze(1).expand(B, C, T))
    g1 = x.gather(2, (i0 + 1).unsqueeze(1).expand(B, C, T))
    return g0 * (1 - w) + g1 * w


POLICIES = {
    "ecg": [baseline_wander, emg_bursts, per_lead_scaling],
    "gen": [gaussian_noise, global_scaling, time_mask, time_warp],
}


class Augment:
    """Apply each transform of a policy independently per sample with probability p. Input (B, 12, T) or (12, T)."""

    def __init__(self, policy, strength="mid", p=0.5):
        self.fns, self.s, self.p = POLICIES[policy], STRENGTH[strength], p
        self.name = f"{policy}-{strength}"

    def __call__(self, x):
        single = x.dim() == 2
        if single:
            x = x.unsqueeze(0)
        for fn in self.fns:
            apply = torch.rand(x.shape[0], 1, 1, device=x.device) < self.p
            x = torch.where(apply, fn(x, self.s), x)
        return x[0] if single else x
