"""Augmentation policies P_ecg and P_gen (research_question_v2.md §5, D17, D20, D23).

All transforms act on one normalised 2.5 s window, a (12, T) float tensor with ~unit per-lead variance,
after preprocessing (0.5–40 Hz band-pass, 100 Hz). They represent residual artefacts or variability
that survives a clinical front end. Random temporal cropping (no resizing) is shared by both policies
and happens in the dataset; resizing is excluded because it changes apparent heart rate (D23).

Strength levels (H3 grid) scale every transform's magnitude. The core factorial uses P_ecg at "mid".
"""
import math
import torch

FS = 100
STRENGTH = {"low": 0.5, "mid": 1.0, "high": 1.5}


# ---- P_ecg: physiologically motivated -------------------------------------------------------
def baseline_wander(x, s):
    """Sum of 3 sinusoids, 0.05–0.5 Hz, random phase, per-lead amplitude up to 0.5*s."""
    t = torch.arange(x.shape[1], dtype=x.dtype) / FS
    f = torch.empty(3).uniform_(0.05, 0.5)
    ph = torch.rand(3) * 2 * math.pi
    wave = torch.sin(2 * math.pi * f[:, None] * t[None] + ph[:, None]).sum(0) / 3
    amp = torch.rand(x.shape[0], 1) * 0.5 * s
    return x + amp * wave[None]


def emg_bursts(x, s):
    """Muscle artefact: Gaussian noise in 1–3 random bursts (0.2–1.0 s), independent per lead, sd up to 0.3*s."""
    out = x.clone()
    T = x.shape[1]
    for lead in range(x.shape[0]):
        for _ in range(int(torch.randint(1, 4, (1,)))):
            L = int(torch.randint(int(0.2 * FS), int(1.0 * FS) + 1, (1,)))
            a = int(torch.randint(0, max(1, T - L), (1,)))
            out[lead, a:a + L] += torch.randn(min(L, T - a)) * float(torch.rand(1)) * 0.3 * s
    return out


def per_lead_scaling(x, s):
    """Electrode-contact variation: each lead scaled independently by U(1 - 0.3s, 1 + 0.3s)."""
    return x * (1 + (torch.rand(x.shape[0], 1) * 2 - 1) * 0.3 * s)


# ---- P_gen: generic signal augmentations ----------------------------------------------------
def gaussian_noise(x, s):
    """Stationary white noise on all leads, sd up to 0.3*s (matched to emg_bursts peak sd)."""
    return x + torch.randn_like(x) * float(torch.rand(1)) * 0.3 * s


def global_scaling(x, s):
    return x * (1 + (float(torch.rand(1)) * 2 - 1) * 0.3 * s)


def time_mask(x, s):
    """Zero one random segment across all leads, up to 25%*s of the window."""
    T = x.shape[1]
    L = int(float(torch.rand(1)) * 0.25 * s * T)
    if L == 0:
        return x
    a = int(torch.randint(0, T - L + 1, (1,)))
    out = x.clone(); out[:, a:a + L] = 0
    return out


def time_warp(x, s, n_knots=4):
    """Smooth random time warp, local speed change up to ±10%*s, length preserved."""
    T = x.shape[1]
    speed = 1 + (torch.rand(n_knots) * 2 - 1) * 0.10 * s
    speed = torch.nn.functional.interpolate(speed[None, None], size=T, mode="linear", align_corners=True)[0, 0]
    pos = torch.cumsum(speed, 0); pos = (pos - pos[0]) / (pos[-1] - pos[0]) * (T - 1)
    i0 = pos.floor().long().clamp(0, T - 2); w = (pos - i0)[None]
    return x[:, i0] * (1 - w) + x[:, i0 + 1] * w


POLICIES = {
    "ecg": [baseline_wander, emg_bursts, per_lead_scaling],
    "gen": [gaussian_noise, global_scaling, time_mask, time_warp],
}


class Augment:
    """Apply each transform of a policy independently with probability p."""

    def __init__(self, policy, strength="mid", p=0.5):
        self.fns, self.s, self.p = POLICIES[policy], STRENGTH[strength], p
        self.name = f"{policy}-{strength}"

    def __call__(self, x):
        for fn in self.fns:
            if float(torch.rand(1)) < self.p:
                x = fn(x, self.s)
        return x
