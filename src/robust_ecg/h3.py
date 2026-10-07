"""H3: ECG-specific vs generic pretraining augmentation (docs/h3_protocol_v1.md, frozen 2026-10-07, D37).

Constants and small pure functions shared by scripts/linear_probe.py, scripts/run_h3.py and scripts/bootstrap_h3.py.
"""
import os
import numpy as np
import torch

from . import augment
from .models import ConcatPool, Encoder

POLICIES = ("ecg", "gen")
LEVELS = ("low", "mid", "high")           # augment.STRENGTH: 0.5, 1.0, 1.5
SEEDS = (0, 1, 2, 3, 4)
FT_POLICY, FT_LEVEL = "ecg", "mid"        # fine-tuning augmentation of BOTH arms (judgement 1)

# linear probe (judgement 3)
PROBE_STEPS, PROBE_LR, PROBE_WD, PROBE_SEED = 1000, 1e-2, 0.0, 0   # amended before freezing, see D37
TIE_MARGIN = 0.0005                        # fold-9 macro-AUROC differences below this are ties -> lower level


def ssl_run_id(policy, level, seed):
    return f"SSL_{policy}-{level}_s{seed}"


def ft_run_id(policy, level, seed):
    """Fine-tuned H3 run (C1 configuration, 100% budget), stored under runs_h3/."""
    return f"C1_b1_s{seed}_lik0_h3-{policy}-{level}"


def core_run_id(seed):
    """The core-grid run that IS the H3 run for (ecg, mid): same encoder, same fine-tuning."""
    return f"C1_b1_s{seed}_lik0"


def is_core(policy, level):
    return (policy, level) == (FT_POLICY, FT_LEVEL)


def select_level(scores):
    """scores: {level: fold-9 linear-probe macro-AUROC}. Highest wins; a level within TIE_MARGIN of the best
    counts as tied with it, and ties go to the lower level."""
    missing = [l for l in LEVELS if l not in scores]
    if missing:
        raise ValueError(f"probe scores missing for {missing}")
    best = max(scores[l] for l in LEVELS)
    return next(l for l in LEVELS if best - scores[l] < TIE_MARGIN)


def needed_runs(selection):
    """(policy, level, seed, in_primary) for every fine-tuned run of the protocol: the 3 x 2 grid at seed 0 and
    the two selected policy-levels at all seeds."""
    out = []
    for p in POLICIES:
        for l in LEVELS:
            for s in SEEDS:
                sel = selection[p] == l
                if s == 0 or sel:
                    out.append((p, l, s, sel))
    return out


class FrozenFeatures(torch.nn.Module):
    """Encoder + concatenated max/average pooling: the 512-d feature used by the linear probe."""

    def __init__(self):
        super().__init__()
        self.encoder, self.pool = Encoder(), ConcatPool()

    def forward(self, x):
        return self.pool(self.encoder(x))


def load_encoder(path, device):
    net = FrozenFeatures()
    sd = torch.load(path, map_location="cpu")
    net.encoder.load_state_dict(sd["encoder"] if "encoder" in sd else sd, strict=True)
    return net.to(device).eval()


def fit_probe(Ftr, Ytr, Fva, device="cpu"):
    """Standardise with training statistics, train one linear layer full-batch, return fold-9 probabilities."""
    Ftr, Fva = np.asarray(Ftr, np.float32), np.asarray(Fva, np.float32)
    mu, sd = Ftr.mean(0, keepdims=True), Ftr.std(0, keepdims=True) + 1e-6
    xtr = torch.from_numpy((Ftr - mu) / sd).to(device)
    xva = torch.from_numpy((Fva - mu) / sd).to(device)
    ytr = torch.from_numpy(np.asarray(Ytr, np.float32)).to(device)
    torch.manual_seed(PROBE_SEED)
    lin = torch.nn.Linear(xtr.shape[1], ytr.shape[1]).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=PROBE_LR, weight_decay=PROBE_WD)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(PROBE_STEPS):
        loss = lossf(lin(xtr), ytr)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        return torch.sigmoid(lin(xva)).cpu().numpy(), float(loss)


@torch.no_grad()
def view_distortion(X, idx, mean, std, policy, level, n_batches=20, bs=512, seed=0, device="cpu"):
    """Diagnostic only: mean squared change of a normalised 2.5 s training window under the policy (each transform
    with p = 0.5), relative to the window's mean square. Uses no labels and decides nothing."""
    from .data import CROP
    rng = np.random.default_rng(seed)
    aug = augment.Augment(policy, level)
    torch.manual_seed(seed)
    num = den = 0.0
    for _ in range(n_batches):
        rows = np.sort(rng.choice(idx, size=min(bs, len(idx)), replace=False))
        starts = rng.integers(0, X.shape[2] - CROP + 1, size=len(rows))
        x = np.stack([(np.asarray(X[r, :, s:s + CROP], np.float32) - mean) / std for r, s in zip(rows, starts)])
        x = torch.from_numpy(x).to(device)
        num += float((aug(x) - x).pow(2).mean()); den += float(x.pow(2).mean())
    return num / den


def iut_p(p_a, p_b):
    """Intersection-union test for a conjunction (judgement 5): the larger p-value."""
    return float(max(p_a, p_b))


def run_dir(root, policy, level, seed, core_dir="runs", h3_dir="runs_h3"):
    if is_core(policy, level):
        return os.path.join(root, core_dir, core_run_id(seed))
    return os.path.join(root, h3_dir, ft_run_id(policy, level, seed))
