"""Cache loading and PyTorch datasets. Caches: <root>/<name>_X.npy (N,12,1000) + <name>_meta.csv."""
import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CROP = 250          # 2.5 s at 100 Hz
STRIDE = 125        # 1.25 s


def load_cache(root, name, mmap=True):
    X = np.load(os.path.join(root, f"{name}_X.npy"), mmap_mode="r" if mmap else None)
    meta = pd.read_csv(os.path.join(root, f"{name}_meta.csv"))
    if len(X) != len(meta):
        raise ValueError(f"{name}: {len(X)} signals vs {len(meta)} metadata rows")
    return X, meta


def load_norm(root):
    with open(os.path.join(root, "norm_ptbxl_folds1-8.json")) as f:
        d = json.load(f)
    return np.asarray(d["mean"], np.float32)[:, None], np.asarray(d["std"], np.float32)[:, None]


class TrainCrops(Dataset):
    """Random 2.5 s crop per access; optional augmentation callable on a (12, CROP) tensor."""

    def __init__(self, X, Y, idx, mean, std, augment=None):
        self.X, self.Y, self.idx = X, Y, np.asarray(idx)
        self.mean, self.std, self.augment = mean, std, augment

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        r = self.idx[i]
        start = np.random.randint(0, self.X.shape[2] - CROP + 1)
        x = (np.asarray(self.X[r, :, start:start + CROP], np.float32) - self.mean) / self.std
        x = torch.from_numpy(x)
        if self.augment is not None:
            x = self.augment(x)
        y = torch.from_numpy(self.Y[r]) if self.Y is not None else torch.empty(0)
        return x, y


class EvalWindows(Dataset):
    """All sliding windows of each record: returns (n_windows, 12, CROP). Predictions are averaged."""

    def __init__(self, X, idx, mean, std, corrupt=None):
        self.X, self.idx, self.mean, self.std, self.corrupt = X, np.asarray(idx), mean, std, corrupt
        self.starts = list(range(0, X.shape[2] - CROP + 1, STRIDE))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x = np.asarray(self.X[self.idx[i]], np.float32)
        if self.corrupt is not None:          # corruptions act on the full 10 s record, in mV
            x = self.corrupt(x, record_index=int(self.idx[i]))
        x = (x - self.mean) / self.std
        return torch.from_numpy(np.stack([x[:, s:s + CROP] for s in self.starts]))


@torch.no_grad()
def predict_windows(model, loader, device):
    """Sigmoid outputs averaged over windows -> (N, S) numpy."""
    model.eval()
    out = []
    for xb in loader:                         # (B, W, 12, CROP)
        b, w = xb.shape[:2]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(xb.reshape(b * w, *xb.shape[2:]).to(device, non_blocking=True))
        out.append(torch.sigmoid(logits.float()).reshape(b, w, -1).mean(1).cpu())
    return torch.cat(out).numpy()
