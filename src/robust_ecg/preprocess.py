"""Signal preprocessing shared by PTB-XL and SPH (docs/preprocessing_v1.md)."""
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

FS_IN = 500
FS_OUT = 100
SECONDS = 10
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
_SOS = butter(3, [0.5, 40.0], btype="bandpass", fs=FS_IN, output="sos")


def preprocess(x, fs=FS_IN):
    """x: (12, L) in mV at 500 Hz, L >= 10 s. Returns float32 (12, 1000) at 100 Hz."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != 12:
        raise ValueError(f"expected (12, L), got {x.shape}")
    if fs != FS_IN:
        raise ValueError(f"expected {FS_IN} Hz input, got {fs}")
    n = FS_IN * SECONDS
    if x.shape[1] < n:
        raise ValueError(f"record shorter than {SECONDS} s: {x.shape[1]} samples")
    x = x[:, :n]                                   # first 10 s
    x = sosfiltfilt(_SOS, x, axis=1)               # 0.5-40 Hz, zero-phase
    x = resample_poly(x, up=1, down=FS_IN // FS_OUT, axis=1)  # anti-aliased 500 -> 100 Hz
    return x.astype(np.float32)


def lead_stats(X):
    """Per-lead mean/std over records and time. X: (N, 12, T)."""
    X = np.asarray(X, dtype=np.float64)
    return X.mean(axis=(0, 2)), X.std(axis=(0, 2))


def amplitude_report(X, name):
    X = np.asarray(X)
    bad = ~np.isfinite(X).all(axis=(1, 2))
    flat = (np.nan_to_num(X).std(axis=2) < 1e-6).any(axis=1)
    a = np.abs(np.nan_to_num(X))
    med = np.median(a, axis=(0, 2))
    p99 = np.percentile(a.max(axis=2), 99, axis=0)
    print(f"[{name}] records={len(X)}  non-finite={int(bad.sum())}  with-flat-lead={int(flat.sum())}")
    print(f"[{name}] median |x| per lead (mV): " + " ".join(f"{m:.3f}" for m in med))
    print(f"[{name}] p99 of per-record max |x| (mV): " + " ".join(f"{p:.2f}" for p in p99))
    return dict(nonfinite=int(bad.sum()), flat=int(flat.sum()))
