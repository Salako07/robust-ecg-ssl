"""Build the SPH cache (docs/preprocessing_v1.md). Run sph_dedup.py first.

  python scripts/prepare_sph.py --sph-dir /content/sph --exclude results/sph_exclude.txt \
                                --out /content/drive/MyDrive/robust-ecg-ssl/cache

--sph-dir must contain metadata.csv and records/*.h5. Writes sph_X.npy and sph_meta.csv.
"""
import argparse, os, sys, time
import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from robust_ecg.preprocess import amplitude_report, preprocess  # noqa: E402
from robust_ecg.sph import resolve_all  # noqa: E402


def read_h5(path):
    with h5py.File(path, "r") as f:
        keys = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
        if len(keys) != 1:
            raise ValueError(f"{path}: expected one dataset, found {keys}")
        x = np.asarray(f[keys[0]][()], dtype=np.float64)
    if x.shape[0] != 12 and x.shape[1] == 12:
        raise ValueError(f"{path}: array is (L, 12); expected (12, L) per the SPH paper")
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sph-dir", required=True)
    ap.add_argument("--exclude", required=True, help="results/sph_exclude.txt from sph_dedup.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    meta = pd.read_csv(os.path.join(a.sph_dir, "metadata.csv"))
    if not os.path.isfile(a.exclude):
        raise FileNotFoundError(f"{a.exclude} missing: run sph_dedup.py first")
    with open(a.exclude) as f:
        excl = {l.strip() for l in f if l.strip()}
    n0 = len(meta)
    meta = meta[~meta["ECG_ID"].isin(excl)].sort_values("ECG_ID").reset_index(drop=True)
    print(f"SPH: {n0} records, {n0 - len(meta)} excluded by dedup, {len(meta)} kept")

    paths = resolve_all(os.path.join(a.sph_dir, "records"), list(meta["ECG_ID"]))
    X = np.empty((len(meta), 12, 1000), np.float32)
    t0 = time.time()
    for i, e in enumerate(meta["ECG_ID"]):
        X[i] = preprocess(read_h5(paths[e]))
        if (i + 1) % 2000 == 0:
            print(f"{i + 1}/{len(meta)} ({time.time() - t0:.0f}s)", flush=True)
    amplitude_report(X, "SPH")
    np.save(os.path.join(a.out, "sph_X.npy"), X)
    meta.to_csv(os.path.join(a.out, "sph_meta.csv"), index=False)
    print(f"saved {X.shape} to {a.out}")


if __name__ == "__main__":
    main()
