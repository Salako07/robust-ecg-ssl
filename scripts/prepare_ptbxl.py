"""Build the PTB-XL cache (docs/preprocessing_v1.md).

  python scripts/prepare_ptbxl.py --ptbxl-dir /content/ptbxl --out /content/drive/MyDrive/robust-ecg-ssl/cache

Writes ptbxl_X.npy (N,12,1000 float32), ptbxl_meta.csv, ptbxl_statements.json,
norm_ptbxl_folds1-8.json. Resumable: finished chunks are kept in <out>/_ptbxl_chunks.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import wfdb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from robust_ecg.preprocess import LEADS, amplitude_report, lead_stats, preprocess  # noqa: E402

CHUNK = 1000


def read_record(ptbxl_dir, rel):
    sig, f = wfdb.rdsamp(os.path.join(ptbxl_dir, rel))
    names = [n.upper() for n in f["sig_name"]]
    if names != LEADS:
        raise ValueError(f"{rel}: lead order {names}")
    if f["fs"] != 500 or any(u.lower() != "mv" for u in f["units"]):
        raise ValueError(f"{rel}: fs={f['fs']} units={f['units']}")
    return preprocess(sig.T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptbxl-dir", required=True, help="folder with ptbxl_database.csv and records500/")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    chunk_dir = os.path.join(a.out, "_ptbxl_chunks"); os.makedirs(chunk_dir, exist_ok=True)

    meta = pd.read_csv(os.path.join(a.ptbxl_dir, "ptbxl_database.csv")).sort_values("ecg_id").reset_index(drop=True)
    scp = pd.read_csv(os.path.join(a.ptbxl_dir, "scp_statements.csv"), index_col=0)
    n_chunks = int(np.ceil(len(meta) / CHUNK))
    t0 = time.time()
    for c in range(n_chunks):
        p = os.path.join(chunk_dir, f"{c:03d}.npy")
        if os.path.exists(p):
            continue
        rows = meta.iloc[c * CHUNK:(c + 1) * CHUNK]
        np.save(p, np.stack([read_record(a.ptbxl_dir, r) for r in rows["filename_hr"]]))
        print(f"chunk {c + 1}/{n_chunks} done ({time.time() - t0:.0f}s)", flush=True)

    X = np.concatenate([np.load(os.path.join(chunk_dir, f"{c:03d}.npy")) for c in range(n_chunks)])
    amplitude_report(X, "PTB-XL")
    np.save(os.path.join(a.out, "ptbxl_X.npy"), X)
    meta.to_csv(os.path.join(a.out, "ptbxl_meta.csv"), index=False)
    with open(os.path.join(a.out, "ptbxl_statements.json"), "w") as f:
        json.dump(list(scp.index), f)
    scp.to_csv(os.path.join(a.out, "scp_statements.csv"))

    train = meta["strat_fold"].between(1, 8).to_numpy()
    mean, std = lead_stats(X[train])
    with open(os.path.join(a.out, "norm_ptbxl_folds1-8.json"), "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist(), "n_records": int(train.sum())}, f, indent=1)
    print(f"saved {X.shape} to {a.out}; per-lead std (mV): " + " ".join(f"{s:.3f}" for s in std))


if __name__ == "__main__":
    main()
