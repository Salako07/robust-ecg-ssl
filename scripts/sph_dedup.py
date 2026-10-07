"""
E3 Gate 3: find exact-duplicate ECG signals in SPH before it is used as a test set.

Duplicates are identified by hashing the signal array itself (shape + dtype + bytes),
not by metadata: identical metadata does not guarantee identical signals, and vice versa.

Policy (see docs/decisions/log.md):
  - within a group of identical signals, keep the first ECG_ID (sorted) and exclude the rest;
  - if members of a group carry different AHA *primary* statements, exclude the whole group
    (label conflict). Modifier-only differences (e.g. 50 vs 50+346) are not conflicts.

Usage:
  python scripts/sph_dedup.py --sph-dir data/sph/records --meta data/sph/metadata.csv \
                              --out-dir results
Writes results/sph_duplicate_groups.csv and results/sph_exclude.txt (one ECG_ID per line).
"""
import argparse, hashlib, os, sys
import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from robust_ecg.sph import resolve_all  # noqa: E402

MODIFIER_MIN = 300  # AHA modifiers are 3xx; primary statements are < 300


def primary_codes(aha_field):
    """'50+346;147' -> frozenset({'50','147'}); primary/modifier order not assumed."""
    return frozenset(p for stmt in str(aha_field).split(";") for p in stmt.split("+")
                     if p.strip().isdigit() and int(p) < MODIFIER_MIN)


def signal_hash(path):
    with h5py.File(path, "r") as f:
        keys = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
        if len(keys) != 1:
            raise ValueError(f"{path}: expected one dataset, found {keys}")
        x = np.ascontiguousarray(f[keys[0]][()])
    h = hashlib.sha256()
    h.update(str(x.shape).encode()); h.update(str(x.dtype).encode()); h.update(x.tobytes())
    return h.hexdigest(), x.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sph-dir", required=True, help="folder containing the SPH .h5 files")
    ap.add_argument("--meta", required=True, help="SPH metadata.csv")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()

    meta = pd.read_csv(a.meta)
    paths = resolve_all(a.sph_dir, list(meta["ECG_ID"]))   # fails loudly if any file is missing
    rows = []
    for i, (ecg_id, p) in enumerate(paths.items()):
        h, shape = signal_hash(p)
        rows.append((ecg_id, h, shape[0], shape[1]))
        if (i + 1) % 5000 == 0:
            print(f"hashed {i + 1}/{len(paths)}", flush=True)

    df = pd.DataFrame(rows, columns=["ECG_ID", "sha256", "n_leads", "n_samples"]).merge(meta, on="ECG_ID")
    bad_leads = (df["n_leads"] != 12).sum()
    if bad_leads:
        print(f"WARNING: {bad_leads} records are not 12 x L (check array orientation)")

    dup = df[df.duplicated("sha256", keep=False)].sort_values(["sha256", "ECG_ID"]).copy()
    dup["primary"] = dup["AHA_Code"].apply(primary_codes)
    dup["label_conflict"] = dup.groupby("sha256")["primary"].transform(lambda s: s.nunique() > 1)
    exclude = []
    for _, g in dup.groupby("sha256"):
        exclude += list(g["ECG_ID"]) if g["label_conflict"].iloc[0] else list(g["ECG_ID"].iloc[1:])
    dup["excluded"] = dup["ECG_ID"].isin(exclude)
    dup["primary"] = dup["primary"].apply(lambda f: ";".join(sorted(f)))

    os.makedirs(a.out_dir, exist_ok=True)
    dup.to_csv(os.path.join(a.out_dir, "sph_duplicate_groups.csv"), index=False)
    with open(os.path.join(a.out_dir, "sph_exclude.txt"), "w") as f:
        f.write("\n".join(sorted(exclude)) + ("\n" if exclude else ""))

    n_groups = dup["sha256"].nunique()
    print(f"records hashed: {len(df)} | duplicate groups: {n_groups} | records in groups: {len(dup)}")
    print(f"groups spanning >1 patient: {(dup.groupby('sha256')['Patient_ID'].nunique() > 1).sum()}")
    print(f"groups with label conflict: {int(dup.groupby('sha256')['label_conflict'].first().sum())}")
    print(f"excluded: {len(exclude)} -> {len(df) - len(exclude)} SPH records remain")


if __name__ == "__main__":
    main()
