"""Linear probe of a frozen SSL encoder on PTB-XL (H3 protocol v1 §4).

  python scripts/linear_probe.py --cache <cache> --encoder <ssl run>/encoder.pt --out <file>.json

Feature of a recording: concatenated max- and average-pooled encoder output (512-d), averaged over the seven
evaluation windows. One linear layer over the 71 statements is trained on folds 1-8 (full batch, AdamW, lr 1e-2,
no weight decay, 1,000 steps, no early stopping, seed 0) and scored by macro-AUROC on fold 9.
Fold 10 and SPH are never read.
"""
import argparse, hashlib, json, os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from robust_ecg import data, h3, labels, metrics, runinfo  # noqa: E402


@torch.no_grad()
def features(net, X, idx, mean, std, dev, workers):
    dl = DataLoader(data.EvalWindows(X, idx, mean, std), batch_size=128, num_workers=workers,
                    pin_memory=dev.type == "cuda")
    out = []
    for xb in dl:                                            # (B, W, 12, CROP)
        b, w = xb.shape[:2]
        f = net(xb.reshape(b * w, *xb.shape[2:]).to(dev, non_blocking=True).float())
        out.append(f.reshape(b, w, -1).mean(1).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print(f"{a.out}: already done, skipping"); return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, meta = data.load_cache(a.cache, "ptbxl")
    scp = pd.read_csv(os.path.join(a.cache, "scp_statements.csv"), index_col=0)
    Y, _ = labels.ptbxl_multihot(meta, scp, 0.0)
    mean, std = data.load_norm(a.cache)
    fold = meta["strat_fold"].to_numpy()
    tr, va = np.flatnonzero(fold <= 8), np.flatnonzero(fold == 9)

    net = h3.load_encoder(a.encoder, dev)
    Ftr, Fva = features(net, X, tr, mean, std, dev, a.workers), features(net, X, va, mean, std, dev, a.workers)
    P, final_loss = h3.fit_probe(Ftr, Y[tr], Fva, dev)
    auc, n_lab = metrics.macro_auroc(Y[va], P)
    res = dict(encoder=os.path.abspath(a.encoder), encoder_sha256=hashlib.sha256(open(a.encoder, "rb").read()).hexdigest(),
               val_macro_auroc=auc, val_n_labels=n_lab, n_train=int(len(tr)), n_val=int(len(va)),
               feature_dim=int(Ftr.shape[1]), final_train_loss=final_loss, steps=h3.PROBE_STEPS, lr=h3.PROBE_LR,
               weight_decay=h3.PROBE_WD, seed=h3.PROBE_SEED, **runinfo.info())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(a.out + ".tmp", a.out)
    print(json.dumps({k: res[k] for k in ("val_macro_auroc", "val_n_labels", "final_train_loss")}))


if __name__ == "__main__":
    main()
