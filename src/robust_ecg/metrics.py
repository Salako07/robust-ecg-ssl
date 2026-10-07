"""Evaluation metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score


def per_label_auroc(Y, P):
    """AUROC per column; NaN where a column has only one class."""
    out = np.full(Y.shape[1], np.nan)
    for j in range(Y.shape[1]):
        if 0 < Y[:, j].sum() < len(Y):
            out[j] = roc_auc_score(Y[:, j], P[:, j])
    return out


def macro_auroc(Y, P):
    """Mean AUROC over labels with at least one positive and one negative (benchmark convention)."""
    a = per_label_auroc(Y, P)
    return float(np.nanmean(a)), int(np.isfinite(a).sum())
