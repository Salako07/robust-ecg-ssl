"""Seed-level summary of the core grid from results/runs/registry.csv (descriptive; D33).

  python scripts/analyze_grid.py --registry results/runs/registry.csv --out results/analysis

Writes arm_means.csv (mean and SD over seeds per arm x budget x metric), contrasts_seed_paired.csv (paired
by seed: mean difference, t-based 95% interval over seeds, seeds with a positive difference, per-seed values) and
per_label_100pct.csv. Runs with the same seed share the budget subset, so differences are paired by seed.

The seed-level interval reflects variation between training runs only. The pre-registered inference (RQ v2 §6:
paired bootstrap over test patients, Holm over the primary contrasts) is a separate analysis that needs the labels.
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["test_macro_auroc", "test_e3_macro_auroc", "sph_e3_macro_auroc", "e3_degradation"]
CONTRASTS = [("C1", "S1", "SSL effect, augmentation matched (primary)"),
             ("C0", "S0", "SSL effect, no augmentation (as usually reported)"),
             ("S1", "S0", "augmentation effect, random init"),
             ("C1", "C0", "augmentation effect, SSL init")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    r = pd.read_csv(a.registry)
    assert not r.run_id.duplicated().any()
    n = r.groupby(["arm", "budget"]).seed.nunique()
    assert n.nunique() == 1, f"unequal seeds per cell:\n{n}"

    g = r.groupby(["budget", "arm"])[METRICS].agg(["mean", "std"]).round(5)
    g.columns = [f"{m}_{s}" for m, s in g.columns]
    g.reset_index().to_csv(os.path.join(a.out, "arm_means.csv"), index=False)

    rows = []
    for x, y, what in CONTRASTS:
        for b in sorted(r.budget.unique()):
            for m in METRICS:
                dx = r[(r.arm == x) & (r.budget == b)].set_index("seed")[m]
                dy = r[(r.arm == y) & (r.budget == b)].set_index("seed")[m]
                d = (dx - dy).sort_index().to_numpy()
                h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                rows.append(dict(contrast=f"{x}-{y}", meaning=what, budget=b, metric=m, n_seeds=len(d),
                                 mean_diff=d.mean(), ci95_lo=d.mean() - h, ci95_hi=d.mean() + h,
                                 seeds_positive=int((d > 0).sum()), per_seed=" ".join(f"{v:+.4f}" for v in d)))
    c = pd.DataFrame(rows)
    c.round(5).to_csv(os.path.join(a.out, "contrasts_seed_paired.csv"), index=False)

    lab = [k for k in r.columns if k.startswith(("test_auroc_", "sph_auroc_"))]
    p = r[r.budget == 1.0].groupby("arm")[lab].mean().T.round(4)
    p.to_csv(os.path.join(a.out, "per_label_100pct.csv"))

    pd.set_option("display.width", 200)
    prim = c[(c.contrast == "C1-S1") & (((c.budget == 0.05) & (c.metric == "test_macro_auroc")) |
                                        ((c.budget == 1.0) & (c.metric == "e3_degradation")))]
    print("Primary contrasts available from this grid (seed-level, descriptive):")
    print(prim[["contrast", "budget", "metric", "mean_diff", "ci95_lo", "ci95_hi", "seeds_positive", "per_seed"]]
          .round(4).to_string(index=False))
    print("\nPer-label AUROC at 100% (mean over seeds):"); print(p.to_string())


if __name__ == "__main__":
    main()
