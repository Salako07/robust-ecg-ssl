"""Patient-level paired bootstrap for between-arm contrasts (RQ v2 §6; E3 protocol "Statistical inference").

  python scripts/bootstrap_contrasts.py --runs results/runs --cache <cache dir> --out results/analysis

The cache dir must hold ptbxl_meta.csv, scp_statements.csv and sph_meta.csv (the files listed in
results/cache_manifest.json). Predictions are read from <runs>/<run_id>/predictions.npz.

Method (fixed in this script before it was first run on real predictions; see the decision-log entry):
- Resampling unit: the PATIENT. A resample draws n patients with replacement from the n test patients; a patient
  drawn k times contributes all of his or her recordings with weight k. PTB-XL fold 10 and SPH are resampled
  independently of each other.
- Pairing: one resample is applied to both arms of a contrast and to all seeds, so arm differences are paired
  on the same patients.
- Statistic: mean over seeds of the per-seed difference (seed k of arm A is paired with seed k of arm B, which
  share the labelled subset). Its bootstrap distribution describes test-set sampling error for these trained
  models. Seed-to-seed variation is reported separately (scripts/analyze_grid.py), as pre-specified.
- Secondary interval ("seeds+patients"): seeds are also resampled with replacement in every resample.
- Macro-AUROC in a resample averages over the labels that have at least one positive and one negative in that
  resample (same convention as robust_ecg.metrics.macro_auroc); both arms use the same label set.
- Interval: percentile, 95%. p-value: two-sided, 2 * min(P(d* <= 0), P(d* >= 0)), with the +1 correction.
- Multiplicity: the family has four pre-specified primary contrasts (H1, H2a, H2b, H3). Holm's procedure needs
  all four p-values. Until H2a and H3 exist, the script prints the Bonferroni bound (p * 4), which is valid
  whatever the missing p-values turn out to be, and Holm-adjusted values over the contrasts available.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from robust_ecg.labels import (E3_GROUPS, E3_PRIMARY, e3_scores, e3_truth_ptbxl, ptbxl_multihot,  # noqa: E402
                               sph_e3_labels)

ARMS = ("S0", "S1", "C0", "C1")
CONTRASTS = (("C1", "S1"), ("C0", "S0"), ("S1", "S0"), ("C1", "C0"))
FAMILY_SIZE = 4  # H1, H2a, H2b, H3 (RQ v2 §6)


# ---------------------------------------------------------------------------------------------------------
class WeightedAUC:
    """AUROC of one score column under integer record weights, for many resamples at once.

    Ties get the usual 0.5 credit. The sort and the tie groups are computed once; a resample then costs one
    gather and one cumulative sum over the records, plus work proportional to the number of positives."""

    def __init__(self, y, s):
        order = np.argsort(s, kind="stable")
        ss, yy = s[order], y[order] > 0
        starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
        ends = np.r_[starts[1:], len(ss)]
        grp = np.searchsorted(starts, np.arange(len(ss)), side="right") - 1
        self.order = order
        self.pos = np.flatnonzero(yy)                      # sorted positions of the positives
        self.gs, self.ge = starts[grp[self.pos]], ends[grp[self.pos]]   # tie group of each positive
        self.kb = np.searchsorted(self.pos, self.gs, side="left")      # positives below the group
        self.ke = np.searchsorted(self.pos, self.ge, side="left")      # positives below the group's end

    def __call__(self, W):
        """W: (B, n) record weights in original record order -> (B,) AUROC, NaN if a class is absent."""
        w = np.take(W, self.order, axis=1)
        C = np.zeros((w.shape[0], w.shape[1] + 1), np.float32)       # exact: integer sums far below 2**24
        np.cumsum(w, axis=1, out=C[:, 1:])
        wp = w[:, self.pos].astype(np.float64)
        CP = np.zeros((w.shape[0], len(self.pos) + 1))
        np.cumsum(wp, axis=1, out=CP[:, 1:])
        neg_below = C[:, self.gs] - CP[:, self.kb]
        neg_tied = (C[:, self.ge] - C[:, self.gs]) - (CP[:, self.ke] - CP[:, self.kb])
        u = (wp * (neg_below + 0.5 * neg_tied)).sum(1)
        tp = CP[:, -1]
        tn = C[:, -1].astype(np.float64) - tp
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where((tp > 0) & (tn > 0), u / (tp * tn), np.nan)


def patient_weights(patients, B, rng, chunk):
    """Yield (b, n_records) float32 weight matrices: patient multiplicities expanded to records."""
    codes, _ = pd.factorize(patients)
    n_pat = codes.max() + 1
    done = 0
    while done < B:
        b = min(chunk, B - done)
        draws = rng.integers(0, n_pat, size=(b, n_pat))
        mult = np.zeros((b, n_pat), np.float32)
        np.add.at(mult, (np.repeat(np.arange(b), n_pat), draws.ravel()), 1.0)
        yield mult[:, codes]
        done += b


def boot_label_aucs(Y, preds, patients, B, rng, chunk):
    """preds: {(arm, seed): (n, L) scores}. Returns point {(arm, seed): (L,)} and boot {(arm, seed): (B, L)}."""
    L = Y.shape[1]
    keep = [j for j in range(L) if 0 < Y[:, j].sum() < len(Y)]
    calc = {k: [WeightedAUC(Y[:, j], P[:, j].astype(np.float64)) for j in keep] for k, P in preds.items()}
    ones = np.ones((1, len(Y)), np.float32)
    point, boot = {}, {k: np.full((B, L), np.nan, np.float32) for k in preds}
    for k in preds:
        point[k] = np.full(L, np.nan)
        point[k][keep] = [c(ones)[0] for c in calc[k]]
    done = 0
    for W in patient_weights(patients, B, rng, chunk):
        for k in preds:
            for jj, j in enumerate(keep):
                boot[k][done:done + len(W), j] = calc[k][jj](W)
        done += len(W)
    return point, boot


def summarise(point, boot, seed_idx=None):
    """point: (S,) per-seed differences; boot: (B, S). Returns dict with estimate, CI and p-value."""
    if seed_idx is not None:  # resample seeds too
        d = np.take_along_axis(boot, seed_idx, axis=1).mean(1)
    else:
        d = boot.mean(1)
    d = d[np.isfinite(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min(((d <= 0).sum() + 1) / (len(d) + 1), ((d >= 0).sum() + 1) / (len(d) + 1))
    return dict(estimate=float(np.mean(point)), ci95_lo=float(lo), ci95_hi=float(hi), p_two_sided=float(min(p, 1.0)),
                boot_sd=float(d.std(ddof=1)), n_boot=int(len(d)))


def holm(pvals):
    order = np.argsort(pvals)
    adj, run = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(pvals) - rank) * pvals[i])
        adj[i] = min(run, 1.0)
    return adj


# ---------------------------------------------------------------------------------------------------------
def load(a):
    reg = pd.read_csv(os.path.join(a.runs, "registry.csv"))
    reg = reg[reg.min_likelihood == 0.0]
    seeds = sorted(reg.seed.unique())
    S = len(seeds)

    meta = pd.read_csv(os.path.join(a.cache, "ptbxl_meta.csv"))
    scp = pd.read_csv(os.path.join(a.cache, "scp_statements.csv"), index_col=0)
    sph = pd.read_csv(os.path.join(a.cache, "sph_meta.csv"))
    Yall, stmts = ptbxl_multihot(meta, scp, 0.0)
    Ysph = sph_e3_labels(sph)
    e3_names = list(E3_PRIMARY)
    groups = {"all": list(range(9)), **{g: [e3_names.index(l) for l in ls] for g, ls in E3_GROUPS.items()}}

    return reg, seeds, S, meta, sph, Yall, stmts, Ysph, e3_names, groups


def run_budget(args):
    a, b = args
    t0 = time.time()
    reg, seeds, S, meta, sph, Yall, stmts, Ysph, e3_names, groups = load(a)
    rows, label_rows, checks = [], [], []
    cell = reg[np.isclose(reg.budget, b)]
    P71, Pe3, Psph, idx = {}, {}, {}, None
    for _, r in cell.iterrows():
        z = np.load(os.path.join(a.runs, r.run_id, "predictions.npz"))
        if idx is None:
            idx = z["idx_test"]
        assert np.array_equal(idx, z["idx_test"]) and len(z["P_sph_e3"]) == len(sph)
        k = (r.arm, int(r.seed))
        P71[k] = z["P_test"].astype(np.float32)
        Pe3[k] = e3_scores(P71[k], stmts)
        Psph[k] = z["P_sph_e3"].astype(np.float32)
    assert len(P71) == len(ARMS) * S, f"budget {b}: incomplete cell"
    Y71 = Yall[idx]
    Ye3 = e3_truth_ptbxl(Y71, stmts)
    pat_t = meta["patient_id"].to_numpy()[idx]
    pat_s = sph["Patient_ID"].to_numpy()

    rng = np.random.default_rng([a.rng_seed, int(round(b * 1000))])
    # fold 10: one set of patient resamples for the 71-statement and the E3 scores (same records)
    both = {k: np.concatenate([P71[k], Pe3[k]], 1) for k in P71}
    pt, bt = boot_label_aucs(np.concatenate([Y71, Ye3], 1), both, pat_t, a.n_boot, rng, a.chunk)
    ps, bs = boot_label_aucs(Ysph, Psph, pat_s, a.n_boot, rng, max(20, a.chunk // 8))
    seed_idx = rng.integers(0, S, size=(a.n_boot, S))
    n71 = Y71.shape[1]

    def metric(point, boot, cols):
        """macro over cols -> point (S,) per arm, boot (B, S) per arm"""
        return ({arm: np.array([np.nanmean(point[(arm, s)][cols]) for s in seeds]) for arm in ARMS},
                {arm: np.stack([np.nanmean(boot[(arm, s)][:, cols], 1) for s in seeds], 1) for arm in ARMS})

    M = {"test_macro_auroc": metric(pt, bt, list(range(n71)))}
    for g, cols in groups.items():
        tp, tb = metric(pt, bt, [n71 + c for c in cols])
        sp, sb = metric(ps, bs, cols)
        suf = "" if g == "all" else f"_{g}"
        M[f"test_e3_macro_auroc{suf}"] = (tp, tb)
        M[f"sph_e3_macro_auroc{suf}"] = (sp, sb)
        M[f"e3_degradation{suf}"] = ({x: tp[x] - sp[x] for x in ARMS}, {x: tb[x] - sb[x] for x in ARMS})

    # reproduce the registry's point values from predictions + rebuilt labels (pipeline check)
    for arm in ARMS:
        for i, s in enumerate(seeds):
            r = cell[(cell.arm == arm) & (cell.seed == s)].iloc[0]
            for m in ("test_macro_auroc", "test_e3_macro_auroc", "sph_e3_macro_auroc", "e3_degradation"):
                checks.append(abs(M[m][0][arm][i] - r[m]))

    for x, y in CONTRASTS:
        for m, (pm, bm) in M.items():
            dp, db = pm[x] - pm[y], bm[x] - bm[y]
            base = dict(contrast=f"{x}-{y}", budget=b, metric=m, n_seeds=S)
            rows.append({**base, "resampling": "patients", **summarise(dp, db)})
            rows.append({**base, "resampling": "seeds+patients", **summarise(dp, db, seed_idx)})
        for j, lab in enumerate(e3_names):
            tp = {arm: np.array([pt[(arm, s)][n71 + j] for s in seeds]) for arm in (x, y)}
            sp = {arm: np.array([ps[(arm, s)][j] for s in seeds]) for arm in (x, y)}
            tb = {arm: np.stack([bt[(arm, s)][:, n71 + j] for s in seeds], 1) for arm in (x, y)}
            sb = {arm: np.stack([bs[(arm, s)][:, j] for s in seeds], 1) for arm in (x, y)}
            for m, dp, db in (("test_auroc", tp[x] - tp[y], tb[x] - tb[y]),
                              ("sph_auroc", sp[x] - sp[y], sb[x] - sb[y]),
                              ("degradation", (tp[x] - sp[x]) - (tp[y] - sp[y]),
                               (tb[x] - sb[x]) - (tb[y] - sb[y]))):
                label_rows.append(dict(contrast=f"{x}-{y}", budget=b, label=lab, metric=m, n_seeds=S,
                                       resampling="patients", **summarise(dp, db)))
    print(f"budget {b}: done ({time.time() - t0:.0f}s)", flush=True)

    info = dict(n_test_records=int(len(idx)), n_test_patients=int(pd.unique(pat_t).size),
                n_sph_records=int(len(sph)), n_sph_patients=int(pd.unique(pat_s).size), seeds=[int(s) for s in seeds])
    return rows, label_rows, checks, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--rng-seed", type=int, default=20261006)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.25, 0.50, 1.0])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    from multiprocessing import Pool
    jobs = [(a, b) for b in a.budgets]
    if a.workers > 1:
        with Pool(a.workers) as pool:
            out = pool.map(run_budget, jobs, chunksize=1)
    else:
        out = [run_budget(j) for j in jobs]
    rows = [r for o in out for r in o[0]]
    label_rows = [r for o in out for r in o[1]]
    checks = [c for o in out for c in o[2]]
    info = out[0][3]
    # Predictions are stored as float16 and the registry was computed from float32 scores, so small differences
    # are expected (largest at the 1% budget, where near-constant outputs tie after rounding).
    worst = max(checks)
    print(f"registry reproduction: max |difference| over {len(checks)} values = {worst:.2e}")

    res = pd.DataFrame(rows)
    res.round(6).to_csv(os.path.join(a.out, "bootstrap_contrasts.csv"), index=False)
    pd.DataFrame(label_rows).round(6).to_csv(os.path.join(a.out, "bootstrap_contrasts_per_label.csv"), index=False)

    prim = res[(res.contrast == "C1-S1") & (res.resampling == "patients") &
               (((np.isclose(res.budget, 0.05)) & (res.metric == "test_macro_auroc")) |
                ((np.isclose(res.budget, 1.0)) & (res.metric == "e3_degradation")))].copy()
    prim.insert(0, "hypothesis", ["H1" if m == "test_macro_auroc" else "H2b" for m in prim.metric])
    prim["p_bonferroni_m4"] = np.minimum(prim.p_two_sided * FAMILY_SIZE, 1.0)
    prim["p_holm_available_only"] = holm(prim.p_two_sided.to_numpy())
    prim.round(6).to_csv(os.path.join(a.out, "bootstrap_primary.csv"), index=False)
    with open(os.path.join(a.out, "bootstrap_config.json"), "w") as f:
        json.dump(dict(n_boot=a.n_boot, rng_seed=a.rng_seed, budgets=a.budgets, **info,
                       registry_reproduction_max_abs_diff=float(worst), family_size=FAMILY_SIZE), f, indent=1)
    assert worst < 1e-3, "point estimates do not reproduce the registry: labels or row order are wrong"
    pd.set_option("display.width", 220)
    print("\nPrimary contrasts available (patient bootstrap of the seed-averaged difference):")
    print(prim.drop(columns=["resampling", "n_seeds", "boot_sd"]).to_string(index=False))


if __name__ == "__main__":
    main()
