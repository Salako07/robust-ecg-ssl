"""Patient-level paired bootstrap for H2a: degradation under held-out corruptions (corruption protocol v1 §4; D34, D35).

  python scripts/bootstrap_h2a.py --runs results/runs --cache <cache dir> --out results/analysis

Needs, for every run of the chosen budget, predictions.npz (clean) and corruptions.npz (scripts/eval_corruptions.py).

Statistic (frozen in D35): degradation = clean macro-AUROC - corrupted macro-AUROC on PTB-XL fold 10. The primary
H2a contrast is C1 - S1 in the MEAN degradation over the nine held-out conditions, 71 statements, 100% budget,
averaged over seeds. A negative value means C1 loses less than S1.

Inference is that of D34 (scripts/bootstrap_contrasts.py): patients are resampled, one resample is applied to the
clean and all corrupted predictions of both arms and all seeds, 10,000 resamples, percentile 95% interval,
two-sided p-value. If results/analysis/bootstrap_primary.csv exists, the H1 and H2b p-values are read from it and
the Bonferroni bound (x 4) and the Holm adjustment over the available contrasts are printed for all three.
"""
import argparse, json, os, sys
from multiprocessing import Pool
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from bootstrap_contrasts import FAMILY_SIZE, boot_label_aucs, holm, summarise  # noqa: E402
from robust_ecg import corrupt  # noqa: E402
from robust_ecg.labels import E3_PRIMARY, e3_scores, e3_truth_ptbxl, ptbxl_multihot  # noqa: E402

CONTRASTS = (("C1", "S1"), ("C0", "S0"), ("S1", "S0"), ("C1", "C0"))
TAGS = [f"{n}_s{s}" for n in corrupt.HELD_OUT + corrupt.SEEN for s in corrupt.SEVERITIES]


def scopes():
    """Named sets of conditions whose degradations are averaged with equal weight."""
    held = [f"{n}_s{s}" for n in corrupt.HELD_OUT for s in corrupt.SEVERITIES]
    seen = [f"{n}_s{s}" for n in corrupt.SEEN for s in corrupt.SEVERITIES]
    out = {"held_out_mean": held, "seen_mean": seen}
    for n in corrupt.HELD_OUT + corrupt.SEEN:
        out[f"{n}_mean"] = [f"{n}_s{s}" for s in corrupt.SEVERITIES]
    out.update({t: [t] for t in TAGS})
    return out


def work(args):
    """One worker handles a subset of conditions. All workers draw identical patient resamples (same RNG seed)."""
    a, tags, cell = args
    meta = pd.read_csv(os.path.join(a.cache, "ptbxl_meta.csv"))
    scp = pd.read_csv(os.path.join(a.cache, "scp_statements.csv"), index_col=0)
    Yall, stmts = ptbxl_multihot(meta, scp, 0.0)
    preds, idx = {}, None
    for rid, arm, seed in cell:
        z = np.load(os.path.join(a.runs, rid, "predictions.npz"))
        c = np.load(os.path.join(a.runs, rid, "corruptions.npz"))
        if idx is None:
            idx = z["idx_test"]
        assert np.array_equal(idx, z["idx_test"]) and np.array_equal(idx, c["idx_test"]), rid
        for t in tags:
            P = (z["P_test"] if t == "clean" else c[f"P_{t}"]).astype(np.float32)
            preds[(arm, seed, t)] = np.concatenate([P, e3_scores(P, stmts)], 1)
    Y71 = Yall[idx]
    Y = np.concatenate([Y71, e3_truth_ptbxl(Y71, stmts)], 1)
    rng = np.random.default_rng([a.rng_seed, int(round(a.budget * 1000)), 2])
    point, boot = boot_label_aucs(Y, preds, meta["patient_id"].to_numpy()[idx], a.n_boot, rng, a.chunk)
    n71 = Y71.shape[1]
    cols = {"macro_auroc": list(range(n71)), "e3_macro_auroc": list(range(n71, n71 + len(E3_PRIMARY)))}
    out = {}
    for k in preds:                                   # reduce to macro scores: (point, (B,)) per metric
        out[k] = {m: (float(np.nanmean(point[k][c])), np.nanmean(boot[k][:, c], 1).astype(np.float32))
                  for m, c in cols.items()}
    return out, int(len(idx)), int(pd.unique(meta["patient_id"].to_numpy()[idx]).size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--arms", nargs="+", default=["S0", "S1", "C0", "C1"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--rng-seed", type=int, default=20261006)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--repro-tol", type=float, default=1e-3,
                    help="allowed difference to corruptions.json (float16 storage); raise only for tiny synthetic sets")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    reg = pd.read_csv(os.path.join(a.runs, "registry.csv"))
    reg = reg[(reg.min_likelihood == 0.0) & np.isclose(reg.budget, a.budget) & reg.arm.isin(a.arms)]
    seeds = sorted(int(s) for s in reg.seed.unique())
    assert len(reg) == len(a.arms) * len(seeds), "incomplete cell: every arm needs every seed"
    cell = [(r.run_id, r.arm, int(r.seed)) for r in reg.itertuples()]
    missing = [rid for rid, _, _ in cell if not os.path.exists(os.path.join(a.runs, rid, "corruptions.npz"))]
    if missing:
        sys.exit(f"corruptions.npz missing for {len(missing)} runs, e.g. {missing[:3]}: run eval_corruptions.py first")

    parts = [["clean"] + list(p) for p in np.array_split(TAGS, max(1, a.workers))]
    jobs = [(a, p, cell) for p in parts]
    if a.workers > 1:
        with Pool(a.workers) as pool:
            res = pool.map(work, jobs, chunksize=1)
    else:
        res = [work(j) for j in jobs]
    S = {}
    for out, n_rec, n_pat in res:
        for k, v in out.items():
            if k in S:                                # "clean" is computed by every worker: must agree exactly
                assert all(np.array_equal(S[k][m][1], v[m][1], equal_nan=True) for m in v), "workers disagree"
            S[k] = v

    # reproduce the per-run metrics written by eval_corruptions.py (float16 predictions -> small differences)
    worst = 0.0
    for rid, arm, seed in cell:
        d = json.load(open(os.path.join(a.runs, rid, "corruptions.json")))
        for t in TAGS:
            for m in ("macro_auroc", "e3_macro_auroc"):
                worst = max(worst, abs(S[(arm, seed, t)][m][0] - d["conditions"][t][m]))
    print(f"reproduction of corruptions.json: max |difference| = {worst:.2e}")

    rng = np.random.default_rng([a.rng_seed, int(round(a.budget * 1000)), 3])
    seed_idx = rng.integers(0, len(seeds), size=(a.n_boot, len(seeds)))
    rows = []
    for m in ("macro_auroc", "e3_macro_auroc"):
        def deg(arm, tags):                           # (S,) point and (B, S) boot of the mean degradation
            p = np.array([np.mean([S[(arm, s, "clean")][m][0] - S[(arm, s, t)][m][0] for t in tags]) for s in seeds])
            b = np.stack([np.mean([S[(arm, s, "clean")][m][1] - S[(arm, s, t)][m][1] for t in tags], 0)
                          for s in seeds], 1)
            return p, b
        for scope, tags in scopes().items():
            d = {arm: deg(arm, tags) for arm in a.arms}
            for arm in a.arms:                        # absolute degradation per arm (descriptive)
                rows.append(dict(kind="arm", contrast=arm, budget=a.budget, scope=scope, metric=m, n_seeds=len(seeds),
                                 resampling="patients", **summarise(d[arm][0], d[arm][1])))
            for x, y in CONTRASTS:
                if x in d and y in d:
                    dp, db = d[x][0] - d[y][0], d[x][1] - d[y][1]
                    base = dict(kind="contrast", contrast=f"{x}-{y}", budget=a.budget, scope=scope, metric=m,
                                n_seeds=len(seeds))
                    rows.append({**base, "resampling": "patients", **summarise(dp, db)})
                    rows.append({**base, "resampling": "seeds+patients", **summarise(dp, db, seed_idx)})
    res = pd.DataFrame(rows)
    res.round(6).to_csv(os.path.join(a.out, "bootstrap_h2a.csv"), index=False)
    with open(os.path.join(a.out, "bootstrap_h2a_config.json"), "w") as f:
        json.dump(dict(n_boot=a.n_boot, rng_seed=a.rng_seed, budget=a.budget, seeds=seeds, arms=a.arms,
                       n_test_records=n_rec, n_test_patients=n_pat,
                       conditions=TAGS, suite_seed=corrupt.SUITE_SEED, protocol_frozen=corrupt.FROZEN,
                       reproduction_max_abs_diff=worst, family_size=FAMILY_SIZE), f, indent=1)
    assert worst < a.repro_tol, "macro-AUROCs do not reproduce corruptions.json"

    pd.set_option("display.width", 220)
    sel = res[(res.kind == "contrast") & (res.contrast == "C1-S1") & (res.metric == "macro_auroc")]
    h2a = sel[(sel.scope == "held_out_mean") & (sel.resampling == "patients")]
    if len(h2a) and np.isclose(a.budget, 1.0):
        prim = pd.DataFrame([dict(hypothesis="H2a", contrast="C1-S1", budget=a.budget,
                                  metric="held_out_mean_degradation", **{k: h2a.iloc[0][k] for k in
                                  ("estimate", "ci95_lo", "ci95_hi", "p_two_sided", "n_boot")})])
        prev = os.path.join(a.out, "bootstrap_primary.csv")
        if os.path.exists(prev):
            old = pd.read_csv(prev)
            old = old[old.hypothesis != "H2a"][["hypothesis", "contrast", "budget", "metric", "estimate", "ci95_lo",
                                                "ci95_hi", "p_two_sided", "n_boot"]]
            prim = pd.concat([old, prim], ignore_index=True)
        prim["p_bonferroni_m4"] = np.minimum(prim.p_two_sided * FAMILY_SIZE, 1.0)
        prim["p_holm_available_only"] = holm(prim.p_two_sided.to_numpy())
        prim.round(6).to_csv(os.path.join(a.out, "bootstrap_primary_with_h2a.csv"), index=False)
        print("\nPrimary contrasts available (Holm is final only when all four exist):")
        print(prim.to_string(index=False))
    print("\nC1 - S1 degradation by scope (71 statements, patients resampled; negative = C1 more robust):")
    print(sel[sel.resampling == "patients"][["scope", "estimate", "ci95_lo", "ci95_hi", "p_two_sided"]]
          .round(5).to_string(index=False))


if __name__ == "__main__":
    main()
