"""Patient-level paired bootstrap for H3 (docs/h3_protocol_v1.md §6; inference as in D34).

  python scripts/bootstrap_h3.py --root results --cache <cache dir> --out results/analysis

<root> holds h3/manifest.json (written by scripts/run_h3.py) and the run folders it names (runs/, runs_h3/), each
with predictions.npz and corruptions.npz.

Two outcomes at the 100% budget, each as (P_ecg arm) - (P_gen arm), mean over seeds; negative = the encoder
pretrained with P_ecg is more robust:
  H3-corr  mean degradation over the nine held-out corruption conditions, fold 10, 71 statements
  H3-sph   degradation from fold 10 to SPH on the nine E3 labels
H3 is a conjunction, so the p-value it contributes to the Holm family is the larger of the two (intersection-union
test), and H3 counts as supported only if both estimates are negative. If results/analysis/
bootstrap_primary_with_h2a.csv exists, the final Holm adjustment over all four primary contrasts is printed.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from bootstrap_contrasts import FAMILY_SIZE, boot_label_aucs, holm, summarise  # noqa: E402
from robust_ecg import corrupt, h3  # noqa: E402
from robust_ecg.labels import E3_PRIMARY, e3_scores, e3_truth_ptbxl, ptbxl_multihot, sph_e3_labels  # noqa: E402

HELD = [f"{n}_s{s}" for n in corrupt.HELD_OUT for s in corrupt.SEVERITIES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--rng-seed", type=int, default=20261006)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--repro-tol", type=float, default=1e-3,
                    help="allowed difference to stored metrics (float16 storage); raise only for tiny synthetic sets")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    man = json.load(open(os.path.join(a.root, "h3", "manifest.json")))
    sel = man["selection"]
    prim = [r for r in man["runs"] if r["in_primary"]]
    seeds = sorted({r["seed"] for r in prim})
    by = {(r["policy"], r["seed"]): r for r in prim}
    assert all((p, s) in by for p in h3.POLICIES for s in seeds) and len(prim) == 2 * len(seeds), "incomplete primary cell"
    assert all(by[(p, s)]["level"] == sel[p] for p in h3.POLICIES for s in seeds)

    meta = pd.read_csv(os.path.join(a.cache, "ptbxl_meta.csv"))
    scp = pd.read_csv(os.path.join(a.cache, "scp_statements.csv"), index_col=0)
    sph = pd.read_csv(os.path.join(a.cache, "sph_meta.csv"))
    Yall, stmts = ptbxl_multihot(meta, scp, 0.0)
    Ysph = sph_e3_labels(sph)

    P10, Psph, idx, worst = {}, {}, None, 0.0
    for (p, s), r in by.items():
        d = os.path.join(a.root, r["run_dir"])
        z, c = np.load(os.path.join(d, "predictions.npz")), np.load(os.path.join(d, "corruptions.npz"))
        if idx is None:
            idx = z["idx_test"]
        assert np.array_equal(idx, z["idx_test"]) and np.array_equal(idx, c["idx_test"]) and len(z["P_sph_e3"]) == len(sph)
        for t in ["clean"] + HELD:
            P = (z["P_test"] if t == "clean" else c[f"P_{t}"]).astype(np.float32)
            P10[(p, s, t)] = np.concatenate([P, e3_scores(P, stmts)], 1)
        Psph[(p, s)] = z["P_sph_e3"].astype(np.float32)
    Y71 = Yall[idx]
    Y10 = np.concatenate([Y71, e3_truth_ptbxl(Y71, stmts)], 1)
    n71, ne3 = Y71.shape[1], len(E3_PRIMARY)

    rng = np.random.default_rng([a.rng_seed, 1000, 4])
    pt, bt = boot_label_aucs(Y10, P10, meta["patient_id"].to_numpy()[idx], a.n_boot, rng, a.chunk)
    ps, bs = boot_label_aucs(Ysph, Psph, sph["Patient_ID"].to_numpy(), a.n_boot, rng, max(20, a.chunk // 8))
    seed_idx = rng.integers(0, len(seeds), size=(a.n_boot, len(seeds)))
    c71, ce3 = list(range(n71)), list(range(n71, n71 + ne3))

    def m10(p, s, t, cols):                                     # (point, (B,)) macro over cols on fold 10
        return np.nanmean(pt[(p, s, t)][cols]), np.nanmean(bt[(p, s, t)][:, cols], 1)

    def msph(p, s):
        return np.nanmean(ps[(p, s)]), np.nanmean(bs[(p, s)], 1)

    def per_arm(fn):                                            # {policy: (point (S,), boot (B, S))}
        out = {}
        for p in h3.POLICIES:
            v = [fn(p, s) for s in seeds]
            out[p] = (np.array([x[0] for x in v]), np.stack([x[1] for x in v], 1))
        return out

    def corr_deg(tags, cols):
        def fn(p, s):
            cl = m10(p, s, "clean", cols)
            d = [(cl[0] - m10(p, s, t, cols)[0], cl[1] - m10(p, s, t, cols)[1]) for t in tags]
            return np.mean([x[0] for x in d]), np.mean([x[1] for x in d], 0)
        return per_arm(fn)

    def sph_deg(p, s):
        f, e = m10(p, s, "clean", ce3), msph(p, s)
        return f[0] - e[0], f[1] - e[1]

    outcomes = {"h3_corr_held_out_mean_degradation": corr_deg(HELD, c71),
                "h3_sph_e3_degradation": per_arm(sph_deg),
                "clean_macro_auroc": per_arm(lambda p, s: m10(p, s, "clean", c71)),
                "clean_e3_macro_auroc": per_arm(lambda p, s: m10(p, s, "clean", ce3)),
                "sph_e3_macro_auroc": per_arm(msph),
                "corr_held_out_mean_degradation_e3": corr_deg(HELD, ce3)}
    for n in corrupt.HELD_OUT:
        outcomes[f"corr_{n}_mean_degradation"] = corr_deg([f"{n}_s{s}" for s in corrupt.SEVERITIES], c71)

    # reproduce stored per-run metrics
    for (p, s), r in by.items():
        d = os.path.join(a.root, r["run_dir"])
        done, cj = json.load(open(os.path.join(d, "done.json"))), json.load(open(os.path.join(d, "corruptions.json")))
        i = seeds.index(s)
        worst = max(worst, abs(outcomes["clean_macro_auroc"][p][0][i] - done["test_macro_auroc"]),
                    abs(outcomes["sph_e3_macro_auroc"][p][0][i] - done["sph_e3_macro_auroc"]),
                    abs(outcomes["h3_sph_e3_degradation"][p][0][i] - done["e3_degradation"]),
                    abs(outcomes["h3_corr_held_out_mean_degradation"][p][0][i]
                        - np.mean([cj["conditions"][t]["degradation"] for t in HELD])))
    print(f"reproduction of stored metrics: max |difference| = {worst:.2e}")

    rows = []
    for name, d in outcomes.items():
        for p in h3.POLICIES:
            rows.append(dict(kind="arm", arm=f"P_{p}-{sel[p]}", outcome=name, resampling="patients", n_seeds=len(seeds),
                             per_seed=" ".join(f"{v:+.4f}" for v in d[p][0]), **summarise(d[p][0], d[p][1])))
        dp, db = d["ecg"][0] - d["gen"][0], d["ecg"][1] - d["gen"][1]
        base = dict(kind="contrast", arm="P_ecg - P_gen", outcome=name, n_seeds=len(seeds),
                    per_seed=" ".join(f"{v:+.4f}" for v in dp))
        rows.append({**base, "resampling": "patients", **summarise(dp, db)})
        rows.append({**base, "resampling": "seeds+patients", **summarise(dp, db, seed_idx)})
    res = pd.DataFrame(rows)
    res.round(6).to_csv(os.path.join(a.out, "bootstrap_h3.csv"), index=False)

    get = lambda o: res[(res.kind == "contrast") & (res.outcome == o) & (res.resampling == "patients")].iloc[0]
    ca, sp = get("h3_corr_held_out_mean_degradation"), get("h3_sph_e3_degradation")
    p_h3 = h3.iut_p(ca.p_two_sided, sp.p_two_sided)
    both_negative = bool(ca.estimate < 0 and sp.estimate < 0)
    summary = dict(selection=sel, seeds=seeds, n_boot=a.n_boot, rng_seed=a.rng_seed,
                   h3_corr=dict(estimate=ca.estimate, ci95=[ca.ci95_lo, ca.ci95_hi], p=ca.p_two_sided),
                   h3_sph=dict(estimate=sp.estimate, ci95=[sp.ci95_lo, sp.ci95_hi], p=sp.p_two_sided),
                   p_h3_iut=p_h3, both_estimates_negative=both_negative,
                   n_test_records=int(len(idx)), n_sph_records=int(len(sph)), reproduction_max_abs_diff=worst)

    prev = os.path.join(a.out, "bootstrap_primary_with_h2a.csv")
    pd.set_option("display.width", 220)
    if os.path.exists(prev):
        fam = pd.read_csv(prev)[["hypothesis", "contrast", "budget", "metric", "estimate", "ci95_lo", "ci95_hi",
                                 "p_two_sided"]]
        fam = pd.concat([fam[fam.hypothesis != "H3"], pd.DataFrame([dict(
            hypothesis="H3", contrast="P_ecg-P_gen", budget=1.0, metric="conjunction: corruptions and SPH (IUT)",
            estimate=np.nan, ci95_lo=np.nan, ci95_hi=np.nan, p_two_sided=p_h3)])], ignore_index=True)
        if len(fam) == FAMILY_SIZE:
            fam["p_holm"] = holm(fam.p_two_sided.to_numpy())
            fam["direction_as_hypothesised"] = [True if h != "H3" else both_negative for h in fam.hypothesis]
            fam.loc[fam.hypothesis == "H1", "direction_as_hypothesised"] = bool(fam[fam.hypothesis == "H1"].estimate.iloc[0] > 0)
            for h in ("H2a", "H2b"):
                fam.loc[fam.hypothesis == h, "direction_as_hypothesised"] = bool(fam[fam.hypothesis == h].estimate.iloc[0] < 0)
            fam.round(6).to_csv(os.path.join(a.out, "bootstrap_primary_final.csv"), index=False)
            summary["holm"] = {r.hypothesis: r.p_holm for r in fam.itertuples()}
            print("\nPrimary family, Holm-adjusted over all four contrasts:")
            print(fam.to_string(index=False))
    with open(os.path.join(a.out, "bootstrap_h3_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    assert worst < a.repro_tol, "metrics do not reproduce the stored run files"

    print(f"\nH3 (selected levels {sel}); negative = P_ecg encoder more robust:")
    print(res[res.kind == "contrast"][["outcome", "resampling", "estimate", "ci95_lo", "ci95_hi", "p_two_sided", "per_seed"]]
          .round(5).to_string(index=False))
    print(f"\nH3 conjunction p (larger of the two) = {p_h3:.4f}; both estimates negative: {both_negative}")

    # seed-0 grid, descriptive (no bootstrap)
    g = []
    for r in man["runs"]:
        if r["seed"] != 0:
            continue
        d = os.path.join(a.root, r["run_dir"])
        done, cj = json.load(open(os.path.join(d, "done.json"))), json.load(open(os.path.join(d, "corruptions.json")))
        g.append(dict(policy=r["policy"], level=r["level"], selected=r["in_primary"],
                      test_macro_auroc=done["test_macro_auroc"], sph_e3_macro_auroc=done["sph_e3_macro_auroc"],
                      e3_degradation=done["e3_degradation"],
                      held_out_mean_degradation=np.mean([cj["conditions"][t]["degradation"] for t in HELD]),
                      **{f"{n}_degradation": np.mean([cj["conditions"][f"{n}_s{s}"]["degradation"]
                                                      for s in corrupt.SEVERITIES]) for n in corrupt.HELD_OUT}))
    pd.DataFrame(g).round(5).to_csv(os.path.join(a.out, "h3_grid_seed0.csv"), index=False)
    print("\nSeed-0 grid (descriptive):"); print(pd.DataFrame(g).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
