"""
E3 feasibility, Gates 1-2: shared-label support in PTB-XL (train folds, patient-level
nested label budgets) and SPH (whole dataset = zero-adaptation test set).

Usage:
  python scripts/e3_label_support.py --ptbxl data/ptbxl/ptbxl_database.csv \
                             --scp-statements data/ptbxl/scp_statements.csv \
                             --sph data/sph/metadata.csv \
                             --seeds 0 1 2 --min-pos 10 --out e3_label_support.csv

Mapping source: PTB-XL scp_statements.csv "AHA code" column (dataset authors' own
crosswalk) matched against the SPH AHA primary statements (Liu et al., Sci Data 2022,
Table 2). Every mapping below is a documented DECISION; edit here, nowhere else.
"""
import argparse, ast
import numpy as np
import pandas as pd

# label -> (PTB-XL SCP codes, SPH AHA primary codes, role)
SHARED = {
    # primary: code-level match via PTB-XL's AHA crosswalk
    "AF":        ({"AFIB"},  {"50"},  "primary"),
    "PR_PROL":   ({"1AVB", "LPR"}, {"82"}, "primary"),   # SPH 82 = prolonged PR; PTB-XL maps LPR->82, 1AVB uncoded
    "CRBBB":     ({"CRBBB"}, {"106"}, "primary"),
    "IRBBB":     ({"IRBBB"}, {"105"}, "primary"),
    "CLBBB":     ({"CLBBB"}, {"104"}, "primary"),
    "LAFB":      ({"LAFB"},  {"101"}, "primary"),
    "LVH":       ({"LVH"},   {"142"}, "primary"),
    "IMI":       ({"IMI"},   {"161"}, "primary"),
    "ASMI":      ({"ASMI"},  {"165"}, "primary"),
    # exploratory: coarser concepts, need clinical sign-off
    "MI_ANY":    ({"IMI", "ASMI", "AMI", "ALMI", "LMI", "PMI", "ILMI", "IPLMI", "IPMI"},
                  {"160", "161", "165", "166"}, "exploratory"),
    "TWAVE_ABN": ({"TAB_", "INVT"}, {"147"}, "exploratory"),
    "NORM":      ({"NORM"},  {"1"},   "exploratory"),     # semantics differ, see notes
    # secondary: heart-rate-defined, near-ceiling
    "SBRAD":     ({"SBRAD"}, {"22"},  "secondary"),
    "STACH":     ({"STACH"}, {"21"},  "secondary"),
    "SARRH":     ({"SARRH"}, {"23"},  "secondary"),
    # window-dependent: label may refer to beats outside a 10 s SPH crop
    "PVC":       ({"PVC"},   {"60"},  "window_dependent"),
    "PAC":       ({"PAC"},   {"30"},  "window_dependent"),
}
BUDGETS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
MODIFIER_MIN = 300  # AHA modifiers are 3xx; primary statements are < 300


def ptbxl_labels(df, min_likelihood, diag):
    """Threshold applies to diagnostic statements only; form/rhythm statements carry
    likelihood 0 by PTB-XL convention and count whenever listed."""
    codes = df["scp_codes"].apply(ast.literal_eval)
    out = pd.DataFrame(index=df.index)
    for lab, (ptb, _, _) in SHARED.items():
        out[lab] = codes.apply(lambda d: int(any(
            c in d and (c not in diag or d[c] >= min_likelihood) for c in ptb)))
    return out


def sph_primary_codes(aha_field):
    """'60+310;50+346;147' -> {'60','50','147'}; order of primary/modifier is not assumed."""
    prim = set()
    for stmt in str(aha_field).split(";"):
        parts = [p.strip() for p in stmt.split("+") if p.strip().isdigit()]
        prim |= {p for p in parts if int(p) < MODIFIER_MIN}
    return prim


def sph_labels(df):
    prim = df["AHA_Code"].apply(sph_primary_codes)
    return pd.DataFrame({lab: prim.apply(lambda s: int(bool(s & sph)))
                         for lab, (_, sph, _) in SHARED.items()}, index=df.index)


def nested_patient_budgets(patients, seed):
    """One random patient order per seed; budget b = first ceil(b*N) patients (nested)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.asarray(sorted(patients)))
    return {b: set(order[: int(np.ceil(b * len(order)))]) for b in BUDGETS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptbxl", required=True)
    ap.add_argument("--sph")
    ap.add_argument("--scp-statements", help="PTB-XL scp_statements.csv; required if --min-likelihood > 0")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--min-likelihood", type=float, default=0.0,
                    help="likelihood threshold for PTB-XL diagnostic statements only; 0 keeps every listed statement")
    ap.add_argument("--min-pos", type=int, default=10,
                    help="flag (not drop) label/budget cells below this many positives")
    ap.add_argument("--out", default="e3_label_support.csv")
    a = ap.parse_args()

    if a.min_likelihood > 0 and not a.scp_statements:
        ap.error("--scp-statements is required when --min-likelihood > 0")
    diag = set()
    if a.scp_statements:
        scp = pd.read_csv(a.scp_statements, index_col=0)
        diag = set(scp.index[scp["diagnostic"] == 1])
    pd.set_option("display.width", 250)

    ptb = pd.read_csv(a.ptbxl)
    y = ptbxl_labels(ptb, a.min_likelihood, diag)

    # sanity check: likelihood values attached to shared statements
    lik = ptb["scp_codes"].apply(ast.literal_eval)
    allcodes = set().union(*(s for s, _, _ in SHARED.values()))
    vals = pd.Series([v for d in lik for k, v in d.items() if k in allcodes])
    print("PTB-XL likelihood values on shared statements:\n", vals.value_counts().sort_index().to_string())
    if "validated_by_human" in ptb.columns:
        print("\nvalidated_by_human share:", round(ptb["validated_by_human"].astype(bool).mean(), 3))
        print("by strat_fold:", ptb.groupby("strat_fold")["validated_by_human"].mean().round(3).to_dict())

    rows = []
    train = ptb["strat_fold"] <= 8
    for split, mask in [("val_fold9", ptb["strat_fold"] == 9), ("test_fold10", ptb["strat_fold"] == 10)]:
        for lab in SHARED:
            rows.append(dict(dataset="PTB-XL", split=split, budget=None, seed=None, label=lab,
                             role=SHARED[lab][2], records=int(mask.sum()),
                             pos_records=int(y.loc[mask, lab].sum()),
                             pos_patients=int(ptb.loc[mask & (y[lab] == 1), "patient_id"].nunique())))
    train_patients = set(ptb.loc[train, "patient_id"])
    for seed in a.seeds:
        for b, pats in nested_patient_budgets(train_patients, seed).items():
            m = train & ptb["patient_id"].isin(pats)
            for lab in SHARED:
                rows.append(dict(dataset="PTB-XL", split="train", budget=b, seed=seed, label=lab,
                                 role=SHARED[lab][2], records=int(m.sum()),
                                 pos_records=int(y.loc[m, lab].sum()),
                                 pos_patients=int(ptb.loc[m & (y[lab] == 1), "patient_id"].nunique())))

    if a.sph:
        sph = pd.read_csv(a.sph)
        ys = sph_labels(sph)
        for lab in SHARED:
            rows.append(dict(dataset="SPH", split="external_test", budget=None, seed=None, label=lab,
                             role=SHARED[lab][2], records=len(sph), pos_records=int(ys[lab].sum()),
                             pos_patients=int(sph.loc[ys[lab] == 1, "Patient_ID"].nunique())))

    res = pd.DataFrame(rows)
    res["below_min_pos"] = res["pos_records"] < a.min_pos
    res["min_likelihood_diag"] = a.min_likelihood
    res.to_csv(a.out, index=False)

    tr = res[res.split == "train"].pivot_table(index=["role", "label"], columns="budget",
                                               values="pos_records", aggfunc=["min", "max"])
    print("\nPTB-XL train positives (min/max across seeds) by budget:\n", tr.to_string())
    ext = res[res.dataset == "SPH"].set_index("label")[["role", "pos_records", "pos_patients"]]
    if len(ext):
        print("\nSPH positives:\n", ext.to_string())
    print(f"\nCells below {a.min_pos} positives are flagged, not dropped. Written: {a.out}")


if __name__ == "__main__":
    main()
