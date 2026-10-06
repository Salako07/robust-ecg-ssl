"""Evaluate saved best checkpoints on the corruption suite (H2a; docs/corruption_protocol_v1.md).

  python scripts/eval_corruptions.py --cache <cache dir> --runs <work>/runs --budgets 1.0

For every finished run of the requested budgets that still has its ckpt_best.pt, the script
  1. recomputes the clean fold-10 predictions and compares them with the stored predictions.npz (guards against a
     wrong checkpoint, cache or code version);
  2. predicts fold 10 under each corruption x severity (9 held-out and 9 seen conditions);
  3. writes <run>/corruptions.json (metrics) and <run>/corruptions.npz (float16 predictions, for the bootstrap);
  4. rebuilds <runs>/corruptions_registry.csv from all corruptions.json files.
Resumable: a run that already has corruptions.json is skipped. No training happens here and SPH is not used.
"""
import argparse, glob, json, os, sys, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from robust_ecg import corrupt, data, labels, metrics, runinfo  # noqa: E402
from robust_ecg.models import XResNet1d50  # noqa: E402

CLEAN_TOL = 5e-3          # stored predictions are float16; GPU inference is not bitwise reproducible


def scores(Y, P, stmts):
    out = {"macro_auroc": metrics.macro_auroc(Y, P)[0]}
    Ye, Pe = labels.e3_truth_ptbxl(Y, stmts), labels.e3_scores(P, stmts)
    out["e3_macro_auroc"] = metrics.macro_auroc(Ye, Pe)[0]
    for k, v in zip(labels.E3_PRIMARY, metrics.per_label_auroc(Ye, Pe)):
        out[f"auroc_{k}"] = float(v)
    return out


def rebuild_registry(runs):
    rows = []
    for f in sorted(glob.glob(os.path.join(runs, "*", "corruptions.json"))):
        d = json.load(open(f))
        base = {k: d[k] for k in ("run_id", "arm", "budget", "seed")}
        rows.append({**base, "condition": "clean", "corruption": "clean", "severity": 0, "held_out": False, **d["clean"]})
        for tag, m in d["conditions"].items():
            rows.append({**base, "condition": tag, **m})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(runs, "corruptions_registry.csv"), index=False)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--budgets", type=float, nargs="+", default=[1.0])
    ap.add_argument("--arms", nargs="+", default=["S0", "S1", "C0", "C1"])
    ap.add_argument("--min-likelihood", type=float, default=0.0)
    ap.add_argument("--no-seen", action="store_true", help="held-out corruptions only")
    ap.add_argument("--no-preds", action="store_true", help="do not save corrupted predictions")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--allow-unfrozen", action="store_true", help="smoke tests on synthetic data only")
    a = ap.parse_args()
    if not corrupt.FROZEN and not a.allow_unfrozen:
        sys.exit("corrupt.FROZEN is False: the corruption parameters are still a proposal. Freeze them with a "
                 "decision-log entry (set FROZEN = True in the same commit) before evaluating real checkpoints.")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, meta = data.load_cache(a.cache, "ptbxl")
    scp = pd.read_csv(os.path.join(a.cache, "scp_statements.csv"), index_col=0)
    Y, stmts = labels.ptbxl_multihot(meta, scp, a.min_likelihood)
    mean, std = data.load_norm(a.cache)
    te_idx = np.flatnonzero(meta["strat_fold"].to_numpy() == 10)
    Yte = Y[te_idx]
    conds = corrupt.conditions(std, seen=not a.no_seen)

    def predict(model, c=None):
        dl = DataLoader(data.EvalWindows(X, te_idx, mean, std, corrupt=c), batch_size=64, num_workers=a.workers,
                        pin_memory=dev.type == "cuda")
        return data.predict_windows(model, dl, dev)

    todo = []
    for f in sorted(glob.glob(os.path.join(a.runs, "*", "done.json"))):
        d = json.load(open(f))
        rdir = os.path.dirname(f)
        if d["arm"] not in a.arms or not any(np.isclose(d["budget"], b) for b in a.budgets):
            continue
        if d["min_likelihood"] != a.min_likelihood or d["run_id"] != os.path.basename(rdir):
            continue
        if os.path.exists(os.path.join(rdir, "corruptions.json")):
            continue
        if not os.path.exists(os.path.join(rdir, "ckpt_best.pt")):
            sys.exit(f"{d['run_id']}: ckpt_best.pt is missing. Restore the full work archive (work.tar), not the "
                     f"results export, which holds no checkpoints.")
        todo.append((rdir, d))
    print(f"{len(todo)} runs to evaluate on {len(conds)} conditions, {len(te_idx)} test records, device={dev}", flush=True)

    model = XResNet1d50(len(stmts)).to(dev)
    for n, (rdir, d) in enumerate(todo, 1):
        t0 = time.time()
        model.load_state_dict(torch.load(os.path.join(rdir, "ckpt_best.pt"), map_location=dev))
        z = np.load(os.path.join(rdir, "predictions.npz"))
        if not np.array_equal(z["idx_test"], te_idx):
            sys.exit(f"{d['run_id']}: stored test indices differ from this cache")
        Pc = predict(model)
        diff = float(np.abs(Pc - z["P_test"].astype(np.float32)).max())
        if diff > CLEAN_TOL:
            sys.exit(f"{d['run_id']}: clean predictions differ from the stored ones by {diff:.2e} (> {CLEAN_TOL}). "
                     f"Wrong checkpoint, cache or code version; nothing was written for this run.")
        clean = scores(Yte, Pc, stmts)
        res = dict(run_id=d["run_id"], arm=d["arm"], budget=d["budget"], seed=d["seed"],
                   clean_max_abs_diff_vs_stored=diff, clean=clean, conditions={},
                   suite=dict(seed=corrupt.SUITE_SEED, frozen=corrupt.FROZEN, dropout_leads=corrupt.DROPOUT_LEADS,
                              reversal_fraction=corrupt.REVERSAL_FRACTION, step_mv=corrupt.STEP_MV,
                              step_window_s=corrupt.STEP_WINDOW_S, seen_strength=corrupt.SEEN_STRENGTH),
                   **runinfo.info())
        save = {}
        for c in conds:
            P = predict(model, c)
            m = scores(Yte, P, stmts)
            res["conditions"][c.tag] = dict(corruption=c.name, severity=c.severity, held_out=c.name in corrupt.HELD_OUT,
                                            **m, degradation=clean["macro_auroc"] - m["macro_auroc"],
                                            e3_degradation=clean["e3_macro_auroc"] - m["e3_macro_auroc"])
            save[f"P_{c.tag}"] = P.astype(np.float16)
        if not a.no_preds:
            np.savez_compressed(os.path.join(rdir, "corruptions.npz.tmp.npz"), idx_test=te_idx, **save)
            os.replace(os.path.join(rdir, "corruptions.npz.tmp.npz"), os.path.join(rdir, "corruptions.npz"))
        with open(os.path.join(rdir, "corruptions.json.tmp"), "w") as f:
            json.dump(res, f, indent=1, default=float)
        os.replace(os.path.join(rdir, "corruptions.json.tmp"), os.path.join(rdir, "corruptions.json"))   # written last
        ho = np.mean([m["degradation"] for m in res["conditions"].values() if m["held_out"]])
        print(f"[{n}/{len(todo)}] {d['run_id']}: clean {clean['macro_auroc']:.4f} (stored diff {diff:.1e}), "
              f"mean held-out degradation {ho:.4f}, {time.time() - t0:.0f}s", flush=True)

    print(f"corruptions_registry.csv: {rebuild_registry(a.runs)} rows")


if __name__ == "__main__":
    main()
