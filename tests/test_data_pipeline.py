import ast, json, os, subprocess, sys
import numpy as np
import pandas as pd
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from robust_ecg import data, labels, preprocess, splits  # noqa: E402

t = np.arange(5000) / 500.0


def sine(f, amp=1.0):
    return np.tile(amp * np.sin(2 * np.pi * f * t), (12, 1))


# --- preprocessing -------------------------------------------------------------------------
def test_shape_dtype_and_first_10s():
    x = np.random.randn(12, 7500)
    y = preprocess.preprocess(x)
    assert y.shape == (12, 1000) and y.dtype == np.float32


def test_passband_kept_stopband_removed():
    keep = preprocess.preprocess(sine(10))[:, 100:-100]
    assert 0.9 < keep.std() * np.sqrt(2) < 1.05          # 10 Hz amplitude ~preserved
    for f in (0.1, 50, 60):                               # drift and powerline removed
        assert preprocess.preprocess(sine(f))[:, 100:-100].std() < 0.05


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        preprocess.preprocess(np.zeros((5000, 12)))
    with pytest.raises(ValueError):
        preprocess.preprocess(np.zeros((12, 4000)))


# --- budgets -------------------------------------------------------------------------------
def make_meta(n=400, seed=0):
    rng = np.random.default_rng(seed)
    pid = rng.integers(0, 300, n)
    return pd.DataFrame(dict(ecg_id=np.arange(1, n + 1), patient_id=pid, strat_fold=pid % 10 + 1,
                             scp_codes=["{'NORM': 100.0}"] * n))


def test_budgets_nested_patient_level_train_only():
    m = make_meta()
    prev = set()
    for b in (0.01, 0.05, 0.10, 0.25, 0.5, 1.0):
        idx = splits.budget_indices(m, b, seed=3)
        assert m.loc[idx, "strat_fold"].between(1, 8).all()
        pats = set(m.loc[idx, "patient_id"])
        assert prev <= pats
        # all records of a selected patient are included
        assert set(np.flatnonzero(m["patient_id"].isin(pats) & m["strat_fold"].between(1, 8))) == set(idx)
        prev = pats
    assert len(splits.budget_indices(m, 1.0, 0)) == m["strat_fold"].between(1, 8).sum()
    assert not np.array_equal(splits.budget_indices(m, 0.25, 0), splits.budget_indices(m, 0.25, 1))


# --- labels --------------------------------------------------------------------------------
SCP = pd.DataFrame({"diagnostic": [1.0, 1.0, np.nan, np.nan, 1.0]},
                   index=["IMI", "1AVB", "LPR", "AFIB", "NORM"])


def test_multihot_threshold_diagnostic_only():
    m = pd.DataFrame({"scp_codes": ["{'IMI': 15.0, 'LPR': 0.0}", "{'AFIB': 0.0, 'IMI': 100.0}"]})
    Y0, s = labels.ptbxl_multihot(m, SCP, 0)
    Y50, _ = labels.ptbxl_multihot(m, SCP, 50)
    assert s == list(SCP.index)
    assert Y0[0, s.index("IMI")] == 1 and Y50[0, s.index("IMI")] == 0     # diagnostic thresholded
    assert Y50[0, s.index("LPR")] == 1 and Y50[1, s.index("AFIB")] == 1   # form/rhythm untouched


def test_e3_scores_max_and_sph_codes():
    stmts = ["1AVB", "LPR", "AFIB", "CRBBB", "IRBBB", "CLBBB", "LAFB", "LVH", "IMI", "ASMI"]
    P = np.zeros((1, len(stmts))); P[0, 0], P[0, 1] = 0.2, 0.7
    assert labels.e3_scores(P, stmts)[0, list(labels.E3_PRIMARY).index("PR_PROL")] == 0.7
    sm = pd.DataFrame({"AHA_Code": ["346+50;147", "106", "1"]})
    Y = labels.sph_e3_labels(sm)
    k = list(labels.E3_PRIMARY)
    assert Y[0, k.index("AF")] == 1 and Y[1, k.index("CRBBB")] == 1 and Y[2].sum() == 0


# --- datasets ------------------------------------------------------------------------------
def test_train_and_eval_datasets():
    X = np.random.randn(5, 12, 1000).astype(np.float32)
    Y = np.eye(5, 3, dtype=np.float32)
    mean, std = X.mean((0, 2))[:, None], X.std((0, 2))[:, None]
    tr = data.TrainCrops(X, Y, [0, 2, 4], mean, std, augment=lambda x: x * 2)
    x, y = tr[1]
    assert x.shape == (12, data.CROP) and torch.equal(y, torch.from_numpy(Y[2]))
    ev = data.EvalWindows(X, [1, 3], mean, std)
    w = ev[0]
    assert w.shape == (7, 12, data.CROP)
    np.testing.assert_allclose(w[0].numpy(), ((X[1] - mean) / std)[:, :250], rtol=1e-6)
    np.testing.assert_allclose(w[-1].numpy(), ((X[1] - mean) / std)[:, 750:], rtol=1e-6)

    class Const(torch.nn.Module):
        def forward(self, x):
            return x.mean(dim=(1, 2), keepdim=False)[:, None].repeat(1, 3)
    P = data.predict_windows(Const(), torch.utils.data.DataLoader(ev, batch_size=2), torch.device("cpu"))
    assert P.shape == (2, 3) and np.all((P > 0) & (P < 1))


# --- end-to-end PTB-XL preparation on synthetic WFDB records -------------------------------
def test_prepare_ptbxl_end_to_end(tmp_path):
    import wfdb
    d = tmp_path / "ptbxl"; (d / "records500" / "00000").mkdir(parents=True)
    rows = []
    for i in range(1, 13):
        name = f"{i:05d}_hr"
        sig = (np.random.randn(5000, 12) * 0.1 + sine(8, 1.0).T).astype(np.float64)
        wfdb.wrsamp(name, fs=500, units=["mV"] * 12, sig_name=preprocess.LEADS, p_signal=sig,
                    fmt=["16"] * 12, adc_gain=[1000.0] * 12, baseline=[0] * 12,
                    write_dir=str(d / "records500" / "00000"))
        rows.append(dict(ecg_id=i, patient_id=float(i), strat_fold=(i % 10) + 1,
                         scp_codes="{'NORM': 100.0}", filename_hr=f"records500/00000/{name}"))
    pd.DataFrame(rows).to_csv(d / "ptbxl_database.csv", index=False)
    SCP.to_csv(d / "scp_statements.csv")
    out = tmp_path / "cache"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "prepare_ptbxl.py"),
                        "--ptbxl-dir", str(d), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    X, meta = data.load_cache(str(out), "ptbxl")
    assert X.shape == (12, 12, 1000) and len(meta) == 12
    mean, std = data.load_norm(str(out))
    assert mean.shape == (12, 1) and np.all(std > 0)
    assert json.load(open(out / "ptbxl_statements.json")) == list(SCP.index)
