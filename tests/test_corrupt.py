"""Checks for the corruption suite (CPU, synthetic signals)."""
import os, sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from robust_ecg import augment, corrupt, preprocess  # noqa: E402

STD = np.full((12, 1), 0.2, np.float32)


def leads_from_electrodes(v):
    """(9, T) electrode potentials -> (12, T) leads, written out lead by lead (independent of the matrix)."""
    ra, la, ll = v[0], v[1], v[2]
    wct = (ra + la + ll) / 3
    return np.stack([la - ra, ll - ra, ll - la, ra - (la + ll) / 2, la - (ra + ll) / 2, ll - (ra + la) / 2]
                    + [v[3 + i] - wct for i in range(6)])


def record(seed=0):
    rng = np.random.default_rng(seed)
    return leads_from_electrodes(rng.standard_normal((9, corrupt.T))).astype(np.float32)


def test_electrode_matrix_matches_lead_definitions_and_einthoven():
    v = np.random.default_rng(1).standard_normal((9, 50))
    x = corrupt.LEAD_FROM_ELECTRODE @ v
    assert np.allclose(x, leads_from_electrodes(v))
    assert np.allclose(x[0] + x[2], x[1])                    # I + III = II
    assert np.allclose(x[3] + x[4] + x[5], 0)                # aVR + aVL + aVF = 0


def test_la_ra_reversal_equals_swapping_the_electrodes():
    v = np.random.default_rng(2).standard_normal((9, corrupt.T))
    swapped = v.copy(); swapped[[0, 1]] = v[[1, 0]]
    x = leads_from_electrodes(v).astype(np.float32)
    assert np.allclose(corrupt.reverse_la_ra(x), leads_from_electrodes(swapped), atol=1e-6)
    assert np.allclose(corrupt.reverse_la_ra(corrupt.reverse_la_ra(x)), x)          # an involution
    assert np.array_equal(corrupt.reverse_la_ra(x)[5:], x[5:])                       # aVF, V1-V6 untouched


def test_reversal_fraction_per_severity():
    x = record()
    for sev, frac in corrupt.REVERSAL_FRACTION.items():
        c = corrupt.Corruptor("limb_reversal", sev)
        changed = np.array([not np.array_equal(c(x, record_index=i), x) for i in range(6000)])
        assert abs(changed.mean() - frac) < 0.02
        if sev > 1:
            assert (changed | ~prev).all()                     # nested: affected at a lower severity -> affected now
        prev = changed


def test_lead_dropout_zeroes_exactly_k_leads():
    x = record()
    for sev, k in corrupt.DROPOUT_LEADS.items():
        c = corrupt.Corruptor("lead_dropout", sev)
        hit = np.zeros(12)
        for i in range(300):
            y = c(x, record_index=i)
            dead = (y == 0).all(1)
            assert dead.sum() == k and np.array_equal(y[~dead], x[~dead])
            hit += dead
        assert hit.min() > 0                                   # every lead can be dropped


def test_step_equals_preprocessing_a_raw_step_and_respects_lead_geometry():
    # a raw 500 Hz record with a step on the LA electrode, preprocessed, equals the clean record plus the template
    rng = np.random.default_rng(3)
    raw_v = rng.standard_normal((9, 5000))
    step_v = raw_v.copy(); step_v[1, 2000:] += 1.0             # +1 mV on LA at t = 4 s
    clean = preprocess.preprocess(leads_from_electrodes(raw_v))
    stepped = preprocess.preprocess(leads_from_electrodes(step_v))
    added = corrupt.LEAD_FROM_ELECTRODE[:, 1][:, None] * corrupt.step_response(4.0)[None]
    assert np.allclose(stepped - clean, added, atol=1e-4)
    # the filtered step is a transient: large near t0, gone a few seconds away, no constant offset left
    r = corrupt.step_response(4.0)
    assert np.abs(r[380:420]).max() > 0.3 and np.abs(r[:150]).max() < 0.02 and np.abs(r[700:900]).max() < 0.02


def test_step_amplitude_scales_and_keeps_einthoven():
    x = record()
    peaks = []
    for sev in corrupt.SEVERITIES:
        d = corrupt.Corruptor("baseline_step", sev)(x, record_index=7) - x
        assert np.allclose(d[0] + d[2], d[1], atol=1e-5) and np.allclose(d[3] + d[4] + d[5], 0, atol=1e-5)
        peaks.append(np.abs(d).max())
    # record 7 draws a different electrode/time per severity, so compare against the nominal amplitudes loosely
    assert peaks[0] > 0 and all(p <= 1.3 * corrupt.STEP_MV[s] for p, s in zip(peaks, corrupt.SEVERITIES))


def test_deterministic_and_independent_of_call_order():
    x = record()
    for c in corrupt.conditions(STD):
        a = c(x, record_index=11)
        _ = c(x, record_index=3)
        assert np.array_equal(a, c(x, record_index=11))
        assert a.shape == (12, corrupt.T) and a.dtype == np.float32 and np.isfinite(a).all()
    assert len(corrupt.conditions(STD)) == 18 and len(corrupt.conditions(STD, seen=False)) == 9


def test_input_is_not_modified():
    x = record(); x0 = x.copy()
    for c in corrupt.conditions(STD):
        c(x, record_index=0)
    assert np.array_equal(x, x0)


def test_seen_strengths_match_the_training_policy():
    assert corrupt.SEEN_STRENGTH == {1: augment.STRENGTH["low"], 2: augment.STRENGTH["mid"], 3: augment.STRENGTH["high"]}
    x = np.zeros((12, corrupt.T), np.float32)
    lo = np.mean([np.abs(corrupt.Corruptor("seen_baseline_wander", 1, STD)(x, record_index=i)).mean() for i in range(300)])
    hi = np.mean([np.abs(corrupt.Corruptor("seen_baseline_wander", 3, STD)(x, record_index=i)).mean() for i in range(300)])
    assert 2.5 < hi / lo < 3.5
    y = corrupt.Corruptor("seen_emg_bursts", 2, STD)(x, record_index=0)
    assert 0 < (y != 0).mean() < 1                              # bursts, not stationary noise
    ones = np.ones((12, corrupt.T), np.float32)
    f = corrupt.Corruptor("seen_lead_scaling", 2, STD)(ones, record_index=0)[:, 0]
    assert f.min() >= 0.7 and f.max() <= 1.3 and f.std() > 0


def test_held_out_set_is_disjoint_from_both_policies():
    policy_fns = {fn.__name__ for fns in augment.POLICIES.values() for fn in fns}
    assert policy_fns == {"baseline_wander", "emg_bursts", "per_lead_scaling", "gaussian_noise", "global_scaling",
                          "time_mask", "time_warp"}
    assert not set(corrupt.HELD_OUT) & policy_fns


def test_bad_arguments():
    with pytest.raises(ValueError):
        corrupt.Corruptor("lead_dropout", 4)
    with pytest.raises(ValueError):
        corrupt.Corruptor("seen_emg_bursts", 1)                 # std missing
    with pytest.raises(ValueError):
        corrupt.Corruptor("lead_dropout", 1)(np.zeros((12, 250), np.float32), record_index=0)
