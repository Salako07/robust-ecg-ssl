"""Test-time corruption suite for H2a (research_question_v2.md §5, D16; docs/corruption_protocol_v1.md).

Every corruption acts on one preprocessed record x of shape (12, 1000): 10 s at 100 Hz, in mV, BEFORE per-lead
normalisation. It plugs into robust_ecg.data.EvalWindows(corrupt=...), which calls corrupt(x, record_index=i).

Randomness is a function of (SUITE_SEED, corruption, severity, record_index) only. Every model therefore sees
exactly the same corrupted test set, which keeps between-arm comparisons paired.

HELD-OUT corruptions appear in neither augmentation policy and are the only ones used to test H2a:
  lead_dropout   severity 1/2/3 = 1/2/3 leads set to zero for the whole record
  limb_reversal  left-arm and right-arm electrodes exchanged; severity 1/2/3 = 25/50/100% of records affected
                 (nested sets)
  baseline_step  an abrupt offset at one electrode, as it appears after the preprocessing filter;
                 severity 1/2/3 = 0.5/1.0/2.0 mV at the electrode (peak of about 0.55 x that after filtering)
SEEN corruptions are the three transforms of P_ecg at strengths low/mid/high (severity 1/2/3), always applied.
They are reported separately and never used for H2a.

The parameters below were frozen on 2026-10-06 (decision D35), before any real checkpoint was evaluated on a
corruption. Changing them needs a protocol v2 and a new decision-log entry. scripts/eval_corruptions.py refuses to
run if FROZEN is False.
"""
import numpy as np
from scipy.signal import resample_poly, sosfiltfilt

from .preprocess import _SOS, FS_IN, FS_OUT, LEADS, SECONDS

FROZEN = True                                          # frozen 2026-10-06, decision D35
SUITE_SEED = 20261007
T = FS_OUT * SECONDS                                   # 1000 samples
HELD_OUT = ("lead_dropout", "limb_reversal", "baseline_step")
SEEN = ("seen_baseline_wander", "seen_emg_bursts", "seen_lead_scaling")
SEVERITIES = (1, 2, 3)
_CODE = {name: i for i, name in enumerate(HELD_OUT + SEEN)}

# ---- parameters (protocol v1, frozen) -------------------------------------------------------------------------------
DROPOUT_LEADS = {1: 1, 2: 2, 3: 3}                     # number of leads set to zero
REVERSAL_FRACTION = {1: 0.25, 2: 0.50, 3: 1.00}        # share of records with LA/RA exchanged
STEP_MV = {1: 0.5, 2: 1.0, 3: 2.0}                     # size of the electrode offset
STEP_WINDOW_S = (1.0, 9.0)                             # the step occurs uniformly in this interval
SEEN_STRENGTH = {1: 0.5, 2: 1.0, 3: 1.5}               # augment.STRENGTH low / mid / high

# ---- electrode model ----------------------------------------------------------------------------------------
# Leads as linear functions of the nine electrode potentials (RA, LA, LL, V1..V6); the right-leg electrode is
# the reference. Limb leads: I = LA - RA, II = LL - RA, III = LL - LA; augmented leads are referred to the mean
# of the other two limb electrodes; precordial leads to the mean of the three (Wilson's central terminal).
ELECTRODES = ("RA", "LA", "LL", "V1", "V2", "V3", "V4", "V5", "V6")
LEAD_FROM_ELECTRODE = np.zeros((12, 9))
LEAD_FROM_ELECTRODE[0, [0, 1]] = [-1, 1]               # I
LEAD_FROM_ELECTRODE[1, [0, 2]] = [-1, 1]               # II
LEAD_FROM_ELECTRODE[2, [1, 2]] = [-1, 1]               # III
LEAD_FROM_ELECTRODE[3, [0, 1, 2]] = [1, -.5, -.5]      # aVR
LEAD_FROM_ELECTRODE[4, [0, 1, 2]] = [-.5, 1, -.5]      # aVL
LEAD_FROM_ELECTRODE[5, [0, 1, 2]] = [-.5, -.5, 1]      # aVF
for _v in range(6):                                    # V1..V6
    LEAD_FROM_ELECTRODE[6 + _v, [0, 1, 2]] = -1 / 3
    LEAD_FROM_ELECTRODE[6 + _v, 3 + _v] = 1
assert LEADS[:6] == ["I", "II", "III", "AVR", "AVL", "AVF"]


def _rng(name, severity, record_index):
    return np.random.default_rng([SUITE_SEED, _CODE[name], int(severity), int(record_index)])


# ---- held-out corruptions -----------------------------------------------------------------------------------
def lead_dropout(x, severity, rng):
    """Set k randomly chosen leads to zero for the whole record (loss of a stored channel)."""
    y = x.copy()
    y[rng.choice(12, size=DROPOUT_LEADS[severity], replace=False)] = 0.0
    return y


def reverse_la_ra(x):
    """Exchange the left-arm and right-arm electrodes: I -> -I, II <-> III, aVR <-> aVL; aVF and V1-V6 unchanged."""
    y = x.copy()
    y[0] = -x[0]
    y[1], y[2] = x[2], x[1]
    y[3], y[4] = x[4], x[3]
    return y


def limb_reversal(x, severity, record_index):
    """One uniform number per record, shared by all severities, so the affected sets are nested (25% within 50%)."""
    u = _rng("limb_reversal", 0, record_index).random()
    return reverse_la_ra(x) if u < REVERSAL_FRACTION[severity] else x.copy()


def step_response(t0_s, fs_in=FS_IN):
    """A unit step at t0, passed through the preprocessing pipeline (0.5-40 Hz zero-phase band-pass, resampling to
    100 Hz). Because the pipeline is linear, adding this to a preprocessed record equals preprocessing a raw
    record with a step added. The result is a transient around t0; the constant offset is removed."""
    step = (np.arange(fs_in * SECONDS) >= int(round(t0_s * fs_in))).astype(np.float64)
    return resample_poly(sosfiltfilt(_SOS, step), up=1, down=fs_in // FS_OUT)


def baseline_step(x, severity, rng):
    """An abrupt potential offset at one electrode (electrode motion), with random sign and time."""
    e = rng.integers(0, len(ELECTRODES))
    amp = STEP_MV[severity] * rng.choice([-1.0, 1.0])
    r = step_response(rng.uniform(*STEP_WINDOW_S))
    return (x + amp * LEAD_FROM_ELECTRODE[:, e][:, None] * r[None, :]).astype(np.float32)


# ---- seen corruptions: P_ecg transforms on the 10 s record, in units of the per-lead training std -------------
def seen_baseline_wander(x, severity, rng, std):
    s = SEEN_STRENGTH[severity]
    t = np.arange(T) / FS_OUT
    f, ph = rng.uniform(0.05, 0.5, (3, 1)), rng.uniform(0, 2 * np.pi, (3, 1))
    wave = np.sin(2 * np.pi * f * t + ph).mean(0)
    return (x + rng.uniform(0, 0.5 * s, (12, 1)) * std * wave).astype(np.float32)


def seen_emg_bursts(x, severity, rng, std):
    """Training uses 1-3 bursts of 0.2-1.0 s per lead in a 2.5 s window; a 10 s record gets four times as many."""
    s = SEEN_STRENGTH[severity]
    mask = np.zeros((12, T), bool)
    for lead in range(12):
        for _ in range(4 * rng.integers(1, 4)):
            n = rng.integers(int(0.2 * FS_OUT), int(1.0 * FS_OUT) + 1)
            a = rng.integers(0, T - n + 1)
            mask[lead, a:a + n] = True
    noise = rng.standard_normal((12, T)) * rng.uniform(0, 0.3 * s, (12, 1)) * std
    return (x + noise * mask).astype(np.float32)


def seen_lead_scaling(x, severity, rng, std):
    """Scaling acts on the normalised signal in training; the training mean is negligible next to the std, so the
    record is scaled directly."""
    s = SEEN_STRENGTH[severity]
    return (x * rng.uniform(1 - 0.3 * s, 1 + 0.3 * s, (12, 1))).astype(np.float32)


_FN = dict(lead_dropout=lead_dropout, limb_reversal=limb_reversal, baseline_step=baseline_step,
           seen_baseline_wander=seen_baseline_wander, seen_emg_bursts=seen_emg_bursts,
           seen_lead_scaling=seen_lead_scaling)


class Corruptor:
    """Callable for EvalWindows: Corruptor("baseline_step", 2)(x, record_index=i) -> corrupted (12, 1000) float32."""

    def __init__(self, name, severity, std=None):
        if name not in _FN or severity not in SEVERITIES:
            raise ValueError(f"unknown corruption {name!r} or severity {severity!r}")
        if name in SEEN and std is None:
            raise ValueError("seen corruptions need the per-lead training std")
        self.name, self.severity = name, severity
        self.std = None if std is None else np.asarray(std, np.float64).reshape(12, 1)

    @property
    def tag(self):
        return f"{self.name}_s{self.severity}"

    def __call__(self, x, record_index):
        x = np.asarray(x, np.float32)
        if x.shape != (12, T):
            raise ValueError(f"expected (12, {T}), got {x.shape}")
        if self.name == "limb_reversal":
            return np.ascontiguousarray(limb_reversal(x, self.severity, record_index), dtype=np.float32)
        rng = _rng(self.name, self.severity, record_index)
        args = (x, self.severity, rng) + ((self.std,) if self.name in SEEN else ())
        return np.ascontiguousarray(_FN[self.name](*args), dtype=np.float32)


def conditions(std, held_out=True, seen=True):
    """All (name, severity) corruptors in a fixed order."""
    names = (HELD_OUT if held_out else ()) + (SEEN if seen else ())
    return [Corruptor(n, s, std) for n in names for s in SEVERITIES]
