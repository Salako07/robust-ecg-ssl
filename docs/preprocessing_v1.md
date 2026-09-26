# Preprocessing v1

**Status:** v1, 2026-09-26. Applies identically to PTB-XL and SPH. Changes need v2 and a log entry.
Implemented in `src/robust_ecg/preprocess.py`; caches built by `scripts/prepare_ptbxl.py` and
`scripts/prepare_sph.py`.

## Pipeline (per record)

| Step | PTB-XL | SPH | Reason |
|---|---|---|---|
| 1. Source | `records500/` (500 Hz, mV, via `wfdb`) | `.h5` (500 Hz, mV) | Start both from 500 Hz so every later step is shared. The authors' 100 Hz PTB-XL files are **not** used, because their downsampling cannot be reproduced on SPH. |
| 2. Lead order | I, II, III, aVR, aVL, aVF, V1–V6 | same | Verified from both dataset papers; asserted in code for PTB-XL (header names). |
| 3. Length | 10 s (5,000 samples) | **first 10 s** (records are 10–60 s; 94% are 10–15 s) | Same input duration in both datasets. Full-length SPH evaluation is a secondary sensitivity analysis. |
| 4. Band-pass | 0.5–40 Hz, 3rd-order Butterworth, zero-phase (`sosfiltfilt`) | same | Removes baseline drift and high-frequency noise; also equalises PTB-XL with SPH, which was already device-filtered. |
| 5. Resample | 500 → 100 Hz, `resample_poly(up=1, down=5)` (anti-aliasing FIR) | same | 100 Hz matches A1 and the PTB-XL benchmark and keeps T4 cost manageable. |
| 6. Store | `float32`, shape N × 12 × 1000 | same | Cached once on Google Drive. |
| 7. Normalise (at load time) | per-lead mean and std computed on **PTB-XL folds 1–8 only** | same statistics applied | **Not** per-record z-scoring: that erases absolute amplitude, which LVH voltage criteria depend on. |

SPH records in `results/sph_exclude.txt` (deduplication, D14) are dropped before caching.

## Inputs to the model

- **Training:** random 2.5 s crop (250 samples) per record per step (as in A1).
- **Evaluation:** sliding 2.5 s windows with 1.25 s stride over the 10 s record (7 windows);
  sigmoid outputs averaged across windows. Same rule for PTB-XL and SPH.

## Labels

- Multi-hot over all PTB-XL statements in `scp_statements.csv` (71), in file order, saved with the cache.
- Likelihood threshold: 0 (primary); ≥ 50 on diagnostic statements only (sensitivity), all splits (D11).
- **Macro-AUROC on PTB-XL** is averaged over statements with at least one positive in the evaluated
  fold (benchmark convention). A statement with no training positives at a small budget still counts;
  a constant output scores 0.5, which is the honest result.

## Consequence for augmentation policy P_ecg

At 100 Hz, a 50 Hz component lies exactly at the Nyquist frequency and 60 Hz aliases to 40 Hz; both are
also removed by the 0.5–40 Hz filter and SPH's device filter. **Powerline interference is therefore
removed from P_ecg** (D20).

## Sanity checks printed by the preparation scripts

- Per-dataset median absolute amplitude and 99th-percentile amplitude per lead (units check: both should
  be in mV with QRS peaks around 0.5–2 mV).
- Count of records with NaN/inf or flat leads (reported, not silently dropped).
