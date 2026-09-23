# E3 protocol v1: zero-adaptation cross-dataset evaluation (PTB-XL → SPH)

**Status:** v1, frozen 2026-09-23 pending the open items at the end. Changes require a new
version and an entry in [`decisions/log.md`](decisions/log.md).

## Question E3 answers

Does an SSL-pretrained encoder, fine-tuned on PTB-XL labels, lose less discriminative
performance than supervised baselines (with and without matched augmentation) when applied
**without any adaptation** to an external hospital dataset with independently assigned labels?

## Datasets

| Role | Dataset | Version / source | Notes |
|---|---|---|---|
| Train / val / in-distribution test | PTB-XL | v1.0.3, PhysioNet | folds 1–8 train, 9 val, 10 test; folds 9–10 are 100% human-validated, folds 1–8 are 64–68% |
| External test (no adaptation) | SPH (Shandong Provincial Hospital) | Liu et al., *Sci Data* 9:272 (2022), figshare | 25,770 records / 24,666 patients; cardiologist-checked AHA statements; deduplicated before use |

SPH is never used for training, tuning, threshold selection or model selection.

## Training task (Option B)

Models are trained on the **full PTB-XL statement set**. E3 evaluates only the shared labels
below, on both PTB-XL fold 10 and SPH, with identical label definitions.

**PTB-XL label threshold.** Primary: every listed statement counts (threshold 0).
Pre-registered sensitivity analysis: likelihood ≥ 50 applied **to diagnostic statements only**,
on **all** PTB-XL splits (train, fold 9, fold 10). Form and rhythm statements carry likelihood 0
by PTB-XL convention and always count.

## Shared labels

Mapping source: the AHA-code column of PTB-XL `scp_statements.csv` (dataset authors' crosswalk),
matched to SPH AHA primary statements. SPH modifiers (codes ≥ 300) are ignored.

| Label | PTB-XL SCP | SPH AHA | Group | SPH positives |
|---|---|---|---|---:|
| AF | AFIB | 50 | criteria-based | 675 |
| PR_PROL | 1AVB ∪ LPR | 82 | criteria-based | 238 |
| CRBBB | CRBBB | 106 | criteria-based | 710 |
| IRBBB | IRBBB | 105 | criteria-based | 1,259 |
| CLBBB | CLBBB | 104 | criteria-based | 84 |
| LAFB | LAFB | 101 | criteria-based | 154 |
| LVH | LVH | 142 | interpretive | 209 |
| IMI | IMI | 161 | interpretive | 120 |
| ASMI | ASMI | 165 | interpretive | 91 |

SPH counts are before deduplication.

Reported but outside the headline metric: MI_ANY, TWAVE_ABN, NORM (exploratory); SBRAD, STACH,
SARRH (secondary, rate-defined). Excluded: PVC, PAC (label may refer to beats outside the
evaluated window).

**Merged labels.** A merged concept's score is the **maximum** of its constituent output
probabilities (PR_PROL = max(p_1AVB, p_LPR)), applied identically on PTB-XL and SPH.

## Label budgets

Patient-level, nested (1% ⊂ 5% ⊂ … ⊂ 100%), unstratified, drawn from folds 1–8 with one patient
ordering per seed (`scripts/e3_label_support.py`).

- **Primary curve:** 5%, 10%, 25%, 50%, 100%. Every primary label has ≥ 19 training positives in
  every seed at 5% under both thresholds.
- **Exploratory:** 1%, per-label only, for labels with ≥ 10 positives in every seed under
  **both** thresholds: IMI, LVH, LAFB. Report per-seed values, not only means.

## Metrics

- **Headline:** macro-AUROC over the 9 primary labels, also reported separately for the
  criteria-based and interpretive groups.
- **Out-of-distribution degradation:** AUROC(PTB-XL fold 10) − AUROC(SPH), on the same labels,
  compared across arms.
- **Also reported:** per-label AUROC; AUPRC shown next to each label's SPH prevalence;
  F1 at thresholds tuned on PTB-XL fold 9 (labelled as deployment realism);
  precision re-computed at PTB-XL prevalence, PPV(π) = TPR·π / (TPR·π + FPR·(1−π)).

## Statistical inference

Paired bootstrap of between-arm differences on the same SPH records, **resampling patients**,
not records. Seed variation reported separately from bootstrap intervals.

## Open items before the first E3 run

1. SPH deduplication run (`scripts/sph_dedup.py`); record counts in the log.
2. Signal pipeline, applied identically to both datasets: band-pass filter parameters,
   sampling rate, amplitude units check, 10 s windowing and test-time aggregation rule.
3. Number of bootstrap resamples and seeds per arm (compute-dependent).
