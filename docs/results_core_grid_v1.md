# Core grid results v1 (descriptive, seed level)

Status: 2026-10-04. Source: `results/runs/registry.csv` (120 fine-tuning runs) and `results/ssl/` (5 pretraining
runs), all on Kaggle, Tesla T4, torch 2.10.0+cu128. Tables are produced by `scripts/analyze_grid.py` and stored in
`results/analysis/`.

**What this document is not.** RQ v2 §6 pre-registers a paired bootstrap over test patients with Holm correction for
the primary contrasts. That analysis needs the test labels and is not done yet. Everything below is the seed-level
view: means over 5 seeds, differences paired by seed (same seed = same labelled subset), and a t-based 95% interval
over seeds (n = 5, 4 degrees of freedom). It describes variation between training runs, not test-set sampling
error, and no result here is called significant.

## 1. Integrity checks (all passed)

- 120 runs: 4 arms × 6 budgets × 5 seeds, no duplicates, no missing cell. Every `done.json` matches its registry row.
- Every C0/C1 run loaded the encoder of its own seed (`encoder_sha256` equals the SSL run's hash). S0/S1 loaded none.
- Predictions: 2,198 fold-10 records × 71 statements and 25,577 SPH records × 9 labels per run, all finite.
- No run has its best validation score at the first evaluation.
- **Cross-platform reproduction.** The three seed-0 runs at 100% that were also made on Colab (D32) reproduce on Kaggle
  with identical stored predictions (maximum absolute difference 0 at float16), identical best step and identical
  metrics. `SSL_ecg-mid_s0` has the same per-epoch losses and the same encoder SHA-256 on both platforms, although
  the torch versions differ (2.11.0 on Colab, 2.10.0 on Kaggle). The cache files therefore also match.

## 2. Arm means (macro-AUROC, mean over 5 seeds; SD in `results/analysis/arm_means.csv`)

PTB-XL fold 10, 71 statements:

| Budget | S0 | S1 | C0 | C1 |
|---|---|---|---|---|
| 1% (exploratory) | 0.679 | 0.688 | 0.696 | 0.679 |
| 5% | 0.811 | 0.802 | 0.810 | 0.812 |
| 10% | 0.849 | 0.851 | 0.843 | 0.851 |
| 25% | 0.885 | 0.886 | 0.882 | 0.881 |
| 50% | 0.905 | 0.904 | 0.905 | 0.906 |
| 100% | 0.923 | 0.925 | 0.922 | 0.923 |

SPH, 9 shared labels (E3), no adaptation:

| Budget | S0 | S1 | C0 | C1 |
|---|---|---|---|---|
| 1% (exploratory) | 0.848 | 0.863 | 0.887 | 0.879 |
| 5% | 0.935 | 0.942 | 0.943 | 0.946 |
| 10% | 0.954 | 0.957 | 0.958 | 0.958 |
| 25% | 0.966 | 0.968 | 0.968 | 0.966 |
| 50% | 0.968 | 0.970 | 0.969 | 0.969 |
| 100% | 0.969 | 0.971 | 0.968 | 0.972 |

Seed SD of the 71-statement score is about 0.010 at 5%, 0.004–0.007 at 10–50% and 0.001–0.003 at 100%.

## 3. Primary contrasts that this grid can address

| Contrast (RQ v2 §6) | Mean C1 − S1 | 95% interval over seeds | Seeds with C1 > S1 | Per seed |
|---|---|---|---|---|
| H1: fold-10 macro-AUROC at 5% | +0.010 | −0.013 to +0.034 | 3 of 5 | +0.023, −0.012, +0.026, −0.010, +0.024 |
| H2b: E3 degradation (fold 10 − SPH) at 100% | −0.0019 | −0.0029 to −0.0009 | 0 of 5 (C1 degrades less in all 5) | −0.0018, −0.0018, −0.0030, −0.0008, −0.0020 |

**Observations.**
- H1: the difference at 5% changes sign across seeds and its interval includes zero. Across the primary budgets C1 − S1
  is +0.010, −0.001, −0.005, +0.001, −0.002 (5% to 100%). There is no consistent advantage and no trend of a larger
  advantage at smaller budgets. At 25% and 100% C1 is lower than S1 in all five seeds, by 0.005 and 0.002.
- H2b: C1 has a smaller fold-10-to-SPH drop than S1 in all five seeds at 100%. The size is 0.002 AUROC, and it comes
  from SPH (C1 − S1 = +0.0016, 5 of 5 seeds) with fold-10 E3 essentially equal (−0.0003). At 5–50% the degradation
  difference is within ±0.001 and changes sign.

**Interpretation (not established).** The H1 data are compatible with no SSL benefit once the supervised model receives
the same augmentation, at this encoder size and with this pretraining. They are also compatible with a benefit or
harm of up to about 0.03 at 5%, because seed variation is large there. The H2b direction matches the hypothesis but
the effect is at the third decimal place, on a benchmark where both arms are near ceiling (D27), and "degradation" is a
net of opposite per-label effects (D28). Whether it exceeds test-set sampling error is exactly what the patient
bootstrap will tell.

## 4. Secondary contrasts (no significance claims)

| Contrast | Metric | 1% | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|---|
| C0 − S0 (SSL, no augmentation) | fold 10, 71 statements | +0.017 | −0.001 | −0.006 | −0.003 | 0.000 | −0.001 |
| C1 − S1 (SSL, augmentation matched) | fold 10, 71 statements | −0.009 | +0.010 | −0.001 | −0.005 | +0.001 | −0.002 |
| S1 − S0 (augmentation alone) | fold 10, 71 statements | +0.009 | −0.010 | +0.003 | +0.001 | −0.001 | +0.002 |
| C0 − S0 | fold 10, E3 labels | +0.051 | +0.006 | +0.006 | −0.002 | −0.001 | 0.000 |
| C1 − S1 | fold 10, E3 labels | +0.025 | +0.005 | 0.000 | −0.001 | −0.001 | 0.000 |
| S1 − S0 | fold 10, E3 labels | +0.024 | +0.001 | +0.007 | +0.001 | +0.001 | +0.001 |
| C0 − S0 | SPH, E3 labels | +0.039 | +0.007 | +0.003 | +0.002 | 0.000 | −0.002 |
| C1 − S1 | SPH, E3 labels | +0.016 | +0.005 | +0.001 | −0.002 | −0.001 | +0.002 |
| S1 − S0 | SPH, E3 labels | +0.015 | +0.007 | +0.003 | +0.002 | +0.002 | +0.001 |

**Observations.**
- From 10% upward all four arms are within about ±0.005 of each other on every metric.
- At the exploratory 1% budget, the comparison "as usually reported" (C0 − S0) shows an SSL gain on the E3 labels:
  +0.051 on fold 10 and +0.039 on SPH, positive in 5 of 5 seeds. With augmentation matched (C1 − S1) the gain is about
  half: +0.025 (4 of 5 seeds) and +0.016 (4 of 5). Augmentation alone (S1 − S0) gives +0.024 and +0.015.
- On the 71-statement score at 1%, C0 − S0 is +0.017 (5 of 5 seeds) while C1 − S1 is −0.009 (2 of 5).
- At 1%, fine-tuning augmentation lowers the SSL arm's 71-statement score (C1 − C0 = −0.017, 4 of 5 seeds negative),
  while it raises the randomly initialised arm's (S1 − S0 = +0.009).

**Caveats for the 1% rows.** The budget is exploratory by pre-registration (D10): 165–186 training records, with
several labels having fewer than 10 positives, and seed SD of 0.01–0.02. The 71-statement macro average at 1% includes
statements with almost no positive training examples. These rows motivate a hypothesis; they do not test one.

## 5. Per-label AUROC on SPH at 100% (mean over seeds)

IRBBB is the only label with a marked cross-site drop in every arm (fold 10 ≈ 0.975; SPH: S0 0.788, S1 0.799, C0 0.775,
C1 0.810). All other labels are ≥ 0.976 on SPH in every arm. Full table: `results/analysis/per_label_100pct.csv`.
The H2b difference at 100% is therefore largely an IRBBB difference, which the per-label bootstrap must confirm.

## 6. SSL pretraining

Five runs, 21.0–21.7 min each. All saturate in the same way as the pilot (D31): final loss 0.094–0.096, in-batch
retrieval top-1 ≥ 0.9998.

## 7. What remains before any claim

1. Paired patient-level bootstrap and Holm correction for the two contrasts in §3 (needs fold-10 and SPH labels).
2. H2a: held-out corruption suite on the saved best checkpoints (not implemented).
3. H3: P_ecg vs P_gen pretraining grid (not run).
4. D30 check: C1 fine-tuned at lr 1e-3. The null H1 pattern makes this check necessary before the result can be
   attributed to SSL itself and not to the shared fine-tuning recipe.
5. Sensitivity analyses: likelihood ≥ 50 (D11), amplitude outliers (D26).

## 8. Threats specific to these results

- **Saturated pretext task** (D31): the encoder may have learned little that transfers, so a null H1 may reflect this
  SimCLR configuration and not contrastive pretraining in general.
- **Small encoder and in-domain pretraining only**: 886,400 parameters pretrained on the same 17,418 records used for
  fine-tuning at 100%. Prior ECG SSL results (A1, A3) pretrain on larger external collections.
- **Shared fine-tuning recipe** (D30).
- **Five seeds**: enough to see that effects above about 0.02 at 5% and about 0.005 at 100% are absent, not enough to
  resolve smaller ones.
