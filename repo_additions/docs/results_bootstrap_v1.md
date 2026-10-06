# Patient-level bootstrap v1: H1 and H2b (core grid)

Status: 2026-10-06. Source: `results/runs/` (120 runs, likelihood threshold 0), labels and patient IDs rebuilt from
`ptbxl_meta.csv`, `scp_statements.csv` and `sph_meta.csv` (hashes equal to `results/cache_manifest.json`).
Script: `scripts/bootstrap_contrasts.py`; tests: `tests/test_bootstrap.py`. Outputs in `results/analysis/`:
`bootstrap_primary.csv`, `bootstrap_contrasts.csv`, `bootstrap_contrasts_per_label.csv`, `bootstrap_config.json`.
Method and its limits: decision D34.

## 1. Checks

- Point estimates recomputed from the stored predictions and rebuilt labels reproduce every registry value:
  maximum absolute difference 2.3e-4 over 480 values, and at most 2.6e-5 at the primary budgets. The residual is
  float16 storage of predictions (largest at 1%, where near-constant outputs tie after rounding).
- The weighted AUROC used in the bootstrap equals scikit-learn's `roc_auc_score(sample_weight=...)` to 1e-12 on
  synthetic data with ties.
- 10,000 resamples; 1,904 fold-10 patients (2,198 recordings) and 24,642 SPH patients (25,577 recordings).

## 2. Primary contrasts (C1 - S1, mean over 5 seeds)

| Hypothesis | Contrast | Estimate | 95% interval (patients) | p (two-sided) | p x 4 (Bonferroni bound) | 95% interval (seeds + patients) |
|---|---|---|---|---|---|---|
| H1 | fold-10 macro-AUROC, 71 statements, 5% | +0.0103 | +0.0011 to +0.0157 | 0.024 | 0.097 | -0.0084 to +0.0257 |
| H2b | degradation fold 10 - SPH, 100% | -0.0019 | -0.0028 to -0.0010 | 0.0002 | 0.0008 | -0.0032 to -0.0006 |

p = 0.0002 is the smallest value 10,000 resamples can give.

**H2b.** C1 degrades less than S1 from fold 10 to SPH. The contrast is significant whatever the two missing
p-values (H2a, H3) turn out to be, and the interval excludes zero when seeds are resampled as well. The size is
0.002 AUROC.

**H1.** With the five trained models taken as fixed, C1 is ahead of S1 at 5%. Its Holm-adjusted p-value cannot be
computed until H2a and H3 exist: it will lie between 0.024 and 0.073, so the result may or may not survive
correction. Two further facts weaken it. The difference changes sign between seeds (3 of 5 positive), and the
interval that also resamples seeds includes zero. The directional part of H1 (advantage largest at 5% and
shrinking with budget) is not what the curve shows (section 3).

## 3. C1 - S1 across budgets (patients resampled; secondary, no significance claims)

| Budget | Fold 10, 71 statements | Fold 10, E3 labels | SPH, E3 labels | Degradation |
|---|---|---|---|---|
| 1% (exploratory) | -0.0084 (-0.0208, +0.0022) | +0.0248 (+0.0183, +0.0312) | +0.0158 (+0.0117, +0.0198) | +0.0090 (+0.0014, +0.0166) |
| 5% | +0.0103 (+0.0011, +0.0157) | +0.0054 (+0.0022, +0.0086) | +0.0045 (+0.0027, +0.0062) | +0.0009 (-0.0027, +0.0045) |
| 10% | -0.0005 (-0.0060, +0.0059) | -0.0001 (-0.0022, +0.0021) | +0.0012 (+0.0001, +0.0023) | -0.0013 (-0.0037, +0.0012) |
| 25% | -0.0048 (-0.0080, -0.0014) | -0.0010 (-0.0024, +0.0004) | -0.0021 (-0.0027, -0.0015) | +0.0011 (-0.0005, +0.0026) |
| 50% | +0.0011 (-0.0019, +0.0038) | -0.0012 (-0.0023, -0.0001) | -0.0010 (-0.0016, -0.0004) | -0.0002 (-0.0015, +0.0010) |
| 100% | -0.0023 (-0.0043, +0.0001) | -0.0003 (-0.0010, +0.0004) | +0.0016 (+0.0011, +0.0021) | -0.0019 (-0.0028, -0.0010) |

- The 71-statement difference is positive at 5%, near zero at 10% and 50%, and negative at 25% with an interval
  that excludes zero. There is no decreasing trend from an advantage at small budgets.
- On the nine E3 labels C1 is ahead at 1% and 5% on both test sets; from 10% the differences are within about
  +-0.002.
- The smaller degradation of C1 appears only at 100%. At 1% the sign is reversed (C1 gains more on fold 10 than on
  SPH).

## 4. H2b by label and by group at 100% (C1 - S1)

| Label / group | Fold 10 | SPH | Degradation |
|---|---|---|---|
| IRBBB | -0.0022 (-0.0049, +0.0002) | +0.0118 (+0.0086, +0.0150) | -0.0140 (-0.0182, -0.0099) |
| PR_PROL | +0.0028 (+0.0003, +0.0056) | -0.0003 (-0.0011, +0.0004) | +0.0032 (+0.0006, +0.0060) |
| ASMI | -0.0013 (-0.0028, +0.0001) | +0.0004 (-0.0005, +0.0013) | -0.0017 (-0.0035, +0.0000) |
| IMI | -0.0012 (-0.0041, +0.0017) | +0.0011 (-0.0008, +0.0032) | -0.0023 (-0.0058, +0.0012) |
| LVH | -0.0006 (-0.0035, +0.0024) | +0.0009 (+0.0002, +0.0019) | -0.0015 (-0.0045, +0.0016) |
| AF | +0.0002 (-0.0020, +0.0026) | +0.0008 (-0.0001, +0.0025) | -0.0006 (-0.0034, +0.0021) |
| CRBBB, CLBBB, LAFB | within +-0.0003 | within +-0.0002 | within +-0.0001 |
| Criteria-based (6 labels) | +0.0000 (-0.0007, +0.0008) | +0.0020 (+0.0013, +0.0026) | -0.0019 (-0.0029, -0.0009) |
| Interpretive (3 labels) | -0.0010 (-0.0025, +0.0004) | +0.0008 (+0.0001, +0.0016) | -0.0018 (-0.0036, -0.0002) |

The H2b effect is mostly IRBBB: its degradation difference of -0.0140 contributes -0.0016 of the -0.0019 macro
difference (one ninth of the label value), and comes from C1 scoring 0.012 higher than S1 on SPH. PR_PROL goes the
other way. This confirms the expectation recorded in
results_core_grid_v1 section 5.

## 5. Other contrasts (secondary)

- **Conventional comparison C0 - S0 at 1%:** +0.017 on 71 statements (+0.005, +0.028) and +0.039 on SPH E3 labels
  (+0.035, +0.043). With augmentation matched (C1 - S1) the SPH gain is +0.016 and the 71-statement difference is
  -0.008. The 1% budget remains exploratory (D10).
- **Augmentation alone, S1 - S0:** higher SPH E3 macro-AUROC at every budget (+0.015 at 1% down to +0.001 at 100%,
  all intervals above zero).
- Full tables: `results/analysis/bootstrap_contrasts.csv`.

## 6. Caveats

1. **What the patient interval covers.** It is the uncertainty from the finite test sets for these five trained
   models per arm. It does not cover variation between training runs. At 5% the seed-to-seed SD of the 71-statement
   score is about 0.010, the same size as the H1 estimate.
2. **Choices made after the seed-level results were known** (D34): number of resamples, how seeds are combined, the
   p-value definition and the independent resampling of the two test sets. They were fixed before this script was
   run on real predictions, and they follow RQ v2 section 6 and the E3 protocol, but they were not written down
   before the grid was seen.
3. **The 71-statement interval is not centred on its estimate** (H1: estimate +0.0103, interval +0.0011 to
   +0.0157). Several statements have very few positives in fold 10, so they drop out of some resamples and the
   macro average then runs over a different label set. A bias-corrected interval or a version restricted to
   statements with a minimum number of positives would be a sensitivity analysis; neither is pre-specified.
4. **Effect sizes are small.** The H2b difference is 0.002 AUROC with both arms near ceiling on eight of nine SPH
   labels (D27), and degradation is a net of opposite per-label effects (D28).
5. **Still open:** the D30 learning-rate check, H2a, H3 and the sensitivity analyses. No hypothesis is declared
   supported or rejected in the paper until Holm's procedure can be run on all four contrasts.
