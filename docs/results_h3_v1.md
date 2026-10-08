# H3 results: ECG-specific versus generic pretraining augmentation

Status: 2026-10-08. Produced by notebook 05 on Kaggle (Tesla T4, torch 2.11.0+cu128) under `docs/h3_protocol_v1.md`
(frozen 2026-10-07, D37). Inference: `scripts/bootstrap_h3.py` (patients resampled, 10,000 resamples, as in D34).
Files: `results/h3/` (probes, selection, view distortion, manifest), `results/runs_h3/` (9 fine-tuned runs),
`results/ssl/` (9 new pretraining runs), `results/analysis/bootstrap_h3.csv`, `bootstrap_h3_summary.json`,
`bootstrap_primary_final.csv`, `h3_grid_seed0.csv`.

## 1. Integrity checks (all passed)

- Every H3 run used the encoder named in the manifest (SHA-256 checked) and fine-tuned with P_ecg-mid, as frozen.
- The P_ecg-mid arm is the core C1 runs, reused unchanged (selection below).
- Clean predictions recomputed for the corruption suite match the stored predictions within 2.4e-4 (float16).
- The bootstrap reproduces every stored per-run metric within 5.8e-6.

## 2. Selection (seed 0, fold-9 linear probe)

| Level | P_ecg | P_gen |
|---|---|---|
| low | 0.8845 | 0.8691 |
| mid | **0.8907** | 0.8647 |
| high | 0.8862 | **0.8718** |

Selected: P_ecg at mid (so the core C1 runs are the P_ecg arm) and P_gen at high. No ties within the 0.0005
margin. The P_ecg probes score above every P_gen probe.

**View distortion** (mean squared change of a training window relative to its mean square; diagnostic only):
P_ecg 0.007 / 0.029 / 0.065, P_gen 0.18 / 0.39 / 0.55 at low / mid / high. The generic policy changes its views about 9 to
26 times more than the ECG policy, mostly through masking and warping (protocol limitation 1). Even so, every
pretraining run solves the pretext task almost completely (in-batch retrieval top-1 ≥ 0.999 at the end; final loss
0.088–0.101 for P_ecg, 0.093–0.118 for P_gen), as in the core runs (D31).

## 3. Primary contrast H3 (100% budget, P_ecg-mid minus P_gen-high, mean over 5 seeds)

Negative = the encoder pretrained with P_ecg degrades less.

| Outcome | P_ecg-mid | P_gen-high | Difference (95% interval, patients) | p | Interval with seeds resampled |
|---|---|---|---|---|---|
| H3-corr: mean held-out degradation, 71 statements | 0.0257 | 0.0256 | +0.0002 (-0.0011, +0.0013) | 0.83 | -0.0034 to +0.0039 |
| H3-sph: fold-10 to SPH degradation, E3 labels | 0.0040 | 0.0060 | -0.0020 (-0.0029, -0.0011) | 0.0002 | -0.0049 to +0.0006 |

**H3 is not supported.** It requires both outcomes to favour P_ecg. The corruption outcome shows no difference
(seed signs: -0.0052, -0.0001, +0.0023, +0.0056, -0.0019), so the conjunction p-value is 0.83.

The SPH outcome favours P_ecg with test-set sampling alone, but:
- it comes mostly from two seeds (per seed: +0.0007, +0.0001, -0.0066, -0.0040, -0.0001), and the interval that
  also resamples seeds includes zero;
- it is driven by the external score (SPH E3 macro-AUROC 0.9721 vs 0.9700), while fold-10 E3 scores are equal
  (0.9761 vs 0.9759).

## 4. Secondary (no significance claims)

- **Clean fold-10 macro-AUROC, 71 statements:** P_gen-high is slightly higher (0.9250 vs 0.9226; difference -0.0024,
  interval -0.0043 to -0.0001 with patients resampled, including zero with seeds resampled).
- **Per corruption, P_ecg minus P_gen:** lead dropout -0.0004, limb reversal +0.0004, baseline step +0.0004; none
  distinguishable from zero. The large reversal sensitivity of C1 seen in H2a is shared by both encoders.
- **Seed-0 grid** (`results/analysis/h3_grid_seed0.csv`): across the six policy-levels, fold-10 macro-AUROC ranges
  from 0.918 to 0.924, SPH E3 from 0.969 to 0.973, and mean held-out degradation from 0.022 to 0.032. One seed per
  cell; the seed-to-seed spread within the selected levels (0.022 to 0.031) is as large as the spread across the
  grid, so no ordering by policy or strength can be read from it.

## 5. Final state of the primary family (Holm over four contrasts)

| Hypothesis | Contrast | Estimate | 95% interval | p | Holm-adjusted p | Direction as hypothesised |
|---|---|---|---|---|---|---|
| H1 | C1 - S1, fold-10 macro-AUROC at 5% | +0.0103 | +0.0011 to +0.0157 | 0.024 | 0.048 | yes |
| H2a | C1 - S1, held-out corruption degradation at 100% | -0.0032 | -0.0046 to -0.0020 | 0.0002 | 0.0008 | yes |
| H2b | C1 - S1, SPH degradation at 100% | -0.0019 | -0.0028 to -0.0010 | 0.0002 | 0.0008 | yes |
| H3 | P_ecg - P_gen, corruptions and SPH (conjunction) | see section 3 | | 0.83 | 0.83 | no |

Each of H1, H2a and H2b carries caveats that the paper must state with it:
- **H1** is marginal (adjusted p 0.048); the difference changes sign between seeds, its seeds-plus-patients interval
  includes zero, and the predicted decline of the advantage with label budget is absent (`results_bootstrap_v1.md`).
- **H2a** comes entirely from limb-lead reversal, where augmentation makes the supervised model more fragile and
  pretraining offsets part of that; pretraining alone gives no robustness gain (`results_h2a_lrcheck_v1.md`).
- **H2b** is 0.002 AUROC, mostly one label (IRBBB), on a benchmark where both arms are near ceiling.

## 6. Caveats

1. The comparison is between the two levels each policy's probe selected; the seed-0 grid is too noisy to say
   whether another pairing would differ.
2. P_gen distorts its views far more than P_ecg, so "type" and "strength" are not separable here (limitation 1).
3. The pretext task saturates for every policy and level, which may limit how much any augmentation choice can
   change what the encoder learns (limitation 2; D31).
4. Selection by clean probe score (limitation 3) and by seed 0 alone (limitation 4).
5. All four contrasts now have p-values. The remaining pre-specified items are the sensitivity analyses
   (likelihood >= 50, amplitude outliers, full-length SPH); they do not enter the Holm family.
