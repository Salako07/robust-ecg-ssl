# H2a (held-out corruptions) and the D30 learning-rate check

Status: 2026-10-07. Produced by notebook 04 on Kaggle (Tesla T4, torch 2.11.0+cu128) from the saved best
checkpoints of the core grid. Protocol: `docs/corruption_protocol_v1.md` (frozen 2026-10-06, D35). Inference:
`scripts/bootstrap_h2a.py`, as fixed in D34. Files: `results/runs/*/corruptions.json` (120 runs),
`results/runs/*_b1_*/corruptions.npz` (20 runs), `results/runs/corruptions_registry.csv`,
`results/analysis/bootstrap_h2a.csv`, `bootstrap_h2a_config.json`, `bootstrap_primary_with_h2a.csv`,
`results/runs_lrcheck/` (6 runs).

## 1. Integrity checks (all passed)

- 120 runs evaluated on 18 conditions each; 2,280 registry rows (120 clean + 2,160 corrupted).
- Clean predictions recomputed from every 100% checkpoint match the stored grid predictions within 2.4e-4
  (float16 storage), although the grid ran under torch 2.10.0 and this evaluation under 2.11.0.
- Every `corruptions.json` records the frozen suite parameters and seed 20261007.
- The bootstrap reproduces the per-run macro-AUROCs of `corruptions.json` within 1.2e-5.

## 2. Primary contrast H2a

C1 - S1 in mean degradation over the nine held-out conditions, fold 10, 71 statements, 100% budget, mean over five
seeds. Negative means C1 loses less than S1.

| Estimate | 95% interval (patients) | p (two-sided) | p x 4 | 95% interval (seeds + patients) | Seeds with C1 < S1 |
|---|---|---|---|---|---|
| -0.0032 | -0.0046 to -0.0020 | 0.0002 | 0.0008 | -0.0066 to -0.0003 | 3 of 5 |

Per seed: -0.0064, +0.0003, +0.0005, -0.0047, -0.0058. p = 0.0002 is the smallest value 10,000 resamples can give.

The contrast has the hypothesised sign and is significant whatever the H3 p-value turns out to be.

## 3. Where the effect comes from

Mean degradation at 100% (clean macro-AUROC: S0 0.9228, S1 0.9249, C0 0.9220, C1 0.9226):

| Corruption (mean of 3 severities) | S0 | S1 | C0 | C1 | C1 - S1 (95% interval) |
|---|---|---|---|---|---|
| Lead dropout | 0.0209 | 0.0145 | 0.0202 | 0.0156 | +0.0011 (-0.0007, +0.0023) |
| Limb-lead reversal | 0.0566 | 0.0711 | 0.0547 | 0.0604 | -0.0106 (-0.0138, -0.0075) |
| Baseline step | 0.0009 | 0.0012 | 0.0010 | 0.0011 | -0.0001 (-0.0005, +0.0003) |
| **All nine held-out conditions** | 0.0262 | 0.0289 | 0.0253 | 0.0257 | **-0.0032 (-0.0046, -0.0020)** |
| Seen (P_ecg) corruptions | 0.0013 | 0.0005 | 0.0013 | 0.0006 | +0.0001 (-0.0001, +0.0003) |

Observations:

1. **The H2a difference is a limb-lead-reversal difference.** C1 - S1 is -0.0053, -0.0078 and -0.0188 at 25%, 50%
   and 100% of records reversed (all intervals exclude zero; C1 < S1 in 5 of 5 seeds for the reversal mean).
   Lead dropout goes slightly the other way and the baseline step shows no difference.
2. **Augmentation, not initialisation, is the larger factor, and it acts in opposite directions.**
   - Lead dropout: training with P_ecg reduces degradation with either initialisation (S1 - S0 = -0.0064,
     C1 - C0 = -0.0047; both intervals exclude zero).
   - Limb-lead reversal: training with P_ecg *increases* degradation (S1 - S0 = +0.0144; C1 - C0 = +0.0057).
     The increase is smaller with the pretrained initialisation, and that difference is the H2a effect.
3. **Pretraining without augmentation gives no robustness gain:** C0 - S0 = -0.0008 (-0.0026, +0.0008).
4. **C1 is not more robust than the plain baseline S0** (0.0257 vs 0.0262 mean held-out degradation). S1 is the
   least robust arm on the held-out suite because of its reversal result.
5. **The baseline step is too mild to discriminate:** degradation is about 0.001 in every arm (limitation 3 of the
   protocol anticipated this possibility).
6. **Seen corruptions behave as expected by construction:** arms trained with P_ecg lose about 0.0005, arms trained
   without it about 0.0013 (S1 - S0 = -0.0007, interval excluding zero), and initialisation makes no difference.
   This is the pattern that would be mistaken for an SSL robustness benefit if C1 were compared with S0 on noise
   from the pretraining policy.

On the nine E3 labels the picture is the same: C1 - S1 = -0.0023 (-0.0030, -0.0016) for the held-out mean and
-0.0074 (-0.0095, -0.0053) for reversal.

## 4. Other budgets (seed level, descriptive; no bootstrap, no significance claims)

C1 - S1 in mean held-out degradation: 1% -0.0108 (4 of 5 seeds negative), 5% +0.0009 (2 of 5), 10% -0.0046
(5 of 5), 25% -0.0017 (3 of 5), 50% +0.0016 (1 of 5), 100% -0.0032 (3 of 5). The sign is not consistent across
budgets.

## 5. Status of the primary family (three of four contrasts)

| Hypothesis | Estimate | 95% interval | p | Holm-adjusted p, whatever H3 gives |
|---|---|---|---|---|
| H1 (5%, fold 10) | +0.0103 | +0.0011 to +0.0157 | 0.0242 | at most 0.0484 |
| H2a (held-out corruptions, 100%) | -0.0032 | -0.0046 to -0.0020 | 0.0002 | at most 0.0008 |
| H2b (SPH, 100%) | -0.0019 | -0.0028 to -0.0010 | 0.0002 | at most 0.0008 |

With H2a and H2b both at the minimum p-value, H1 is third or fourth in Holm's ordering, so its adjusted value is
0.0484 (if H3's p is larger) or lower. All three therefore pass at the 0.05 level regardless of H3. For H1 this is
marginal and carries the caveats of `results_bootstrap_v1.md`: the sign changes between seeds, the
seeds-plus-patients interval includes zero, and the predicted decline of the advantage with budget is absent.

## 6. D30 check: C1 fine-tuned with max lr 1e-3 (exploratory)

Seeds 0-2. Fold-10 macro-AUROC, 71 statements:

| Budget | C1, lr 1e-3 | C1, lr 1e-2 (grid) | S1, lr 1e-2 (grid) | lr 1e-3 minus grid C1 |
|---|---|---|---|---|
| 5% | 0.752, 0.768, 0.780 | 0.811, 0.802, 0.826 | 0.789, 0.814, 0.800 | -0.046 |
| 100% | 0.902, 0.907, 0.903 | 0.918, 0.925, 0.922 | 0.927, 0.926, 0.923 | -0.018 |

SPH E3 macro-AUROC changes by -0.002 (5%) and -0.003 (100%).

- The lower learning rate makes C1 worse at both budgets, in all three seeds. The concern recorded in D30, that a
  peak rate of 1e-2 overwrites pretrained features and hides an SSL benefit, is not supported by this check.
- What the check cannot show: with 1e-3 the model also learns more slowly throughout (validation 0.62 against
  0.83 at the fifth evaluation point at 100%; training loss higher at every point), so the new classification head
  is probably under-trained in the same number of steps. A recipe with separate rates for encoder and head would
  test the D30 concern more sharply. It was not pre-specified and is not run.
- There is no S1 run at 1e-3, so the check says nothing about C1 - S1 at that rate.
- As pre-specified, these runs do not replace the primary C1 results.

## 7. Caveats

1. The H2a statistic averages nine conditions with equal weight (D35); one corruption supplies the whole effect.
   The per-corruption table must accompany the headline number wherever it is reported.
2. The explanation of observation 2 (why P_ecg augmentation raises sensitivity to reversal) is not known and is
   not claimed. Per-lead amplitude scaling is the only P_ecg transform that treats leads differently, which makes
   it the first candidate to examine; that would be a new, exploratory analysis.
3. Corruptions are synthetic models of acquisition faults (protocol section 6).
4. Patient intervals describe test-set sampling error for these trained models. With seeds resampled as well, the
   H2a interval is -0.0066 to -0.0003.
5. H3 and the sensitivity analyses are still open; no hypothesis is declared supported or rejected in the paper
   until the family of four is complete.
