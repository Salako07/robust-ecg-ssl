# H3 protocol v1: ECG-specific versus generic pretraining augmentation

**Status: v1, frozen 2026-10-07 (decision D37).** The five judgements were approved by the author as proposed. One
amendment was made before freezing, to the linear-probe recipe (section 4, judgement 3). No P_gen encoder had
been pretrained and no H3 run existed when it was frozen. Changes need v2 and a new decision-log entry.

Code: `src/robust_ecg/h3.py`, `scripts/linear_probe.py`, `scripts/run_h3.py`, `scripts/bootstrap_h3.py`,
`tests/test_h3.py`, notebook 05.

Fixes what RQ v2 left open for H3 (§4, §5 "strength matching", §6 primary contrast 4): how strength is matched,
how one level per policy is chosen, how the H3 runs are fine-tuned, how many seeds, and how the two outcomes
combine into one primary contrast.

## 1. Question

Does an encoder pretrained with the ECG-specific policy P_ecg degrade less than one pretrained with the generic
policy P_gen, under held-out corruptions and on SPH, when the two policies are matched for strength?

## 2. What is compared

Only the pretraining augmentation policy differs between the two H3 arms. Everything downstream is the core C1
configuration:

| Step | P_ecg arm | P_gen arm |
|---|---|---|
| SimCLR pretraining (SSL protocol v1, unchanged) | views augmented with P_ecg at level L_ecg | views augmented with P_gen at level L_gen |
| Fine-tuning (training protocol v1, 100% budget) | P_ecg at mid, as in C1 | **the same: P_ecg at mid** |
| Evaluation | fold 10 clean, held-out corruption suite (protocol v1), SPH | the same |

**Judgement 1: fine-tuning augmentation is held fixed at P_ecg-mid for both arms.** RQ v2 says H3 is "evaluated
for the C1 configuration only" and asks about "the augmentation policy used for pretraining". H2a showed that the
fine-tuning augmentation by itself changes robustness substantially (D36). If the P_gen encoder were fine-tuned
with P_gen, a difference between the arms could come from either stage. *Alternative:* each encoder fine-tuned
with its own pretraining policy. It answers a different question (which policy is better end to end) and is not
proposed.

## 3. Strength levels and matching

Levels low, mid and high multiply every magnitude of a policy by 0.5, 1.0 and 1.5 (`augment.STRENGTH`, in the
code since before any SSL run). Each transform is applied with probability 0.5, as in the core design.

What "matched" means here, with the numbers measured from `augment.py` on synthetic input (20,000 windows):

| Component | P_ecg | P_gen | Low | Mid | High |
|---|---|---|---|---|---|
| Additive noise power, in units of signal variance | baseline wander + muscle bursts | stationary Gaussian noise | 0.0064 vs 0.0075 | 0.026 vs 0.031 | 0.058 vs 0.068 |
| Amplitude scaling power | per lead | global | 0.0075 both | 0.030 both | 0.067 both |
| Time masking, mean share of the window zeroed | none | one segment, all leads | 6% | 12% | 19% |
| Time warping, local speed change | none | up to | ±5% | ±10% | ±15% |

- The additive components are matched on signal-to-noise ratio to within 0.7 dB at every level (Gaussian noise has
  1.17 times the power of wander plus bursts). The scaling components are equal by construction.
- **Judgement 2: masking and warping have no counterpart in P_ecg and are not compensated.** They scale with the
  level like everything else, which is the reading of RQ v2 §5 ("matched on the fraction of samples altered")
  that needs no further tuning. P_gen therefore changes its views more than P_ecg at every level. This is a
  property of the two policies, stated as a limitation; the full grid (section 5) shows the effect of strength
  within each policy so the reader can judge how much it matters. *Alternative:* reduce P_gen until a total
  distortion measure equals P_ecg's. Any such measure is dominated by masking and warping, and equalising it would
  leave P_gen with almost no noise, defeating the comparison of types.
- A diagnostic, not used for any decision: the mean squared difference between augmented and unaugmented training
  windows, per policy and level, is computed on PTB-XL folds 1-8 and reported.

## 4. Choosing one level per policy

RQ v2 §5: "the level used for each policy is selected on fold 9 linear-probe macro-AUROC".

1. Pretrain one encoder per policy and level with seed 0 (six encoders; `SSL_ecg-mid_s0` already exists).
2. Linear probe for each: the encoder is frozen; the feature of a recording is the concatenated average- and
   max-pooled encoder output, averaged over the seven evaluation windows; features are standardised with
   training-fold statistics; one linear layer over the 71 statements is trained on folds 1-8 with binary
   cross-entropy (AdamW, learning rate 1e-2, no weight decay, full batch, 1,000 steps, no early stopping, seed 0).
3. The level with the highest fold-9 macro-AUROC is selected, separately for each policy. Ties (difference below
   0.0005) go to the lower level.

Selection uses fold 9 only. Fold 10, SPH and the corruption suite play no part in it, and both policies get the
same selection step. All six probe scores are reported.

**Judgement 3: the probe settings in step 2.** RQ v2 names the linear probe but not its recipe.

*Amendment before freezing.* The recipe first proposed and approved was learning rate 1e-3 for 500 steps. A unit
test on synthetic 512-dimensional features showed that it stops far short of convergence (AUROC 0.83 where a
converged logistic regression reaches 0.97), which would make the selection depend on an under-trained probe.
Learning rate 1e-2 for 1,000 steps reaches the converged value on the same synthetic task (0.96). The change was
made on synthetic data only, before any encoder of the H3 grid existed, and applies equally to both policies.

## 5. Runs

| Stage | Runs | New | Estimated T4 time |
|---|---|---|---|
| A. Seed-0 encoders, all six policy-levels | 6 | 5 | 1.8 h |
| B. Linear probes and selection | 6 | 6 | minutes |
| C. Seeds 1-4 of the two selected policy-levels | 8 | 4 to 8 | 1.4 to 2.8 h |
| D. Fine-tuning at 100%: full 3 x 2 grid at seed 0, selected levels at seeds 0-4 | 14 | up to 13 | up to 1.3 h |
| E. Corruption suite and SPH on every stage-D run | 14 | up to 13 | minutes |

About 4.5 to 6 GPU-hours, so roughly 3 hours on Kaggle's two T4s, in one saved version. If L_ecg is mid, its five
encoders and five fine-tuned runs already exist as the core C1 runs and are reused unchanged.

- Primary contrast: the two selected policy-levels, five seeds each.
- **Judgement 4: the full 3 x 2 grid is run at seed 0 only** and reported descriptively. RQ v2 planned six
  pretraining runs for the grid; five seeds for all six cells would triple the compute.

## 6. Primary contrast and inference

Two outcomes, both at the 100% budget, both as P_ecg arm minus P_gen arm, mean over five seeds (negative =
P_ecg encoder more robust):

- **H3-corr:** difference in mean degradation over the nine held-out corruption conditions, fold 10, 71
  statements (the H2a statistic of D35).
- **H3-sph:** difference in degradation from fold 10 to SPH on the nine E3 labels (the H2b statistic).

Each is tested with the patient bootstrap of D34 (10,000 resamples, percentile interval, two-sided p).

**Judgement 5: H3 is a conjunction.** The hypothesis says the P_ecg encoder shows smaller degradation under
held-out corruptions *and* on SPH. The one p-value H3 contributes to the Holm family of four is therefore the
larger of the two (an intersection-union test): H3 is supported only if both differences are negative and both
are significant. Both components are always reported. *Alternative:* count H3 as supported if either outcome is
significant, with a Bonferroni factor of two; that tests a weaker claim than the one pre-specified.

Secondary, without significance claims: the seed-0 grid; clean fold-10 and SPH scores; per-corruption and
per-label differences; the probe scores; retrieval accuracy and loss of each pretraining run.

## 7. Known limitations

1. P_gen distorts views more than P_ecg at every level (section 3). A P_gen advantage could reflect strength, and
   the seed-0 grid is the only evidence on that.
2. The core SimCLR runs saturate early (D31). If stronger views simply make the pretext task harder, level may
   matter more than policy type.
3. Selection by clean linear-probe score does not select for robustness; a level that is best on clean data need
   not be the most robust.
4. Seed 0 alone decides the selected level.
5. Fine-tuning with P_ecg for both arms (judgement 1) means the P_ecg arm's pretraining and fine-tuning policies
   coincide while the P_gen arm's do not.
6. The protocol is written after H1, H2a and H2b were analysed (D34, D36), and judgement 1 is informed by the H2a
   result. No H3 run existed when it was written.
