# Research question v2, hypotheses and design

**Status:** v2, **frozen 2026-09-26**. Changes require v3 and a decision-log entry.
Supersedes the research question and hypotheses in [`plan_v0.md`](plan_v0.md) §2–3.
Evidence for each claim below is in [`literature_matrix.md`](literature_matrix.md) (IDs A1–B3).

## 1. Gap

Contrastive self-supervised pretraining has been reported to improve 12-lead ECG classification,
label efficiency and robustness to physiological noise (A1), and to transfer across ECG datasets
when fine-tuned on target labels (A3). In these studies the supervised baselines are not trained
with the augmentations the SSL encoder saw during pretraining (A1, A3, A7), the robustness tests
use noise types that were part of the pretraining augmentations (A1), and cross-dataset evaluation
fine-tunes on the target dataset (A3). Evidence from a large private wearable dataset suggests that
augmentation alone accounts for most of the gain at full label budget (A5); an extreme few-shot study
on PTB-XL does not separate augmentation from access to unlabelled data (A8). In natural images,
contrastive learning remained more robust than supervised learning to test-time corruptions under
identical augmentation (B2); whether this holds for ECG has not been tested.

No reviewed ECG study combines (i) an augmentation-matched supervised control, (ii) a controlled
label-budget curve, (iii) corruptions held out from every augmentation policy, and (iv) evaluation
on an external hospital dataset without adaptation.

## 2. Research question

> When supervised and contrastively pretrained ECG models receive the same augmentation policy,
> does self-supervised pretraining still improve (a) diagnostic performance under limited labels,
> (b) robustness to signal corruptions that no model saw during training, and (c) performance on an
> external hospital dataset without adaptation?

Sub-question (H3): does the choice of augmentation policy used for pretraining (ECG-specific vs
generic, matched for strength) change the answers to (b) and (c)?

## 3. Design: 2 × 2 factorial core

| Arm | Initialisation | Augmentation during supervised training / fine-tuning | Role |
|---|---|---|---|
| **S0** | random | none | replicates the usual baseline in prior work |
| **S1** | random | policy P | **augmentation-matched supervised control** |
| **C0** | SSL (pretrained with P) | none | SSL as usually reported |
| **C1** | SSL (pretrained with P) | policy P | SSL, matched to S1 |

- The **primary contrast is C1 − S1**: identical architecture, data, budget, augmentation policy and
  fine-tuning schedule; the only difference is initialisation.
- S1 − S0 estimates the effect of augmentation alone; C0 − S0 reproduces the comparison in prior work.
  The factorial layout also gives the interaction (does SSL add anything once augmentation is present?).
- **Pretraining data:** SSL uses PTB-XL folds 1–8 **without labels** only. No external or test data
  enters pretraining. At the 100% budget C1 and S1 see exactly the same records, so their difference
  isolates the pretraining objective; below 100% it includes access to unlabelled records, which is
  the intended advantage of SSL and is reported as such.
- **Architecture:** xresnet1d50 for all arms (comparable with A1 and the PTB-XL benchmark).
- **SSL method:** SimCLR-style contrastive pretraining (as in A1, A3, A8).

## 4. Hypotheses (directional, pre-registered)

**H1 — Label efficiency.** C1 achieves higher macro-AUROC than S1 on PTB-XL fold 10, with the
difference largest at the smallest primary budget (5%) and decreasing as the budget grows.

**H2a — Held-out corruptions.** Under corruptions excluded from every augmentation policy (§5),
C1 loses less macro-AUROC relative to its clean score than S1 does.
Seen corruptions (those in P) are reported separately and are not used to test H2a.

**H2b — External dataset.** The macro-AUROC drop from PTB-XL fold 10 to SPH, on the E3 shared labels,
is smaller for C1 than for S1. Protocol: [`E3_protocol_v1.md`](E3_protocol_v1.md).

**H3 — Augmentation policy.** For SSL encoders pretrained with the ECG-specific policy P_ecg versus
the generic policy P_gen, matched for strength (§5), the P_ecg encoder shows smaller degradation
under held-out corruptions and on SPH. Evaluated for the C1 configuration only.

Degradation is defined as `AUROC_clean − AUROC_shifted` (absolute) and reported together with both
absolute scores, so that a model with a low clean score cannot appear robust by default.

A result against any hypothesis is reported with the same prominence as a result in favour.
In particular, C1 ≈ S1 would indicate that the SSL advantage reported in prior ECG work is largely
attributable to augmentation, which is itself a publishable finding.

## 5. Augmentation policies and corruption suite

| Transform | P_ecg | P_gen | Seen corruption for | Held-out corruption |
|---|:-:|:-:|---|:-:|
| Baseline wander (low-frequency sinusoids) | ✓ | | P_ecg | |
| Powerline interference (50/60 Hz) | ✓ | | P_ecg | |
| Muscle (EMG-like) noise | ✓ | | P_ecg | |
| Per-lead amplitude scaling, physiological range | ✓ | | P_ecg | |
| Additive Gaussian noise | | ✓ | P_gen | |
| Global amplitude scaling | | ✓ | P_gen | |
| Random time masking | | ✓ | P_gen | |
| Time warping | | ✓ | P_gen | |
| Random resized crop | ✓ | ✓ | both | |
| **Lead dropout (1–3 leads zeroed)** | | | | ✓ |
| **Limb-lead reversal (LA↔RA)** | | | | ✓ |
| **Baseline step shift (electrode motion)** | | | | ✓ |

- **Policy P for the core factorial (H1, H2) is P_ecg.** Decided before any results.
- **Strength matching for H3:** each policy is run at three strength levels chosen so that the mean
  per-lead SNR of augmented views is comparable across policies (masking and warping, which have no
  SNR, are matched on the fraction of samples altered). The level used for each policy is selected on
  fold 9 linear-probe macro-AUROC, and the full 3 × 2 grid is reported, so the comparison cannot be
  explained by one policy being tuned harder.
- **Corruption severity:** each corruption at three severities, applied to PTB-XL fold 10 only.
  Parameters are fixed in code before the first evaluation run.

## 6. Evaluation and inference

- **Budgets:** 5, 10, 25, 50, 100% (primary); 1% exploratory, per-label only (E3 protocol, D10).
- **Labels:** full PTB-XL statement set for training; E3 shared labels for H2b.
- **Headline metric:** macro-AUROC (0.5 for any constant predictor).
  Threshold-dependent metrics are reported next to a constant-predictor reference.
- **Primary contrasts (Holm-corrected as one family):**
  1. C1 − S1 at 5%, PTB-XL fold 10 (H1)
  2. C1 − S1 degradation, held-out corruptions, 100% (H2a)
  3. C1 − S1 degradation, SPH, 100% (H2b)
  4. P_ecg − P_gen degradation, held-out corruptions and SPH, 100% (H3)
  Everything else is secondary and reported without significance claims.
- **Uncertainty:** paired bootstrap over test **patients** for between-arm differences;
  seed-to-seed spread reported separately. Effect sizes with 95% intervals, not only p-values.
- **Training diagnostics:** per-seed results at every budget; runs whose best validation epoch is
  ≤ 1 are flagged (descriptive only, following the concern raised in A8/A9).

## 7. Compute budget

**Hardware:** one NVIDIA T4 (16 GB). All runs use mixed precision (FP16) and must checkpoint and
resume, since hosted T4 sessions are time-limited.

Fine-tuning runs for the core design = 4 arms × 5 budgets × 5 seeds = **100 runs**.

**Planning estimate (to be replaced by measured times after the first baseline runs).** Mehari &
Strodthoff (A1) report ~10 min to fine-tune xresnet1d50 on PTB-XL at 100 Hz on a V100. Assuming a T4
is 2–3× slower:

| Component | Runs | Est. T4 time each | Est. total |
|---|---:|---:|---:|
| Fine-tuning, core (time scales with budget: 5+10+25+50+100 = 1.9× a full run per arm-seed) | 20 arm-seeds | ≈ 1 h | ≈ 20 h |
| SSL pretraining, core (one per seed, PTB-XL folds 1–8, ≈ 300 epochs) | 5 | 2–3 h | 10–15 h |
| SSL pretraining, H3 grid (2 policies × 3 strengths) | 6 | 2–3 h | 12–18 h |
| H3 fine-tuning at 100% + all evaluation (corruptions, SPH) | — | — | ≈ 5 h |
| **Total** | | | **≈ 50–60 GPU-hours** |

- SimCLR batch size is limited by T4 memory; start at 512 and report the value used. Smaller batches
  than A1 (8192) mean fewer negatives, which is stated as a limitation, not tuned away.
- **Order of runs protects the primary contrasts:** S0 sanity check → S1 and C1 (all budgets, 5 seeds)
  → S0 and C0 → corruption and SPH evaluation → H3 grid last.
- Rule if compute is short: drop the 10% and 50% budgets before going below 5 seeds; H3 can be
  reduced to one strength level per policy, reported as a limitation.

## 8. What would change this design

- If the S0 baseline cannot reproduce published PTB-XL performance with xresnet1d50, stop and fix
  the pipeline before any SSL run.
- If SPH deduplication or signal harmonisation fails, H2b falls back to held-out corruptions only,
  and the gap statement loses component (iv).
