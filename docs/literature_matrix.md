# Literature matrix

Source of truth for the literature review. **Verification** says what was actually read:
*full text*, *abstract*, or *search excerpt only* (treat the last as unverified).
Columns H1/H2/H3 and "Aug-matched control" are our reading, not the authors' framing.

Last updated: 2026-09-26.

## A. ECG self-supervised learning

| # | Paper | Venue | Data | H1 label budgets | H2 shift | H3 aug type | Aug-matched supervised control | Verification |
|---|---|---|---|---|---|---|---|---|
| A1 | Mehari & Strodthoff, *Self-supervised representation learning from 12-lead ECG data* | Comput. Biol. Med. 141:105114 (2022); arXiv 2103.12676 | PTB-XL (+ CinC2020, Chapman, Ribeiro for pretraining) | 1–8 folds (≥ 12.5%) | Physiological test noise; same noise family used in "physio" pretraining | Physio vs artificial augs | **No** — supervised baselines use cropping only | full text |
| A2 | Soltanieh, Etemad & Hashemi, *Analysis of augmentations for contrastive ECG representation learning* | IJCNN 2022; arXiv 2206.07656 | PTB-XL | Partial | No | Strength sweeps, not type | No | abstract |
| A3 | Soltanieh, Hashemi & Etemad, *In-distribution and out-of-distribution self-supervised ECG representation learning for arrhythmia detection* | IEEE JBHI 28(2):789–800 (2024); arXiv 2304.06427 | PTB-XL, Chapman, Ribeiro | No | "OOD" = different **pretraining** set; always fine-tuned on target labels. 60 Hz noise test | No | No; reports best F1 across augmentation settings | full text |
| A4 | Gopal et al., *3KG: Contrastive learning of 12-lead ECGs using physiologically-inspired augmentations* | ML4H, PMLR 158:156–167 (2021) | PhysioNet 2020 | Includes 1% | No | VCG augs vs SSL baselines | No | abstract |
| A5 | Lai et al., *Practical intelligent diagnostic algorithm for wearable 12-lead ECG via self-supervised learning on large-scale dataset* | Nat. Commun. 14:3741 (2023) | Private wearable, 658k ECGs | 10–100% | Quality-stratified test; CPSC2018 with retraining | No | **Yes, full labels only**: AUPRC 0.582 / 0.593 (+PW) / 0.637 (+Aug) / 0.646 (+Aug+PW); Aug+PW not significantly better than Aug | full text |
| A6 | Dade et al., *Self-supervised contrastive learning enables robust electrocardiogram-based cardiac classification* | Heart Rhythm O2 7(4):757–770 (2026) | Private, ~1M ECGs; binary LVEF, KCl | 1–100%, 5 seeds | No | No | No | full text |
| A7 | Weimann & Conrad, *Self-supervised pre-training with joint-embedding predictive architecture boosts ECG classification performance* | Comput. Biol. Med. (2025); arXiv 2410.13867 | 10 public DBs, >1M records; PTB-XL eval | **No** | **No** external evaluation | N/A (JEPA avoids hand-crafted augs) | **No** — baselines are random init or other SSL | full text (v1) |
| A8 | Zeng et al., *Stabilizing extreme few-shot ECG classification via self-supervised contrastive pretraining* | Ann. Noninvasive Electrocardiol. (2026), doi 10.1111/anec.70188 | Unverified | N = 70 (extreme few-shot) | Unverified | No | **Partial**: "Aug Only" group vs "SSL, NoAug"; primary outcome is training collapse, not accuracy | **search excerpt only** |
| A9 | Ahmed et al., *Critical appraisal of self-supervised contrastive pretraining for extreme few-shot ECG classification* | Ann. Noninvasive Electrocardiol. (2026), doi 10.1111/anec.70216 | Commentary on A8 | — | — | — | — | **title only; not read** |

## B. Outside ECG: does SSL help once augmentation is controlled?

| # | Paper | Venue | Setting | What it controls | Finding relevant to us | Verification |
|---|---|---|---|---|---|---|
| B1 | Hendrycks, Mazeika, Kadavath & Song, *Using self-supervised learning can improve model robustness and uncertainty* | NeurIPS 2019 | Images; SSL as auxiliary loss | SSL added to supervised training | SSL improves robustness to adversarial examples, label corruption, common corruptions, and near-distribution OOD detection | abstract |
| B2 | Zhong et al., *Is self-supervised learning more robust than supervised learning?* | arXiv 2206.05259 (2022) | Images; contrastive vs supervised | **Same data augmentation across methods** | Contrastive learning more robust than supervised to **downstream (test-time)** corruptions; under **pre-training** corruption the picture reverses for pixel/patch corruptions. Attributed to feature uniformity; uniformity regularization improves supervised robustness | full text (HTML); author list not fully verified |
| B3 | Liu et al., *An empirical study on distribution shift robustness from the perspective of pre-training and data augmentation* | NeurIPS 2022 Workshop on Distribution Shifts | 7 pretrained models × 5 shift datasets × 5 algorithms | Pretraining type and augmentation, jointly | ERM plus data augmentation is competitive when the pretrained model is chosen well | abstract |

## Implications for the gap (current reading)

1. **The core control is not new in general ML.** B2 already compares contrastive and supervised learning under identical augmentation and finds a downstream-robustness advantage for contrastive learning. Our contribution cannot be "first augmentation-matched comparison"; it has to be the **ECG-specific combination**: augmentation-matched supervision × label budgets × held-out ECG corruptions × zero-adaptation transfer to an external hospital.
2. **B2 gives us a testable, pre-registrable prediction:** if B2 transfers to ECG, SSL should keep an advantage on held-out test-time corruptions even against augmentation-matched supervision. A5 (full labels, private data) points the other way for clean performance. The disagreement is itself a reason to run the experiment.
3. **Within ECG, no reviewed study combines the augmentation-matched control with label budgets or external-dataset evaluation.** A5 has the control at full labels only; A8 appears to have a related control only at N = 70 with a stability outcome (unverified); A7 has neither control nor external evaluation.
4. **Still to check before freezing the gap:** read A8 and A9 in full (blocked from our environment; download manually); search time-series SSL benchmarks for augmentation-matched baselines (first search found nothing directly on point).
