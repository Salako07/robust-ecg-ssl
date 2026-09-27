# SSL pretraining protocol v1

Status: frozen before the first real pretraining run (2026-09-27). Code: `scripts/pretrain_simclr.py`,
`robust_ecg.models.SimCLRNet` / `nt_xent`, `robust_ecg.data.PairCrops`, `robust_ecg.augment`.
Decision record: D29 (and D30 for fine-tuning of the SSL arms).

## 1. What is pretrained

| Item | Setting | Source / reason |
|---|---|---|
| Method | SimCLR: two views per record, NT-Xent loss, in-batch negatives | RQ v2 §3 (as in A1, A3, A8); M1 (Chen et al. 2020, method reference) |
| Data | PTB-XL folds 1–8, **labels unused**; folds 9–10 and SPH never seen | RQ v2 §3; prevents test leakage and keeps C1 vs S1 at 100% a pure objective contrast |
| Encoder | the xresnet1d50 encoder of the fine-tuning model (886,400 parameters) | identical weights are transferred; a unit test checks `load_state_dict(strict=True)` |
| Projector | concat pooling (512) → Linear 512 → BN → ReLU → Linear 128 | SimCLR-style two-layer MLP; discarded after pretraining |
| Views | two independent random 2.5 s crops of the same 10 s record, then the augmentation policy applied independently to each | same crop length as fine-tuning (preprocessing_v1) |
| Augmentation | policy P at strength s, each transform applied with p = 0.5, per sample | the **same** `Augment` object used by S1/C1 fine-tuning (RQ v2 §5) |
| Temperature τ | 0.1 | not tuned (no labels may be used for selection) |
| Batch size | 512 pairs (1,022 negatives per anchor) | T4 memory and time budget (RQ v2 §7) |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4 | common SimCLR-for-ECG practice; not tuned |
| Schedule | linear warm-up 10 epochs, then cosine decay to 0, stepped per iteration | |
| Length | 300 epochs, `drop_last` (≈ 34 steps/epoch, ≈ 10,200 steps; exact numbers in each `config.json`) | RQ v2 §7 |
| Precision | FP16 autocast for the network; loss in float32; augmentation in float32 before autocast | |
| Checkpoint used downstream | **final epoch** (`encoder.pt`) | no labelled data is used to select an SSL checkpoint |

Core runs: `SSL_ecg-mid_s{0..4}` (P_ecg, mid), one per seed; fine-tuning seed k uses the encoder of SSL seed k.
H3 runs: P_ecg and P_gen at low/mid/high (RQ v2 §4, H3).

## 2. Diagnostics logged per epoch

Mean NT-Xent loss, top-1 positive-retrieval accuracy within the batch (chance 1/1023), learning rate, minutes.
These are diagnostics only. They are not used to choose between runs or settings.

## 3. Reproducibility

- `--seed` sets the Python, NumPy and torch RNGs. Epoch *e* uses a DataLoader generator seeded by (seed, e).
- The checkpoint (every 5 epochs, atomic write) stores the model, optimiser, scheduler, grad-scaler and
  **all RNG states**. On CPU a killed-and-resumed run reproduces the uninterrupted run's losses exactly (tested
  2026-09-27). On GPU, cuDNN non-determinism means runs are not bitwise reproducible either way.
- Each fine-tuning run of an SSL arm records the SHA-256 of the encoder file it loaded (`encoder_sha256` in
  `config.json`), so every C0/C1 result can be traced to one pretraining run.
- A finished pretraining run writes `config.json`, `log.csv`, `done.json`, `encoder.pt` and a row in
  `registry_ssl.csv`. `--max-steps N` is a timing mode that writes nothing.

## 4. Known limitations (for §7 of the paper)

1. **Batch size.** Chen et al. (2020, M1) state that contrastive learning "benefits from larger batch sizes and more
   training steps compared to supervised learning" (abstract, verified). Our 512 is a hardware constraint. It weakens the SSL arms
   in absolute terms but applies equally to every SSL arm, so the H3 comparison is unaffected. For H1/H2 it is a
   reason the result may understate what SSL could do with more compute. Specific numbers from their
   batch-size experiments are not cited until the full text has been read.
2. **Untuned τ, lr and length.** Tuning them without labels is not possible, and tuning them with fold-9 labels would
   hand the SSL arms a selection advantage that S0/S1 do not get.
3. **Weak views.** The views differ only by crop position and the P augmentations (each applied with p = 0.5).
   P_ecg is mild by SimCLR standards, so the pretext task may be easy; the retrieval accuracy will show this. It is
   a consequence of the design constraint that pretraining and supervised arms share one policy. It is not a bug to fix.
4. **Single SSL method.** Results speak to SimCLR-style contrastive learning with this encoder, not to SSL in general
   (e.g. JEPA-style methods such as A7 do not use hand-crafted augmentations).

## 5. Timing

First run (`SSL_ecg-mid_s0`, Colab T4, torch 2.11.0+cu128): 300 epochs × 34 steps = 10,200 steps in **23.6 min**
in one session (no resume). The RQ v2 §7 estimate of 2–3 h per run was about 6× too high. All 11 planned
pretraining runs (5 core + 6 H3) now fit in ≈ 4.5 GPU-hours.

## 6. Observation from the first run: the pretext task saturates

In-batch retrieval top-1 reached 0.99 at epoch 9 and 0.999 at epoch 26 (chance ≈ 0.001); NT-Xent loss fell from 3.63
(epoch 1) to 0.20 (epoch 20) and only to 0.094 by epoch 300. For 90% of training almost every positive is already
ranked first. This is the risk anticipated in §4 item 3; see D31 for what is and is not concluded from it.
