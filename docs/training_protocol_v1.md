# Training protocol v1

**Status:** v1, 2026-09-26, before any real training run. Implemented in `scripts/train.py`.
Changes need v2 and a decision-log entry.

## Model

xresnet1d50 as in the PTB-XL benchmark (Strodthoff et al., `code/models/xresnet1d.py`): kernel size 5,
stem (32, 32, 64), bottleneck stages [3, 4, 6, 3] with **constant width 64** (expansion 4), concat-pooling
head with one hidden layer of 128 units, dropout 0.25/0.5. Our encoder was checked against the reference
code: same 886,400 parameters, same 156 tensor shapes, identical outputs with shared weights.
Total 962,503 parameters with 71 outputs.

## Optimisation (benchmark defaults unless stated)

| Setting | Value | Source |
|---|---|---|
| Loss | binary cross-entropy over all 71 statements | benchmark |
| Optimiser | AdamW, weight decay 1e-2 | benchmark (`wd=1e-2`) |
| Schedule | one-cycle, max LR 1e-2 | benchmark (`fit_one_cycle(50, 1e-2)`) |
| Batch size | 128 | benchmark |
| Length | **max(50 epochs, 2,000 steps)** | ours: 50 epochs at 5% budget is only ~350 steps |
| Input | random 2.5 s crop per record per step | benchmark |
| Precision | FP16 autocast + gradient scaling on GPU | ours (T4) |

The benchmark used the fastai learning-rate finder; we fix LR = 1e-2 for every arm and budget so that the
arms differ only in initialisation and augmentation.

## Model selection and evaluation

- Validation on **all of PTB-XL fold 9** at 50 evenly spaced points per run; the checkpoint with the best
  validation macro-AUROC is used for testing. The same fold-9 set is used at every budget and for every arm.
  *Limitation:* at small budgets the validation set is larger than the training set, which is unrealistic
  for a label-scarce setting; it affects all arms equally.
- Test-time aggregation: mean of sigmoid outputs over 7 windows (2.5 s, stride 1.25 s). The benchmark
  default is the max over windows; our rule was fixed in preprocessing v1 and is kept.
- Reported per run: fold-10 macro-AUROC over all statements with positives; E3 macro-AUROC on fold 10 and
  SPH, E3 degradation, per-label E3 AUROCs; predictions saved for patient-level bootstrap.
- Runs whose best checkpoint is the first evaluation point are flagged (A8/A9 lesson, descriptive only).

## Reproducibility

- Seeds: the budget subset, batch order and initialisation all derive from `--seed`.
  Epoch *e* uses a permutation seeded by (seed, e).
- Resume: checkpoints at each evaluation point (model, optimiser, scheduler, scaler, step, log).
  A resumed run replays the same batch order; random crops and augmentations are re-drawn, so a
  resumed run is statistically but not bitwise identical to an uninterrupted one.
- Each finished run writes `config.json`, `log.csv`, `done.json`, `predictions.npz` and one row of
  `registry.csv`.

## Sanity check before any SSL work

S0 at 100% labels, seed 0. Target: fold-10 macro-AUROC over all statements near the published
xresnet1d50/xresnet1d101 values (≈ 0.92–0.93). Exact agreement is not expected (PTB-XL v1.0.3 vs earlier
release, D22; fixed LR; mean vs max aggregation). A result below ~0.90 means the pipeline must be debugged
before continuing.
