# robust-ecg-ssl

Does self-supervised pretraining improve 12-lead ECG classification under scarce labels and
distribution shift, **once supervised baselines receive the same augmentations**?

This is a master's research portfolio project: a working paper plus a reproducible codebase.
It does not aim to set state-of-the-art results.

## Research question (v2)

When supervised and contrastively pretrained ECG models receive the same augmentation policy, does
self-supervised pretraining still improve (a) performance under limited labels, (b) robustness to
signal corruptions no model saw during training, and (c) performance on an external hospital dataset
without adaptation? Hypotheses, the 2 × 2 design and primary contrasts:
[`docs/research_question_v2.md`](docs/research_question_v2.md).

## What is decided

- **E3 (cross-dataset, zero adaptation):** train on PTB-XL, evaluate without adaptation on SPH
  (Shandong Provincial Hospital) on 9 shared labels. Full spec: [`docs/E3_protocol_v1.md`](docs/E3_protocol_v1.md).
- Every design decision, with its evidence: [`docs/decisions/log.md`](docs/decisions/log.md).
- Literature matrix (source of truth for the review): [`docs/literature_matrix.md`](docs/literature_matrix.md).
- Lab notebook (chronological record, including failures): [`docs/lab_notebook.md`](docs/lab_notebook.md).
- Paper outline, each section mapped to its evidence: [`docs/paper_outline.md`](docs/paper_outline.md).
- Original plan (superseded, kept for the record): [`docs/plan_v0.md`](docs/plan_v0.md).

## Data (not in this repository)

| Dataset | Source | Place files in |
|---|---|---|
| PTB-XL v1.0.3 | https://physionet.org/content/ptb-xl/1.0.3/ | `data/ptbxl/` (`ptbxl_database.csv`, `scp_statements.csv`, records) |
| SPH | https://doi.org/10.6084/m9.figshare.c.5779802 | `data/sph/metadata.csv`, `data/sph/records/*.h5` |

Both are CC BY 4.0. Cite the original papers when using them.

## Running on Google Colab

Experiments run on a single T4 in Colab. Caches, checkpoints and results live on Google Drive
(`MyDrive/robust-ecg-ssl/`), so an interrupted session can resume.

1. Open `notebooks/00_colab_setup.ipynb` in Colab and run all cells once. It downloads PTB-XL and SPH,
   deduplicates SPH and builds the preprocessed caches ([`docs/preprocessing_v1.md`](docs/preprocessing_v1.md)).
2. `notebooks/01_baseline_sanity.ipynb` (T4): supervised baseline S0 at 100% labels, the check against
   published PTB-XL results ([`docs/training_protocol_v1.md`](docs/training_protocol_v1.md)).
3. `notebooks/02_ssl_pretrain.ipynb` (T4): SimCLR pretraining of the encoder on PTB-XL folds 1–8 without labels
   ([`docs/ssl_protocol_v1.md`](docs/ssl_protocol_v1.md)).
4. `notebooks/03_core_grid.ipynb` (T4): the core grid (4 arms × 5 budgets × 5 seeds, plus the exploratory 1%), run as
   a resumable queue seed by seed.

Never use Colab's *Save a copy in GitHub*: it overwrites the repo's notebooks.

Tests (CPU, synthetic data): `python -m pytest -q tests`.

## Reproduce the current results

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# E3 label support (Gate 2), primary threshold and sensitivity threshold
python scripts/e3_label_support.py --ptbxl data/ptbxl/ptbxl_database.csv \
    --sph data/sph/metadata.csv --seeds 0 1 2 --out results/e3_label_support_lik0.csv
python scripts/e3_label_support.py --ptbxl data/ptbxl/ptbxl_database.csv \
    --scp-statements data/ptbxl/scp_statements.csv --min-likelihood 50 \
    --sph data/sph/metadata.csv --seeds 0 1 2 --out results/e3_label_support_lik50diag.csv

# SPH deduplication (Gate 3)
python scripts/sph_dedup.py --sph-dir data/sph/records --meta data/sph/metadata.csv --out-dir results
```

See [`results/README.md`](results/README.md) for what each output file contains.

## Status

| Stage | State |
|---|---|
| Literature review and gap | done; RQ v2 frozen |
| E3 label harmonisation and support | done |
| SPH deduplication | done: 193 excluded, 25,577 records |
| Signal preprocessing pipeline | specified and tested; caches built on Colab (notebook 00) |
| Supervised baseline | sanity check passed: fold-10 macro-AUROC 0.9228 (published 0.9242) |
| SSL pretraining | first run done (23.6 min on a T4); pretext task saturates early (D31) |
| Core grid (4 arms × 6 budgets × 5 seeds) | done on Kaggle; seed-level results in [`docs/results_core_grid_v1.md`](docs/results_core_grid_v1.md) |
| Patient-level bootstrap, corruptions (H2a), H3 grid | not started |

## License

Code: MIT (see `LICENSE`). Datasets keep their own licenses.
