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
- Original plan (superseded, kept for the record): [`docs/plan_v0.md`](docs/plan_v0.md).

## Data (not in this repository)

| Dataset | Source | Place files in |
|---|---|---|
| PTB-XL v1.0.3 | https://physionet.org/content/ptb-xl/1.0.3/ | `data/ptbxl/` (`ptbxl_database.csv`, `scp_statements.csv`, records) |
| SPH | https://doi.org/10.6084/m9.figshare.c.5779802 | `data/sph/metadata.csv`, `data/sph/records/*.h5` |

Both are CC BY 4.0. Cite the original papers when using them.

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
| Literature review and gap | done; RQ v2 drafted, frozen once compute is confirmed |
| E3 label harmonisation and support | done |
| SPH deduplication | script ready, not yet run |
| Signal preprocessing pipeline | not started |
| Baselines, SSL, experiments | not started |

## License

Code: MIT (see `LICENSE`). Datasets keep their own licenses.
