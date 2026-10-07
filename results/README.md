# Results

| File | Produced by | Contents |
|---|---|---|
| `e3_label_support_lik0.csv` / `e3_lik0.log` | `scripts/e3_label_support.py`, threshold 0 | per label × split × budget × seed: records, positive records, positive patients, flag if < 10 positives |
| `e3_label_support_lik50diag.csv` / `e3_lik50diag.log` | same, `--min-likelihood 50` on diagnostic statements | sensitivity analysis (decision D11) |
| `sph_duplicate_groups.csv`, `sph_exclude.txt` | `scripts/sph_dedup.py` | identical-signal groups in SPH (168 pairs) and the 193 ECG_IDs excluded (D14) |
| `runs/<run_id>/` | `scripts/train.py` on Colab, exported by notebook 01 | `config.json`, `log.csv`, `done.json`, `predictions.npz` (float16) per finished run |
| `ssl/<run_id>/` | `scripts/pretrain_simclr.py` on Colab, exported by notebook 02 | `config.json`, `log.csv` (loss, retrieval top-1, lr per epoch), `done.json`; `encoder.pt` is kept on Drive, its SHA-256 is recorded in `done.json` and in every C0/C1 `config.json` |
| `ssl/registry_ssl.csv` | `scripts/pretrain_simclr.py` | one row per finished pretraining run |
| `runs/registry.csv` | `scripts/train.py` | one row per finished run; the table all results are computed from |

Seeds 0, 1, 2. Budgets are nested and patient-level within PTB-XL folds 1–8.
Logs are the console output of the run that produced the matching CSV.

## Platforms (D32)

- `pilot_colab/`: the first runs, made on Colab (T4) before the grid moved to Kaggle: S0/S1/C1 at 100% seed 0 and
  `SSL_ecg-mid_s0`. They are a **pilot**, not part of the analysed grid. The Kaggle grid reruns the same
  configurations, and the two copies are compared as a cross-platform reproducibility check.
- `runs/`, `ssl/`: the analysed grid, all from one platform (Kaggle, 2 × T4). Each `config.json` records `platform`,
  `gpu`, `torch` and `cudnn`. `cache_manifest.json` gives the SHA-256 of every cache file the grid read.
- `logs/`: console output of every grid run.
- `pilot_colab/02_ssl_pretrain_executed.ipynb`: notebook 02 as executed on Colab (outputs only; its code cells are a stale copy, see lab notebook Failure 6).
