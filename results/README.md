# Results

| File | Produced by | Contents |
|---|---|---|
| `e3_label_support_lik0.csv` / `e3_lik0.log` | `scripts/e3_label_support.py`, threshold 0 | per label × split × budget × seed: records, positive records, positive patients, flag if < 10 positives |
| `e3_label_support_lik50diag.csv` / `e3_lik50diag.log` | same, `--min-likelihood 50` on diagnostic statements | sensitivity analysis (decision D11) |
| `sph_duplicate_groups.csv`, `sph_exclude.txt` | `scripts/sph_dedup.py` | identical-signal groups in SPH (168 pairs) and the 193 ECG_IDs excluded (D14) |
| `runs/<run_id>/` | `scripts/train.py` on Colab, exported by notebook 01 | `config.json`, `log.csv`, `done.json`, `predictions.npz` (float16) per finished run |
| `runs/registry.csv` | `scripts/train.py` | one row per finished run; the table all results are computed from |

Seeds 0, 1, 2. Budgets are nested and patient-level within PTB-XL folds 1–8.
Logs are the console output of the run that produced the matching CSV.
