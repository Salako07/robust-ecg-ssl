# Paper outline (working)

Working title: *Does self-supervised pretraining still help ECG classification once supervised baselines see the
same augmentations? A controlled study of label efficiency, held-out corruptions and cross-hospital transfer.*

Each section lists the source documents it is written from, so every claim in the paper can be traced.
Status: ✅ evidence ready · 🟡 partial · ⬜ pending experiments.

| § | Section | Content | Sources | Status |
|---|---|---|---|---|
| 1 | Introduction | Label scarcity in ECG; SSL claims; the augmentation confound; contribution | literature_matrix (A1, A3, A5, B2); RQ v2 §1–2 | ✅ |
| 2 | Related work | ECG SSL (A1–A8); augmentation-matched controls outside ECG (B1–B3); cross-dataset ECG evaluation | literature_matrix | ✅ |
| 3 | Research question and hypotheses | RQ, H1–H3, 2 × 2 design, primary contrasts | research_question_v2 | ✅ |
| 4.1 | Data | PTB-XL v1.0.3 (21,799 records); SPH (25,770 → 25,577 after dedup); label provenance | E3_protocol_v1; D13, D14, D22 | ✅ |
| 4.2 | Label harmonisation | AHA crosswalk; 9 shared labels; excluded concepts and why | E3_protocol_v1; D3–D8, D12 | ✅ |
| 4.3 | Preprocessing | 500 Hz source, band-pass, 100 Hz, normalisation, windows | preprocessing_v1; D20, D21 | ✅ |
| 4.4 | Model and training | xresnet1d50 (verified), hyperparameters, budgets, selection | training_protocol_v1; D10, D24 | ✅ |
| 4.5 | SSL pretraining | SimCLR on folds 1–8, batch size, epochs | *to write* | ⬜ |
| 4.6 | Augmentations and corruptions | P_ecg, P_gen, strength matching, held-out corruptions | RQ v2 §5; augment.py; D16, D17, D23 | 🟡 corruptions not implemented |
| 4.7 | Evaluation and statistics | macro-AUROC, degradation, patient bootstrap, Holm | RQ v2 §6; E3 protocol | 🟡 bootstrap not implemented |
| 5.1 | Sanity check | S0 0.9228 vs 0.9242 published | D27 | ✅ |
| 5.2 | H1 label efficiency | C1 − S1 across budgets | runs | ⬜ |
| 5.3 | H2a held-out corruptions | degradation C1 vs S1 | runs | ⬜ |
| 5.4 | H2b external dataset | E3 degradation C1 vs S1; per-label; criteria vs interpretive | runs; D27 ceiling note | ⬜ |
| 5.5 | H3 augmentation policy | P_ecg vs P_gen grid | runs | ⬜ |
| 5.6 | Sensitivity analyses | likelihood ≥ 50; amplitude outliers > 20 mV; full-length SPH | D11, D26, D21 | ⬜ |
| 6 | Discussion | interpretation separated from observation | — | ⬜ |
| 7 | Limitations | SPH label noise (25 conflicting duplicates); IRBBB criteria difference; fold-9 larger than small budgets; small SimCLR batch; single architecture; PTB-XL release difference; decisions made after seeing S0 (D26) | lab_notebook; D14, D24, D26, D27 | 🟡 |
| — | Reproducibility statement | code, seeds, caches, registry, hardware | README; training_protocol_v1 | 🟡 results not yet in repo |

## Rule for results

A number enters the paper only if its run folder (`config.json`, `log.csv`, `done.json`, `predictions.npz`) and
its `registry.csv` row are committed under `results/runs/`.
