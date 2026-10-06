# Decision log

Each entry: **Decision → Reason → Evidence → Impact.** Newest last. Never delete entries;
supersede them with a new one.

---

### D1 — 2026-09-23 · Main task is not the five PTB-XL superclasses
- **Reason:** superclasses (NORM, MI, STTC, CD, HYP) have no defensible one-to-one equivalent in
  external datasets; forcing a mapping would make E3 uninterpretable.
- **Evidence:** SPH uses AHA statements; ST-T and MI statements do not map at superclass level
  (see D5, D6).
- **Impact:** supersedes plan v0 §6. Training on the full statement set (D9).

### D2 — 2026-09-23 · External test set is SPH; E3 is zero-adaptation
- **Reason:** prior cross-dataset ECG SSL work (Soltanieh et al., *IEEE JBHI* 2024) fine-tunes on
  target labels, so "OOD" there means different pretraining data. Deployment-relevant shift is
  train-on-PTB-XL, test-elsewhere without adaptation.
- **Evidence:** SPH is public (CC BY 4.0), single-source, cardiologist-checked, has patient IDs,
  500 Hz, 12 leads, and differs from PTB-XL in country, device and era.
  CODE-test (827 records) is too small; CODE-15% labels are semi-automated.
- **Impact:** E3 = PTB-XL fold 10 vs SPH on identical shared labels.

### D3 — 2026-09-23 · Label mapping uses PTB-XL's own AHA crosswalk
- **Reason:** SNOMED harmonisation can silently drop classes: in Leinonen et al. (arXiv
  2403.15012) the PTB/PTB-XL source shows 0 RBBB despite PTB-XL containing CRBBB.
- **Evidence:** `scp_statements.csv` maps CRBBB→106, CLBBB→104, IRBBB→105, AFIB→50, LPR→82,
  LAFB→101, LVH→142, IMI→161, ASMI→165. SPH Table 2 labels 104/106 as (complete) BBB,
  105 as incomplete RBBB.
- **Impact:** complete vs incomplete BBB resolved from primary sources.

### D4 — 2026-09-23 · PR_PROL = PTB-XL 1AVB ∪ LPR ↔ SPH 82
- **Reason:** SPH has no first-degree AV block statement; code 82 is "prolonged PR interval".
  PTB-XL maps LPR→82 and leaves 1AVB without an AHA code.
- **Evidence:** SPH Table 2 (82, n = 238); PTB-XL crosswalk.
- **Impact:** first-degree AV block is defined by PR prolongation, so the merge is clinically
  defensible, but it is a decision, not a code match. Scored as max of the two outputs.

### D5 — 2026-09-23 · ST-T dropped as a concept; T-wave abnormality exploratory only
- **Reason:** many-to-many mapping. SPH splits ST deviation (145), ST deviation with T change (146),
  T-wave abnormality (147), hypertrophy-related ST-T (153). PTB-XL uses ischemic statements (AHA 226,
  absent in SPH) and non-standard codes 501/502 for ST elevation/depression.
- **Impact:** only TAB_/INVT ↔ 147 retained, as exploratory.

### D6 — 2026-09-23 · IMI and ASMI primary; pooled MI exploratory
- **Evidence:** SPH MI category n = 260 (anterior 52, inferior 120, anteroseptal 91,
  extensive anterior 7), almost all with the "old" modifier. PTB-XL combined-location MI
  statements (ILMI, ALMI, IPLMI, IPMI) have no AHA code; lateral/posterior MI have no SPH counterpart.
- **Impact:** low SPH power for IMI/ASMI; headline uses pooled macro-AUROC.

### D7 — 2026-09-23 · PVC and PAC excluded from E3
- **Reason:** SPH records are 10–60 s; a 10 s crop may not contain the labelled ectopic beat,
  producing label noise that would be misread as distribution shift.

### D8 — 2026-09-23 · NORM exploratory only
- **Reason:** SPH "normal ECG" records were verified to carry no abnormal statements; PTB-XL NORM
  is a diagnostic statement stored separately from rhythm statements, so records such as
  NORM + SBRAD are expected (count not yet checked).

### D9 — 2026-09-23 · Option B: train on full PTB-XL statement set
- **Reason:** shared-label training support is identical under A and B (same records), so
  Gate 2 does not discriminate; B keeps comparability with standard PTB-XL tasks.
- **Impact:** merged labels scored as max of constituent outputs (D4).

### D10 — 2026-09-23 · 1% budget moved to exploratory
- **Evidence (`results/e3_lik0.log`):** at 1%, min positives across seeds: CLBBB 1, CRBBB 2,
  AF 6, PR_PROL 7, IRBBB 9. At 5%, every primary label ≥ 19 in every seed.
- **Impact:** primary curve 5/10/25/50/100%; 1% per-label for labels passing under both
  thresholds (D11): IMI, LVH, LAFB.

### D11 — 2026-09-23 · Likelihood threshold: 0 primary, ≥ 50 on diagnostic statements as sensitivity, all splits
- **Evidence:** 4,843 shared-statement occurrences have likelihood 0; these are form/rhythm
  statements (AFIB, LPR, INVT, TAB_, PAC, PVC, SARRH, SBRAD, STACH), so a global threshold would
  delete them. About 36% of IMI and 18% of ASMI/LVH occurrences have likelihood 15 or 35.
- **Correction to earlier note:** the ≥ 50 run also changes criteria-based diagnostic labels
  slightly at 100% (CRBBB 432→431, IRBBB 894→891, PR_PROL 808→805), not only MI, LVH and NORM.
- **Impact at 1% (min positives, thr 0 → ≥ 50):** IMI 18→11, LVH 16→13, LAFB 13→13, ASMI 12→8.
  Fold 10 changes too (IMI 267→175, ASMI 234→194, LVH 214→177), so the threshold must apply to
  every split.

### D12 — 2026-09-23 · Criteria-based vs interpretive label groups
- **Reason:** AUROC is invariant to prevalence shift but not to a change in labelling practice.
  Prevalence gaps are large for IMI (12.30% vs 0.47%), ASMI (10.83% vs 0.35%), LAFB (7.45% vs
  0.60%); CRBBB and IRBBB are similar across datasets (2.48/2.76%, 5.13/4.89%).
- **Impact:** macro-AUROC reported overall and per group. Grouping is an analytic judgement,
  stated as such in the paper.

### D13 — 2026-09-23 · Label provenance
- **Evidence:** `validated_by_human` = 100% in folds 9–10, 64–68% in folds 1–8.
- **Impact:** both evaluation sets are clinician-reviewed; unvalidated labels affect training
  only and equally across arms. Stated as a limitation, not addressed with an extra arm.

### D14 — 2026-09-23 · SPH deduplication policy
- **Decision:** hash signal arrays; keep the first ECG_ID per identical group; exclude the whole
  group if members differ in AHA *primary* statements (modifier-only differences are not conflicts).
- **Reason:** Leinonen et al. report 132 ECGs with identical recording data in SPH.
- **Impact:** pending run; record counts here when done.

### D15 — 2026-09-26 · Core design is a 2 × 2 factorial (initialisation × fine-tuning augmentation)
- **Reason:** prior ECG SSL studies compare SSL against supervised baselines trained without the SSL
  augmentations (A1, A3, A7), so the SSL effect is confounded with augmentation. Crossing
  initialisation (random / SSL) with augmentation (none / P) separates the two and gives their interaction.
- **Evidence:** literature matrix A1, A3, A5, A7, A8, B2.
- **Impact:** primary contrast is C1 − S1 (both with P). Arms S0 and C0 reproduce prior comparisons.

### D16 — 2026-09-26 · Corruption suite split into seen and held-out
- **Reason:** in A1 the robustness test uses the same noise family as the pretraining augmentations,
  so part of the robustness gain is guaranteed by construction.
- **Impact:** H2a is tested only on lead dropout, LA↔RA limb-lead reversal and baseline step shift,
  none of which appears in either augmentation policy.

### D17 — 2026-09-26 · H3 compares strength-matched policies over a shared grid
- **Reason:** A2 shows contrastive ECG performance depends on augmentation strength, so an
  ECG-specific vs generic comparison at arbitrary strengths could be explained by strength alone.
- **Impact:** 3 strength levels per policy, matched on SNR (or fraction altered), selected on fold 9,
  full grid reported.

### D18 — 2026-09-26 · Four Holm-corrected primary contrasts; ≥ 5 seeds target
- **Reason:** many arms × budgets × shifts invite multiple-comparison problems; A8/A9 show that
  three seeds cannot support claims about rates or stability.
- **Impact:** secondary results reported without significance claims. If compute is short, drop the
  10% and 50% budgets before going below 5 seeds.

### D19 — 2026-09-26 · Compute: single T4; RQ v2 frozen
- **Evidence:** user's hardware; planning estimate ≈ 50–60 GPU-hours from A1's reported V100 fine-tuning time, assumed 2–3× slower on T4.
- **Impact:** FP16 and checkpoint/resume are mandatory; SimCLR batch ≈ 512; run order puts the primary contrasts (S1, C1) first and H3 last. Estimates are replaced by measured times after the first baseline runs.

### D20 — 2026-09-26 · Powerline interference removed from P_ecg (RQ v2 → v2.1, before any results)
- **Reason:** at 100 Hz sampling a 50 Hz component lies exactly at the Nyquist frequency and 60 Hz
  aliases to 40 Hz; both are also removed by the 0.5–40 Hz band-pass and by SPH's device filter.
  An augmentation that the pipeline removes, or that appears at a false frequency, cannot be
  interpreted as "physiologically motivated".
- **Impact:** P_ecg = baseline wander, EMG-like noise, per-lead amplitude scaling, random resized crop.

### D21 — 2026-09-26 · Preprocessing v1 shared by PTB-XL and SPH
- **Decision:** start both datasets from 500 Hz; first 10 s; 0.5–40 Hz zero-phase Butterworth
  (order 3); anti-aliased resampling to 100 Hz; per-lead normalisation with statistics from PTB-XL
  folds 1–8 only; 2.5 s training crops; 7 sliding windows (1.25 s stride) averaged at evaluation.
- **Reason:** identical processing is required for E3 (SPH is device-filtered, PTB-XL is not);
  per-record z-scoring would erase absolute amplitude, which LVH voltage criteria depend on;
  the authors' 100 Hz PTB-XL files cannot be reproduced on SPH.
- **Impact:** full-length SPH evaluation becomes a secondary sensitivity analysis.
- **Evidence:** `docs/preprocessing_v1.md`; filter behaviour tested in `tests/test_data_pipeline.py`.

### D22 — 2026-09-26 · PTB-XL v1.0.3 has 21,799 records (18,869 patients), not 21,837
- **Evidence:** PhysioNet v1.0.3 page and changelog: v1.0.2 removed 36 records with identical raw waveforms,
  preferentially from test folds 9–10; v1.0.3 removed two further duplicates. Our cache size matches
  21,799 × 12 × 1,000 float32 exactly.
- **Impact:** A1 and A3 used the earlier 21,837-record release, so their fold-10 test sets are not identical
  to ours. The baseline sanity check (reproducing ≈ 0.92–0.93 macro-AUROC) is therefore approximate;
  a difference of a few thousandths is not evidence of a pipeline error.

### D14 result — 2026-09-26 · SPH deduplication run
- **Result:** 168 groups of identical signals, all pairs (336 records). 143 clean pairs → one copy dropped each;
  25 pairs carried different primary statements → both copies dropped. **193 excluded; 25,577 SPH records remain.**
  No duplicate group spans two patients.
- **Impact:** the 25 conflicting pairs (0.1% of SPH) are direct evidence of label noise in the external test set;
  reported as a limitation. SPH label counts quoted earlier (e.g. CLBBB 84) are pre-deduplication.

### D21 check — 2026-09-26 · Units and amplitude after preprocessing
- **Result:** both datasets in mV with matching scale. p99 of per-record max |x|: PTB-XL limb leads 1.5–2.4,
  precordial 2.7–4.1 mV; SPH 1.4–2.0 and 2.9–4.2 mV. No non-finite values; one PTB-XL record with a flat lead (kept).

### D23 — 2026-09-26 · Resized cropping removed; muscle noise defined as bursts (RQ v2.2, before any results)
- **Reason:** resizing a crop stretches the time axis and changes apparent heart rate, so in the
  augmentation-matched supervised arm a sinus-tachycardia record could be shown with a normal-rate appearance
  under its original label. Muscle noise modelled as stationary Gaussian noise would be identical to the generic
  policy's Gaussian noise, making H3 uninterpretable.
- **Impact:** both policies share plain random 2.5 s crops (which SimCLR also uses to form two views).
  P_ecg = baseline wander, muscle-noise bursts, per-lead scaling. P_gen = stationary Gaussian noise, global
  scaling, time masking, time warping (±10% local speed, which can shift rate-defined labels; accepted as a
  property of generic augmentation).

### D24 — 2026-09-26 · Training protocol v1
- **Decision:** xresnet1d50 identical to the benchmark encoder; benchmark hyperparameters (AdamW, wd 1e-2,
  one-cycle LR 1e-2, batch 128) with a fixed LR for all arms; length max(50 epochs, 2,000 steps); selection by
  best fold-9 macro-AUROC over 50 evaluation points; mean aggregation over windows.
- **Evidence:** reference code (`helme/ecg_ptbxl_benchmarking`); encoder verified identical (outputs equal
  with shared weights). Details: `docs/training_protocol_v1.md`.
- **Impact:** the S0 100% run is the sanity check against published PTB-XL results before any SSL work.

### D25 — 2026-09-27 · Evaluation runs in float32
- **Evidence:** first real S0 run (100%, seed 0) trained normally (best fold-9 macro-AUROC 0.9246, 6.1 min for
  6,800 steps on a T4) but SPH prediction under FP16 autocast produced NaN. Suspected cause: FP16 overflow on a
  few high-amplitude SPH records (to be confirmed with the amplitude diagnostic in notebook 01).
- **Decision:** all prediction (validation, test, SPH, corruptions) runs in float32; training stays in FP16.
  Any remaining non-finite output raises an error naming the records instead of producing a metric.
- **Impact:** negligible cost; removes a precision difference between datasets from the comparison.

### D25 result — 2026-09-27 · FP16-overflow explanation confirmed
- **Evidence:** max |x| per record after preprocessing: PTB-XL median 1.76, p99.9 11.1, max 44.2 mV (7 records > 20 mV);
  **SPH median 1.71, p99.9 43.0, max 1,084.6 mV (39 records > 20 mV, 25 > 100 mV).** Values above ~10 mV are not
  physiological; these are corrupted recordings (probable scaling or saturation errors) and explain the FP16 overflow.

### D26 — 2026-09-27 · Amplitude outliers kept in the primary analysis; excluded in a sensitivity analysis
- **Decision:** primary analysis keeps every record (no exclusion rule was pre-specified, and the S0 result with
  these records included has already been seen, so excluding them now would be a post-hoc choice).
  Pre-registered sensitivity analysis: exclude records whose max |x| after preprocessing exceeds **20 mV**, applied
  identically to PTB-XL test, fold 9 and SPH (PTB-XL: 7 records in total; SPH: 39).
- **Reason:** 20 mV is roughly double the p99.9 of PTB-XL and well above any physiological QRS amplitude.
- **Impact:** evaluation code reports both versions; training data unchanged.

### D27 — 2026-09-27 · Sanity check passed (S0, 100%, seed 0)
- **Result:** fold-10 macro-AUROC over 71 statements **0.9228** (best fold-9 0.9246 at step 6,664 of 6,800;
  6.1 min on a T4). Published xresnet1d50 on the earlier PTB-XL release: 0.9242 (A1, Table 2). Pipeline accepted.
- **First E3 numbers (single seed, no CI, not for inference):** fold-10 E3 macro-AUROC 0.976, SPH 0.968,
  degradation 0.008. SPH per-label AUROC: AF 0.999, PR_PROL 0.994, CRBBB 0.993, IRBBB 0.778, CLBBB 0.999,
  LAFB 0.990, LVH 0.989, IMI 0.976, ASMI 0.993.
- **Implication for H2b (flagged, design unchanged):** at 100% labels the supervised baseline is near ceiling on SPH
  for 8 of 9 labels, so between-arm differences in degradation will be small at this budget. The low-budget end of
  the curve and IRBBB carry most of the information. IRBBB's drop despite near-identical prevalence in both datasets
  (5.1% vs 4.9%) is consistent with different site criteria for incomplete RBBB, which AUROC cannot absorb.
- **Compute revision:** a 100% run takes ~6 min; smaller budgets run 2,000 steps (~2 min). The core fine-tuning grid
  (4 arms × 5 budgets × 5 seeds) is estimated at ~7 GPU-hours instead of ~20; to be updated after the first SSL run.

### D28 — 2026-09-27 · "Degradation" is a net of opposite per-label effects; H2b stays a between-arm contrast
- **Evidence (S0, 100%, seed 0, `results/runs/S0_b1_s0_lik0`):** per-label AUROC fold 10 → SPH:
  IRBBB 0.975 → 0.778 (drop 0.197); CRBBB 0.997 → 0.993; six labels score *higher* on SPH
  (LVH 0.949 → 0.989, IMI 0.937 → 0.976, PR_PROL 0.974 → 0.994, ASMI 0.977 → 0.993, AF 0.987 → 0.999,
  LAFB 0.986 → 0.990). The macro drop of 0.008 is IRBBB's loss partly cancelled by gains elsewhere.
- **Interpretation (not yet tested):** SPH looks *easier* for most labels, consistent with a case-mix difference:
  54% of SPH records are normal ECGs and its statements are definitive cardiologist diagnoses, whereas PTB-XL fold 10
  includes low-likelihood diagnostic statements (e.g. 36% of IMI occurrences have likelihood 15 or 35, D11) and more
  comorbid negatives. The ≥ 50 likelihood sensitivity analysis (D11) partly tests this for IMI and LVH.
- **Consequences:**
  1. Absolute degradation values must not be read as "the shift hurts performance"; negative values mean the external
     set is easier for that label. Report per-label values alongside the macro value everywhere.
  2. H2b is unaffected in design: it compares degradation *between arms* on the same two test sets, so test-set
     difficulty cancels to first order. The pre-registered contrast stands.
  3. IRBBB is the one label with a real cross-site drop at similar prevalence (5.1% vs 4.9%); site-specific criteria for
     incomplete RBBB are the leading explanation (to be discussed, not claimed).

### D29 — 2026-09-27 · SSL pretraining protocol v1 frozen; augmentation moved to the GPU
- **Decision:** SimCLR on PTB-XL folds 1–8 without labels, batch 512, τ = 0.1, AdamW lr 1e-3 / wd 1e-4,
  10-epoch warm-up + cosine, 300 epochs, final-epoch encoder transferred (full spec: `docs/ssl_protocol_v1.md`).
  No SSL hyperparameter is tuned, and no labelled data is used to choose an SSL checkpoint.
- **Reason:** tuning with fold-9 labels would give SSL arms a selection step the supervised arms do not have;
  tuning without labels is not possible. Batch 512 is the T4 constraint (RQ v2 §7). M1's abstract states that
  contrastive learning benefits from larger batches and longer training, so this is recorded as a limitation that
  may understate SSL, not as a neutral choice.
- **Engineering change (no protocol change):** augmentations were rewritten as batched GPU operations applied after
  the batch is moved to the device. The per-sample CPU version was too slow for SimCLR on Colab's 2 CPU cores
  (two augmented views per record per step). Distributions are unchanged except one detail: in `emg_bursts` the noise
  sd is now drawn once per lead and shared by that lead's bursts (previously once per burst). S0 is unaffected
  (no augmentation). No augmented run had been made before the change.
- **Evidence (synthetic smoke data, CPU):** timing mode, full short run, kill-and-resume (identical per-epoch losses
  to the uninterrupted run once RNG states were added to the checkpoint), and C1 fine-tuning from the saved encoder.
  21 unit tests pass, including NT-Xent sanity checks (identical views → retrieval accuracy 1; flat similarities →
  loss = log(2n − 1)) and strict weight transfer from `SimCLRNet.encoder` to `XResNet1d50.encoder`.
- **Impact:** C0/C1 runs record `encoder_sha256`; every SSL-arm result is traceable to one pretraining run.

### D30 — 2026-09-27 · SSL arms are fine-tuned with exactly the supervised optimisation
- **Decision:** C0/C1 fine-tune the whole network with the training protocol v1 recipe used for S0/S1
  (AdamW, OneCycle max lr 1e-2, same length, same selection on fold 9). No frozen-encoder phase, no lower encoder
  learning rate, no linear probing.
- **Reason:** the 2 × 2 design attributes C1 − S1 to the initialisation. Any SSL-specific fine-tuning recipe would
  add a second difference between the arms.
- **Risk (recorded before any SSL result):** a peak lr of 1e-2 may overwrite pretrained features, especially at
  large budgets, which would bias C1 − S1 towards zero. If H1 is null, this is the first alternative explanation a
  reviewer will raise.
- **Planned check (exploratory, not a primary contrast, not Holm-corrected):** C1 fine-tuned with max lr 1e-3 at the
  5% and 100% budgets, seeds 0–2, compared with C1 at 1e-2. It is reported whatever its outcome and is not used to
  replace the primary C1 results.

### D31 — 2026-09-27 · First SSL run and first matched pair (100%, seed 0): observations, no protocol change
- **SSL run `SSL_ecg-mid_s0`** (`results/ssl/`): 23.6 min on a T4 (estimate was 2–3 h). The contrastive task
  saturates early: retrieval top-1 ≥ 0.99 from epoch 9 and ≥ 0.999 from epoch 26; final loss 0.094.
- **Fine-tuning, fold 10, 71 statements (single seed, no CI):** S0 0.9228, S1 0.9265, C1 0.9183.
  Fold-9 best: S1 0.9271, C1 0.9268 (essentially equal). E3 on SPH: S0 0.9677, S1 0.9722, C1 0.9724;
  degradation S0 0.0078, S1 0.0053, C1 0.0035. C1 loaded the encoder with SHA-256 `0cc027b0…`, matching the SSL run.
- **Observation:** C1 is ahead of S1 on fold 9 early in fine-tuning (step 136: 0.596 vs 0.556; step 680: 0.831 vs
  0.799), and the gap has closed by ~2,000 steps.
- **What is not concluded:** nothing about H1 or H2. One seed at one budget cannot separate the 0.008 fold-10
  difference from seed variance, and fold-9 and fold-10 rank C1 and S1 differently. The 100% budget was already
  expected to be the least informative for H1 (D27).
- **Interpretations recorded for testing, not claimed:**
  1. *Saturation.* Retrieval may be solvable from record-specific cues (patient morphology across 12 leads) and from
     overlap between crops: two uniform 2.5 s starts within 10 s overlap with probability 1 − (500/750)² ≈ 0.56, and
     with p = 0.5 per transform 12.5% of views are unaugmented. If so, most epochs add little. The H3 grid's
     high-strength runs partly test whether stronger views change downstream results.
  2. *Early advantage that fades* is consistent with the D30 concern (high fine-tuning lr overwriting pretrained
     features) and with pretraining acting mainly as a better starting point at full labels. The D30 lr check and
     the low budgets distinguish these.
- **Decision:** no change to the frozen protocol. Changing SSL settings after seeing a C1 result would be a post-hoc
  choice. Any additional SSL variant (e.g. non-overlapping crops) is added only as a labelled exploratory arm and
  reported whatever its outcome.
- **Compute:** SSL 24 min per run and fine-tuning ~6 min at 100% (~2 min below) put the full core grid plus H3 at
  roughly 12 GPU-hours.

### D32 — 2026-09-27 · Core grid moves to Kaggle (2 × T4); Colab runs become a pilot
- **Decision:** the core grid runs on Kaggle with two T4 GPUs and one job per GPU (`scripts/run_queue.py`,
  notebook 03). All grid runs, including seed 0, are made there. The five runs already made on Colab
  (S0/S1/C1 at 100% seed 0 and `SSL_ecg-mid_s0`) move to `results/pilot_colab/`. They are reported as a pilot and
  are not pooled with the grid.
- **Reason:** the user moved to Kaggle. Pooling seed-0 runs from Colab with seeds 1–4 from Kaggle would mix software
  stacks within one arm. Rerunning seed 0 costs about 35 GPU-minutes and gives a free cross-platform check: the
  same configuration on the same GPU type under a different environment.
- **Safeguards:**
  1. Every `config.json` now records `platform`, `gpu`, `torch` and `cudnn` (`robust_ecg.runinfo`).
  2. The grid writes `cache_manifest.json` with the SHA-256 of each cache file. The Kaggle cache is a copy of the
     Drive cache, so the hashes should match the Drive files.
  3. Concurrent jobs would race when appending to `registry.csv`, so the queue rebuilds the registry from `done.json`
     files after every job.
  4. Kaggle outputs survive only as a version's output, so the queue re-archives the whole work folder to
     `/kaggle/working/work.tar` after every finished job. A later version restores it and skips finished runs.
- **Note on earlier entries:** D28 and D31 cite `results/runs/…` and `results/ssl/…`. Those files now live under
  `results/pilot_colab/` with unchanged contents.
- **Timing (estimate):** ≈ 7.5 GPU-hours for 125 jobs, so ≈ 4 h wall time on two GPUs.

### D33 — 2026-10-04 · Core grid complete (Kaggle); descriptive results recorded, inference pending
- **Evidence:** `results/runs/` (120 runs), `results/ssl/` (5 runs), `results/logs/`, `results/cache_manifest.json`,
  `results/analysis/`. Summary and caveats: `docs/results_core_grid_v1.md`.
- **Reproducibility finding:** the seed-0 runs made on Colab (pilot, D32) reproduce on Kaggle with identical
  predictions and an identical SSL encoder hash, across torch 2.11.0 and 2.10.0. The pilot and the grid are the same
  results, so nothing depends on which copy is used. The grid copy is the analysed one.
- **Seed-level observations (descriptive):** C1 − S1 on fold 10 at 5% is +0.010 with a seed interval of −0.013 to
  +0.034 (3 of 5 seeds positive). C1 − S1 degradation on SPH at 100% is −0.0019 (C1 smaller in 5 of 5 seeds). From
  10% upward all arms are within about ±0.005. At the exploratory 1% budget the unmatched comparison (C0 − S0) shows a
  larger SSL gain on E3 labels than the matched one (C1 − S1): +0.051 vs +0.025 on fold 10.
- **Decision:** no hypothesis is declared supported or rejected from these tables. The pre-registered patient-level
  bootstrap with Holm correction is run first. The 1% observation stays exploratory.
- **Decision:** the D30 check (C1 at lr 1e-3; 5% and 100%; seeds 0–2) is now required before H1 is written up,
  because the H1 pattern is null. It remains exploratory and is reported whatever it shows.
- **Protocol unchanged:** no setting of the SSL or fine-tuning protocol is altered in response to these results.

### D34 — 2026-10-06 · Bootstrap specification fixed; H1 and H2b analysed (inference still incomplete)
- **Decision:** `scripts/bootstrap_contrasts.py` implements the inference of RQ v2 §6 and the E3 protocol as follows.
  Resampling unit: patient (all recordings of a drawn patient, with multiplicity). Fold 10 and SPH are resampled
  independently. One resample is applied to both arms and to all five seeds. Statistic: mean over seeds of the
  seed-paired difference. 10,000 resamples, RNG seed 20261006. Percentile 95% interval; two-sided p-value
  2·min(P(d* ≤ 0), P(d* ≥ 0)) with the +1 correction. Macro-AUROC in a resample averages over the labels that have
  both classes in that resample. A second interval that also resamples seeds is reported as secondary.
- **Reason:** RQ v2 §6 fixed the unit (patients), the pairing and the Holm family, but left the number of resamples,
  the handling of seeds and the two-test-set case open (E3 protocol, open item 3).
- **Timing (stated as a limitation):** these choices were made after the seed-level tables of D33 had been seen.
  They were fixed in the script before it was run on real predictions and were not changed afterwards. One
  implementation change was made after a first complete run: the tolerance of the registry-reproduction check was
  raised from 2e-4 to 1e-3, because float16 storage gives a 2.3e-4 difference at the 1% budget. That first run
  stopped at the check before writing any file, so no result from it was seen; the reported run uses the same RNG
  seed.
- **Evidence:** `docs/results_bootstrap_v1.md`; `results/analysis/bootstrap_*.csv`, `bootstrap_config.json`;
  `tests/test_bootstrap.py` (weighted AUROC equal to scikit-learn's to 1e-12, whole-patient resampling, Holm).
- **Result:** H2b (C1 − S1 degradation, 100%): −0.0019, 95% interval −0.0028 to −0.0010, p = 0.0002; significant
  for any values of the two missing p-values (p × 4 = 0.0008); driven mainly by IRBBB. H1 (C1 − S1, 5%, 71
  statements): +0.0103, interval +0.0011 to +0.0157, p = 0.024; Holm-adjusted p will lie between 0.024 and 0.073
  depending on H2a and H3; sign changes across seeds and the seeds-plus-patients interval includes zero; no
  decreasing trend across budgets (C1 − S1 is −0.0048 at 25%, interval excluding zero).
- **Impact:** no hypothesis is declared supported or rejected until all four primary p-values exist. The D30
  learning-rate check remains required before H1 is written up.

### D35 — 2026-10-06 · Corruption protocol v1 and H2a statistic frozen
- **Decision:** `docs/corruption_protocol_v1.md` is frozen as proposed, with no amendment. Held-out suite on PTB-XL
  fold 10: lead dropout (1/2/3 leads), LA↔RA reversal (25/50/100% of records, nested), baseline step at one
  electrode passed through the preprocessing filter (0.5/1.0/2.0 mV). Seen suite: P_ecg transforms at
  low/mid/high. H2a statistic: C1 − S1 in mean degradation over the nine held-out conditions, 71 statements,
  100% budget; inference as in D34 (`scripts/bootstrap_h2a.py`).
- **Reason:** RQ v2 §5 fixed the three corruption types and "three severities" but not their parameters, and
  RQ v2 §6 did not say how conditions are combined into the one primary contrast.
- **Judgements recorded:** reversal severity is the share of records affected (it has no magnitude); the step is
  modelled at the electrode and filtered, not as a constant offset on single leads; the nine conditions are
  averaged with equal weight.
- **Safeguard:** `corrupt.FROZEN` was `False` until this entry, and `scripts/eval_corruptions.py` exits while it
  is. No real checkpoint had been evaluated on any corruption. Code tested on synthetic data only (unit tests;
  end-to-end smoke runs with randomly trained models).
- **Timing (limitation):** frozen after the clean and SPH results of the core grid were known (D33, D34), before
  any corrupted result.
- **Impact:** RQ v2 §5 "parameters are fixed in code before the first evaluation run" is satisfied by this entry.
