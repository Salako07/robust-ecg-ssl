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
