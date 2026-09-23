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
