# Corruption protocol v1 (H2a)

**Status: v1, frozen 2026-10-06 (decision D35).** Approved by the author as proposed, with no amendment. No real
checkpoint had been evaluated on any corruption when it was frozen. Changes need v2 and a new decision-log entry.
`corrupt.FROZEN` is `True`; `scripts/eval_corruptions.py` refuses to run otherwise.

Code: `src/robust_ecg/corrupt.py`, `scripts/eval_corruptions.py`, `scripts/bootstrap_h2a.py`, `tests/test_corrupt.py`,
notebook 04.
Fixes what RQ v2 §5 left open ("each corruption at three severities ... parameters are fixed in code before the
first evaluation run") and the statistic for primary contrast 2 (RQ v2 §6).

## 1. Where corruptions are applied

To the preprocessed 10 s record (12 x 1000, 100 Hz, mV), before per-lead normalisation and before the seven
evaluation windows are cut. PTB-XL fold 10 only (2,198 records). Labels are unchanged. Clean and corrupted
predictions come from the same saved best checkpoint of each run; nothing is retrained or re-selected.

Randomness depends only on (suite seed 20261007, corruption, severity, record index). Every arm and seed
therefore sees the same corrupted test set, so comparisons stay paired.

## 2. Held-out corruptions (the only ones used for H2a)

| Corruption | What it does | Severity 1 | Severity 2 | Severity 3 |
|---|---|---|---|---|
| Lead dropout | k randomly chosen leads set to zero for the whole record | k = 1 | k = 2 | k = 3 |
| Limb-lead reversal | left-arm and right-arm electrodes exchanged: I -> -I, II <-> III, aVR <-> aVL; aVF and V1-V6 unchanged | 25% of records | 50% | 100% |
| Baseline step | abrupt offset at one randomly chosen electrode (RA, LA, LL, V1-V6), random sign, at a time uniform in 1-9 s | 0.5 mV | 1.0 mV | 2.0 mV |

Choices that are judgements:

1. **Reversal has no natural magnitude**, so severity is the share of records affected. The affected sets are
   nested (the 25% are among the 50%). Severity 3 is the plain "all leads reversed" test. *Alternative:* one
   condition only (100%), which would make the suite 7 conditions instead of 9.
2. **The step is modelled at the electrode and passed through the preprocessing filter.** A step on one electrode
   changes several leads at once (for example a left-arm step changes I, III, aVR, aVL, aVF and, through Wilson's
   central terminal, every precordial lead by a third). In a real pipeline the 0.5 Hz high-pass turns a step into
   a transient: with our zero-phase filter the peak is about 0.55 x the step size and it stays above 10% of the
   step for about 0.9 s. A unit test checks that adding the template equals preprocessing a raw record with a
   step. *Alternative:* a plain constant offset on single leads after filtering, which no real recording chain
   would produce.
3. **Step sizes** 0.5, 1 and 2 mV bracket typical QRS amplitudes (p99 of per-record maxima is 1.5-4 mV, D21 check).
4. **Lead dropout zeroes stored leads**, as in RQ v2 §5, and is not derived from an electrode model.

None of these appears in P_ecg or P_gen (a unit test asserts the name sets are disjoint). The nearest
augmentations are per-lead scaling by 0.55-1.45 at most (P_ecg, high strength) and time masking of all leads for
up to 0.94 s (P_gen).

## 3. Seen corruptions (reported separately, never used for H2a)

The three P_ecg transforms applied to every record at strengths low / mid / high (0.5, 1.0, 1.5) as severities
1-3: baseline wander, muscle-noise bursts, per-lead scaling. Magnitudes are sampled as in training. Bursts are
drawn 4-12 per lead over the 10 s record, which matches the training density of 1-3 per 2.5 s window.

## 4. Metrics and inference

- Per run and condition: fold-10 macro-AUROC over the 71 statements (headline) and over the nine E3 labels;
  degradation = clean - corrupted, reported with both absolute scores.
- **Primary statistic for H2a (contrast 2 of RQ v2 §6):** mean degradation over the nine held-out conditions
  (three corruptions x three severities), 71 statements, 100% budget, C1 - S1, mean over the five seeds.
  A negative value means C1 loses less than S1.
- Inference as fixed in D34: paired bootstrap over fold-10 patients, 10,000 resamples, one resample applied to
  the clean and all corrupted predictions of both arms and all seeds; percentile 95% interval; two-sided p-value.
  It joins H1, H2b and H3 in the Holm family of four.
- Secondary, without significance claims: each corruption and severity separately; E3 labels; other budgets;
  C0 - S0 and S1 - S0; the seen corruptions.

## 5. Integrity checks built into the evaluation

- The clean predictions recomputed from each checkpoint must match the stored `predictions.npz` within 5e-3
  (float16 storage, GPU non-determinism); otherwise the run is not written.
- Stored test indices must equal the cache's fold-10 indices.
- Each `corruptions.json` records the suite parameters, the seed and the software versions.

## 6. Known limitations

1. Synthetic corruptions are a model of real artefacts. Real lead-off conditions, for instance, usually produce
   saturation or noise and not an exact zero.
2. Three corruptions cannot represent all acquisition faults; conclusions are limited to these.
3. Severity levels were chosen without looking at any model's behaviour, so some may turn out too mild or too
   strong to discriminate between arms. They are reported whatever they show.
4. The protocol is written after the clean and SPH results of the core grid were seen (D33, D34). No corrupted
   result existed when it was written.
