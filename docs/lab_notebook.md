# Lab notebook

Chronological record of what was done, what failed and what was learned. Decisions themselves are in
[`decisions/log.md`](decisions/log.md) (IDs D#); this file is the narrative the Methods and Limitations
sections are written from. Failed attempts are kept on purpose.

---

## 2026-09-22 — Plan v0
Original plan (`plan_v0.md`): SSL vs supervised on the five PTB-XL superclasses; external dataset undecided;
H3 defined only loosely.

## 2026-09-23 — Literature check and redesign
- Prior work already showed SSL improves label efficiency and noise robustness on PTB-XL (A1) and "OOD"
  transfer (A3). Close reading showed both lack an augmentation-matched supervised baseline, A1 tests
  robustness on noise seen in pretraining, and A3's "OOD" always fine-tunes on the target. The gap was
  narrowed accordingly (D1, D2).
- External dataset feasibility study: SPH chosen (D2). Label harmonisation through PTB-XL's own AHA crosswalk
  (D3–D8). Training-support counts run on the user's machine: at 1% budget five labels have 1–9 positives,
  so 1% became exploratory (D10). Likelihood threshold analysis (D11).
- **Failure:** first label-support run used the old budget list; its CSV was later deleted from `results/`
  to avoid confusion with the correct runs.

## 2026-09-24 → 26 — Remaining literature, frozen question
- Outside ECG, contrastive vs supervised learning under identical augmentation had already been compared on
  images (B2), so "first augmentation-matched comparison" cannot be claimed; the contribution is the ECG-specific
  combination (RQ v2 §1).
- Zeng et al. 2026 (A8) and the letter criticising it (A9) read in full: A8's augmentation-only arm is not an
  augmentation-matched control, and its headline Macro-F1 (0.192) is barely above a constant predictor (≈ 0.18).
- RQ v2 frozen with a 2 × 2 design (D15–D19). Compute: one T4 on Colab.

## 2026-09-26 — Data pipeline
- Preprocessing v1 specified (D21). **Design error caught before any result:** powerline interference cannot be
  represented at 100 Hz (50 Hz = Nyquist, 60 Hz aliases to 40 Hz), so it was removed from P_ecg (D20, RQ v2.1).
- **Failure 1 — SPH file lookup.** Scripts were written against a sample metadata file whose IDs included `.h5`;
  the real metadata uses `A00001`. Worse, the deduplication script only *warned* about missing files and wrote an
  empty exclusion list. Fixed: IDs resolved with or without extension, and any missing file now aborts (commit 5e78a21).
- **Failure 2 — Colab overwrote the repo.** Saving the notebook to GitHub from Colab replaced the fixed notebook with
  a stale session copy, silently reverting the fix. Restored; notebooks now warn against saving from Colab.
- **Finding:** PTB-XL v1.0.3 has 21,799 records, not the 21,837 quoted in earlier papers; duplicates were removed in
  v1.0.2–1.0.3, mostly from test folds 9–10, so published test sets differ slightly from ours (D22).
- SPH deduplication: 168 identical pairs; 193 records excluded; 25 pairs carried conflicting labels (D14 result).
- Units check: both datasets in mV on the same scale (D21 check).

## 2026-09-26 — Model and training code
- xresnet1d50 ported from the PTB-XL benchmark code and verified identical (same 886,400 encoder parameters,
  identical outputs with shared weights). The benchmark uses constant width 64 in every stage, not ImageNet widths.
- **Design error caught before any result:** random *resized* crop changes apparent heart rate, which would give the
  augmentation-matched supervised arm mislabelled examples for rate-defined statements. Replaced by plain cropping;
  muscle noise redefined as bursts so it differs from generic Gaussian noise (D23, RQ v2.2).
- Resume logic tested by killing a run mid-training and after training (evaluation-only resume).

## 2026-09-26/27 — First real run: S0, 100%, seed 0
- Training: 6,800 steps in 6.1 min on a T4; best fold-9 macro-AUROC 0.9246 at step 6,664.
- **Failure 3 — NaN on SPH.** Prediction under FP16 autocast produced NaN on SPH. Diagnosis: 25 SPH records exceed
  100 mV (max 1,084.6 mV), far outside physiological range, overflowing FP16. Fix: all prediction in float32; any
  remaining non-finite output aborts with the record indices (D25).
- Outlier policy decided *after* seeing the S0 result, so the primary analysis keeps all records and exclusion
  above 20 mV is a sensitivity analysis only (D26).
- **Result:** fold-10 macro-AUROC 0.9228 vs 0.9242 published for the same architecture (sanity check passed, D27).
  First E3 numbers: fold-10 0.976, SPH 0.968; IRBBB is the only label that drops markedly (0.778).
- **Concern recorded, design unchanged:** the supervised baseline is near ceiling on SPH at 100% labels, so H2b has
  little room at that budget (D27).

## 2026-09-27 — S0 results committed; per-label reading
- Run files for S0 (100%, seed 0) and the SPH dedup outputs committed under `results/` (export from Colab,
  checked: `done.json` matches the console output; 2,198 fold-10 and 25,577 SPH predictions, all finite).
- Per-label comparison shows six of nine E3 labels score higher on SPH than on fold 10; only IRBBB drops markedly.
  The macro "degradation" of 0.008 is a net of opposite effects (D28). Case mix is the leading explanation for the
  gains; this is recorded as an interpretation to test, not a finding.
- Environment of the run: Colab, T4, torch 2.11.0+cu128.
- **Failure 4 — results silently not committed.** `.gitignore` excluded every `runs/` folder and all `*.npz`
  files, so the first results commit (38f00fb) omitted `results/runs/`. Caught by listing the pushed tree;
  fixed in the next commit. Check after every results commit: `git ls-tree -r --name-only origin/main results/runs`.

## 2026-09-27 — SSL pretraining code
- SimCLR pretraining written (`scripts/pretrain_simclr.py`) and the protocol frozen before any real run
  (`ssl_protocol_v1.md`, D29). SSL arms get the supervised fine-tuning recipe unchanged (D30).
- **Engineering problem:** the per-sample augmentation ran in DataLoader workers on the CPU. SimCLR needs two
  augmented views per record per step and Colab gives two CPU cores, so the CPU would have been the bottleneck.
  Augmentations were rewritten as batched GPU tensor operations. One distributional detail changed (EMG burst noise sd
  per lead instead of per burst, D29); no augmented run existed yet.
- **Bug caught in testing:** the first resume test showed the LR schedule continued correctly but losses after the
  resume differed from the uninterrupted run, because RNG states (crops, augmentations) were not checkpointed. Added;
  a killed-and-resumed CPU run now reproduces the uninterrupted run's losses exactly. (Fine-tuning runs still
  re-draw crops/augmentations after a resume, as documented in training_protocol_v1.)
- **Naming clash avoided:** the SimCLR paper was first entered in the literature matrix as "C1", which is also an arm
  name. Method references now use the prefix M (M1).
- Verified from the SimCLR abstract that contrastive learning benefits from larger batches and longer training;
  cited only for that sentence until the full text is read.
- **Failure 5 — Drive mount dropped during the cache copy** (first attempt at notebook 02): `OSError [Errno 107]
  Transport endpoint is not connected` while copying `ptbxl_X.npy` (~1 GB) from Drive to local disk. An
  infrastructure fault, not a data or code fault; notebook 01 had copied the same file without error. Fix: the copy
  now goes through `robust_ecg.colab_utils.copy_dir_with_retry`, which writes to a `.part` file, checks the size,
  remounts Drive on error and retries. A unit test simulates the drop.

## 2026-09-27 — First SSL run and first S1/C1 pair
- Notebook 02 ran after the Drive fix. `SSL_ecg-mid_s0` finished in 23.6 min, far below the 2–3 h planned.
- **Unexpected observation:** the contrastive pretext task is essentially solved by epoch ~10 (retrieval top-1 0.99).
  Recorded in D31 with two candidate explanations; protocol unchanged because a C1 result had already been seen.
- First matched pair at 100%, seed 0: S1 0.9265, C1 0.9183 on fold 10; equal on fold 9 (0.9271 vs 0.9268); E3 SPH
  0.9722 vs 0.9724. C1 learns faster early in fine-tuning; the advantage has gone by ~2,000 steps. Single seed: not evidence.
- Augmentation alone (S1 vs S0) is +0.004 on fold 10 and +0.005 on SPH E3 at 100%, also single seed.
- Run files committed: `results/runs/{S1,C1}_b1_s0_lik0`, `results/ssl/SSL_ecg-mid_s0` (encoder kept on Drive, hash recorded).
- The predictions files hold scores and row indices but not labels, so the bootstrap analysis has to rebuild labels
  from the caches (on Colab or with the metadata). Noted for the analysis script.

## 2026-09-27 — Move to Kaggle
- The user started notebook 03 on Kaggle, which the Colab-only notebook did not support (no Drive, read-only
  inputs, outputs kept only per saved version). Rewrote it to detect the platform and wrote `scripts/run_queue.py`.
  It is a resumable scheduler with one job per GPU and SSL dependencies. It rebuilds the registry and archives the
  outputs after each job, and stops its child jobs cleanly on interrupt.
- Tested on synthetic data with two parallel slots, a kill partway through and a restart. The restart skipped
  finished jobs, resumed the rest and left no orphaned processes; the registry had no duplicates. The first version
  left orphaned child processes when killed, which is why signal handling was added.
- Colab runs moved to `results/pilot_colab/` (D32). Seed 0 will be rerun on Kaggle and compared with the pilot.
- **Failure 6 — Colab "Save a copy in GitHub" again** (commit d88368f, 01:00). The saved notebook 02 was a stale copy,
  and its cache-copy cell had reverted to the pre-Failure-5 code. Caught because the next push was rejected; the repo
  version was kept (merge with `-s ours`). The executed copy is kept as a record in
  `results/pilot_colab/02_ssl_pretrain_executed.ipynb`: it holds the timing check (0.204 s/step, projected 0.58 h;
  actual 23.6 min) and the full per-epoch pretraining log. The grid now runs on Kaggle, where this cannot happen.
- Cache moved to Kaggle as the private dataset `olamidesalako/robust-ecg-ssl-cache` (2026-09-28). It was uploaded
  directly from Colab with the Kaggle API, so it never passed through a local download: the 7 cache files only,
  without the intermediate `_ptbxl_chunks`. Byte sizes checked against the Colab listing before upload; all matched.
  Content identity is verified by the SHA-256 manifest that notebook 03 writes on Kaggle. A first manual download
  from Drive had been incomplete (4 of 7 files, plus the chunk folder), which is why the API route was used.
