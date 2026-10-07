> **Status: SUPERSEDED (v0, 2026-09-22).** Kept for the record of how the design evolved.
> Current E3 design: [`E3_protocol_v1.md`](E3_protocol_v1.md). Rationale for each change: [`decisions/log.md`](decisions/log.md).
> Converted from the original .docx with pandoc; content unchanged.

**Research Project Plan  
Self-Supervised ECG Representation Learning Under Distribution Shift**

*Working Paper / Master's Research Portfolio Project  
*Target: Working paper + reproducible GitHub repository + professor outreach

# 1. Project Objective

Build a compact but rigorous machine-learning research study investigating whether self-supervised representation learning improves label efficiency and robustness to distribution shift in 12-lead ECG classification. The goal is not to claim a new state-of-the-art foundation model, but to demonstrate the ability to formulate a research question, reproduce meaningful baselines, design controlled experiments, analyze results, and communicate findings in a paper-quality working paper.

# 2. Core Research Question

Does self-supervised pretraining produce ECG representations that generalize better than conventional supervised learning when labeled data is scarce and the test distribution differs from the training distribution?

# 3. Research Hypotheses

- H1 — Label efficiency: Self-supervised pretraining will outperform supervised training when only a small fraction of labels is available.

- H2 — Distribution shift: Self-supervised representations will degrade less than supervised representations under cross-dataset and controlled signal shifts.

- H3 — Augmentation: Physiologically motivated ECG augmentations will produce more robust representations than arbitrary generic augmentations.

# 4. Scope — What We Will and Will Not Build

In scope:

- PTB-XL as the primary dataset.

- A supervised 1D CNN/ResNet baseline.

- A transformer-based baseline if computationally practical.

- One contrastive self-supervised method, initially SimCLR-style.

- Optional second SSL method only if the core experiments finish early.

- Label-budget experiments: 100%, 50%, 25%, 10%, 5%, and 1%.

- Cross-dataset evaluation where preprocessing and label mapping are defensible.

- Controlled signal corruptions: noise, baseline wander, lead masking, and temporal/signal masking.

- Ablation of important augmentation components.

- Paper-quality plots, tables, failure analysis, and reproducibility documentation.

Explicitly out of scope for the first version:

- Training a large ECG foundation model from scratch.

- Clinical deployment or diagnosis claims.

- A web application or polished product UI.

- Large-scale uncertainty/fairness/explainability studies.

- Five or more competing SSL algorithms.

- Trying to beat every published state-of-the-art result.

# 5. Experimental Pipeline

Overall flow:

1.  Literature review → define gap and hypothesis.

2.  Dataset acquisition → preprocessing → quality checks.

3.  Establish supervised baseline.

4.  Implement self-supervised pretraining.

5.  Evaluate representation using downstream ECG classification.

6.  Run label-efficiency experiments.

7.  Run distribution-shift experiments.

8.  Run corruption/robustness experiments.

9.  Run ablation studies.

10. Analyze failure cases and statistical significance.

11. Write working paper and release reproducible repository.

# 6. Dataset and Task

Primary dataset: PTB-XL. Use a well-defined multi-label ECG classification task, with clinically meaningful label grouping chosen from the dataset's available diagnostic superclass/subclass structure. Keep the label mapping fixed throughout the experiments.

- Document patient-level splitting to prevent leakage.

- Document sampling rate, number of leads, signal length, normalization, missing-value handling, and augmentation policy.

- Use the official or widely accepted PTB-XL split protocol where possible.

- Keep a fixed validation/test protocol so every model is compared fairly.

# 7. Models

| Model                  | Purpose                    | Training                    | Priority |
|------------------------|----------------------------|-----------------------------|----------|
| 1D CNN                 | Simple supervised baseline | Supervised                  | Must     |
| ResNet1D               | Strong supervised baseline | Supervised                  | Must     |
| Transformer/TS encoder | Sequence-model baseline    | Supervised                  | Should   |
| Contrastive encoder    | Main research method       | Self-supervised → fine-tune | Must     |
| Second SSL method      | Robustness comparison      | Self-supervised             | Optional |

# 8. Self-Supervised Learning Design

Start with a SimCLR-style contrastive objective. Two augmented views of the same ECG are passed through a shared encoder and projection head. Positive pairs are different views of the same recording; other recordings form negative examples.

- Encoder → projection head → contrastive loss.

- Pretrain without diagnostic labels.

- Discard the projection head and fine-tune/evaluate the encoder on the downstream task.

- Record pretraining configuration, batch size, temperature, optimizer, epochs, and augmentation probabilities.

# 9. ECG Augmentations

Start with a small controlled set:

- Gaussian/noise injection.

- Baseline wander.

- Amplitude scaling.

- Temporal masking/cropping.

- Lead masking.

- Signal dropout.

Important: do not assume every augmentation is physiologically valid. Each augmentation must be justified and tested. The ablation study should determine which transformations actually contribute to robustness.

# 10. Experiment Matrix

| Experiment            | Train  | Test                 | Main Metric         | Question                                    |
|-----------------------|--------|----------------------|---------------------|---------------------------------------------|
| E1 — Baseline         | PTB-XL | PTB-XL               | AUROC/F1            | How strong are the supervised baselines?    |
| E2 — Label efficiency | PTB-XL | PTB-XL               | AUROC/F1            | Does SSL help with fewer labels?            |
| E3 — Cross-domain     | PTB-XL | External ECG dataset | AUROC/F1            | Does SSL generalize across datasets?        |
| E4 — Noise robustness | PTB-XL | Corrupted PTB-XL     | Performance drop    | How fast does performance degrade?          |
| E5 — Lead masking     | PTB-XL | Masked PTB-XL        | Performance drop    | Can representations tolerate missing leads? |
| E6 — Ablation         | PTB-XL | PTB-XL/OOD           | Delta vs full model | Which components actually matter?           |

# 11. Label-Efficiency Study

Train/fine-tune using:

- 100% labels

- 50% labels

- 25% labels

- 10% labels

- 5% labels

- 1% labels

Use multiple random seeds where computationally feasible. Report mean ± standard deviation rather than a single lucky run.

# 12. Distribution-Shift Study

Three levels of shift:

- In-distribution: normal held-out PTB-XL test data.

- Cross-dataset: train on PTB-XL and evaluate on a compatible external ECG dataset.

- Synthetic acquisition shift: noise, baseline wander, lead masking, temporal masking, and signal dropout.

Primary robustness statistic: relative performance degradation from the clean test condition. Report both absolute performance and percentage drop.

# 13. Evaluation Metrics

- AUROC — primary ranking metric.

- AUPRC — especially useful for class imbalance.

- Macro F1 / weighted F1 — classification performance.

- Per-class metrics — identify which diagnostic groups fail.

- Performance degradation under corruption — robustness.

- Mean ± standard deviation across seeds where feasible.

# 14. Ablation Study

Start with the complete SSL pipeline and remove one component at a time:

- Remove contrastive learning → supervised-only baseline.

- Remove lead masking augmentation.

- Remove temporal augmentation.

- Remove noise augmentation.

- Use generic augmentations instead of ECG-specific augmentations.

The purpose is to show that the observed result is attributable to identifiable components rather than an arbitrary training configuration.

# 15. Failure Analysis

Do not hide failures. Analyze:

- Diagnostic classes with the largest performance drop.

- Signal conditions that cause errors.

- Examples of false positives and false negatives.

- Whether OOD samples produce systematically worse confidence/performance.

- Cases where supervised learning outperforms SSL.

# 16. Reproducibility Requirements

- Fixed random seeds for reported runs.

- Configuration files for experiments.

- Exact preprocessing pipeline.

- Requirements/environment file.

- Training and evaluation commands.

- Dataset preparation instructions.

- Saved metrics and experiment logs.

- Scripts to reproduce the main figures and tables.

# 17. Repository Structure

> robust-ecg-ssl/
>
> ├── configs/
>
> ├── data/
>
> ├── preprocessing/
>
> ├── models/
>
> ├── losses/
>
> ├── training/
>
> ├── evaluation/
>
> ├── experiments/
>
> ├── figures/
>
> ├── results/
>
> ├── notebooks/
>
> ├── paper/
>
> ├── requirements.txt
>
> └── README.md

# 18. Working Paper Structure

12. Abstract

13. 1\. Introduction

14. 2\. Related Work

15. 3\. Research Questions and Hypotheses

16. 4\. Dataset and Preprocessing

17. 5\. Methodology

18. 6\. Experimental Setup

19. 7\. Results

20. 8\. Distribution Shift and Robustness Analysis

21. 9\. Ablation and Failure Analysis

22. 10\. Discussion

23. 11\. Limitations

24. 12\. Conclusion

25. References

# 19. Timeline — September 22 to November 20, 2026

| Dates        | Milestone                    | Deliverable                                             |
|--------------|------------------------------|---------------------------------------------------------|
| Sep 22–25    | Literature + research design | 10–12 paper literature matrix; frozen research question |
| Sep 26–Oct 3 | Data + baseline              | Preprocessing pipeline + supervised baseline            |
| Oct 4–12     | SSL implementation           | Contrastive pretraining + downstream evaluation         |
| Oct 13–20    | Label efficiency             | 1–100% label experiments + plots                        |
| Oct 21–27    | Distribution shift           | Cross-dataset + corruption experiments                  |
| Oct 28–Nov 3 | Ablations + analysis         | Ablation tables + failure analysis                      |
| Nov 4–10     | Paper                        | Complete working-paper draft                            |
| Nov 10–15    | Release                      | Clean GitHub + reproducibility package                  |
| Nov 15–20    | Outreach                     | Professor-specific research emails + paper/GitHub       |

# 20. Definition of Done

- A clear research question and hypothesis.

- At least two credible supervised baselines.

- One working self-supervised method.

- Label-efficiency results.

- At least one defensible distribution-shift experiment.

- At least two robustness/corruption tests.

- An ablation study.

- Failure analysis.

- Reproducible GitHub repository.

- 8–12 page working paper.

- A 1-page research summary for professor outreach.

# 21. Priority Rule

If time becomes tight, protect the core scientific story. Cut optional models, UI, uncertainty estimation, and extra datasets before cutting baselines, label-efficiency experiments, distribution-shift evaluation, ablations, or analysis. The project wins on experimental rigor, not feature count.

# 22. Immediate Next Actions — Start Today

26. Create the GitHub repository using the proposed structure.

27. Create a project tracker with the timeline above.

28. Read and summarize 10–12 directly relevant papers.

29. Download and inspect PTB-XL.

30. Write the exact label mapping and patient-level split protocol.

31. Implement preprocessing and one simple 1D CNN baseline.

32. Run one end-to-end experiment before adding any SSL code.

33. Create an experiment log so every result is reproducible.

**Core principle:** Do less, measure better, and write down what you learn.
