# Reference and claim check (draft v1, 2026-10-05)

Covers `main.tex` / `references.bib` and the Word and PDF copies built from them.

## 1. Bibliographic details

**Confirmed online on 2026-10-05** (authors, title, venue, volume, pages, DOI checked against OpenAlex, PMC or the publisher's proceedings page): 21 entries.

| Key | Reference | Source checked |
|---|---|---|
| mehari2022 | Mehari & Strodthoff, Comput. Biol. Med. 141:105114 | OpenAlex |
| soltanieh2022aug | Soltanieh, Etemad & Hashemi, IJCNN 2022, pp. 1–10 | OpenAlex |
| soltanieh2024ood | Soltanieh, Hashemi & Etemad, IEEE JBHI 28(2):789–800 | OpenAlex |
| gopal2021 | Gopal et al., ML4H, PMLR 158:156–167 | PMLR page |
| lai2023 | Lai et al., Nat. Commun. 14:3741 | OpenAlex |
| dade2026 | Dade et al., Heart Rhythm O2 7(4):757–770 | PMC |
| weimann2025 | Weimann & Conrad, Comput. Biol. Med. 196:110809 | OpenAlex |
| zeng2026 | Zeng et al., Ann. Noninvasive Electrocardiol. 31(3):e70188 | OpenAlex |
| ahmed2026 | Ahmed et al., Ann. Noninvasive Electrocardiol. 31(4):e70216 | OpenAlex |
| kiyasseh2021 | Kiyasseh, Zhu & Clifton, ICML, PMLR 139:5606–5615 | PMLR page |
| raghu2022 | Raghu et al., CHIL, PMLR 174:282–310 | PMLR page |
| zhong2022 | Zhong, Tang, Chen, Peng & Wang, arXiv:2206.05259 | OpenAlex |
| liu2022shift | Liu et al., NeurIPS 2022 DistShift workshop, arXiv:2205.12753 | search listing (authors, title only) |
| wagner2020 | Wagner et al., Sci. Data 7:154 | OpenAlex |
| strodthoff2021 | Strodthoff et al., IEEE JBHI 25(5):1519–1528 | OpenAlex |
| liu2022sph | Liu et al., Sci. Data 9:272 | OpenAlex |
| leinonen2024 | Leinonen et al., Comput. Biol. Med. 183:109271 | OpenAlex |
| goldberger2000 | Goldberger et al., Circulation 101(23):e215–e220 | OpenAlex (end page from memory) |
| mason2007 | Mason, Hancock & Gettes, JACC 49(10):1128–1135 | OpenAlex |
| hannun2019 | Hannun et al., Nat. Med. 25(1):65–69 | OpenAlex |
| perezalday2020 | Perez Alday et al., Physiol. Meas. 41(12):124003 | OpenAlex |

**Not confirmed online; written from memory. Check each against its DOI or proceedings page before submission:** 24 entries.

chen2020simclr, oord2018cpc, hendrycks2019ssl, zoph2020, he2019rethinking, oliver2018, hendrycks2019corruptions, koh2021, zech2018, ribeiro2020, zheng2020, wagner2022physionet (PhysioNet v1.0.3 DOI 10.13026/kfzx-aw45), kligfield2007, moody1984, batchvarov2007, he2016, he2019bag, loshchilov2019adamw, loshchilov2017sgdr, smith2019, micikevicius2018, holm1979, efron1993, bouthillier2021.

Lookups that failed: Kligfield 2007 and Batchvarov 2007 (rate limited), the arXiv page for Leinonen (rate limited; the journal version was confirmed instead), PhysioNet's PTB-XL page and Semantic Scholar (permission prompt timed out).

## 2. Claims about cited papers

Statements about prior work follow `docs/literature_matrix.md` and inherit its verification level.

- **Read at abstract level only in the matrix:** Soltanieh 2022 (IJCNN), Gopal 2021 (3KG), Hendrycks 2019 (SSL robustness), Liu 2022 (distribution shift), Chen 2020 (SimCLR). Their rows in Table 1 and the sentences describing them should be checked against the full text.
- **Added by me from general knowledge, not in the matrix:** the sentences citing He 2019 (*Rethinking ImageNet pre-training*), Zoph 2020, Oliver 2018, Raghu 2022 (abstract confirmed), Zech 2018, Koh 2021, Hendrycks & Dietterich 2019, Perez Alday 2020, Leinonen 2024 (abstract confirmed), Moody 1984, Batchvarov 2007 and Kligfield 2007. Read each before submission, in particular:
  - Kligfield 2007 is cited for "the band-pass is narrower than the bandwidth recommended for diagnostic recording".
  - Mehari & Strodthoff is cited for comparing "several SSL methods, including SimCLR and contrastive predictive coding".

## 3. Numbers

Every number in Section 4 comes from the repository at commit `890d233` (2026-10-04) or was recomputed from `ptbxl_database.csv`, `scp_statements.csv`, SPH `metadata.csv` and `results/sph_exclude.txt` with the repo's own `labels.py` and `splits.py`.

Recomputed for this draft (not previously in the repo docs):

- Table 3: SPH positives after deduplication (the E3 protocol lists pre-deduplication counts), fold-10 positives and prevalences.
- SPH after deduplication: 24,642 patients.
- Budget sizes over the five seeds: patients 151 / 752 / 1,503 / 3,756 / 7,512 / 15,023; recordings 165–186, 860–887, 1,717–1,766, 4,326–4,396, 8,705–8,752, 17,418.

One discrepancy with the repo docs: `preprocessing_v1.md` says 94% of SPH recordings are 10–15 s; the metadata gives 96%. The draft avoids the figure.

## 4. Open items marked [TO DO] in the draft

1. Author names, affiliations, e-mail.
2. Abstract.
3. Findings sentence at the end of the Introduction.
4. Severity parameters of the three held-out corruptions (not yet in the code).
5. H3 strength-matching and level-selection procedure as implemented.
6. Bootstrap details: number of resamples, how seeds are combined, resampling for two-test-set contrasts.
7. Two cross-references to the Results section.

## 5. Wording choices to confirm

- The draft says the hypotheses were "fixed" or "pre-specified" in a public repository, not "pre-registered", because no registry was used.
- Section 3 and the H2a/H3 parts of Section 4 are written in the past tense as they will read in the finished paper, although the corruption suite, the H3 grid and the bootstrap have not been run yet.
- British spelling throughout, matching the repo docs.
