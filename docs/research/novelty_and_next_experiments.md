# Raising the novelty of the project: literature scan and a prioritised experiment programme

Status: **research planning document, not a result.** Nothing here has been run. Written 2026-11-20 from ten parallel literature scouts (one angle each), then merged and checked by the project lead. Every experiment below carries a precommitted pass/fail rule so that a negative result is still a result.

## 1. How to read this document, and how far to trust it

- The scouts used web search and page fetches. ENCODE, GEO, PMC and bioRxiv blocked most automated fetches, so **most sources were read at abstract or snippet level only.** None of the scouts read a full text with its tables.
- I re-checked the existence, title, first author and year of the references in section 3 against arXiv and Crossref metadata. That confirms the papers exist and are what they claim to be. It does **not** confirm the claims attributed to them. Read the paper before quoting a number or a finding.
- "Not found in a limited search" means exactly that. It never means "does not exist". Every novelty statement below should be written "to our knowledge" and re-checked with a systematic search before submission.
- Everything marked UNVERIFIED must be opened and read before it goes into the manuscript. Accessions, genome builds and licences in particular must be confirmed in a browser.

## 2. What "novelty" can and cannot mean here

Using a language model, a bigger CNN, or a newer architecture is not novelty by itself. Novelty comes from a question the literature has not answered, answered with a design that could have come out the other way.

**Already known; do not claim as new**
- Negative-set choice changes reported performance, and imperfect GC matching leaves GC as a proxy (Krützfeldt et al. 2020, scouts opened the page).
- Simple k-mer baselines are competitive with CNNs and with DNA language models on regulatory tasks (Krützfeldt 2020; DART-Eval 2024, abstract-level; Tang et al. 2025, snippet-level).
- Random splits leak and leave-one-chromosome-out (LOCO) exposes it (LOCO-EPI 2025, abstract-level; enhancer-promoter task, not cCRE classification).
- Equal-budget tuning comparisons of trees against deep models exist for tabular data (Grinsztajn et al. 2022).
- Our "tuned XGBoost is about as good as a CNN" finding is therefore **consistent with prior work, not a discovery.**

**Plausibly ours, to our knowledge (needs a systematic search before any claim)**
- A joint GC + CpG obs/exp + repeat matching ladder evaluated across model classes, with a composition-only reference collapsing to about 0.55 on the matched sets.
- A false-discovery-rate-controlled count of CNN first-layer filters matching JASPAR (0.4% of 3,699), with the caveat in section 5 that this measures our pipeline as much as the models.
- Class-resolved (PLS/pELS/dELS) SHAP showing a CpG gradient, with the circularity caveat in section 5.
- Tuning-parity and shortcut controls as the *primary variable* of a regulatory-element benchmark, not an afterthought.

**Recommended framing (from scouts 2 and 8, and I agree):** a protocol, a quantitative score and an open benchmark for shortcut-controlled, tuning-parity evaluation of regulatory-element classifiers, with a mechanistic follow-up on what the models do use. Not "we discovered shortcuts".

## 3. Evidence status of the main references

Existence, title, first author and year checked by the project lead against arXiv or Crossref. Content claims are the scouts' reading and remain to be verified.

| Reference | Metadata checked | What the scouts report (verify) |
|---|---|---|
| Krützfeldt, Schubach, Kircher 2020, PLoS ONE 15:e0237412 | Crossref | Shuffled negatives inflate in-distribution AUROC and fail on natural background (gkm-SVM 0.96 to 0.64); strict GC matching still under-matches high-GC regions. Page opened by three scouts (partial text). |
| ENCODE Consortium (Moore et al.) 2020, Nature 583:699 | Crossref | cCRE registry; labels are annotations, not ground truth; about 44% MPRA validation in GM12878 vs 12% background. |
| Patel et al. 2024, DART-Eval, arXiv 2412.05430 | arXiv | DNA language models give inconsistent results, usually no clear gain over simple baselines. Whether its cCRE task uses composition-matched controls is **unconfirmed** (search snippet only); if so it is closer prior art than we assumed. |
| Tahir et al. 2025, LOCO-EPI, arXiv 2504.00306 | arXiv | Random splits leak; performance drops under LOCO. Enhancer-promoter task. |
| Tang, Somia, Yu, Koo 2025, Genome Biology | Crossref | Reported (snippet only) to find pre-trained DNA models not clearly better than tuned one-hot baselines. Closest genomics precedent for our baseline argument. |
| Grinsztajn, Oyallon, Varoquaux 2022, arXiv 2207.08815 | arXiv | Trees beat deep models on medium tabular data under a tuning-cost-aware protocol. |
| Dodge et al. 2019, arXiv 1909.03004 | arXiv | Expected best validation score vs tuning budget; template for budget curves. |
| Bouthillier et al. 2021, arXiv 2103.03098 | arXiv | Variance from splits, initialisation and tuning; detection of real improvements. |
| Nguyen et al. 2023, HyenaDNA, arXiv 2306.15794 | arXiv | hg38-pretrained, 14.2M parameter checkpoint; fits a 16 GB GPU (model card, per scout). |
| Ferenc et al. 2025, inMOTIFin, arXiv 2506.20769 | arXiv | Motif insertion with controlled spacing, orientation and co-occurrence on real or synthetic sequence. |
| Tseng et al. 2024, arXiv 2410.06211 | arXiv | Architecture whose weights encode motifs and combination rules. |
| Liu et al. 2018, arXiv 1812.04778 | arXiv | ONION and domain-adversarial debiasing of confounded biological data; not a regulatory-sequence model. |
| Kathail et al. 2024 (bioRxiv; Genome Biology per scout) | not checked | Models are less accurate in cell-type-specific accessible regions than ubiquitous ones. Opened by one scout. |
| Majdandzic et al. 2022/2023 (attribution consistency and gradient correction) | not checked | Accuracy does not imply consistent attributions; gradient noise correction. Abstract-level. |
| Ramu et al. 2026, ICLR LMRL workshop | not checked | Attributions can localise motifs yet be unfaithful to grammar logic. Abstract-level. |

Search-snippet-only and unopened (do not cite): hashFrag homology-leakage preprint (authors/venue unconfirmed), GUANinE, Genomic Benchmarks (Grešová 2023), Koo and Eddy 2019, Koo and Ploenzke 2021, BPNet, ChromBPNet, Sei, ChromTransfer, DeepCAPE, ReFeaFi, DeePromoter, Rozenberg 2008, Zeng 2021 (promoter GC), Sullivan 2023 / Zoonomia 2020, Chen 2024 (Gnocchi), Villar 2015, the Weber 2019 benchmarking guidelines, and the DOME paper beyond its arXiv abstract.

## 4. The experiment programme

Ranked by (novelty to our knowledge) x (feasibility on Kaggle with existing assets) x (informativeness if negative). Waves are the proposed order. Notation: "matched" is the GC+CpG matched set, "rep-matched" adds repeats; the bars to beat are always the **composition-only model** and the **tuned XGBoost** (`docs/results/xgb_wide.md`).

### Wave 1 - rigour and calibration (existing models and tables, under about one GPU-day)

**W1.1 Uncertainty that a single split hides.** (scouts 8 and 9)
- Chromosome-block bootstrap (resample test chromosomes, B = 1,000, paired differences between models) and leave-one-chromosome-out / grouped-chromosome folds, with per-chromosome AUROC regressed on GC, gene density and repeat fraction.
- Hypothesis: the CNN-minus-XGBoost gaps reported so far (-0.013 to +0.012) have chromosome-level confidence intervals that include 0 in most cells, and chr19-22 are an atypical (GC-rich, gene-dense) draw.
- Pass: we can state, per cell, whether the gap is distinguishable from 0 at the chromosome level. Fail/negative: intervals exclude 0 for the GM12878 matched sets, which would make that edge real.
- Risk: only four test chromosomes, so intervals will be wide; that is itself the finding.

**W1.2 Negative contamination probe (PU-style).** (scout 9)
- Count how many GC-matched "negatives" of one biosample are cCREs in the other two, and compare model scores on those against clean negatives; re-estimate AUPRC after removing them.
- Hypothesis: a non-trivial fraction is contaminated and scores higher.
- Failure: overlap reflects genuine specificity, not label noise; three biosamples make a weak probe. Report both readings.

**W1.3 Calibrating our own motif pipeline.** (scout 3)
- Plant JASPAR motifs at 1%, 5% and 20% frequency into real GC-matched backgrounds, train the CNN, and measure per-base attribution AUPRC and filter-level recovery with our matcher and with an established matcher (Tomtom or equivalent, after verifying the tool).
- Why: it tells us whether 0.4% is a property of the models or a ceiling of the method. Until this is done, the motif result should be quoted as "our pipeline found almost nothing", not "the models contain no motifs".

**W1.4 Shuffle ladder on the saved models.** (scout 10)
- Score the saved CNNs and XGBoost on (i) original, (ii) dinucleotide shuffles, (iii) approximately 6-mer-preserving shuffles, (iv) segment shuffles that keep local composition and break long-range order; also in-silico knockout of JASPAR hits.
- Precommitted reading: the GM12878 CNN edge is "structural" if the CNN-XGBoost gap survives 6-mer-preserving shuffles (CI above 0) and collapses under segment shuffles; it is falsified if the CNN equals XGBoost under 6-mer-preserving shuffles.
- Caveat: shuffled windows are off-distribution; report as robustness curves, not accuracy.

### Wave 2 - the candidate headline contributions

**W2.1 Specificity under composition control.** (scout 6; our strongest new-question candidate)
- Question: within a fixed class (dELS first, then PLS/pELS), can sequence separate biosample-specific elements from elements active in all three biosamples, once matched on class, GC, CpG, repeats and length? Plus a pairwise "which biosample" task.
- Why it is promising: "is it a cCRE" is dominated by composition; specificity is the biology we care about. Our transfer-by-class run already found about 70% of the cross-biosample gap remains within class. Kathail et al. (abstract-level) suggest models struggle exactly here.
- Falsifiable predictions: (a) specificity AUPRC exceeds the composition-only model by a margin with chromosome-block CI above 0; (b) pairwise difficulty follows lineage distance (K562 vs T cell easier than GM12878 vs T cell).
- Cautions: "specific" depends on assay coverage, so use several definitions (only-in-X; X vs all-3; wider ENCODE panel if available) and report sensitivity; report per class, base rates and AUPRC; do not downsample before splitting; exclude adjacent cCREs.
- Lineage-motif ablation (GATA1/TAL1; PU.1/EBF1/PAX5; TCF7/LEF1/RUNX) against random GC-matched ablation, with permutation nulls.

**W2.2 Shortcut-reliance score and strictness curve.** (scout 2)
- Define SRS = (AUROC_random - AUROC_matched) / (AUROC_random - 0.5) with bootstrap CI; sweep matching granularity (4x4, 8x8, 16x16) and add repeat and mappability constraints; plot AUROC against residual confounding (composition-only AUROC on the same set).
- Pass: model rank stability across biosamples (Kendall tau of at least 0.7) and SRS predicting the drop on an independent control not used in matching (for example mappability-matched). Report matched-pair counts; fine bins get sparse.
- This is the protocol contribution of section 2; its novelty is moderate and rests on a systematic search.

**W2.3 Planted-grammar benchmark on real backgrounds.** (scout 10)
- inMOTIFin: insert motif pairs with spacing/orientation rules into GC-matched genomic backgrounds; negatives carry the same motifs with the rule violated (identical k-mer composition).
- Models: ShallowCNN, AttnCNN, a dilated CNN, XGBoost with and without spaced/gapped k-mer features, composition-only.
- Precommit: CNN "wins" if AUROC is at least 0.05 above XGBoost at spacings beyond 6 bp; if XGBoost with spaced k-mers ties, the claim is that k-mers suffice. Output: a map of the regime where grammar beats composition.

**W2.4 Tuning-budget curves.** (scout 8)
- Random search at N = 1, 2, 4 ... 64 trials for each family; expected-best validation score vs budget with search-order resampling; the CNN needs a properly specified space (filters, kernel, depth, dilation, dropout), which also closes our stated "no architecture search" limitation.
- Hypothesis: the CNN-minus-XGBoost gap tends to 0 as budgets equalise.

### Wave 3 - orthogonal validation (new data; verify access first)

**W3.1 K562 functional labels.** (scout 4)
- K562 CRISPRi element-gene benchmark (EngreitzLab `CRISPR_comparison` and `ENCODE_CRISPR_data`, GRCh38 per file name) and ENCODE K562 STARR-seq records.
- Questions: does a model's score, after GC/CpG/repeat matching, add rank correlation with STARR activity beyond composition (target: partial Spearman gain above 0.05 with CI excluding 0)? Among elements that are all cCREs, does it separate CRISPRi hits from tested non-hits better than distance-to-TSS and DNase signal (AUPRC primary; hundreds of positives, wide CIs expected)?
- Limits: **K562 only; no verified GM12878 or T-cell functional set. Do not extrapolate.** Plasmid STARR-seq favours GC-rich fragments, and library and benchmark selection use the same marks that define cCREs (circular); always compare with a DNase-only baseline.

**W3.2 Conservation as an independent axis.** (scout 5)
- Per-window phyloP (100-way, 30-way, Zoonomia 241-way) via remote `pyBigWig` queries (files are 5-9 GB; do not download), cached to a small table; keep the sign.
- Questions: does constraint add to k-mers within the matched sets (prediction: largest gain for dELS, smallest for PLS)? Is attribution mass enriched at conserved **non-CpG** bases?
- Pitfalls: CpG hypermutability, repeats and alignment masking; conservation must never be used as a label (lineage-specific enhancers are real but poorly conserved); fit thresholds on training chromosomes only.

**W3.3 Clustered re-split.** (scout 9) Re-split by 8-mer Jaccard clusters (validate the similarity measure on a sample against an alignment-based tool) and compare with our chromosome split; calibration per GC decile.

### Wave 4 - held back or lower novelty

- **W4.1 Foundation-model probes (held back by the project owner; planning only).** Frozen embeddings plus a linear probe against tuned XGBoost with matched effort, random-initialised copies of each architecture as controls, a composition-decodability test of the embeddings, and a precommitted win rule (lower CI bound of the matched-set gain over XGBoost above 0). Candidate small checkpoints, as reported by scouts and still to be confirmed on the model cards: HyenaDNA (14.2M, BSD-3), Caduceus (about 7.7M, Apache-2.0), Nucleotide Transformer v2 100M (non-commercial licence), DNABERT-2 (licence unconfirmed). **Pretraining leakage:** HyenaDNA and Caduceus saw all of hg38 unlabeled, including our test chromosomes; state this in any write-up. Do not start without the owner's go-ahead.
- **W4.2 Training-time debiasing** (gradient reversal against GC/CpG bins, reweighting, matched-negative arm, with XGBoost on the same reweighted data as a fair control). Modest novelty; likely a supplement. Removing CpG may remove real promoter signal, so report per class and the full-set drop.
- **W4.3 Biological follow-ups on the CpG gradient** (K562/GM12878 whole-genome bisulfite, TF occupancy of the same windows, JASPAR motif add-ins for dELS). WGBS accessions seen only in snippets (ENCSR765JPC, ENCSR890UQO); TF ChIP accessions not found. High circularity (see section 5).
- **W4.4 Learning curves** (1% to 100% of training data, filters 8 to 256, 5 seeds): "when do CNNs beat k-mers". Publishable as a null or crossover result but about 20 GPU-hours; do after W2.4.

## 5. Weaknesses in our current findings that these experiments address

1. **The CpG gradient may partly reflect how the labels are defined.** PLS requires H3K4me3, which is deposited at unmethylated CpG-rich sites (background knowledge, not verified this session). So a CG signal in PLS may be the label definition, not regulatory mechanism. SHAP explains the model, not the cell. Write it that way; W4.3 and W3.1 are the checks.
2. **The 0.4% motif match rate is as much about the method as the models.** First-layer filters are known to be architecture-dependent (Koo and Eddy, unverified), a 7.9k-parameter CNN may lack the capacity for motif-like filters, and our matcher is non-standard. W1.3 calibrates it.
3. **A single test split.** Four chromosomes, one draw. W1.1 and W3.3.
4. **Labels are annotations.** cCREs come from DNase and histone marks; "negatives" are GC-matched windows outside the biosample's cCREs, not validated non-regulatory DNA. W1.2.
5. **Tuning is still unequal.** XGBoost has had two searches; the CNNs have had a learning-rate sweep only. W2.4.

## 6. Where this could be published (scope pages opened by scout 8; check current calls)

- **NeurIPS Evaluations and Datasets track:** explicitly welcomes negative results, auditing and critical analyses and does not require a new model; needs a dataset on Hugging Face or Kaggle and Croissant metadata. The 2026 deadlines have passed; the next cycle is the target.
- **NAR Genomics and Bioinformatics:** accepts benchmarking only with a new approach or insight; a pure tuning-parity paper risks falling outside scope, so lead with the controlled-evaluation approach. Requires code on GitHub plus Zenodo and a licence.
- Genome Biology and Bioinformatics: fit unconfirmed; check article types with the editors before committing.

## 7. Suggested next step

Run Wave 1 first. It needs no new data, mostly reuses saved models, and settles whether our existing headline numbers survive proper uncertainty and calibration. Then W2.1 (specificity), the strongest new question. W3 needs access checks in a browser before any code is written. W4.1 stays on hold until the project owner decides.

Each experiment would follow the usual routine: Kaggle notebook with a smoke test, run record, results document with limitations, private model/dataset publication, and a PR. The hypotheses and pass/fail rules above should be copied unchanged into each notebook's header *before* it runs.
