# 1. Introduction (draft)

> Draft status: see `README.md`. `[CITE: ...]` = citation still to be added and verified against the primary source.

Candidate cis-regulatory elements (cCREs) mark the non-coding sequences that control when and where genes are active, and cataloguing them
across cell types is a central output of large consortium projects [CITE: ENCODE registry of cCREs / SCREEN: the consortium paper describing the
cCRE classes promoter-like (PLS), proximal enhancer-like (pELS) and distal enhancer-like (dELS)]. A long line of work asks whether the DNA
sequence alone predicts such elements, first with k-mer and motif features and then with convolutional and attention-based neural networks
[CITE: early sequence-based CNN predictors of regulatory activity from DNA, e.g. DeepSEA / Basset / DeepBind class of models] and, more recently,
with large pretrained genomic language models [CITE: DNABERT / DNABERT-2 / Nucleotide Transformer / HyenaDNA; state which are cited for what]. Because
these models are opaque, interpretation is usually added after the fact: first-layer convolutional filters are converted to position weight
matrices and compared with transcription-factor motif databases [CITE: TOMTOM; JASPAR release used], and per-feature attributions are computed
[CITE: SHAP / TreeSHAP; integrated gradients if used].

Three practical problems make reported results in this area hard to interpret.

**(i) The task depends on how negatives are built.** Random genomic windows differ from cCREs in simple composition (GC and CpG content), so a
classifier can score well without learning anything regulatory [CITE: work documenting GC / composition confounds in sequence-based regulatory
prediction, if available; otherwise state this as our own observation]. In our data, random negatives inflate AUROC by 0.07-0.09 relative to
GC-matched negatives, and two composition features alone separate promoter-like elements from GC-matched background at AUROC 0.88-0.90
(`docs/results/phase1_unified_baselines.md`, `docs/results/multiclass_shap.md`).

**(ii) Baselines are often weaker than they could be.** A neural network is usually compared with an untuned classical model. We find that giving
gradient-boosted trees on k-mer frequencies a modest hyperparameter search removes most of the advantage a tuned CNN appeared to have
(`docs/results/cnn_tuned.md`, `docs/results/xgb_tuned.md`).

**(iii) Interpretation claims are rarely tested against a null.** Filter-to-motif matches are commonly reported at permissive thresholds without
correction for the many motifs searched. When we control the false discovery rate against a shuffled-motif null, only 0.4% of 3,699 first-layer
filters resemble a known motif (`docs/results/motif_fdr.md`).

We therefore treat this as an evaluation problem rather than a modelling one. **Our contribution is a controlled, reproducible evaluation
protocol and the negative and nuanced results it produces**, not a new model. Specifically we (1) build confound-aware benchmark tables for three
biosamples with chromosome-level splits, near-duplicate flags and three negative strategies; (2) evaluate classical, convolutional and (as a
planned extension) pretrained models under GC-, CpG- and repeat-matched test sets; (3) give the classical baseline a hyperparameter search
comparable to the CNN's; (4) test interpretability claims with an explicit null and FDR control; and (5) publish every run, with logs,
code versions and artefacts, so that each number can be reproduced.

Research questions (each is answered by a results document):

- **RQ1.** How much of the apparent discrimination of cCREs from GC-matched background is explained by GC, CpG and repeat content?
  (`cpg_ablation.md`, `repeat_control.md`)
- **RQ2.** Does a CNN outperform a tuned k-mer baseline once composition is controlled? (`cnn_tuned.md`, `xgb_tuned.md`)
- **RQ3.** Is cross-cell-type transfer inflated by elements shared between cell types, and by class composition? (`transfer_by_class.md`)
- **RQ4.** What sequence features separate promoter-like, proximal and distal enhancer-like elements once composition is controlled?
  (`multiclass_shap.md`)
- **RQ5.** Do first-layer CNN filters correspond to known transcription-factor motifs at a controlled false discovery rate? (`motif_fdr.md`)

[TODO for authors: a related-work paragraph with verified references; a statement of what is and is not novel relative to that literature
(the individual components are standard; the combination of confound controls, baseline parity, FDR-controlled filter matching and full
run-level reproducibility is the claimed contribution); and a decision on whether pretrained language models are in scope. They are held back in
the current work.]
