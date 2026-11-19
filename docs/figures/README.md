# Figures

Regenerate all figures from the committed run records with `python tools/figures/make_figures.py` (needs `pandas`, `matplotlib`).
Every number is read from `experiments/runs/*/results/*.csv`; the same numbers are tabulated in `docs/results/*.md` (table view).
AUROC is drawn as dots, not truncated bars, with chance (0.5) marked. Colours: blue / orange / aqua with distinct marker shapes, gray for reference or
control series (palette checked for colour-blind separation; aqua is below 3:1 contrast on the light surface, hence the legends and marker shapes).

| Figure | What it shows | Source | Write-up |
|---|---|---|---|
| `fig1_shortcut_ladder.png` | Test AUROC of tuned XGBoost (wide search), ShallowCNN and AttnCNN as controls are added (full, GC+CpG matched, + repeat matched), with a six-feature composition-and-repeat baseline in gray | runs 08, 11 | `docs/results/repeat_control.md` |
| `fig2_tuning_effect.png` | ShallowCNN minus XGBoost AUROC against an untuned, a tuned (nb09) and a wide-search tuned (nb11) XGBoost, full and matched test sets | runs 08, 09, 11 | `docs/results/xgb_wide.md` |
| `fig3_motif_claim.png` | Share of 3,699 CNN filters whose best JASPAR motif passes raw p < 0.05, raw p < 0.01, and BH q < 0.05 | run 10 | `docs/results/motif_fdr.md` |
| `fig4_multiclass_auroc.png` | AUROC of each element class against background: XGBoost on the full test set, XGBoost on the GC+CpG-matched subset, and a GC+CpG-only model on the matched subset | run 12 | `docs/results/multiclass_shap.md` |
| `fig5_cpg_gradient.png` | Share of mean absolute SHAP on CG-containing 4-mers per class, against the share expected if such k-mers were used in proportion to their number | run 12 | `docs/results/multiclass_shap.md` |

Caveats that apply to the figures: the matched subsets are sub-populations (not comparable with full-task AUROC, and not mutually comparable across
classes in figure 4); the matching is coarse, so matched AUROCs are upper bounds on composition-independent signal; in figure 2 the "untuned" XGBoost
is the notebook-08 XGBoost k=6 (3 seeds) and the tuned ones are notebook 09 and notebook 11, whose searches were truncated (see `xgb_tuned.md`, `xgb_wide.md`).
