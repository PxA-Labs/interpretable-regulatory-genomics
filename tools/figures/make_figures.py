"""Regenerate the project's summary figures from the committed run records (experiments/runs/*/results/*.csv).

  python tools/figures/make_figures.py [--out docs/figures]

Palette: first three slots of the validated reference palette (blue, orange, aqua) + neutral gray; validated with the dataviz
skill's validate_palette.js (adjacent CVD dE 9.2, normal-vision dE 27.6; aqua is below 3:1 contrast on the light surface, so every
chart carries a legend plus distinct marker shapes, and the underlying tables are in docs/results/*.md). AUROC is drawn as dots
(not truncated bars) with chance (0.5) marked. Titles state the finding; all numbers come from the CSVs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = Path("experiments/runs")
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BIO = ["tcell", "k562", "gm12878"]
BIO_LABEL = {"tcell": "T cell", "k562": "K562", "gm12878": "GM12878"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 10.5})


def csv(run, name):
    return pd.read_csv(RUNS / run / "results" / name)


def tidy(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(length=0)


def fig_ladder(out):
    """AUROC as controls are added: full -> GC+CpG matched -> +repeat matched, per biosample."""
    rc = csv("20261119_irg-08-repeat-control", "repeat_control.csv")
    xt = csv("20261119_irg-09-xgb-tuned", "xgb_tuned.csv")
    xt = xt[xt.model == "xgb_k6_tuned"]
    cols = ["auroc_full", "auroc_gc_cpg_matched", "auroc_gc_cpg_rep_matched"]
    series = [("XGBoost k=6 (tuned)", xt, BLUE, "o", None), ("ShallowCNN", rc[rc.model == "shallow_cnn"], ORANGE, "s", None),
              ("AttnCNN", rc[rc.model == "attn_cnn"], AQUA, "^", None), ("GC + CpG + repeat features only", rc[rc.model == "feat6_lr"], GRAY, "D", "--")]
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 4.1), sharey=True)
    for ax, b in zip(axs, BIO):
        for name, df, c, mk, ls in series:
            r = df[df.biosample == b].iloc[0]
            ax.plot(range(3), [r[k] for k in cols], color=c, marker=mk, markersize=7, linewidth=2, linestyle=ls or "-", label=name,
                    markeredgecolor=SURF, markeredgewidth=1.2, zorder=3)
        ax.axhline(0.5, color=GRID, linewidth=1.2)
        ax.text(2.08, 0.5, "chance", color=INK2, fontsize=8.5, va="center")
        ax.set_xticks(range(3), ["full\ntest set", "GC+CpG\nmatched", "+ repeat\nmatched"])
        ax.set_xlim(-0.25, 2.55)
        ax.set_title(BIO_LABEL[b], loc="left")
        tidy(ax)
    axs[0].set_ylabel("test AUROC")
    axs[0].set_ylim(0.45, 0.95)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Controlling composition and repeats removes much of the apparent signal, for every model", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(out / "fig1_shortcut_ladder.png", dpi=200)
    plt.close(fig)


def fmt(v):
    """Signed 3-decimal label; values that round to zero are shown without a sign."""
    return "0.000" if abs(v) < 0.0005 else f"{v:+.3f}"


def fig_tuning(out):
    """ShallowCNN minus XGBoost AUROC, before and after tuning XGBoost (untuned = notebook-08 XGBoost k=6, 3 seeds; tuned = notebook 09)."""
    rc = csv("20261119_irg-08-repeat-control", "repeat_control.csv")
    xt = csv("20261119_irg-09-xgb-tuned", "xgb_tuned.csv")
    xt = xt[xt.model == "xgb_k6_tuned"]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    for ax, (col, title) in zip(axs, [("auroc_full", "full test set"), ("auroc_gc_cpg_matched", "GC + CpG matched test subset")]):
        for i, b in enumerate(BIO):
            cnn = rc[(rc.model == "shallow_cnn") & (rc.biosample == b)].iloc[0][col]
            untuned = rc[(rc.model == "xgb_k6") & (rc.biosample == b)].iloc[0][col]
            tuned = xt[xt.biosample == b].iloc[0][col]
            d0, d1 = cnn - untuned, cnn - tuned
            ax.plot([d0, d1], [i, i], color=GRID, linewidth=3, zorder=1, solid_capstyle="round")
            ax.plot(d0, i, "o", color=GRAY, markersize=9, markeredgecolor=SURF, markeredgewidth=1.5, zorder=3, label="vs untuned XGBoost" if i == 0 else None)
            ax.plot(d1, i, "o", color=BLUE, markersize=9, markeredgecolor=SURF, markeredgewidth=1.5, zorder=3, label="vs tuned XGBoost" if i == 0 else None)
            ax.text(d0, i + 0.22, fmt(d0), ha="center", fontsize=8.5, color=INK2)
            ax.text(d1, i - 0.34, fmt(d1), ha="center", fontsize=8.5, color=BLUE)
        ax.axvline(0, color=INK2, linewidth=1.2, zorder=2)
        ax.set_yticks(range(3), [BIO_LABEL[b] for b in BIO])
        ax.set_ylim(-0.7, 2.6)
        ax.set_xlim(-0.01, 0.025)
        ax.set_xticks([-0.01, 0, 0.01, 0.02], ["-0.01", "0", "+0.01", "+0.02"])
        ax.set_title(title, loc="left")
        ax.set_xlabel("ShallowCNN minus XGBoost, test AUROC  (right of 0 = CNN ahead)")
        tidy(ax, "x")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("The CNN's advantage largely disappears once XGBoost is tuned too", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(out / "fig2_tuning_effect.png", dpi=200)
    plt.close(fig)


def fig_motifs(out):
    """Share of CNN filters whose best JASPAR hit passes each criterion."""
    h = csv("20261119_irg-10-motif-fdr", "motif_hits_best_per_filter.csv")
    vals = [(h.p < 0.05).mean(), (h.p < 0.01).mean(), (h.min_q_filter < 0.05).mean()]
    labels = ["raw p < 0.05\n(no multiple-testing\ncorrection)", "raw p < 0.01\n(no multiple-testing\ncorrection)", "q < 0.05\n(Benjamini-Hochberg\nFDR, shuffle null)"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    bars = ax.bar(range(3), vals, color=[GRAY, GRAY, BLUE], width=0.58, edgecolor=SURF, linewidth=2)
    for r in bars:
        r.set_zorder(3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.025, f"{100 * v:.1f}%", ha="center", fontsize=11, fontweight="bold", color=BLUE if i == 2 else INK)
    ax.set_xticks(range(3), labels)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel(f"share of {len(h):,} CNN filters (30 models)\nwhose best JASPAR motif passes")
    ax.set_title("Nearly every filter matches some motif at raw p < 0.05;\nalmost none survive multiple-testing control", loc="left", fontsize=10.5)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(out / "fig3_motif_claim.png", dpi=200)
    plt.close(fig)


def fig_multiclass(out):
    m = csv("20261119_irg-12-multiclass-shap", "multiclass_metrics.csv")
    cls = ["PLS", "pELS", "dELS"]
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 4.0), sharey=True)
    spec = [("xgb_k4_multiclass", "auroc_vs_bg_full", "XGBoost, full test set", BLUE, "o"), ("xgb_k4_multiclass", "auroc_vs_bg_matched", "XGBoost, GC+CpG matched", ORANGE, "s"),
            ("gc_cpg_lr", "auroc_vs_bg_matched", "GC + CpG only, GC+CpG matched", GRAY, "D")]
    for ax, b in zip(axs, BIO):
        for name, col, label, c, mk in spec:
            d = m[(m.biosample == b) & (m.model == name)].set_index("class").loc[cls, col]
            ax.plot(range(3), d.values, linestyle="none", marker=mk, markersize=9, color=c, markeredgecolor=SURF, markeredgewidth=1.4, label=label, zorder=3)
        ax.axhline(0.5, color=GRID, linewidth=1.2)
        ax.text(-0.46, 0.483, "chance", color=INK2, fontsize=8.5, va="top", ha="left")
        ax.set_xticks(range(3), cls)
        ax.set_xlim(-0.5, 2.5)
        ax.set_title(BIO_LABEL[b], loc="left")
        tidy(ax)
    axs[0].set_ylabel("AUROC, class vs background")
    axs[0].set_ylim(0.45, 1.0)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Much of the promoters' apparent advantage is GC/CpG composition; distal enhancers stay hardest", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(out / "fig4_multiclass_auroc.png", dpi=200)
    plt.close(fig)


def fig_cpg_share(out):
    s = csv("20261119_irg-12-multiclass-shap", "shap_cg_share.csv")
    g = s.groupby(["biosample", "class"]).cg_share_of_abs_shap.mean().unstack()
    exp = float(s.cg_share_expected_if_no_preference.iloc[0])
    cls = ["PLS", "pELS", "dELS"]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    w = 0.25
    cols, mks = [BLUE, ORANGE, AQUA], ["T cell", "K562", "GM12878"]
    for j, b in enumerate(BIO):
        vals = [g.loc[b, c] for c in cls]
        bars = ax.bar(np.arange(3) + (j - 1) * w, vals, width=w - 0.02, color=cols[j], edgecolor=SURF, linewidth=2, label=BIO_LABEL[b], zorder=3)
        for x, v in zip(np.arange(3) + (j - 1) * w, vals):
            ax.text(x, v + 0.008, f"{v:.2f}", ha="center", fontsize=8.5, color=INK2)
    ax.hlines(exp, -0.55, 2.5, color=INK2, linestyle="--", linewidth=1.3, zorder=2)
    ax.text(2.58, exp, f"expected if CG-containing\nk-mers were used in proportion\nto their number ({exp:.2f})", ha="left", va="center", fontsize=8.5, color=INK2)
    ax.set_xticks(range(3), cls)
    ax.set_ylim(0, 0.5)
    ax.set_xlim(-0.55, 3.45)
    ax.set_ylabel("share of mean |SHAP| on CG-containing 4-mers")
    ax.set_title("The class signal follows a CpG gradient: promoters > proximal > distal enhancers", loc="left", fontsize=10.5)
    ax.legend(frameon=False, loc="upper right", ncol=3, bbox_to_anchor=(0.74, 1.0))
    tidy(ax)
    fig.tight_layout()
    fig.savefig(out / "fig5_cpg_gradient.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig_ladder, fig_tuning, fig_motifs, fig_multiclass, fig_cpg_share):
        f(out)
        print("wrote", f.__name__)
