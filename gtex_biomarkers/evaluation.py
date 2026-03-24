"""Plotting and evaluation utilities for model results."""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    auc as sk_auc,
    confusion_matrix, ConfusionMatrixDisplay,
)

from gtex_biomarkers.config import Config


# ── Single-model plots ────────────────────────────────────────────────────────

def plot_roc_folds(results, title="", ax=None):
    """Plot per-fold ROC curves + mean on a single axis."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    mean_fpr = np.linspace(0, 1, 101)
    interp_tprs = []

    for i, (fpr, tpr, a) in enumerate(
        zip(results["fold_fprs"], results["fold_tprs"], results["fold_aucs"]), 1
    ):
        ax.plot(fpr, tpr, alpha=0.35, label=f"Fold {i} ({a:.3f})")
        interp_tprs.append(np.interp(mean_fpr, fpr, tpr))

    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
    mean_auc_val = sk_auc(mean_fpr, mean_tpr)

    ax.plot(mean_fpr, mean_tpr, lw=2, label=f"Mean ({mean_auc_val:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(
        f"{title}\nAUC = {results['mean_auc']:.3f} ± {results['std_auc']:.3f}",
        fontsize=9,
    )
    ax.legend(fontsize=6, loc="lower right")
    return ax


# ── Multi-model grid plots ───────────────────────────────────────────────────

def plot_roc_grid(results_dict, suptitle="", save_path=None, ncols=4):
    """ROC grid — one subplot per model."""
    tags = list(results_dict.keys())
    n = len(tags)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    for idx, tag in enumerate(tags):
        r, c = divmod(idx, ncols)
        plot_roc_folds(results_dict[tag], title=tag, ax=axes[r, c])
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(suptitle, y=1.02, fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_pr_grid(results_dict, suptitle="", save_path=None, ncols=4):
    """Precision-Recall grid — one subplot per model."""
    tags = list(results_dict.keys())
    n = len(tags)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    for idx, tag in enumerate(tags):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        res = results_dict[tag]
        mask = ~np.isnan(res["oof"])
        y_true = res["y"].values[mask]
        y_score = res["oof"][mask]
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        prevalence = y_true.mean()
        ax.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
        ax.axhline(prevalence, ls="--", color="grey", label=f"Baseline = {prevalence:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{tag}\nAP = {ap:.3f}", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6, loc="upper right")
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(suptitle, y=1.02, fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_cm_grid(results_dict, suptitle="", save_path=None, ncols=4):
    """Confusion matrix grid at Youden's J threshold."""
    tags = list(results_dict.keys())
    n = len(tags)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    for idx, tag in enumerate(tags):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        res = results_dict[tag]
        mask = ~np.isnan(res["oof"])
        thresh = res["optimal_threshold"]
        y_pred = (res["oof"][mask] >= thresh).astype(int)
        cm = confusion_matrix(res["y"].values[mask], y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
        disp.plot(ax=ax, values_format="d", colorbar=False)
        ax.set_title(f"{tag}\n(thresh = {thresh:.3f})", fontsize=8)
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(suptitle, y=1.02, fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_boxplot_grid(results_dict, suptitle="", save_path=None, ncols=4):
    """Box plots of predicted probability by ground-truth label."""
    tags = list(results_dict.keys())
    n = len(tags)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
    for idx, tag in enumerate(tags):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        res = results_dict[tag]
        y_true = res["y"].values
        y_score = res["oof"]
        mask = ~np.isnan(y_score)
        p0 = y_score[mask & (y_true == 0)]
        p1 = y_score[mask & (y_true == 1)]
        ax.boxplot([p0, p1], vert=False, tick_labels=["No", "Yes"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Ground truth")
        ax.set_title(tag, fontsize=9)
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(suptitle, y=1.02, fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


# ── Summary / comparison plots ────────────────────────────────────────────────

def plot_ranked_barplot(summary_df, auc_cutoff=None, save_path=None):
    """Horizontal bar chart ranking all models by mean AUC."""
    auc_cutoff = auc_cutoff or Config.AUC_CUTOFF
    s = summary_df.sort_values("mean_auc", ascending=True).copy()
    s["label"] = s["tissue"] + " | " + s["category"]
    colors = ["#2ecc71" if v >= auc_cutoff else "#95a5a6" for v in s["mean_auc"]]

    fig, ax = plt.subplots(figsize=(10, max(6, len(s) * 0.35)))
    ax.barh(range(len(s)), s["mean_auc"], color=colors,
            xerr=s["std_auc"], capsize=2, ecolor="black", alpha=0.85)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s["label"], fontsize=7)
    ax.axvline(auc_cutoff, ls="--", color="red", lw=1.2, label=f"AUC = {auc_cutoff}")
    ax.axvline(0.5, ls=":", color="grey", lw=0.8, label="Random (0.5)")
    ax.set_xlabel("Mean ROC-AUC (5-fold CV)")
    ax.set_title("All Models — Ranked by AUC", fontsize=12)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0.3, 1.0)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_auc_heatmap(summary_df, save_path=None):
    """Heatmap of AUC: tissue (rows) × category (columns)."""
    pivot = summary_df.pivot_table(index="tissue", columns="category", values="mean_auc")

    fig, ax = plt.subplots(
        figsize=(max(8, pivot.shape[1] * 1.2), max(6, pivot.shape[0] * 0.45))
    )
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.55 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Mean ROC-AUC")
    ax.set_title("AUC by Tissue × Category", fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_roc_overlay(results_dict, tags, title="", save_path=None):
    """Overlay ROC curves for selected models on one plot."""
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.colormaps["tab20"].resampled(max(1, len(tags)))

    for i, tag in enumerate(tags):
        res = results_dict[tag]
        mask = ~np.isnan(res["oof"])
        fpr, tpr, _ = roc_curve(res["y"].values[mask], res["oof"][mask])
        ax.plot(fpr, tpr, lw=1.8, color=cmap(i),
                label=f"{tag}  ({res['mean_auc']:.3f})")

    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_comparison_scatter(comp_df, save_path=None):
    """Scatter plot: LR AUC (x) vs RF AUC (y) per model."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(comp_df["mean_auc_lr"], comp_df["mean_auc_rf"],
               s=40, alpha=0.7, edgecolors="black", lw=0.5)
    ax.plot([0.3, 1], [0.3, 1], ls="--", color="grey", lw=1, label="Equal performance")
    ax.axhline(0.65, ls=":", color="red", lw=0.8, alpha=0.5)
    ax.axvline(0.65, ls=":", color="red", lw=0.8, alpha=0.5)

    top = comp_df.nlargest(5, "auc_diff")
    for _, row in top.iterrows():
        ax.annotate(f"{row['tissue'][:15]}|{row['category'][:10]}",
                    (row["mean_auc_lr"], row["mean_auc_rf"]),
                    fontsize=6, alpha=0.8, xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Logistic Regression AUC", fontsize=11)
    ax.set_ylabel("Random Forest AUC", fontsize=11)
    ax.set_title("Model Comparison: LR vs RF", fontsize=12)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_comparison_barplot(comp_df, save_path=None):
    """Horizontal bar chart of AUC difference (RF - LR) per model."""
    s = comp_df.dropna(subset=["auc_diff"]).sort_values("auc_diff", ascending=True)
    s["label"] = s["tissue"] + " | " + s["category"]
    colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in s["auc_diff"]]

    fig, ax = plt.subplots(figsize=(10, max(6, len(s) * 0.35)))
    ax.barh(range(len(s)), s["auc_diff"], color=colors, alpha=0.85)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s["label"], fontsize=7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("AUC difference (RF - LR)")
    ax.set_title("Per-Model AUC Change: RF vs LR", fontsize=12)
    ax.legend(handles=[Patch(color="#2ecc71", label="RF better"),
                       Patch(color="#e74c3c", label="LR better")],
              loc="lower right", fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig
