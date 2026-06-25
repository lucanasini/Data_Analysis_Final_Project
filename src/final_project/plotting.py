"""
plotting.py
===========
Visualization module.
"""
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
logger = logging.getLogger(f"{'plotting':<16}")


def _draw_heatmap(ax, corr, labels, title, annotate=True):
    """
    Draw a heatmap of the correlation matrix with annotations.

    Args:
        ax: matplotlib axis to draw on.
        corr: 2D array of correlation coefficients.
        labels: list of variable names for axes.
        title: title of the plot.
    
    Returns:
        im: image object from imshow (for colorbar).
    """
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=11)
    if annotate:
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = corr[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)
    return im


def plot_correlations(
    df: pd.DataFrame,
    output_dir: str | Path,
    max_vars: int = 7139,
) -> None:
    """
    Plot Pearson correlation matrices for jet and track variables.

    Args:
        df (DataFrame): DataFrame of data file.
        output_dir (str | Path): Directory where PDFs are saved.

    Warnings:
        If more than `max_vars` numeric columns are found, only the first `max_vars`
        are plotted (to avoid unreadable / oversized figures).
    """
    logger.info("Plotting correlations ...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        logger.warning("No numeric columns found. Skipping correlation plot.")
        return

    if numeric_df.shape[1] > max_vars:
        logger.warning(
            "%d numeric columns found. Plotting the first %d only.",
            numeric_df.shape[1], max_vars,
        )
        numeric_df = numeric_df.iloc[:, :max_vars]

    corr_mat = numeric_df.corr(method="pearson")
    labels   = numeric_df.columns.tolist()
    n        = len(labels)

    fig, ax = plt.subplots(figsize=(min(20, max(8, n * 0.55)), min(20, max(7, n * 0.55))))
    im = _draw_heatmap(ax, corr_mat.values, labels, "Variables - Correlation", annotate=(n <= 40))
    if n > 40:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = output_dir / "correlation_matrix.pdf"
    fig.savefig(out)
    plt.close(fig)

    logger.info("Saved: %s", out)


def plot_norm_stats(
    norm_stats: dict[str, np.ndarray],
    feature_names: list[str],
    output_dir: str | Path,
    max_features: int = 100,
) -> None:
    """
    Plot per-feature mean and standard deviation used for normalization.

    NaN entries in ``norm_stats`` (e.g. columns excluded from scaling) are
    skipped automatically.

    Args:
        norm_stats (dict): dict with keys ``"mean"``, ``"sigma"`` (np.ndarray,
            aligned with ``feature_names``).
        feature_names (list[str]): column names, same order/length as the
            arrays in ``norm_stats``.
        output_dir (str | Path): directory where the PDF is saved.
        max_features (int): if more than this many valid features are found,
            only the first ``max_features`` are plotted (avoids unreadable /
            oversized figures).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Plotting normalization statistics ...")

    mean  = np.asarray(norm_stats["mean"])
    sigma = np.asarray(norm_stats["sigma"])

    mask = np.isfinite(mean) & np.isfinite(sigma)
    if not mask.any():
        logger.warning("No valid (non-NaN) normalization stats to plot.")
        return

    names = np.array(feature_names)[mask]
    mean  = np.sort(mean[mask])
    sigma = np.sort(sigma[mask])

    if len(names) > max_features:
        logger.warning(
            "%d normalized features found; plotting only the first %d.",
            len(names), max_features,
        )
        names, mean, sigma = names[:max_features], mean[:max_features], sigma[:max_features]

    n = len(names)
    x = np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(min(24, max(8, n * 0.3)), 8), sharex=True)

    ax = axes[0]
    ax.bar(x, mean, color="steelblue")
    ax.set_ylabel("Mean", fontsize=12)
    ax.set_title("Per-feature normalization statistics (training set)", fontsize=13)
    if n > 40:
        ax.set_xticks([])
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(x, sigma, color="darkorange")
    ax.set_ylabel("Std. dev.", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    if n > 40:
        ax.set_xticks([])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = output_dir / "norm_stats.pdf"
    fig.savefig(out)
    plt.close(fig)

    logger.info("Saved: %s", out)


def plot_inclusion_probabilities(
    rfe_inclusion_prob: np.ndarray,
    fdr_inclusion_prob: np.ndarray,
    lasso_inclusion_prob: np.ndarray,
    output_dir: str | Path,
) -> None:
    """
    Plot the full distribution of feature inclusion probabilities
    (across all genes) for both RFE and FDR selection, plus a ranked
    comparison of the top genes.
 
    Args:
        rfe_inclusion_prob (np.ndarray): inclusion probability per feature, RFE.
        fdr_inclusion_prob (np.ndarray): inclusion probability per feature, FDR.
        lasso_inclusion_prob (np.ndarray): inclusion probability per feature, Lasso.
        output_dir (str | Path): directory where PDFs are saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Plotting full inclusion-probability map ...")
 
    # 1. histograms: how many genes at each stability level
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
 
    bins = np.linspace(0, 1, 21)
    ax = axes[0]
    ax.hist(rfe_inclusion_prob, bins=bins, color="steelblue", edgecolor="k", alpha=0.8)
    ax.set_title("SVM-RFE: inclusion probability distribution")
    ax.set_xlabel("Inclusion probability")
    ax.set_ylabel("Number of genes")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
 
    ax = axes[1]
    ax.hist(fdr_inclusion_prob, bins=bins, color="darkorange", edgecolor="k", alpha=0.8)
    ax.set_title("FDR (BH): inclusion probability distribution")
    ax.set_xlabel("Inclusion probability")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.hist(lasso_inclusion_prob, bins=bins, color="green", edgecolor="k", alpha=0.8)
    ax.set_title("Lasso: inclusion probability distribution")
    ax.set_xlabel("Inclusion probability")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
 
    fig.tight_layout()
    out = output_dir / "inclusion_prob_histograms.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved: %s", out)
 
    # 2. sorted rank plot: shows the "cliff" between stable/unstable genes
    order = np.argsort(rfe_inclusion_prob)[::-1]
    sorted_rfe = rfe_inclusion_prob[order]
    sorted_fdr = fdr_inclusion_prob[order]
    sorted_lasso = lasso_inclusion_prob[order]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(sorted_rfe))
    ax.plot(x, sorted_rfe, label="SVM-RFE", color="steelblue", linewidth=1.5)
    ax.plot(x, sorted_fdr, label="FDR (BH)", color="darkorange", linewidth=1.5, alpha=0.8)
    ax.plot(x, sorted_lasso, label="Lasso", color="green", linewidth=1.5, alpha=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="50% threshold")
    ax.set_xlabel("Gene rank (sorted by RFE inclusion prob.)")
    ax.set_ylabel("Inclusion probability")
    ax.set_title("Ranked inclusion probabilities across all genes")
    ax.legend()
    ax.grid(True, alpha=0.3)
 
    fig.tight_layout()
    out = output_dir / "inclusion_prob_ranked.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved: %s", out)


def plot_bias_comparison(
    n_genes_list: list[int],
    biased_acc: list[float],
    unbiased_acc: list[float],
    chance_level: float,
    output_dir: str | Path,
    title_suffix: str = "",
    filename: str = "bias_comparison.pdf",
) -> None:
    """
    Plotta il confronto tra accuratezza "biased" (selezione fuori CV) e
    "unbiased" (selezione dentro CV) in funzione del numero di geni
    selezionati, replicando la Fig. 2 di Ambroise & McLachlan (2002).

    Args:
        n_genes_list (list[int]): numeri di geni testati (asse x).
        biased_acc (list[float]): accuratezza media, procedura biased.
        unbiased_acc (list[float]): accuratezza media, procedura unbiased.
        chance_level (float): accuratezza attesa per puro caso (1/n_classi).
        output_dir (str | Path): directory dove salvare il PDF.
        title_suffix (str): testo aggiuntivo per il titolo (es. " (permutato)").
        filename (str): nome del file di output.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Plotting bias comparison ...")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_genes_list, biased_acc, marker="o", color="crimson",
            label="Selezione FUORI CV (biased)")
    ax.plot(n_genes_list, unbiased_acc, marker="o", color="steelblue",
            label="Selezione DENTRO CV (unbiased)")
    ax.axhline(chance_level, color="gray", linestyle="--", linewidth=1,
               label=f"Chance level ({chance_level:.2f})")

    ax.set_xlabel("Numero di geni selezionati")
    ax.set_ylabel("Accuratezza media (CV)")
    ax.set_xscale("log")
    ax.set_title(f"Selection bias: biased vs unbiased{title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = output_dir / filename
    fig.savefig(out)
    plt.close(fig)

    logger.info("Saved: %s", out)


if __name__ == "__main__":
    pass
