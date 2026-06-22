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
logger = logging.getLogger("plotting")


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


def plot_learning_curves(
    history: dict[str, list[float]],
    output_dir: str | Path,
) -> None:
    """
    Plot training and validation loss curves + LR schedule.

    Args:
        history (dict): keys ``"train_loss"``, ``"val_loss"``, ``"lr"``.
        output_dir (str | Path): Directory where the PDF is saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Plotting learning curve ...")

    with plt.rc_context({"axes.autolimit_mode": "data"}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # loss
        ax = axes[0]
        ax.plot(history["train_loss"], c='r', ls='-', label="Train", linewidth=1.5)
        ax.plot(history["val_loss"], c='b', ls='--', label="Validation", linewidth=1.5)
        ax.set_xlabel("Epoch", fontsize=14)
        ax.set_ylabel("Loss (CE)", fontsize=14)
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # lr
        ax = axes[1]
        ax.plot(history["lr"], color="darkorange", linewidth=1.5)
        ax.set_xlabel("Epoch", fontsize=14)
        ax.set_ylabel("Learning Rate", fontsize=14)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        out = output_dir / "learning_curves.pdf"
        fig.savefig(out)
        plt.close(fig)

    logger.info("Saved: %s", out)


# def plot_score_distributions(
#     proba: np.ndarray,
#     labels: np.ndarray,
#     output_dir: str | Path,
# ) -> None:
#     """
#     Plot softmax score distributions for each output node.

#     One figure is produced with one panel per class (P_b, P_c, P_u, P_tau).
#     Inside each panel, the distribution is shown separately for every
#     true-label class, allowing direct reading of signal/background separation.

#     Args:
#         proba (np.ndarray): shape ``(N, n_classes)``, softmax probabilities.
#         labels (np.ndarray): shape ``(N,)``, true class labels.
#         output_dir (str | Path): directory where the PDF is saved.
#     """
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#     logger.info("Plotting output score distributions ...")

#     n_classes = proba.shape[1]
#     classes   = sorted(FLAVOUR_LABELS.keys())
#     bins      = np.linspace(0, 1, 50)

#     nrows = 2
#     ncols = int(n_classes / nrows)
#     fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
#     axes = axes.flatten()
#     if n_classes == 1:
#         axes = [axes]

#     for label_idx, ax in enumerate(axes):
#         label_name = FLAVOUR_LABELS.get(label_idx, f"class {label_idx}")
#         for cls in classes:
#             mask = labels == cls
#             if mask.sum() == 0:
#                 continue
#             ax.hist(
#                 proba[mask, label_idx],
#                 bins=bins,
#                 density=True,
#                 histtype="step",
#                 linewidth=1.5,
#                 color=FLAVOUR_COLORS[cls],
#                 label=FLAVOUR_LABELS[cls],
#             )
#         ax.set_xlabel(f"P({label_name})", fontsize=14)
#         ax.set_ylabel("Normalised entries", fontsize=14)
#         ax.set_yscale("log")
#         ax.legend()

#     fig.tight_layout()
#     out = output_dir / "score_distributions.pdf"
#     fig.savefig(out)
#     plt.close(fig)

#     logger.info("Saved: %s", out)


# def plot_confusion_matrix(
#     labels: np.ndarray,
#     preds: np.ndarray,
#     output_dir: str | Path,
# ) -> None:
#     """
#     Plot and save a normalised confusion matrix.

#     Args:
#         labels (np.ndarray): true class labels.
#         preds (np.ndarray): predicted class labels.
#         output_dir (str | Path): output directory.
#     """
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#     logger.info("Plotting confusion matrix ...")

#     classes     = sorted(FLAVOUR_LABELS.keys())
#     class_names = [FLAVOUR_LABELS[c] for c in classes]
#     conf_mat    = confusion_matrix(labels, preds, labels=classes, normalize="true")

#     fig, ax = plt.subplots(figsize=(6, 5))
#     im = ax.imshow(conf_mat, vmin=0, vmax=1, cmap="Blues", aspect="auto")
#     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

#     ax.set_xticks(range(len(class_names)))
#     ax.set_yticks(range(len(class_names)))
#     ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=11)
#     ax.set_yticklabels(class_names, fontsize=11)
#     ax.set_xlabel("Predicted class", fontsize=14)
#     ax.set_ylabel("True class", fontsize=14)

#     # write values in cells (white text for high values, black for low)
#     for i in range(len(class_names)):
#         for j in range(len(class_names)):
#             val   = conf_mat[i, j]
#             color = "white" if val > 0.6 else "black"
#             ax.text(j, i, f"{val:.2f}", ha="center", va="center",
#                     fontsize=11, color=color)

#     fig.tight_layout()
#     out = output_dir / "confusion_matrix.pdf"
#     fig.savefig(out)
#     plt.close(fig)

#     logger.info("Saved: %s", out)


def _roc_rejection(
    scores: np.ndarray,
    labels: np.ndarray,
    signal_class: int,
    bg_class: int,
    n_points: int = 200,
    eff_range: list = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate signal efficiency and background rejection for ROC curve.

    Args:
        scores (np.ndarray): Discriminant scores for all jets.
        labels (np.ndarray): True class labels for all jets.
        signal_class (int): Class index of the signal (e.g. ``b-jets``).
        bg_class (int): Class index of the background (e.g. ``c-jets``).
        n_points (int): Number of points on the ROC curve.
        eff_range (list): Minimum and maximum signal efficiency to include in the output
            (default ``[0.5, 1.0]``).

    Returns:
        eff (np.ndarray): Signal efficiency values.
        rej (np.ndarray): Background rejection values (``1 / bg efficiency``).
    """
    if eff_range is None:
        eff_range = [0.5, 1.0]
    thresholds = np.linspace(scores.min(), scores.max(), n_points)
    total_sig  = (labels == signal_class).sum()
    total_bg   = (labels == bg_class).sum()

    eff, rej = [], []
    for thr in thresholds:
        tagged_sig = ((scores >= thr) & (labels == signal_class)).sum()
        tagged_bg  = ((scores >= thr) & (labels == bg_class)).sum()
        sig_eff = tagged_sig / total_sig if total_sig > 0 else 0
        bg_eff  = tagged_bg  / total_bg  if total_bg  > 0 else 0
        eff.append(sig_eff)
        rej.append(1. / bg_eff if bg_eff > 0 else np.nan)

    eff = np.array(eff)
    rej = np.array(rej)

    mask = (eff >= eff_range[0]) & (eff <= eff_range[1]) & np.isfinite(rej)
    return eff[mask], rej[mask]


def _plot_roc(
    scores: np.ndarray,
    labels: np.ndarray,
    signal_class: int,
    bg_classes: list[tuple[int, str, str]],
    discriminant_type: str,
    output_dir: str | Path,
    eff_range: list = None,
) -> None:
    """
    Plot a ROC curve (signal efficiency vs background rejection).

    Args:
        scores (np.ndarray): discriminant scores for all jets.
        labels (np.ndarray): true class labels.
        signal_class (int): index of the signal class.
        bg_classes (list): list of ``(class_index, linestyle, legend_label)`` tuples.
        discriminant_type (str): name of the discriminant (e.g. "b" or "c", used for
            axis label and filename).
        output_dir (str | Path): output directory.
        eff_range (list): Minimum and maximum signal efficiency to include in the plot
            (default ``[0.5, 1.0]``).
    """
    if eff_range is None:
        eff_range = [0.5, 1.0]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Plotting ROC curve (%s-tag) ...", discriminant_type)

    fig, ax = plt.subplots(figsize=(7, 6))

    for bg_cls, linestyle, bg_label in bg_classes:
        eff, rej = _roc_rejection(scores, labels, signal_class, bg_cls, eff_range=eff_range)
        if eff.size == 0:
            logger.warning("No valid ROC points for `class %d` vs `class %d`.",
                           signal_class, bg_cls)
            continue
        ax.plot(eff, rej, linestyle=linestyle, linewidth=1.8, label=bg_label)

    ax.set_xlabel(f"{discriminant_type}-jet tagging efficiency", fontsize=14)
    ax.set_ylabel("Background rejection", fontsize=14)
    ax.set_yscale("log")
    ax.set_xlim(*eff_range)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=1.0)
    fig.tight_layout()
    out = output_dir / f"roc_d{discriminant_type}.pdf"
    fig.savefig(out)
    plt.close(fig)

    logger.info("Saved: %s", out)



if __name__ == "__main__":
    pass
