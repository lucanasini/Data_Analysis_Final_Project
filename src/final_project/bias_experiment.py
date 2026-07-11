"""
bias_experiment.py
===================
Replica dell'esperimento di Ambroise & McLachlan (2002): confronto tra
selezione delle feature "biased" (fuori dalla CV) e "unbiased" (dentro
la CV) per evidenziare il selection bias.
"""
import logging
from copy import deepcopy
from itertools import product
from pathlib import Path

import pandas as pd

from .cross_validation import CrossValidationBiased, CrossValidationUnbiased
from .plotting import plot_bias_comparison

logger = logging.getLogger(f"{'bias-experiment':<16}")


def run_bias_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    n_repeats: int,
    parameters: list,
    plot_dir: str | Path = None,
):
    """
    Runs the selection bias experiment comparing biased and unbiased feature
    selection methods (RFE and Lasso) on the provided dataset.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        n_splits (int): Number of splits for CV.
        n_repeats (int): Number of repeats for CV.
        parameters (list, optional): List of parameters for feature selection and modeling
            (``genes``, ``alpha``, ``C``).
        plot_dir (str | Path, optional): Directory to save the plot.

    Returns:
        results (pd.DataFrame): DataFrame containing the results of the biase
            and unbiased feature selection methods.
        unbiased_inclusion_probs (list): List of inclusion probabilities for
            each gene from the unbiased method
    """
    genes, alphas, Cs = parameters[0], parameters[1], parameters[2]

    n_classes = y.nunique()
    chance_level = 1.0 / n_classes
    logger.info("Chance level (random guessing): %.4f", chance_level)

    base = {
        "rfe": {"train": [], "val": []},
        "lasso": {"train": [], "val": []}
    }
    biased_metrics = {
        "accuracy":  deepcopy(base), "loss":    deepcopy(base),
        "precision": deepcopy(base), "recall":  deepcopy(base),
        "f1-score":  deepcopy(base), "roc-auc": deepcopy(base)
    }
    unbiased_metrics = {
        "accuracy":  deepcopy(base), "loss":    deepcopy(base),
        "precision": deepcopy(base), "recall":  deepcopy(base),
        "f1-score":  deepcopy(base), "roc-auc": deepcopy(base)
    }
    unbiased_inclusion_probs = []
    parameter                = []
    for gene, alpha, C in product(genes, alphas, Cs):
        logger.info("=== n_genes = %d | alpha = %f | C = %f ===", gene, alpha, C)
        train_biased_metric, val_biased_metric = CrossValidationBiased(
            X, y,
            n_splits=n_splits,
            n_repeats=n_repeats,
            parameters=[gene, alpha, C],
        )
        unbiased_inclusion_prob, train_unbiased_metric, val_unbiased_metric = CrossValidationUnbiased(
            X, y,
            n_splits=n_splits,
            n_repeats=n_repeats,
            parameters=[gene, alpha, C],
            plot_dir=plot_dir,
        )

        biased_metrics["accuracy"]["rfe"]["train"]      += [train_biased_metric[0]]
        biased_metrics["loss"]["rfe"]["train"]          += [train_biased_metric[1]]
        biased_metrics["precision"]["rfe"]["train"]     += [train_biased_metric[2]]
        biased_metrics["recall"]["rfe"]["train"]        += [train_biased_metric[3]]
        biased_metrics["f1-score"]["rfe"]["train"]      += [train_biased_metric[4]]
        biased_metrics["roc-auc"]["rfe"]["train"]       += [train_biased_metric[5]]
        biased_metrics["accuracy"]["lasso"]["train"]    += [train_biased_metric[6]]
        biased_metrics["loss"]["lasso"]["train"]        += [train_biased_metric[7]]
        biased_metrics["precision"]["lasso"]["train"]   += [train_biased_metric[8]]
        biased_metrics["recall"]["lasso"]["train"]      += [train_biased_metric[9]]
        biased_metrics["f1-score"]["lasso"]["train"]    += [train_biased_metric[10]]
        biased_metrics["roc-auc"]["lasso"]["train"]     += [train_biased_metric[11]]
        unbiased_metrics["accuracy"]["rfe"]["train"]    += [train_unbiased_metric[0]]
        unbiased_metrics["loss"]["rfe"]["train"]        += [train_unbiased_metric[1]]
        unbiased_metrics["precision"]["rfe"]["train"]   += [train_unbiased_metric[2]]
        unbiased_metrics["recall"]["rfe"]["train"]      += [train_unbiased_metric[3]]
        unbiased_metrics["f1-score"]["rfe"]["train"]    += [train_unbiased_metric[4]]
        unbiased_metrics["roc-auc"]["rfe"]["train"]     += [train_unbiased_metric[5]]
        unbiased_metrics["accuracy"]["lasso"]["train"]  += [train_unbiased_metric[6]]
        unbiased_metrics["loss"]["lasso"]["train"]      += [train_unbiased_metric[7]]
        unbiased_metrics["precision"]["lasso"]["train"] += [train_unbiased_metric[8]]
        unbiased_metrics["recall"]["lasso"]["train"]    += [train_unbiased_metric[9]]
        unbiased_metrics["f1-score"]["lasso"]["train"]  += [train_unbiased_metric[10]]
        unbiased_metrics["roc-auc"]["lasso"]["train"]   += [train_unbiased_metric[11]]

        biased_metrics["accuracy"]["rfe"]["val"]        += [val_biased_metric[0]]
        biased_metrics["loss"]["rfe"]["val"]            += [val_biased_metric[1]]
        biased_metrics["precision"]["rfe"]["val"]       += [val_biased_metric[2]]
        biased_metrics["recall"]["rfe"]["val"]          += [val_biased_metric[3]]
        biased_metrics["f1-score"]["rfe"]["val"]        += [val_biased_metric[4]]
        biased_metrics["roc-auc"]["rfe"]["val"]         += [val_biased_metric[5]]
        biased_metrics["accuracy"]["lasso"]["val"]      += [val_biased_metric[6]]
        biased_metrics["loss"]["lasso"]["val"]          += [val_biased_metric[7]]
        biased_metrics["precision"]["lasso"]["val"]     += [val_biased_metric[8]]
        biased_metrics["recall"]["lasso"]["val"]        += [val_biased_metric[9]]
        biased_metrics["f1-score"]["lasso"]["val"]      += [val_biased_metric[10]]
        biased_metrics["roc-auc"]["lasso"]["val"]       += [val_biased_metric[11]]
        unbiased_metrics["accuracy"]["rfe"]["val"]      += [val_unbiased_metric[0]]
        unbiased_metrics["loss"]["rfe"]["val"]          += [val_unbiased_metric[1]]
        unbiased_metrics["precision"]["rfe"]["val"]     += [val_unbiased_metric[2]]
        unbiased_metrics["recall"]["rfe"]["val"]        += [val_unbiased_metric[3]]
        unbiased_metrics["f1-score"]["rfe"]["val"]      += [val_unbiased_metric[4]]
        unbiased_metrics["roc-auc"]["rfe"]["val"]       += [val_unbiased_metric[5]]
        unbiased_metrics["accuracy"]["lasso"]["val"]    += [val_unbiased_metric[6]]
        unbiased_metrics["loss"]["lasso"]["val"]        += [val_unbiased_metric[7]]
        unbiased_metrics["precision"]["lasso"]["val"]   += [val_unbiased_metric[8]]
        unbiased_metrics["recall"]["lasso"]["val"]      += [val_unbiased_metric[9]]
        unbiased_metrics["f1-score"]["lasso"]["val"]    += [val_unbiased_metric[10]]
        unbiased_metrics["roc-auc"]["lasso"]["val"]     += [val_unbiased_metric[11]]

        unbiased_inclusion_probs += [unbiased_inclusion_prob]
        parameter                += [[gene, alpha, C]]


    results = pd.DataFrame({
        "parameters": parameter,

        "biased_rfe_train_acc":           biased_metrics["accuracy"]["rfe"]["train"],
        "biased_rfe_val_acc":             biased_metrics["accuracy"]["rfe"]["val"],

        "biased_rfe_train_loss":          biased_metrics["loss"]["rfe"]["train"],
        "biased_rfe_val_loss":            biased_metrics["loss"]["rfe"]["val"],

        "biased_rfe_train_precision":     biased_metrics["precision"]["rfe"]["train"],
        "biased_rfe_val_precision":       biased_metrics["precision"]["rfe"]["val"],

        "biased_rfe_train_recall":        biased_metrics["recall"]["rfe"]["train"],
        "biased_rfe_val_recall":          biased_metrics["recall"]["rfe"]["val"],

        "biased_rfe_train_f1_score":      biased_metrics["f1-score"]["rfe"]["train"],
        "biased_rfe_val_f1_score":        biased_metrics["f1-score"]["rfe"]["val"],

        "biased_rfe_train_roc_auc":       biased_metrics["roc-auc"]["rfe"]["train"],
        "biased_rfe_val_roc_auc":         biased_metrics["roc-auc"]["rfe"]["val"],

        "biased_lasso_train_acc":         biased_metrics["accuracy"]["lasso"]["train"],
        "biased_lasso_val_acc":           biased_metrics["accuracy"]["lasso"]["val"],

        "biased_lasso_train_loss":        biased_metrics["loss"]["lasso"]["train"],
        "biased_lasso_val_loss":          biased_metrics["loss"]["lasso"]["val"],

        "biased_lasso_train_precision":   biased_metrics["precision"]["lasso"]["train"],
        "biased_lasso_val_precision":     biased_metrics["precision"]["lasso"]["val"],

        "biased_lasso_train_recall":      biased_metrics["recall"]["lasso"]["train"],
        "biased_lasso_val_recall":        biased_metrics["recall"]["lasso"]["val"],

        "biased_lasso_train_f1_score":    biased_metrics["f1-score"]["lasso"]["train"],
        "biased_lasso_val_f1_score":      biased_metrics["f1-score"]["lasso"]["val"],

        "biased_lasso_train_roc_auc":     biased_metrics["roc-auc"]["lasso"]["train"],
        "biased_lasso_val_roc_auc":       biased_metrics["roc-auc"]["lasso"]["val"],

        "unbiased_rfe_train_acc":         unbiased_metrics["accuracy"]["rfe"]["train"],
        "unbiased_rfe_val_acc":           unbiased_metrics["accuracy"]["rfe"]["val"],

        "unbiased_rfe_train_loss":        unbiased_metrics["loss"]["rfe"]["train"],
        "unbiased_rfe_val_loss":          unbiased_metrics["loss"]["rfe"]["val"],

        "unbiased_rfe_train_precision":   unbiased_metrics["precision"]["rfe"]["train"],
        "unbiased_rfe_val_precision":     unbiased_metrics["precision"]["rfe"]["val"],

        "unbiased_rfe_train_recall":      unbiased_metrics["recall"]["rfe"]["train"],
        "unbiased_rfe_val_recall":        unbiased_metrics["recall"]["rfe"]["val"],

        "unbiased_rfe_train_f1_score":    unbiased_metrics["f1-score"]["rfe"]["train"],
        "unbiased_rfe_val_f1_score":      unbiased_metrics["f1-score"]["rfe"]["val"],

        "unbiased_rfe_train_roc_auc":     unbiased_metrics["roc-auc"]["rfe"]["train"],
        "unbiased_rfe_val_roc_auc":       unbiased_metrics["roc-auc"]["rfe"]["val"],

        "unbiased_lasso_train_acc":       unbiased_metrics["accuracy"]["lasso"]["train"],
        "unbiased_lasso_val_acc":         unbiased_metrics["accuracy"]["lasso"]["val"],

        "unbiased_lasso_train_loss":      unbiased_metrics["loss"]["lasso"]["train"],
        "unbiased_lasso_val_loss":        unbiased_metrics["loss"]["lasso"]["val"],

        "unbiased_lasso_train_precision": unbiased_metrics["precision"]["lasso"]["train"],
        "unbiased_lasso_val_precision":   unbiased_metrics["precision"]["lasso"]["val"],

        "unbiased_lasso_train_recall":    unbiased_metrics["recall"]["lasso"]["train"],
        "unbiased_lasso_val_recall":      unbiased_metrics["recall"]["lasso"]["val"],

        "unbiased_lasso_train_f1_score":  unbiased_metrics["f1-score"]["lasso"]["train"],
        "unbiased_lasso_val_f1_score":    unbiased_metrics["f1-score"]["lasso"]["val"],

        "unbiased_lasso_train_roc_auc":   unbiased_metrics["roc-auc"]["lasso"]["train"],
        "unbiased_lasso_val_roc_auc":     unbiased_metrics["roc-auc"]["lasso"]["val"],
    })

    # logger.info("\n%s", results.to_string(index=False))

    if plot_dir is not None:
        plot_bias_comparison(
            results=results,
            chance_level=chance_level,
            output_dir=plot_dir,
        )

    return results, unbiased_inclusion_probs
