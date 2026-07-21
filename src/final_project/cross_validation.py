import logging
from copy import deepcopy

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ._constants import SEED
from .feature_selection import lasso_selection, rfe_svm_selection, statistical_fdr_selection
from .utils import calculate_metrics

logger = logging.getLogger(f"{'cross-validation':<17}")


def _run_one_fold_biased(X, y, train_idx, val_idx, kernel):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val,   y_val   = X.iloc[val_idx],   y.iloc[val_idx]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),
                                  columns=X_train.columns,
                                  index=X_train.index)
    X_val_scaled   = pd.DataFrame(scaler.transform(X_val),
                                  columns=X_val.columns,
                                  index=X_val.index)

    model = CalibratedClassifierCV(SVC(kernel=kernel, random_state=SEED), ensemble=False)
    model.fit(X_train_scaled, y_train)

    train_preds, train_probs = model.predict(X_train_scaled), model.predict_proba(X_train_scaled)
    val_preds,   val_probs   = model.predict(X_val_scaled),   model.predict_proba(X_val_scaled)

    return {
        "train": calculate_metrics(y_train, train_preds, train_probs, model),
        "val": calculate_metrics(y_val, val_preds, val_probs, model),
    }

def _run_one_fold_unbiased(X, y, train_idx, val_idx, kernel, alpha, n_genes, C):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val,   y_val   = X.iloc[val_idx],   y.iloc[val_idx]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),
                                  columns=X_train.columns,
                                  index=X_train.index)
    X_val_scaled   = pd.DataFrame(scaler.transform(X_val),
                                  columns=X_val.columns,
                                  index=X_val.index)

    fdr_mask, _ = statistical_fdr_selection(X_train_scaled, y_train, alpha=alpha)
    rfe_mask    = rfe_svm_selection(X_train_scaled, y_train, n_genes=n_genes)
    lasso_mask  = lasso_selection(X_train_scaled, y_train, C=C)

    X_train_sel_rfe   = X_train_scaled.iloc[:, rfe_mask]
    X_val_sel_rfe     = X_val_scaled.iloc[:, rfe_mask]
    X_train_sel_lasso = X_train_scaled.iloc[:, lasso_mask]
    X_val_sel_lasso   = X_val_scaled.iloc[:, lasso_mask]

    model_rfe   = CalibratedClassifierCV(SVC(kernel=kernel, random_state=SEED), ensemble=False)
    model_rfe.fit(X_train_sel_rfe, y_train)
    model_lasso = CalibratedClassifierCV(SVC(kernel=kernel, random_state=SEED), ensemble=False)
    model_lasso.fit(X_train_sel_lasso, y_train)

    logger.debug(
        "Fold feature counts - RFE: %d | FDR: %d | Lasso: %d",
        rfe_mask.sum(), fdr_mask.sum(), lasso_mask.sum()
    )

    return {
        "fdr_mask": fdr_mask, "rfe_mask": rfe_mask, "lasso_mask": lasso_mask,
        "rfe":   {
            "train": calculate_metrics(y_train, model_rfe.predict(X_train_sel_rfe),
                                       model_rfe.predict_proba(X_train_sel_rfe), model_rfe),
            "val":   calculate_metrics(y_val,   model_rfe.predict(X_val_sel_rfe),
                                       model_rfe.predict_proba(X_val_sel_rfe),   model_rfe),
        },
        "lasso": {
            "train": calculate_metrics(y_train, model_lasso.predict(X_train_sel_lasso),
                                       model_lasso.predict_proba(X_train_sel_lasso), model_lasso),
            "val":   calculate_metrics(y_val,   model_lasso.predict(X_val_sel_lasso),
                                       model_lasso.predict_proba(X_val_sel_lasso),   model_lasso),
        },
    }


def cross_validation_biased(
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits: int,
    n_repeats: int,
    parameters: list,
    kernel: str = "linear",
    n_jobs: int = -1,
):
    """
    Run cross-validation with external feature selection (RFE + Lasso) and training of a linear SVM.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.DataFrame): Target vector.
        n_splits (int): Number of splits for CV.
        n_repeats (int): Number of repeats for CV.
        parameters (list): List of parameters for feature selection and modeling
            (``num_features``, ``alpha``, ``C``).
        kernel (str, optional): Kernel type for SVM. (Default: ``linear``)
    
    Returns:
        train_metrics (list): List containing mean training accuracy and loss for RFE and Lasso.
        val_metrics (list): List containing mean validation accuracy and loss for RFE and Lasso.
    """
    n_genes, _, lasso_C = parameters

    # 1. feature selection
    # SVM-RFE
    rfe_mask         = rfe_svm_selection(X, y, n_genes=n_genes)
    X_selected_rfe   = X.iloc[:, rfe_mask]
    # Lasso
    lasso_mask       = lasso_selection(X, y, C=lasso_C)
    X_selected_lasso = X.iloc[:, lasso_mask]

    # 2. CV
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)

    base = {"rfe": [], "lasso": []}
    train_metrics = {
        "accuracy":  deepcopy(base), "loss":    deepcopy(base),
        "precision": deepcopy(base), "recall":  deepcopy(base),
        "f1_score":  deepcopy(base), "roc_auc": deepcopy(base)
    }
    val_metrics   = {
        "accuracy":  deepcopy(base), "loss":    deepcopy(base),
        "precision": deepcopy(base), "recall":  deepcopy(base),
        "f1_score":  deepcopy(base), "roc_auc": deepcopy(base)
    }

    fold_results_rfe = Parallel(n_jobs=n_jobs)(
        delayed(_run_one_fold_biased)(X_selected_rfe, y, train_idx, val_idx, kernel)
        for train_idx, val_idx in rskf.split(X_selected_rfe, y)
    )

    for res in fold_results_rfe:
        for metric in train_metrics:
            train_metrics[metric]["rfe"].append(res["train"][metric])
            val_metrics[metric]["rfe"].append(res["val"][metric])
        
    fold_results_lasso = Parallel(n_jobs=n_jobs)(
        delayed(_run_one_fold_biased)(X_selected_lasso, y, train_idx, val_idx, kernel)
        for train_idx, val_idx in rskf.split(X_selected_lasso, y)
    )

    for res in fold_results_lasso:
        for metric in train_metrics:
            train_metrics[metric]["lasso"].append(res["train"][metric])
            val_metrics[metric]["lasso"].append(res["val"][metric])

    train_rfe_mean_accuracy    = float(np.mean(train_metrics["accuracy"]["rfe"]))
    train_rfe_mean_loss        = float(np.mean(train_metrics["loss"]["rfe"]))
    train_rfe_mean_precision   = float(np.mean(train_metrics["precision"]["rfe"]))
    train_rfe_mean_recall      = float(np.mean(train_metrics["recall"]["rfe"]))
    train_rfe_mean_f1_score    = float(np.mean(train_metrics["f1_score"]["rfe"]))
    train_rfe_mean_roc_auc     = float(np.mean(train_metrics["roc_auc"]["rfe"]))
    train_lasso_mean_accuracy  = float(np.mean(train_metrics["accuracy"]["lasso"]))
    train_lasso_mean_loss      = float(np.mean(train_metrics["loss"]["lasso"]))
    train_lasso_mean_precision = float(np.mean(train_metrics["precision"]["lasso"]))
    train_lasso_mean_recall    = float(np.mean(train_metrics["recall"]["lasso"]))
    train_lasso_mean_f1_score  = float(np.mean(train_metrics["f1_score"]["lasso"]))
    train_lasso_mean_roc_auc   = float(np.mean(train_metrics["roc_auc"]["lasso"]))

    val_rfe_mean_accuracy    = float(np.mean(val_metrics["accuracy"]["rfe"]))
    val_rfe_mean_loss        = float(np.mean(val_metrics["loss"]["rfe"]))
    val_rfe_mean_precision   = float(np.mean(val_metrics["precision"]["rfe"]))
    val_rfe_mean_recall      = float(np.mean(val_metrics["recall"]["rfe"]))
    val_rfe_mean_f1_score    = float(np.mean(val_metrics["f1_score"]["rfe"]))
    val_rfe_mean_roc_auc     = float(np.mean(val_metrics["roc_auc"]["rfe"]))
    val_lasso_mean_accuracy  = float(np.mean(val_metrics["accuracy"]["lasso"]))
    val_lasso_mean_loss      = float(np.mean(val_metrics["loss"]["lasso"]))
    val_lasso_mean_precision = float(np.mean(val_metrics["precision"]["lasso"]))
    val_lasso_mean_recall    = float(np.mean(val_metrics["recall"]["lasso"]))
    val_lasso_mean_f1_score  = float(np.mean(val_metrics["f1_score"]["lasso"]))
    val_lasso_mean_roc_auc   = float(np.mean(val_metrics["roc_auc"]["lasso"]))
    logger.info(
        "Biased CV completed. Mean Validation Metrics: RFE Accuracy = %.4f, RFE Loss = %.4f, "
        "Lasso Accuracy = %.4f, Lasso Loss = %.4f",
        val_rfe_mean_accuracy, val_rfe_mean_loss, val_lasso_mean_accuracy, val_lasso_mean_loss
    )
    return ([train_rfe_mean_accuracy, train_rfe_mean_loss,
             train_rfe_mean_precision, train_rfe_mean_recall,
             train_rfe_mean_f1_score, train_rfe_mean_roc_auc,
             train_lasso_mean_accuracy, train_lasso_mean_loss,
             train_lasso_mean_precision, train_lasso_mean_recall,
             train_lasso_mean_f1_score, train_lasso_mean_roc_auc],
            [val_rfe_mean_accuracy, val_rfe_mean_loss,
             val_rfe_mean_precision, val_rfe_mean_recall,
             val_rfe_mean_f1_score, val_rfe_mean_roc_auc,
             val_lasso_mean_accuracy, val_lasso_mean_loss,
             val_lasso_mean_precision, val_lasso_mean_recall,
             val_lasso_mean_f1_score, val_lasso_mean_roc_auc])


def cross_validation_unbiased(
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits: int,
    n_repeats: int,
    parameters: list,
    kernel: str = "linear",
    n_jobs: int = -1,
):
    """
    Run cross-validation with internal feature selection (RFE + RFE + Lasso)
    and training of a linear SVM.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.DataFrame): Target vector.
        n_splits (int): Number of splits for CV.
        n_repeats (int): Number of repeats for CV.
        parameters (list): List of parameters for feature selection and modeling
            (``num_features``, ``alpha``, ``C``).
        kernel (str, optional): Kernel type for SVM. (Default: ``linear``)

    Returns:
        inclusion_probs (np.ndarray): Array of inclusion probabilities for each gene.
        train_metrics (list): List containing mean training accuracy and loss for RFE and Lasso.
        val_metrics (list): List containing mean validation accuracy and loss for RFE and Lasso
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)

    n_genes, fdr_alpha, lasso_C = parameters

    rfe_selection_counts   = np.zeros(X.shape[1])
    fdr_selection_counts   = np.zeros(X.shape[1])
    lasso_selection_counts = np.zeros(X.shape[1])

    base = {"rfe": [], "lasso": []}
    train_metrics = {
        "accuracy":  deepcopy(base), "loss":    deepcopy(base),
        "precision": deepcopy(base), "recall":  deepcopy(base),
        "f1_score":  deepcopy(base), "roc_auc": deepcopy(base)
    }
    val_metrics   = {
        "accuracy":  deepcopy(base), "loss":    deepcopy(base),
        "precision": deepcopy(base), "recall":  deepcopy(base),
        "f1_score":  deepcopy(base), "roc_auc": deepcopy(base)
    }

    fold_results = Parallel(n_jobs=n_jobs)(
        delayed(_run_one_fold_unbiased)(X, y, train_idx, val_idx, kernel,
                                        fdr_alpha, n_genes, lasso_C)
        for train_idx, val_idx in rskf.split(X, y)
    )

    for res in fold_results:
        fdr_selection_counts   += res["fdr_mask"]
        rfe_selection_counts   += res["rfe_mask"]
        lasso_selection_counts += res["lasso_mask"]
        for metric in train_metrics:
            train_metrics[metric]["rfe"].append(res["rfe"]["train"][metric])
            val_metrics[metric]["rfe"].append(res["rfe"]["val"][metric])
            train_metrics[metric]["lasso"].append(res["lasso"]["train"][metric])
            val_metrics[metric]["lasso"].append(res["lasso"]["val"][metric])

    # calculate inclusion probabilities
    total_runs = n_splits * n_repeats
    rfe_inclusion_prob   = rfe_selection_counts   / total_runs
    fdr_inclusion_prob   = fdr_selection_counts   / total_runs
    lasso_inclusion_prob = lasso_selection_counts / total_runs

    # top 10 biomarkers for stability (SVM-RFE)
    top_rfe_idx = np.argsort(rfe_inclusion_prob)[::-1][:10]
    logger.info("=== TOP 10 BIOMARKS FOR STABILITY (SVM-RFE) ===")
    for idx in top_rfe_idx:
        logger.info("Gene: %-16s | Probability of Inclusion: %.2f",
                    X.columns[idx], rfe_inclusion_prob[idx])
    # top 10 biomarkers for stability (SVM-Lasso)
    top_lasso_idx = np.argsort(lasso_inclusion_prob)[::-1][:10]
    logger.info("=== TOP 10 BIOMARKS FOR STABILITY (SVM-Lasso) ===")
    for idx in top_lasso_idx:
        logger.info("Gene: %-16s | Probability of Inclusion: %.2f",
                    X.columns[idx], lasso_inclusion_prob[idx])

    # comparison of inclusion probabilities between methods
    all_three = np.sum(
        (rfe_inclusion_prob   > 0.5) &
        (fdr_inclusion_prob   > 0.5) &
        (lasso_inclusion_prob > 0.5)
    )
    logger.info(f"Stable genes (>50%) in common between FDR, RFE and Lasso: {all_three}")

    train_rfe_mean_accuracy    = float(np.mean(train_metrics["accuracy"]["rfe"]))
    train_rfe_mean_loss        = float(np.mean(train_metrics["loss"]["rfe"]))
    train_rfe_mean_precision   = float(np.mean(train_metrics["precision"]["rfe"]))
    train_rfe_mean_recall      = float(np.mean(train_metrics["recall"]["rfe"]))
    train_rfe_mean_f1_score    = float(np.mean(train_metrics["f1_score"]["rfe"]))
    train_rfe_mean_roc_auc     = float(np.mean(train_metrics["roc_auc"]["rfe"]))
    train_lasso_mean_accuracy  = float(np.mean(train_metrics["accuracy"]["lasso"]))
    train_lasso_mean_loss      = float(np.mean(train_metrics["loss"]["lasso"]))
    train_lasso_mean_precision = float(np.mean(train_metrics["precision"]["lasso"]))
    train_lasso_mean_recall    = float(np.mean(train_metrics["recall"]["lasso"]))
    train_lasso_mean_f1_score  = float(np.mean(train_metrics["f1_score"]["lasso"]))
    train_lasso_mean_roc_auc   = float(np.mean(train_metrics["roc_auc"]["lasso"]))

    val_rfe_mean_accuracy    = float(np.mean(val_metrics["accuracy"]["rfe"]))
    val_rfe_mean_loss        = float(np.mean(val_metrics["loss"]["rfe"]))
    val_rfe_mean_precision   = float(np.mean(val_metrics["precision"]["rfe"]))
    val_rfe_mean_recall      = float(np.mean(val_metrics["recall"]["rfe"]))
    val_rfe_mean_f1_score    = float(np.mean(val_metrics["f1_score"]["rfe"]))
    val_rfe_mean_roc_auc     = float(np.mean(val_metrics["roc_auc"]["rfe"]))
    val_lasso_mean_accuracy  = float(np.mean(val_metrics["accuracy"]["lasso"]))
    val_lasso_mean_loss      = float(np.mean(val_metrics["loss"]["lasso"]))
    val_lasso_mean_precision = float(np.mean(val_metrics["precision"]["lasso"]))
    val_lasso_mean_recall    = float(np.mean(val_metrics["recall"]["lasso"]))
    val_lasso_mean_f1_score  = float(np.mean(val_metrics["f1_score"]["lasso"]))
    val_lasso_mean_roc_auc   = float(np.mean(val_metrics["roc_auc"]["lasso"]))
    logger.info(
        "Unbiased CV completed. Mean Validation Metrics: RFE Accuracy = %.4f, RFE Loss = %.4f, "
        "Lasso Accuracy = %.4f, Lasso Loss = %.4f\n",
        val_rfe_mean_accuracy, val_rfe_mean_loss, val_lasso_mean_accuracy, val_lasso_mean_loss
    )
    return ([rfe_inclusion_prob,
             fdr_inclusion_prob,
             lasso_inclusion_prob],
            [train_rfe_mean_accuracy, train_rfe_mean_loss,
             train_rfe_mean_precision, train_rfe_mean_recall,
             train_rfe_mean_f1_score, train_rfe_mean_roc_auc,
             train_lasso_mean_accuracy, train_lasso_mean_loss,
             train_lasso_mean_precision, train_lasso_mean_recall,
             train_lasso_mean_f1_score, train_lasso_mean_roc_auc],
            [val_rfe_mean_accuracy, val_rfe_mean_loss,
             val_rfe_mean_precision, val_rfe_mean_recall,
             val_rfe_mean_f1_score, val_rfe_mean_roc_auc,
             val_lasso_mean_accuracy, val_lasso_mean_loss,
             val_lasso_mean_precision, val_lasso_mean_recall,
             val_lasso_mean_f1_score, val_lasso_mean_roc_auc])
