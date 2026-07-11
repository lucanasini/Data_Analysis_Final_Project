import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ._constants import SEED
from .feature_selection import lasso_selection, rfe_svm_selection, statistical_fdr_selection
from .plotting import plot_inclusion_probabilities

logger = logging.getLogger(f"{'cross-validation':<16}")


def CrossValidationBiased(
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits: int,
    n_repeats: int,
    parameters: list,
    kernel: str = "linear",
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
    n_features_to_select, _, lasso_C = parameters

    # 1. feature selection
    # SVM-RFE
    rfe_mask         = rfe_svm_selection(X, y, n_features_to_select=n_features_to_select)
    X_selected_rfe   = X.iloc[:, rfe_mask]
    # Lasso
    lasso_mask       = lasso_selection(X, y, C=lasso_C)
    X_selected_lasso = X.iloc[:, lasso_mask]

    # 2. CV
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)

    base = {
        "rfe":   [],
        "lasso": []
    }
    train_metrics = {
        "accuracy":  deepcopy(base), "loss":    deepcopy(base),
        "precision": deepcopy(base), "recall":  deepcopy(base),
        "f1-score":  deepcopy(base), "roc-auc": deepcopy(base)
    }
    val_metrics   = {
        "accuracy":  deepcopy(base), "loss":    deepcopy(base),
        "precision": deepcopy(base), "recall":  deepcopy(base),
        "f1-score":  deepcopy(base), "roc-auc": deepcopy(base)
    }
    for _, (train_idx, val_idx) in enumerate(rskf.split(X_selected_rfe, y)):
        X_train, y_train = X_selected_rfe.iloc[train_idx], y.iloc[train_idx]
        X_val,   y_val   = X_selected_rfe.iloc[val_idx],   y.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_val_scaled   = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )

        base_svc_rfe = SVC(kernel=kernel, random_state=SEED)
        model_rfe    = CalibratedClassifierCV(base_svc_rfe, ensemble=False)
        model_rfe.fit(X_train_scaled, y_train)

        train_preds_rfe = model_rfe.predict(X_train_scaled)
        train_probs_rfe = model_rfe.predict_proba(X_train_scaled)
        val_preds_rfe   = model_rfe.predict(X_val_scaled)
        val_probs_rfe   = model_rfe.predict_proba(X_val_scaled)

        train_metrics["accuracy"]["rfe"]  += [accuracy_score(y_train, train_preds_rfe)]
        train_metrics["loss"]["rfe"]      += [log_loss(y_train, train_probs_rfe,
                                                       labels=model_rfe.classes_)]
        train_metrics["precision"]["rfe"] += [precision_score(y_train, train_preds_rfe,
                                                              pos_label=model_rfe.classes_[1])]
        train_metrics["recall"]["rfe"]    += [recall_score(y_train, train_preds_rfe,
                                                           pos_label=model_rfe.classes_[1])]
        train_metrics["f1-score"]["rfe"]  += [f1_score(y_train, train_preds_rfe,
                                                       pos_label=model_rfe.classes_[1])]
        train_metrics["roc-auc"]["rfe"]   += [roc_auc_score(y_train, train_probs_rfe[:, 1],
                                                            labels=model_rfe.classes_)]
        
        val_metrics["accuracy"]["rfe"]    += [accuracy_score(y_val, val_preds_rfe)]
        val_metrics["loss"]["rfe"]        += [log_loss(y_val, val_probs_rfe,
                                                       labels=model_rfe.classes_)]
        val_metrics["precision"]["rfe"]   += [precision_score(y_val, val_preds_rfe,
                                                              pos_label=model_rfe.classes_[1])]
        val_metrics["recall"]["rfe"]      += [recall_score(y_val, val_preds_rfe,
                                                           pos_label=model_rfe.classes_[1])]
        val_metrics["f1-score"]["rfe"]    += [f1_score(y_val, val_preds_rfe,
                                                       pos_label=model_rfe.classes_[1])]
        val_metrics["roc-auc"]["rfe"]     += [roc_auc_score(y_val, val_probs_rfe[:, 1],
                                                            labels=model_rfe.classes_)]
        
    for _, (train_idx, val_idx) in enumerate(rskf.split(X_selected_lasso, y)):
        X_train, y_train = X_selected_lasso.iloc[train_idx], y.iloc[train_idx]
        X_val,   y_val   = X_selected_lasso.iloc[val_idx],   y.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_val_scaled   = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        base_svc_lasso = SVC(kernel=kernel, random_state=SEED)
        model_lasso    = CalibratedClassifierCV(base_svc_lasso, ensemble=False)
        model_lasso.fit(X_train_scaled, y_train)
        
        train_preds_lasso = model_lasso.predict(X_train_scaled)
        train_probs_lasso = model_lasso.predict_proba(X_train_scaled)
        val_preds_lasso   = model_lasso.predict(X_val_scaled)
        val_probs_lasso   = model_lasso.predict_proba(X_val_scaled)

        train_metrics["accuracy"]["lasso"]  += [accuracy_score(y_train, train_preds_lasso)]
        train_metrics["loss"]["lasso"]      += [log_loss(y_train, train_probs_lasso,
                                                         labels=model_lasso.classes_)]
        train_metrics["precision"]["lasso"] += [precision_score(y_train, train_preds_lasso,
                                                                pos_label=model_lasso.classes_[1])]
        train_metrics["recall"]["lasso"]    += [recall_score(y_train, train_preds_lasso,
                                                             pos_label=model_lasso.classes_[1])]
        train_metrics["f1-score"]["lasso"]  += [f1_score(y_train, train_preds_lasso,
                                                         pos_label=model_lasso.classes_[1])]
        train_metrics["roc-auc"]["lasso"]   += [roc_auc_score(y_train, train_probs_lasso[:, 1],
                                                              labels=model_lasso.classes_)]
        
        val_metrics["accuracy"]["lasso"]    += [accuracy_score(y_val, val_preds_lasso)]
        val_metrics["loss"]["lasso"]        += [log_loss(y_val, val_probs_lasso,
                                                         labels=model_lasso.classes_)]
        val_metrics["precision"]["lasso"]   += [precision_score(y_val, val_preds_lasso,
                                                                pos_label=model_lasso.classes_[1])]
        val_metrics["recall"]["lasso"]      += [recall_score(y_val, val_preds_lasso,
                                                             pos_label=model_lasso.classes_[1])]
        val_metrics["f1-score"]["lasso"]    += [f1_score(y_val, val_preds_lasso,
                                                         pos_label=model_lasso.classes_[1])]
        val_metrics["roc-auc"]["lasso"]     += [roc_auc_score(y_val, val_probs_lasso[:, 1],
                                                              labels=model_lasso.classes_)]

    train_rfe_mean_accuracy    = float(np.mean(train_metrics["accuracy"]["rfe"]))
    train_rfe_mean_loss        = float(np.mean(train_metrics["loss"]["rfe"]))
    train_rfe_mean_precision   = float(np.mean(train_metrics["precision"]["rfe"]))
    train_rfe_mean_recall      = float(np.mean(train_metrics["recall"]["rfe"]))
    train_rfe_mean_f1_score    = float(np.mean(train_metrics["f1-score"]["rfe"]))
    train_rfe_mean_roc_auc     = float(np.mean(train_metrics["roc-auc"]["rfe"]))
    train_lasso_mean_accuracy  = float(np.mean(train_metrics["accuracy"]["lasso"]))
    train_lasso_mean_loss      = float(np.mean(train_metrics["loss"]["lasso"]))
    train_lasso_mean_precision = float(np.mean(train_metrics["precision"]["lasso"]))
    train_lasso_mean_recall    = float(np.mean(train_metrics["recall"]["lasso"]))
    train_lasso_mean_f1_score  = float(np.mean(train_metrics["f1-score"]["lasso"]))
    train_lasso_mean_roc_auc   = float(np.mean(train_metrics["roc-auc"]["lasso"]))

    val_rfe_mean_accuracy    = float(np.mean(val_metrics["accuracy"]["rfe"]))
    val_rfe_mean_loss        = float(np.mean(val_metrics["loss"]["rfe"]))
    val_rfe_mean_precision   = float(np.mean(val_metrics["precision"]["rfe"]))
    val_rfe_mean_recall      = float(np.mean(val_metrics["recall"]["rfe"]))
    val_rfe_mean_f1_score    = float(np.mean(val_metrics["f1-score"]["rfe"]))
    val_rfe_mean_roc_auc     = float(np.mean(val_metrics["roc-auc"]["rfe"]))
    val_lasso_mean_accuracy  = float(np.mean(val_metrics["accuracy"]["lasso"]))
    val_lasso_mean_loss      = float(np.mean(val_metrics["loss"]["lasso"]))
    val_lasso_mean_precision = float(np.mean(val_metrics["precision"]["lasso"]))
    val_lasso_mean_recall    = float(np.mean(val_metrics["recall"]["lasso"]))
    val_lasso_mean_f1_score  = float(np.mean(val_metrics["f1-score"]["lasso"]))
    val_lasso_mean_roc_auc   = float(np.mean(val_metrics["roc-auc"]["lasso"]))
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


def CrossValidationUnbiased(
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits: int,
    n_repeats: int,
    parameters: list,
    kernel: str = "linear",
    plot_dir: str | Path = None,
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
        plot_dir (str | Path, optional): Directory to save plots.

    Returns:
        inclusion_probs (np.ndarray): Array of inclusion probabilities for each gene.
        train_metrics (list): List containing mean training accuracy and loss for RFE and Lasso.
        val_metrics (list): List containing mean validation accuracy and loss for RFE and Lasso
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)

    n_features_to_select, fdr_alpha, lasso_C = parameters

    rfe_selection_counts   = np.zeros(X.shape[1])
    fdr_selection_counts   = np.zeros(X.shape[1])
    lasso_selection_counts = np.zeros(X.shape[1])

    base = {
        "rfe":   [],
        "lasso": []
    }
    train_metrics = {
        "accuracy":  deepcopy(base), "loss":    deepcopy(base),
        "precision": deepcopy(base), "recall":  deepcopy(base),
        "f1-score":  deepcopy(base), "roc-auc": deepcopy(base)
    }
    val_metrics   = {
        "accuracy":  deepcopy(base), "loss":    deepcopy(base),
        "precision": deepcopy(base), "recall":  deepcopy(base),
        "f1-score":  deepcopy(base), "roc-auc": deepcopy(base)
    }
    for _, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val,   y_val   = X.iloc[val_idx],   y.iloc[val_idx]

        # 1. standardization
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_val_scaled   = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )

        # 2. internal feature selection
        # 1st approach: FDR
        fdr_mask, _ = statistical_fdr_selection(X_train_scaled, y_train, alpha=fdr_alpha)
        fdr_selection_counts += fdr_mask
        # 2nd approach: SVM-RFE
        rfe_mask = rfe_svm_selection(X_train_scaled, y_train,
                                     n_features_to_select=n_features_to_select)
        rfe_selection_counts += rfe_mask
        # 3rd approach: Lasso
        lasso_mask = lasso_selection(X_train_scaled, y_train, C=lasso_C)
        lasso_selection_counts += lasso_mask

        # 3. training
        # on RFE gene
        X_train_sel_rfe = X_train_scaled.iloc[:, rfe_mask]
        X_val_sel_rfe   = X_val_scaled.iloc[:, rfe_mask]
        base_svc_rfe = SVC(kernel=kernel, random_state=SEED)
        model_rfe = CalibratedClassifierCV(base_svc_rfe, ensemble=False)
        model_rfe.fit(X_train_sel_rfe, y_train)
        # on Lasso gene
        X_train_sel_lasso = X_train_scaled.iloc[:, lasso_mask]
        X_val_sel_lasso   = X_val_scaled.iloc[:, lasso_mask]
        base_svc_lasso = SVC(kernel=kernel, random_state=SEED)
        model_lasso = CalibratedClassifierCV(base_svc_lasso, ensemble=False)
        model_lasso.fit(X_train_sel_lasso, y_train)

        # 4. validation
        train_preds_rfe   = model_rfe.predict(X_train_sel_rfe)
        train_probs_rfe   = model_rfe.predict_proba(X_train_sel_rfe)
        train_preds_lasso = model_lasso.predict(X_train_sel_lasso)
        train_probs_lasso = model_lasso.predict_proba(X_train_sel_lasso)
        val_preds_rfe     = model_rfe.predict(X_val_sel_rfe)
        val_probs_rfe     = model_rfe.predict_proba(X_val_sel_rfe)
        val_preds_lasso   = model_lasso.predict(X_val_sel_lasso)
        val_probs_lasso   = model_lasso.predict_proba(X_val_sel_lasso)
        
        train_metrics["accuracy"]["rfe"]    += [accuracy_score(y_train, train_preds_rfe)]
        train_metrics["loss"]["rfe"]        += [log_loss(y_train, train_probs_rfe,
                                                         labels=model_rfe.classes_)]
        train_metrics["precision"]["rfe"]   += [precision_score(y_train, train_preds_rfe,
                                                                pos_label=model_rfe.classes_[1])]
        train_metrics["recall"]["rfe"]      += [recall_score(y_train, train_preds_rfe,
                                                             pos_label=model_rfe.classes_[1])]
        train_metrics["f1-score"]["rfe"]    += [f1_score(y_train, train_preds_rfe,
                                                         pos_label=model_rfe.classes_[1])]
        train_metrics["roc-auc"]["rfe"]     += [roc_auc_score(y_train, train_probs_rfe[:, 1],
                                                              labels=model_rfe.classes_)]
        train_metrics["accuracy"]["lasso"]  += [accuracy_score(y_train, train_preds_lasso)]
        train_metrics["loss"]["lasso"]      += [log_loss(y_train, train_probs_lasso,
                                                         labels=model_lasso.classes_)]
        train_metrics["precision"]["lasso"] += [precision_score(y_train, train_preds_lasso,
                                                                pos_label=model_lasso.classes_[1])]
        train_metrics["recall"]["lasso"]    += [recall_score(y_train, train_preds_lasso,
                                                             pos_label=model_lasso.classes_[1])]
        train_metrics["f1-score"]["lasso"]  += [f1_score(y_train, train_preds_lasso,
                                                         pos_label=model_lasso.classes_[1])]
        train_metrics["roc-auc"]["lasso"]   += [roc_auc_score(y_train, train_probs_lasso[:, 1],
                                                              labels=model_lasso.classes_)]

        val_metrics["accuracy"]["rfe"]      += [accuracy_score(y_val, val_preds_rfe)]
        val_metrics["loss"]["rfe"]          += [log_loss(y_val, val_probs_rfe,
                                                         labels=model_rfe.classes_)]
        val_metrics["precision"]["rfe"]     += [precision_score(y_val, val_preds_rfe,
                                                                pos_label=model_rfe.classes_[1])]
        val_metrics["recall"]["rfe"]        += [recall_score(y_val, val_preds_rfe,
                                                             pos_label=model_rfe.classes_[1])]
        val_metrics["f1-score"]["rfe"]      += [f1_score(y_val, val_preds_rfe,
                                                         pos_label=model_rfe.classes_[1])]
        val_metrics["roc-auc"]["rfe"]       += [roc_auc_score(y_val, val_probs_rfe[:, 1],
                                                              labels=model_rfe.classes_)]

        val_metrics["accuracy"]["lasso"]    += [accuracy_score(y_val, val_preds_lasso)]
        val_metrics["loss"]["lasso"]        += [log_loss(y_val, val_probs_lasso,
                                                         labels=model_lasso.classes_)]
        val_metrics["precision"]["lasso"]   += [precision_score(y_val, val_preds_lasso,
                                                                pos_label=model_lasso.classes_[1])]
        val_metrics["recall"]["lasso"]      += [recall_score(y_val, val_preds_lasso,
                                                             pos_label=model_lasso.classes_[1])]
        val_metrics["f1-score"]["lasso"]    += [f1_score(y_val, val_preds_lasso,
                                                         pos_label=model_lasso.classes_[1])]
        val_metrics["roc-auc"]["lasso"]     += [roc_auc_score(y_val, val_probs_lasso[:, 1],
                                                              labels=model_lasso.classes_)]

    # calculate inclusion probabilities
    total_runs = n_splits * n_repeats
    rfe_inclusion_prob   = rfe_selection_counts   / total_runs
    fdr_inclusion_prob   = fdr_selection_counts   / total_runs
    lasso_inclusion_prob = lasso_selection_counts / total_runs

    if plot_dir is not None:
        plot_inclusion_probabilities(
            rfe_inclusion_prob   = rfe_inclusion_prob,
            fdr_inclusion_prob   = fdr_inclusion_prob,
            lasso_inclusion_prob = lasso_inclusion_prob,
            output_dir = plot_dir,
        )

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
    train_rfe_mean_f1_score    = float(np.mean(train_metrics["f1-score"]["rfe"]))
    train_rfe_mean_roc_auc     = float(np.mean(train_metrics["roc-auc"]["rfe"]))
    train_lasso_mean_accuracy  = float(np.mean(train_metrics["accuracy"]["lasso"]))
    train_lasso_mean_loss      = float(np.mean(train_metrics["loss"]["lasso"]))
    train_lasso_mean_precision = float(np.mean(train_metrics["precision"]["lasso"]))
    train_lasso_mean_recall    = float(np.mean(train_metrics["recall"]["lasso"]))
    train_lasso_mean_f1_score  = float(np.mean(train_metrics["f1-score"]["lasso"]))
    train_lasso_mean_roc_auc   = float(np.mean(train_metrics["roc-auc"]["lasso"]))

    val_rfe_mean_accuracy    = float(np.mean(val_metrics["accuracy"]["rfe"]))
    val_rfe_mean_loss        = float(np.mean(val_metrics["loss"]["rfe"]))
    val_rfe_mean_precision   = float(np.mean(val_metrics["precision"]["rfe"]))
    val_rfe_mean_recall      = float(np.mean(val_metrics["recall"]["rfe"]))
    val_rfe_mean_f1_score    = float(np.mean(val_metrics["f1-score"]["rfe"]))
    val_rfe_mean_roc_auc     = float(np.mean(val_metrics["roc-auc"]["rfe"]))
    val_lasso_mean_accuracy  = float(np.mean(val_metrics["accuracy"]["lasso"]))
    val_lasso_mean_loss      = float(np.mean(val_metrics["loss"]["lasso"]))
    val_lasso_mean_precision = float(np.mean(val_metrics["precision"]["lasso"]))
    val_lasso_mean_recall    = float(np.mean(val_metrics["recall"]["lasso"]))
    val_lasso_mean_f1_score  = float(np.mean(val_metrics["f1-score"]["lasso"]))
    val_lasso_mean_roc_auc   = float(np.mean(val_metrics["roc-auc"]["lasso"]))
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
