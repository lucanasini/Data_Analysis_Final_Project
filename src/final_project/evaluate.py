"""
evaluate.py
===========
This module contains functions for evaluating the performance of machine
learning models on a test set. It includes functionality for selecting
stable genes based on their inclusion probabilities from Recursive Feature
Elimination (RFE) and plotting correlations among these genes.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ._constants import SEED
from .plotting import plot_correlations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(f"{'evaluate':<16}")


def evaluate(
    X_cv: pd.DataFrame,
    y_cv: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    n_features_to_select: int,
    rfe_inclusion_prob: np.ndarray,
    lasso_inclusion_prob: np.ndarray,
    kernel: str = "linear",
    plot_dir: str | Path = None
):
    """
    Evaluates the performance of a machine learning model on a test set using
    stable genes selected based on their inclusion probabilities from RFE.

    Args:
        X_cv (pd.DataFrame): Cross-validation feature matrix.
        y_cv (pd.DataFrame): Cross-validation target vector.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.DataFrame): Test target vector.
        n_features_to_select (int): Number of features to select based on
            RFE inclusion probabilities.
        rfe_inclusion_prob (np.ndarray): Array of inclusion probabilities
            for each gene from RFE.
        lasso_inclusion_prob (np.ndarray): Array of inclusion probabilities
            for each gene from Lasso.
        plot_dir (str | Path): Directory to save correlation plots (optional, default is ``None``).
    """
    stable_genes_mask_rfe = np.argsort(rfe_inclusion_prob)[::-1][:n_features_to_select]
    stable_genes_mask_lasso = np.argsort(lasso_inclusion_prob)[::-1][:n_features_to_select]

    stable_gene_names_rfe     = X_cv.columns[stable_genes_mask_rfe].tolist()
    df_stable_rfe             = X_cv[stable_gene_names_rfe].copy()
    df_stable_rfe["cancer"]   = y_cv.values
    stable_gene_names_lasso   = X_cv.columns[stable_genes_mask_lasso].tolist()
    df_stable_lasso           = X_cv[stable_gene_names_lasso].copy()
    df_stable_lasso["cancer"] = y_cv.values

    if plot_dir is not None:
        plot_correlations(
            df=df_stable_rfe.select_dtypes(include="number"),
            output_dir=plot_dir,
            plot_name="stable_genes_rfe"
        )
        plot_correlations(
            df=df_stable_lasso.select_dtypes(include="number"),
            output_dir=plot_dir,
            plot_name="stable_genes_lasso"
        )
        
    scaler = StandardScaler()
    X_cv   = scaler.fit_transform(X_cv)
    X_test = scaler.transform(X_test)

    final_model_rfe = SVC(kernel=kernel, probability=True, random_state=SEED)
    final_model_rfe.fit(X_cv[:, stable_genes_mask_rfe], y_cv)
    final_model_lasso = SVC(kernel=kernel, probability=True, random_state=SEED)
    final_model_lasso.fit(X_cv[:, stable_genes_mask_lasso], y_cv)

    test_preds_rfe   = final_model_rfe.predict(X_test[:, stable_genes_mask_rfe])
    test_probs_rfe   = final_model_rfe.predict_proba(X_test[:, stable_genes_mask_rfe])
    test_preds_lasso = final_model_lasso.predict(X_test[:, stable_genes_mask_lasso])
    test_probs_lasso = final_model_lasso.predict_proba(X_test[:, stable_genes_mask_lasso])

    test_rfe_accuracy    = accuracy_score(y_test, test_preds_rfe)
    test_lasso_accuracy  = accuracy_score(y_test, test_preds_lasso)
    test_rfe_loss        = log_loss(y_test, test_probs_rfe, labels=final_model_rfe.classes_)
    test_lasso_loss      = log_loss(y_test, test_probs_lasso, labels=final_model_lasso.classes_)
    test_rfe_precision   = precision_score(y_test, test_preds_rfe, pos_label=final_model_rfe.classes_[1])
    test_lasso_precision = precision_score(y_test, test_preds_lasso, pos_label=final_model_lasso.classes_[1])
    test_rfe_recall      = recall_score(y_test, test_preds_rfe, pos_label=final_model_rfe.classes_[1])
    test_lasso_recall    = recall_score(y_test, test_preds_lasso, pos_label=final_model_lasso.classes_[1])
    test_rfe_f1_score    = f1_score(y_test, test_preds_rfe, pos_label=final_model_rfe.classes_[1])
    test_lasso_f1_score  = f1_score(y_test, test_preds_lasso, pos_label=final_model_lasso.classes_[1])
    test_rfe_roc_auc     = roc_auc_score(y_test, test_probs_rfe[:, 1], labels=final_model_rfe.classes_)
    test_lasso_roc_auc   = roc_auc_score(y_test, test_probs_lasso[:, 1], labels=final_model_lasso.classes_)
    

    logger.info(
        "Test RFE   -> Acc: %.4f | Loss: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | AUC: %.4f",
        test_rfe_accuracy, test_rfe_loss, test_rfe_precision, test_rfe_recall, test_rfe_f1_score, test_rfe_roc_auc
    )
    logger.info(
        "Test Lasso -> Acc: %.4f | Loss: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | AUC: %.4f",
        test_lasso_accuracy, test_lasso_loss, test_lasso_precision, test_lasso_recall, test_lasso_f1_score, test_lasso_roc_auc
    )

    return pd.DataFrame([{
        "rfe":   dict(accuracy=test_rfe_accuracy, loss=test_rfe_loss, precision=test_rfe_precision,
                    recall=test_rfe_recall, f1=test_rfe_f1_score, roc_auc=test_rfe_roc_auc),
        "lasso": dict(accuracy=test_lasso_accuracy, loss=test_lasso_loss, precision=test_lasso_precision,
                    recall=test_lasso_recall, f1=test_lasso_f1_score, roc_auc=test_lasso_roc_auc),
    }])
