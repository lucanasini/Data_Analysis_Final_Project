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

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ._constants import SEED
from .plotting import plot_correlations
from .utils import calculate_metrics

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
    stable_genes_mask_lasso = np.argsort(lasso_inclusion_prob)[::-1][:]

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

    rfe_metrics   = calculate_metrics(y_test, test_preds_rfe,   test_probs_rfe,   final_model_rfe)
    lasso_metrics = calculate_metrics(y_test, test_preds_lasso, test_probs_lasso, final_model_lasso)

    joblib.dump(final_model_rfe, plot_dir / "run" / "rfe_model.joblib")
    joblib.dump(final_model_lasso, plot_dir / "run" / "lasso_model.joblib")

    logger.info(
        "Test RFE   -> Acc: %.4f | Loss: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | AUC: %.4f",
        rfe_metrics["accuracy"], rfe_metrics["loss"], rfe_metrics["precision"],
        rfe_metrics["recall"], rfe_metrics["f1_score"], rfe_metrics["roc_auc"]
    )
    logger.info(
        "Test Lasso -> Acc: %.4f | Loss: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | AUC: %.4f",
        lasso_metrics["accuracy"], lasso_metrics["loss"], lasso_metrics["precision"],
        lasso_metrics["recall"], lasso_metrics["f1_score"], lasso_metrics["roc_auc"]
    )

    return pd.DataFrame([{
        "rfe":   rfe_metrics,
        "lasso": lasso_metrics,
    }])
