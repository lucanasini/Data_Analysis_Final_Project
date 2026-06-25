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
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

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
    fdr_inclusion_prob: np.ndarray,
    rfe_inclusion_prob: np.ndarray,
    lasso_inclusion_prob: np.ndarray,
    plot_dir: str | Path = None
):
    """
    Evaluates the performance of a machine learning model on a test set using stable genes selected based on their inclusion probabilities from RFE.

    Args:
        X_cv (pd.DataFrame): Cross-validation feature matrix.
        y_cv (pd.DataFrame): Cross-validation target vector.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.DataFrame): Test target vector.
        n_features_to_select (int): Number of features to select based on RFE inclusion probabilities.
        fdr_inclusion_prob (np.ndarray): Array of inclusion probabilities for each gene from FDR.
        rfe_inclusion_prob (np.ndarray): Array of inclusion probabilities for each gene from RFE.
        lasso_inclusion_prob (np.ndarray): Array of inclusion probabilities for each gene from Lasso.
        plot_dir (str | Path): Directory to save correlation plots (optional, default is ``None``).
    """
    # select genes that have a global stability > 50% in the CV process
    stable_genes_mask_rfe = rfe_inclusion_prob > 0.5
    if np.sum(stable_genes_mask_rfe) == 0:              # Fallback if none exceed the 50% threshold
        stable_genes_mask_rfe = np.argsort(rfe_inclusion_prob)[::-1][:n_features_to_select]
    stable_genes_mask_lasso = lasso_inclusion_prob > 0.5
    if np.sum(stable_genes_mask_lasso) == 0:            # Fallback if none exceed the 50% threshold
        stable_genes_mask_lasso = np.argsort(lasso_inclusion_prob)[::-1][:n_features_to_select]

    stable_gene_names_rfe   = X_cv.columns[stable_genes_mask_rfe].tolist()
    df_stable_rfe           = X_cv[stable_gene_names_rfe].copy()
    df_stable_rfe["cancer"] = y_cv.values
    stable_gene_names_lasso   = X_cv.columns[stable_genes_mask_lasso].tolist()
    df_stable_lasso           = X_cv[stable_gene_names_lasso].copy()
    df_stable_lasso["cancer"] = y_cv.values

    if plot_dir is not None:
        plot_correlations(
            df=df_stable_rfe.select_dtypes(include="number"),
            output_dir=plot_dir / "stable_genes_correlation_rfe",
        )
        plot_correlations(
            df=df_stable_lasso.select_dtypes(include="number"),
            output_dir=plot_dir / "stable_genes_correlation_lasso",
        )
        
    scaler = StandardScaler()
    X_cv   = scaler.fit_transform(X_cv)
    X_test = scaler.transform(X_test)

    final_model_rfe = SVC(kernel="linear", probability=True, random_state=42)
    final_model_rfe.fit(X_cv[:, stable_genes_mask_rfe], y_cv)
    final_model_lasso = SVC(kernel="linear", probability=True, random_state=42)
    final_model_lasso.fit(X_cv[:, stable_genes_mask_lasso], y_cv)

    test_preds_rfe   = final_model_rfe.predict(X_test[:, stable_genes_mask_rfe])
    test_preds_lasso = final_model_lasso.predict(X_test[:, stable_genes_mask_lasso])

    logger.info(f"Test Set Accuracy RFE (Stable Genes):   {accuracy_score(y_test, test_preds_rfe):.4f}")
    logger.info(f"Test Set Accuracy Lasso (Stable Genes): {accuracy_score(y_test, test_preds_lasso):.4f}")
