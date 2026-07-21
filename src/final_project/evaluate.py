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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ._constants import SEED
from .plotting import plot_correlations
from .utils import calculate_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(f"{'evaluate':<17}")


def evaluate(
    X_cv: pd.DataFrame,
    y_cv: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    n_genes: int,
    rfe_inclusion_prob: np.ndarray,
    lasso_inclusion_prob: np.ndarray,
    kernel: str = "linear",
    plot_dir: str | Path = None,
    run_dir: str | Path = None
):
    """
    Evaluates the performance of a machine learning model on a test set using
    stable genes selected based on their inclusion probabilities from RFE.

    Args:
        X_cv (pd.DataFrame): Cross-validation feature matrix.
        y_cv (pd.DataFrame): Cross-validation target vector.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.DataFrame): Test target vector.
        n_genes (int): Number of features to select based on
            RFE inclusion probabilities.
        rfe_inclusion_prob (np.ndarray): Array of inclusion probabilities
            for each gene from RFE.
        lasso_inclusion_prob (np.ndarray): Array of inclusion probabilities
            for each gene from Lasso.
        plot_dir (str | Path): Directory to save correlation plots (optional, default is ``None``).
    """
    logger.info("=== Evaluate ===")

    stable_genes_mask_rfe = np.argsort(rfe_inclusion_prob)[::-1][:n_genes]
    stable_genes_mask_lasso = np.argsort(lasso_inclusion_prob)[::-1]
    stable_genes_mask_lasso = stable_genes_mask_lasso[
                                lasso_inclusion_prob[stable_genes_mask_lasso] != 0
                              ][:n_genes]

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

    final_model_rfe = CalibratedClassifierCV(SVC(kernel=kernel, random_state=SEED),
                                             ensemble=False)
    final_model_rfe.fit(X_cv[:, stable_genes_mask_rfe], y_cv)
    final_model_lasso = CalibratedClassifierCV(SVC(kernel=kernel, random_state=SEED),
                                               ensemble=False)
    final_model_lasso.fit(X_cv[:, stable_genes_mask_lasso], y_cv)

    train_preds_rfe   = final_model_rfe.predict(X_cv[:, stable_genes_mask_rfe])
    train_probs_rfe   = final_model_rfe.predict_proba(X_cv[:, stable_genes_mask_rfe])
    test_preds_rfe    = final_model_rfe.predict(X_test[:, stable_genes_mask_rfe])
    test_probs_rfe    = final_model_rfe.predict_proba(X_test[:, stable_genes_mask_rfe])
    train_preds_lasso = final_model_lasso.predict(X_cv[:, stable_genes_mask_lasso])
    train_probs_lasso = final_model_lasso.predict_proba(X_cv[:, stable_genes_mask_lasso])
    test_preds_lasso  = final_model_lasso.predict(X_test[:, stable_genes_mask_lasso])
    test_probs_lasso  = final_model_lasso.predict_proba(X_test[:, stable_genes_mask_lasso])

    train_rfe_metrics   = calculate_metrics(y_cv,   train_preds_rfe,
                                            train_probs_rfe,   final_model_rfe)
    test_rfe_metrics    = calculate_metrics(y_test, test_preds_rfe,
                                            test_probs_rfe,    final_model_rfe)
    train_lasso_metrics = calculate_metrics(y_cv,   train_preds_lasso,
                                            train_probs_lasso, final_model_lasso)
    test_lasso_metrics  = calculate_metrics(y_test, test_preds_lasso,
                                            test_probs_lasso,  final_model_lasso)

    joblib.dump(final_model_rfe, run_dir / "rfe_model.joblib")
    joblib.dump(final_model_lasso, run_dir / "lasso_model.joblib")

    logger.info(
        "Train RFE   -> Acc: %.4f | Loss: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | AUC: %.4f",
        train_rfe_metrics["accuracy"], train_rfe_metrics["loss"], train_rfe_metrics["precision"],
        train_rfe_metrics["recall"], train_rfe_metrics["f1_score"], train_rfe_metrics["roc_auc"]
    )
    logger.info(
        "Test RFE    -> Acc: %.4f | Loss: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | AUC: %.4f",
        test_rfe_metrics["accuracy"], test_rfe_metrics["loss"], test_rfe_metrics["precision"],
        test_rfe_metrics["recall"], test_rfe_metrics["f1_score"], test_rfe_metrics["roc_auc"]
    )
    logger.info(
        "Train Lasso -> Acc: %.4f | Loss: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | AUC: %.4f",
        train_lasso_metrics["accuracy"], train_lasso_metrics["loss"],
        train_lasso_metrics["precision"], train_lasso_metrics["recall"],
        train_lasso_metrics["f1_score"], train_lasso_metrics["roc_auc"]
    )
    logger.info(
        "Test Lasso  -> Acc: %.4f | Loss: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | AUC: %.4f",
        test_lasso_metrics["accuracy"], test_lasso_metrics["loss"],
        test_lasso_metrics["precision"], test_lasso_metrics["recall"],
        test_lasso_metrics["f1_score"], test_lasso_metrics["roc_auc"]
    )

    return pd.DataFrame([
        {"method": "rfe",   **test_rfe_metrics},
        {"method": "lasso", **test_lasso_metrics},
    ])
