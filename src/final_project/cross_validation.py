import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold

from .feature_selection import statistical_fdr_selection, rfe_svm_selection, lasso_selection
from .plotting import plot_inclusion_probabilities

logger = logging.getLogger(f"{'cross-validation':<16}")

def CrossValidation(
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits: int,
    n_repeats: int,
    parameters: list,
    plot_dir: str | Path = None,
):
    """
    Run cross-validation with internal feature selection (FDR + RFE + Lasso) and training of a linear SVM.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.DataFrame): Target vector.
        n_splits (int): Number of splits for cross-validation.
        n_repeats (int): Number of repeats for cross-validation.
        parameters (list): List of parameters for feature selection and modeling.
        plot_dir (str | Path, optional): Directory to save plots.

    Returns:
        rfe_inclusion_prob (np.ndarray): Array with inclusion probabilities for features selected by RFE.
        fdr_inclusion_prob (np.ndarray): Array with inclusion probabilities for features selected by FDR.
        lasso_inclusion_prob (np.ndarray): Array with inclusion probabilities for features selected by Lasso.
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    n_features_to_select, fdr_alpha, lasso_C = parameters

    rfe_selection_counts   = np.zeros(X.shape[1])
    fdr_selection_counts   = np.zeros(X.shape[1])
    lasso_selection_counts = np.zeros(X.shape[1]) 

    logger.info("Starting Cross-Validation (%d total iterations)...", n_splits * n_repeats)

    val_accuracies = {"rfe": 0., "lasso": 0.}
    val_losses     = {"rfe": 0., "lasso": 0.}
    for i, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val     = X.iloc[val_idx],   y.iloc[val_idx]
        
        # 1. standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 2. internal feature selection
        # 1st approach: FDR
        fdr_mask, _ = statistical_fdr_selection(X_train, y_train, alpha=fdr_alpha)
        fdr_selection_counts += fdr_mask
        # 2nd approach: SVM-RFE
        rfe_mask = rfe_svm_selection(X_train_scaled, y_train, n_features_to_select=n_features_to_select)
        rfe_selection_counts += rfe_mask
        # 3rd approach: Lasso
        lasso_mask = lasso_selection(X_train_scaled, y_train, C=lasso_C)
        lasso_selection_counts += lasso_mask
        
        # 3. training
        # on RFE gene
        X_train_sel_rfe = X_train_scaled[:, rfe_mask]
        X_val_sel_rfe   = X_val_scaled[:, rfe_mask]
        model_rfe = SVC(kernel="linear", probability=True, random_state=42)
        model_rfe.fit(X_train_sel_rfe, y_train)
        # on Lasso gene
        X_train_sel_lasso = X_train_scaled[:, lasso_mask]
        X_val_sel_lasso   = X_val_scaled[:, lasso_mask]
        model_lasso = SVC(kernel="linear", probability=True, random_state=42)
        model_lasso.fit(X_train_sel_lasso, y_train)
        
        # 4. validation
        preds_rfe   = model_rfe.predict(X_val_sel_rfe)
        probs_rfe   = model_rfe.predict_proba(X_val_sel_rfe)
        preds_lasso = model_lasso.predict(X_val_sel_lasso)
        probs_lasso = model_lasso.predict_proba(X_val_sel_lasso)
        
        val_accuracies["rfe"]   += accuracy_score(y_val, preds_rfe)
        val_losses["rfe"]       += log_loss(y_val, probs_rfe)
        val_accuracies["lasso"] += accuracy_score(y_val, preds_lasso)
        val_losses["lasso"]     += log_loss(y_val, probs_lasso)

        if i % 5 == 0:
            logger.debug("Iteration %d/%d completed", i + 1, n_splits * n_repeats)

    logger.info("CV completata. Val Accuracy Media = %.4f", np.mean(list(val_accuracies.values())))

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
        logger.info(f"Gene: {X.columns[idx]} | Probability of Inclusion: {rfe_inclusion_prob[idx]:.2f}")
    # top 10 biomarkers for stability (SVM-lasso)
    top_lasso_idx = np.argsort(lasso_inclusion_prob)[::-1][:10]
    logger.info("=== TOP 10 BIOMARKS FOR STABILITY (SVM-lasso) ===")
    for idx in top_lasso_idx:
        logger.info(f"Gene: {X.columns[idx]} | Probability of Inclusion: {lasso_inclusion_prob[idx]:.2f}")

    # comparison of inclusion probabilities between methods
    all_three = np.sum(
        (rfe_inclusion_prob   > 0.5) &
        (fdr_inclusion_prob   > 0.5) &
        (lasso_inclusion_prob > 0.5)
    )
    logger.info(f"Stable genes (>50%) in common between FDR, RFE and Lasso: {all_three}")

    return rfe_inclusion_prob, fdr_inclusion_prob, lasso_inclusion_prob

def CrossValidationBiased(
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits: int,
    n_repeats: int,
    n_features_to_select: int,
):
    """
    Replica della procedura "distorta" descritta in Ambroise & McLachlan (2002):
    la feature selection (SVM-RFE) viene eseguita UNA SOLA VOLTA su tutto il
    dataset, PRIMA della cross-validation. Ogni fold della CV usa quindi lo
    stesso identico sottoinsieme di geni, già scelto guardando anche i dati
    che finiranno nel validation set: questo introduce un bias ottimistico
    nella stima dell'errore.

    Args:
        X (pd.DataFrame): Feature matrix (tutto il dataset di CV).
        y (pd.DataFrame): Target vector.
        n_splits (int): Numero di split per la CV.
        n_repeats (int): Numero di ripetizioni della CV.
        n_features_to_select (int): Numero di geni da selezionare con RFE.

    Returns:
        mean_accuracy (float): Accuratezza media di validazione sui fold.
    """
    # 1. standardizzazione su TUTTO il dataset (già di per sé fonte di leakage,
    #    ma qui il problema centrale è la feature selection, quindi la teniamo
    #    coerente con l'errore metodologico descritto nel paper)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. feature selection UNA SOLA VOLTA su tutto il dataset
    selected_mask = rfe_svm_selection(
        X_scaled, y, n_features_to_select=n_features_to_select
    )
    X_selected = X_scaled[:, selected_mask]

    # 3. CV ripetuta usando SEMPRE lo stesso sottoinsieme di geni
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42
    )

    accuracies = []
    for train_idx, val_idx in rskf.split(X_selected, y):
        X_train, X_val = X_selected[train_idx], X_selected[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = SVC(kernel="linear", probability=True, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        accuracies.append(accuracy_score(y_val, preds))

    mean_accuracy = float(np.mean(accuracies))
    logger.info(
        "[BIASED] n_genes=%d | Val Accuracy Media = %.4f",
        n_features_to_select, mean_accuracy,
    )
    return mean_accuracy


def CrossValidationUnbiasedSimple(
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits: int,
    n_repeats: int,
    n_features_to_select: int,
):
    """
    Versione "leggera" della procedura corretta (unbiased), usata per
    l'esperimento di confronto con CrossValidationBiased: la feature
    selection (SVM-RFE) viene rifatta da zero in ogni fold, usando solo
    i dati di training del fold stesso.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.DataFrame): Target vector.
        n_splits (int): Numero di split per la CV.
        n_repeats (int): Numero di ripetizioni della CV.
        n_features_to_select (int): Numero di geni da selezionare con RFE.

    Returns:
        mean_accuracy (float): Accuratezza media di validazione sui fold.
    """
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42
    )

    accuracies = []
    for train_idx, val_idx in rskf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val     = X.iloc[val_idx],   y.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled   = scaler.transform(X_val)

        # feature selection ricalcolata SOLO sul training del fold
        mask = rfe_svm_selection(
            X_train_scaled, y_train, n_features_to_select=n_features_to_select
        )

        model = SVC(kernel="linear", probability=True, random_state=42)
        model.fit(X_train_scaled[:, mask], y_train)
        preds = model.predict(X_val_scaled[:, mask])
        accuracies.append(accuracy_score(y_val, preds))

    mean_accuracy = float(np.mean(accuracies))
    logger.info(
        "[UNBIASED] n_genes=%d | Val Accuracy Media = %.4f",
        n_features_to_select, mean_accuracy,
    )
    return mean_accuracy
