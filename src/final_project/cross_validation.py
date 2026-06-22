import logging

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold

from .feature_selection import statistical_fdr_selection, rfe_svm_selection


logger = logging.getLogger("cross-validation")

def CrossValidation(
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits: int,
    n_repeats: int,
    n_features_to_select: int,
):
    """
    Run cross-validation with internal feature selection (FDR + RFE) and training of a linear SVM.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.DataFrame): Target vector.
        n_splits (int): Number of splits for cross-validation.
        n_repeats (int): Number of repeats for cross-validation.
        n_features_to_select (int): Number of features to select.

    Returns:
        rfe_selection_counts (np.ndarray): Array with counts of feature selection by RFE.
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
        
    # Array per mappare la stabilità (frequenza di inclusione delle feature)
    rfe_selection_counts = np.zeros(X.shape[1])
    fdr_selection_counts = np.zeros(X.shape[1])

    val_accuracies = []
    val_losses = []

    logger.info("Inizio ciclo di Cross-Validation (%d iterazioni totali)...", n_splits * n_repeats)

    for _, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # 1. standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 2. internal feature selection
        # 1st approach: Controllo FDR
        fdr_mask, _ = statistical_fdr_selection(X_train, y_train, alpha=0.05)
        fdr_selection_counts += fdr_mask
        # 2nd approach: SVM-RFE
        rfe_mask = rfe_svm_selection(X_train_scaled, y_train, n_features_to_select=n_features_to_select)
        rfe_selection_counts += rfe_mask
        
        # 3. training (on RFE gene)
        X_train_sel = X_train_scaled[:, rfe_mask]
        X_val_sel = X_val_scaled[:, rfe_mask]
        
        model = SVC(kernel="linear", probability=True, random_state=42)
        model.fit(X_train_sel, y_train)
        
        # 4. validation
        preds = model.predict(X_val_sel)
        probs = model.predict_proba(X_val_sel)
        
        val_accuracies.append(accuracy_score(y_val, preds))
        val_losses.append(log_loss(y_val, probs))

    logger.info("CV completata. Val Accuracy Media = %.4f", np.mean(val_accuracies))

    return rfe_selection_counts, fdr_selection_counts

    