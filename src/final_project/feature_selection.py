import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests
from sklearn.feature_selection import RFE
from sklearn.svm import SVC


def statistical_fdr_selection(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 0.05,
):
    """
    Calculates the t-test for each gene and applies the Benjamini-Hochberg correction (FDR).
    Returns a boolean mask of significant genes and the q-values.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        alpha (float): Significance level for FDR correction (default: ``0.0``).
    
    Returns:
        reject (np.ndarray): Boolean mask of significant genes.
        q_values (np.ndarray): Array of q-values for each gene.
    """
    p_values = []
    groups = [X[y == cls] for cls in y.unique()]
    for col in X.columns:
        samples = [g[col].values for g in groups]
        _, p = f_oneway(*samples)
        p_values.append(p if not np.isnan(p) else 1.0)

    p_values = np.array(p_values)
    reject, q_values, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    return reject, q_values


def rfe_svm_selection(
    X_scaled: np.ndarray,
    y: pd.Series,
    n_features_to_select: int = 30
):
    """
    RFE with Linear SVM (inspired by Guyon et al., 2002).
    
    Args:
        X_scaled (np.ndarray): Scaled feature matrix.
        y (pd.Series): Target vector.
        n_features_to_select (int): Number of features to select (default: ``30``).
    
    Returns:
        selector.support_ (np.ndarray): Boolean mask of selected features.
    """
    estimator = SVC(kernel="linear", random_state=42)
    selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=0.1)
    selector.fit(X_scaled, y)
    return selector.support_


def lasso_selection(
    X: np.ndarray,
    y: pd.Series,
    C: float = 0.1,
) -> np.ndarray:
    """
    L1-penalized logistic regression (Lasso) feature selection.

    A third, independent selection criterion alongside FDR (statistical)
    and SVM-RFE (wrapper ML method): genes with a non-zero coefficient in
    at least one class are considered "selected". With 3 classes, `coef_`
    has shape (n_classes, n_features) under the multinomial scheme, so a
    feature is selected if ANY class has a non-zero weight for it.
    Smaller C -> stronger penalty -> sparser selection.

    Args:
        X (np.ndarray): feature matrix.
        y (pd.Series): target vector (binary or multiclass).
        C (float): inverse regularization strength (default 0.1).

    Returns:
        np.ndarray: boolean mask of selected features.
    """
    model = LogisticRegression(
        penalty="l1",
        solver="saga",          # supports l1 + multinomial multiclass
        C=C,
        random_state=42,
        max_iter=10000,         # saga needs more iterations to converge than liblinear
    )
    model.fit(X, y)
    coefs = model.coef_            # shape (n_classes, n_features) for multiclass
    return np.any(coefs != 0, axis=0)
