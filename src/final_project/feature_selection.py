import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.feature_selection import RFE
from sklearn.svm import SVC


def statistical_fdr_selection(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 0.0,
):
    """
    Calcola il t-test per ogni gene e applica la correzione Benjamini-Hochberg (FDR).
    Ritorna una maschera booleana dei geni significativi e i q-values.
    """
    p_values = []
    class_0 = X[y == 0]
    class_1 = X[y == 1]
    
    for col in X.columns:
        _, p = ttest_ind(class_0[col], class_1[col], equal_var=False)
        p_values.append(p if not np.isnan(p) else 1.0)
        
    p_values = np.array(p_values)
    # Correzione Benjamini-Hochberg (FDR)
    reject, q_values, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    
    return reject, q_values


def rfe_svm_selection(
    X_scaled: np.ndarray,
    y: pd.Series,
    n_features_to_select: int = 30
):
    """
    RFE con SVM Lineare (ispirato a Guyon et al., 2002).
    """
    estimator = SVC(kernel="linear", random_state=42)
    selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=0.1)
    selector.fit(X_scaled, y)
    return selector.support_