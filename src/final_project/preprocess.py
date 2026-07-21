"""
preprocess.py
=============
Standalone preprocessing script.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ._constants import CLINICAL_COLS

logger = logging.getLogger(f"{'preprocess':<17}")


def compute_normalization_stats(
    X: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """
    Compute per-feature mean std on the **training set only**.

    Statistics are computed only on numerical columns and exclusively on the training
    set to prevent data leakage.

    Args:
        X (pd.DataFrame): Train dataset dataframe.

    Returns:
        dict with keys ``"mean"``, ``"sigma"`` (each a ``np.ndarray``).
    """
    all_cols = X.columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    scaler = StandardScaler()
    scaler.fit(X[num_cols])

    mean  = np.full(len(all_cols), np.nan)
    sigma = np.full(len(all_cols), np.nan)

    numeric_positions = [all_cols.index(c) for c in num_cols]
    mean[numeric_positions]  = scaler.mean_
    sigma[numeric_positions] = scaler.scale_

    stats = {"mean": mean, "sigma": sigma}
    logger.info("Normalization stats computed on %s entries (%d numeric cols).",
                f"{len(X):,}", len(num_cols))
    return stats


def run_preprocess(df: pd.DataFrame, config: dict) -> None:
    """
    Run the full preprocessing pipeline.

    Reads all settings from ``config`` (already-parsed)
    and performs all preprocessing steps, including train/val/test
    splitting and cropping of clinical columns.

    Config keys read:
        - `data`:
            - `cv_fraction` (float)
            - `test_fraction` (float)
            - `shuffle` (bool, default ``False``)
            - `split_seed` (int, default ``42``)

    Args:
        df (pd.DataFrame): Full dataset dataframe.
        config (dict): Full configuration dict.

    Warnings:
        ValueError: If the train + val / test fractions do not sum to 1.
    """
    logger.info("=== Preprocess ===")

    df = df.drop(columns=["Samples"], errors="ignore")
    X = df.drop(columns=["cancer"] + CLINICAL_COLS)
    y = df["cancer"]

    y = y.replace({"allB": "ALL", "allT": "ALL"})

    # 1. load configuration
    data_config = config["data"]

    cv_frac   = data_config["cv_fraction"]
    test_frac = data_config["test_fraction"]
    shuffle   = data_config.get("shuffle", True)
    seed      = data_config.get("split_seed", 42)

    total = cv_frac + test_frac
    if abs(total - 1.) > 1e-6:
        logger.warning(
            "Fractions sum to %.6f, normalizing to 1.0", total
        )
        cv_frac   /= total
        test_frac /= total

    # 2. train + val / test split
    X_cv, X_test, y_cv, y_test = train_test_split(
        X, y,
        train_size   = cv_frac,
        random_state = seed,
        shuffle      = shuffle,
        stratify     = y,
    )

    logger.info("Preprocessing complete.")

    return X_cv, X_test, y_cv, y_test
