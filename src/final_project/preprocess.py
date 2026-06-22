"""
preprocess.py
=============
Standalone preprocessing script.

Run this script ONCE before training. It will:
    1. Load data from the dataset file.
    3. Split valid indices into train + val / test sets.
    4. Compute normalization statistics (mean, sigma) on the cross-validation set only.
    5. Save indices and norm stats to disk.

Outputs (under ``config["output"]["preprocess_dir"]``):

.. code-block:: text

    preprocess_dir/
    ├── indices/
    │   ├── cv_indices.npy
    │   └── test_indices.npy
    └── norm_stats.json
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("preprocess")


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


def save_norm_stats(output_dir: str | Path, norm_stats: dict[str, np.ndarray]) -> None:
    """
    Serialize normalization statistics (numpy arrays) to ``norm_stats.json``.

    Args:
        output_dir (str | Path): Directory in which ``norm_stats.json`` will be written.
        norm_stats (dict): Dict mapping stat name to numpy array.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "norm_stats.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({k: v.tolist() for k, v in norm_stats.items()}, f, indent=4)

    logger.info("Normalization stats saved to %s", out_path)


def run_preprocess(df: pd.DataFrame, config: dict) -> None:
    """
    Run the full preprocessing pipeline.

    Reads all settings from ``config`` (already-parsed)
    and apply One Hot Encoding and split.

    Config keys read:
        - `data`:
            - `cv_fraction` (float)
            - `test_fraction` (float)
            - `shuffle` (bool, default ``False``)
            - `split_seed` (int, default ``42``)

    Args:
        df (pd.DataFrame): Full dataset dataframe.
        config (dict): Full configuration dict.

    Raises:
        ValueError: If the train + val / test fractions do not sum to 1.
    """
    df = df.drop(columns=["Samples"], errors="ignore")
    X = df.drop("cancer", axis=1)
    y = df["cancer"]

    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols)

    # 1. load configuration
    data_config = config["data"]

    cv_frac = data_config["cv_fraction"]
    test_frac  = data_config["test_fraction"]
    shuffle = data_config.get("shuffle", False)
    seed    = data_config.get("split_seed", 42)

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
    )

    logger.info("Preprocessing complete.")

    return X_cv, X_test, y_cv, y_test


if __name__ == "__main__":
    pass
