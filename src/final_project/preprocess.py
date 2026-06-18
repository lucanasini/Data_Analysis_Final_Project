"""
preprocess.py
=============
Standalone preprocessing script.

Run this script ONCE before training. It will:
    1. Load data from the dataset file.
    3. Split valid indices into train / val / test sets.
    4. Compute normalization statistics (mean, sigma) on the training set only.
    5. Save indices and norm stats to disk.

Outputs (under ``config["output"]["preprocess_dir"]``):

.. code-block:: text

    preprocess_dir/
    ├── indices/
    │   ├── train_indices.npy
    │   ├── val_indices.npy
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
    file_path: str | Path,
    train_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute per-feature mean and std on the **training set only**.

    Statistics are computed exclusively on the training set to prevent data leakage.

    Args:
        file_path (str | Path): Path to the dataset CSV file.
        train_indices (np.ndarray): sorted array of training jet indices.

    Returns:
        dict with keys ``"mean"``, ``"sigma"`` (each a ``np.ndarray``).
    
    Raises:
        FileNotFoundError: If the data file is not found.
    """

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error("Data file not found: %s", file_path)
        raise

    scaler   = StandardScaler()
    scaler.fit(df)

    stats = {
        "mean":  scaler.mean_,
        "sigma": scaler.scale_,
    }
    logger.info("Normalization stats computed on %s entries.", f"{len(train_indices):,}")
    return stats


def save_indices(
    output_dir: str | Path,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> None:
    """
    Save train / val / test index arrays as ``.npy`` files.

    Args:
        output_dir (str | Path): Base directory to save indices.
        train (np.ndarray): Array of training indices.
        val (np.ndarray): Array of validation indices.
        test (np.ndarray): Array of test indices.
    """
    idx_dir = Path(output_dir) / "indices"
    idx_dir.mkdir(parents=True, exist_ok=True)

    np.save(idx_dir / "train_indices.npy", train)
    np.save(idx_dir / "val_indices.npy",   val)
    np.save(idx_dir / "test_indices.npy",  test)

    logger.info("Indices saved to %s", idx_dir)
    logger.debug("    Train : %s jets", f"{len(train):>8,}")
    logger.debug("    Val   : %s jets", f"{len(val):>8,}")
    logger.debug("    Test  : %s jets", f"{len(test):>8,}")


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


def run_preprocess(config: dict) -> None:
    """
    Run the full preprocessing pipeline.

    Reads all settings from ``config`` (already-parsed)
    and writes the preprocessing artifacts to disk.

    Config keys read:
        - `data`:
            - `file_path` (str)
            - `train_fraction` (float)
            - `val_fraction` (float)
            - `test_fraction` (float)
            - `shuffle` (bool, default ``False``)
            - `split_seed` (int, default ``42``)
            - `norm_batch_size` (int, default ``10000``)
        - `output`:
            - `preprocess_dir` (str)

    Args:
        config (dict): Full configuration dict.

    Raises:
        ValueError: If the train/val/test fractions do not sum to 1.
    """
    # 1. load configuration
    data_config = config["data"]
    output_dir = Path(config["output"]["preprocess_dir"])

    file_path  = Path(data_config["file_path"])
    df = pd.read_csv(file_path)

    train_frac = data_config["train_fraction"]
    val_frac   = data_config["val_fraction"]
    test_frac  = data_config["test_fraction"]

    total = train_frac + val_frac + test_frac
    if abs(total - 1.) > 1e-6:
        logger.warning(
            "Fractions sum to %.6f, normalizing to 1.0", total
        )
        train_frac /= total
        val_frac   /= total
        test_frac  /= total

    shuffle = data_config.get("shuffle", False)
    seed    = data_config.get("split_seed", 42)

    # 2. train / val / test split
    train_val_indices, test_indices = train_test_split(
        df,
        train_size   = train_frac + val_frac,
        random_state = seed,
        shuffle      = shuffle,
    )
    train_indices, val_indices = train_test_split(
        train_val_indices,
        train_size   = train_frac / (train_frac + val_frac),
        random_state = seed,
        shuffle      = shuffle,
    )
    save_indices(output_dir, train_indices, val_indices, test_indices)

    # 3. normalization statistics (training set only)
    logger.info("Computing normalization statistics on training set ...")
    norm_stats = compute_normalization_stats(
        file_path     = file_path,
        train_indices = train_indices,
    )
    save_norm_stats(output_dir, norm_stats)

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    pass
