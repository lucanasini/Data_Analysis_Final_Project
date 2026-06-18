"""
utils.py
========
Utility functions for the transformer jet tagging project.
"""
import datetime
print(f"{datetime.datetime.now()} Importing utils.py")
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import SGD, Adam, AdamW

from ._constants import (
    SUPPORTED_ACTIVATIONS,
    SUPPORTED_DEVICES,
    SUPPORTED_OPTIMIZERS,
)

logger = logging.getLogger("utils")


def load_config_json(filepath: str | Path) -> dict:
    """
    Load and return a JSON configuration file.

    Args:
        filepath (str | Path): path to the JSON configuration file.

    Returns:
        dict: parsed configuration.

    Raises:
        FileNotFoundError: if the file does not exist.
        json.JSONDecodeError: if the file is not valid JSON.
    """
    filepath = Path(filepath)
    try:
        with open(filepath, encoding="utf-8") as f:
            config = json.load(f)
        logger.info("Config loaded: %s", filepath)
        return config
    except FileNotFoundError:
        logger.error("Config file not found: %s", filepath)
        raise
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in config file %s: %s", filepath, exc)
        raise



def get_device(device_str: str = "auto") -> torch.device:
    """
    Resolve a device string from config into a ``torch.device``.

    Accepted values:
        - ``"auto"``  - CUDA if available, otherwise CPU.
        - ``"cuda"``  - NVIDIA GPU.
        - ``"cpu"``   - always CPU.

    Args:
        device_str (str): device string from config (default ``"auto"``).

    Returns:
        torch.device: resolved device.

    Raises:
        ValueError: if ``device_str`` is not one of the supported values.
        RuntimeError: if the requested device is not available on this machine.
    """
    key = device_str.lower()
    if key not in SUPPORTED_DEVICES:
        raise ValueError(
            f"Unknown device '{device_str}'. Supported: {SUPPORTED_DEVICES}"
        )

    if key == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    elif key == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' requested but CUDA is not available on this machine."
            )
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.debug("Using device: %s", device)
    return device


def get_activation(name: str) -> torch.nn:
    """
    Build and return an activation from the activation name.

    Supported activation names: ``"relu"``, ``"leakyrelu"``,
    ``"sigmoid"``, ``"tanh"``, ``"softplus"``.

    Args:
        name (str): activation name from config.

    Returns:
        torch.nn: configured activation instance.

    Raises:
        ValueError: if ``activation`` is not one of the supported names.
    """
    name = name.lower()
    if name not in SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. Supported: {SUPPORTED_ACTIVATIONS}"
        )

    if name == "relu":
        activation = torch.nn.ReLU()
    elif name == "leakyrelu":
        activation = torch.nn.LeakyReLU()
    elif name == "sigmoid":
        activation = torch.nn.Sigmoid()
    elif name == "tanh":
        activation = torch.nn.Tanh()
    elif name == "softplus":
        activation = torch.nn.Softplus()

    logger.debug("Activation: %s", name.upper())
    return activation


def get_optimizer(
    model: nn.Module,
    name: str = "adam",
    lr: float = 5e-4,
    wd: float = 1e-5,
) -> torch.optim.Optimizer:
    """
    Build and return an optimizer from the config.

    Supported optimizer names: ``"adamw"``, ``"adam"``, ``"sgd"``.

    Config keys read:
        - `optimizer` (str,   default ``"adamw"``)
        - `lr_peak` (float, default ``5e-4``, used as the initial lr)
        - `weight_decay` (float, default ``1e-5``)

    Args:
        model (nn.Module): the model whose parameters will be optimized.
        name (str): the name of the optimizer to use.
        lr (float): learning rate (default ``5e-4``).
        wd (float): weight decay (default ``1e-5``).

    Returns:
        torch.optim.Optimizer: configured optimizer instance.

    Raises:
        ValueError: if ``optimizer`` is not one of the supported names.
    """
    name = name.lower()
    if name not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{name}'. Supported: {SUPPORTED_OPTIMIZERS}"
        )

    params = model.parameters()

    if name == "adamw":
        optimizer = AdamW(params, lr=lr, weight_decay=wd)
    elif name == "adam":
        optimizer = Adam(params, lr=lr, weight_decay=wd)
    elif name == "sgd":
        optimizer = SGD(params, lr=lr, weight_decay=wd)

    logger.debug(
        "Optimizer: %s | lr_peak=%.2e | weight_decay=%.2e",
        name.upper(), lr, wd,
    )
    return optimizer


def check_artifacts(paths: list[str | Path]) -> bool:
    """
    Return ``True`` if every path in *paths* exists, ``False`` otherwise.

    Args:
        paths: list of ``Path`` objects to check.

    Returns:
        bool: ``True`` if all paths exist.
    """
    missing = [Path(p) for p in paths if not p.exists()]
    if missing:
        for p in missing:
            logger.warning("Missing artifact: %s", p)
        return False
    return True


def artifact_paths(preprocess_dir: str | Path) -> dict[str, Path]:
    """
    Return the paths for all preprocessing artifacts.

    Args:
        preprocess_dir (str | Path): root preprocessing output directory
            (``config["output"]["preprocess_dir"]``).

    Returns:
        dict with keys ``"train_indices"``, ``"val_indices"``,
        ``"test_indices"``, ``"norm_stats"``.
    """
    idx_dir = Path(preprocess_dir) / "indices"
    return {
        "train_indices": idx_dir / "train_indices.npy",
        "val_indices":   idx_dir / "val_indices.npy",
        "test_indices":  idx_dir / "test_indices.npy",
        "norm_stats":    preprocess_dir / "norm_stats.json",
    }


def load_indices(preprocess_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the train / val / test index arrays saved by the preprocessing step.

    Args:
        preprocess_dir (str | Path): root preprocessing output directory.

    Returns:
        tuple: ``(train_indices, val_indices, test_indices)`` as ``np.ndarray``.

    Raises:
        FileNotFoundError: if any index file is missing.
    """
    preprocess_dir = Path(preprocess_dir)
    paths = artifact_paths(preprocess_dir)
    for key in ("train_indices", "val_indices", "test_indices"):
        if not paths[key].exists():
            raise FileNotFoundError(
                f"Index file not found: {paths[key]}. Run preprocessing first."
            )
    train = np.load(paths["train_indices"])
    val   = np.load(paths["val_indices"])
    test  = np.load(paths["test_indices"])
    logger.info("Indices loaded - Train: %s, Val: %s, Test: %s",
                f"{len(train):,}", f"{len(val):,}", f"{len(test):,}")
    return train, val, test


def load_norm_stats(preprocess_dir: str | Path) -> dict[str, np.ndarray]:
    """
    Load normalization statistics from ``norm_stats.json``.

    Args:
        preprocess_dir (str | Path): root preprocessing output directory.

    Returns:
        dict mapping stat name to ``np.ndarray`` (keys: ``"mean"``, ``"sigma"``).

    Raises:
        FileNotFoundError: if ``norm_stats.json`` does not exist.
    """
    preprocess_dir = Path(preprocess_dir)
    path = artifact_paths(preprocess_dir)["norm_stats"]
    if not path.exists():
        raise FileNotFoundError(
            f"Norm stats not found: {path}. Run preprocessing first."
        )
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    stats = {k: np.array(v) for k, v in raw.items()}
    logger.info("Norm stats loaded from %s", path)
    return stats
