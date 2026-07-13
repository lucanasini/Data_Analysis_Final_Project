"""
utils.py
========
Utility functions for the transformer jet tagging project.
"""
import json
import logging
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(f"{'utils':<16}")


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

def calculate_metrics(y_true, y_pred, y_prob, model):
    """
    Calculate and return a dictionary of evaluation metrics.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        y_prob (np.ndarray): Predicted probabilities for the positive class.
        model (SVC or LogisticRegression): The trained model used for predictions.
    
    Returns:
        dict: Dictionary containing accuracy, loss, precision, recall, F1 score, and ROC AUC.
    """
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "loss":      log_loss(y_true, y_prob, labels=model.classes_),
        "precision": precision_score(y_true, y_pred, pos_label=model.classes_[1]),
        "recall":    recall_score(y_true, y_pred, pos_label=model.classes_[1]),
        "f1_score":  f1_score(y_true, y_pred, pos_label=model.classes_[1]),
        "roc_auc":   roc_auc_score(y_true, y_prob[:, 1], labels=model.classes_),
    }
    return metrics