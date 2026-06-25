"""
utils.py
========
Utility functions for the transformer jet tagging project.
"""
import json
import logging
from pathlib import Path


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
