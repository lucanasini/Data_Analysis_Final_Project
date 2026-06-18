import datetime
print(f"{datetime.datetime.now()} Importing main.py")
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .plotting import plot_correlations
from .utils import (
    artifact_paths,
    check_artifacts,
    get_device,
    load_config_json,
    load_indices,
    load_norm_stats,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("MAIN")


def main():
    parser = argparse.ArgumentParser(
        prog="final project",
        description="Data Analysis final project",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to the JSON configuration file.",
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation on the test set instead of training.",
    )

    parser.add_argument(
        "--debug-frac",
        type=float,
        default=1.0,
        help="Fraction of data to use",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    debug_frac  = args.debug_frac

    # load configuration
    config = load_config_json(config_path)

    file_path = Path(config["data"]["file_path"])
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error("Data file not found: %s", file_path)
        raise

    preprocess_dir = Path(config["output"]["preprocess_dir"])

    batch_size  = config["training"].get("batch_size", 1024)
    shuffle_var = config["data"].get("shuffle", False)

    # preprocessing
    paths = artifact_paths(preprocess_dir)
    if not check_artifacts(list(paths.values())):
        logger.info("Preprocessing not found: running preprocess")
        from .preprocess import run_preprocess
        run_preprocess(config)

    if not check_artifacts(list(paths.values())):
        raise FileNotFoundError("Preprocessing failed: artifacts still missing.")

    train_indices, val_indices, test_indices = load_indices(preprocess_dir)
    norm_stats = load_norm_stats(preprocess_dir)

    if debug_frac < 1.0:
        rng = np.random.default_rng(seed=42)

        train_indices = rng.choice(
            train_indices,
            size=int(len(train_indices) * debug_frac),
            replace=False,
        )
        val_indices = rng.choice(
            val_indices,
            size=int(len(val_indices) * debug_frac),
            replace=False,
        )
        test_indices = rng.choice(
            test_indices,
            size=int(len(test_indices) * debug_frac),
            replace=False,
        )

        logger.info("Debug mode: %s", f"{debug_frac:.1%}")

    train_indices = np.sort(train_indices)
    val_indices   = np.sort(val_indices)
    test_indices  = np.sort(test_indices)

    logger.info(
        "Train=%s, Val=%s, Test=%s",
        f"{len(train_indices):,}",
        f"{len(val_indices):,}",
        f"{len(test_indices):,}",
    )

    # # datasets and dataloaders
    # common_kwargs = dict(
    #     h5_file_path    = file_path,
    #     max_tracks      = config["data"].get("max_tracks", 40),
    #     jet_vars        = jet_vars,
    #     track_vars      = track_vars,
    #     jet_flavour     = label_vars,
    #     jet_flavour_map = label_map,
    #     stats           = norm_stats,
    # )

    # loader_kwargs = dict(
    #     batch_size  = batch_size,
    #     num_workers = config["training"].get("num_workers", 0),
    #     pin_memory  = config["training"].get("device", "auto") in ("gpu", "auto")
    #                   and torch.cuda.is_available(),
    #     drop_last   = config["data"].get("drop_last", False),
    # )

    # train_dataset = GN2Dataset(jet_indices=train_indices, **common_kwargs)
    # val_dataset   = GN2Dataset(jet_indices=val_indices,   **common_kwargs)
    # test_dataset  = GN2Dataset(jet_indices=test_indices,  **common_kwargs)

    # train_loader = gn2_dataloader(train_dataset, **loader_kwargs, shuffle=shuffle_var)
    # val_loader   = gn2_dataloader(val_dataset,   **loader_kwargs, shuffle=False)
    # test_loader  = gn2_dataloader(test_dataset,  **loader_kwargs, shuffle=False)

    if config["output"].get("save_plots", False):
        plot_dir = Path(config["output"].get("plots_dir", "outputs/plots"))

        plot_correlations(
            df         = df,
            output_dir = plot_dir,
        )


if __name__ == "__main__":
    main()
