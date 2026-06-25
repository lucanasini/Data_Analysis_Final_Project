import argparse
import logging
from pathlib import Path

import pandas as pd

from . import __version__
from .bias_experiment import run_bias_experiment
from .cross_validation import CrossValidation
from .evaluate import evaluate
from .plotting import plot_correlations, plot_norm_stats
from .preprocess import run_preprocess, compute_normalization_stats
from .utils import load_config_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(f"{'MAIN':<16}")


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
    args = parser.parse_args()

    config_path = Path(args.config)

    # load configuration and data
    config = load_config_json(config_path)

    file_path = Path(config["data"]["file_path"])
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error("Data file not found: %s", file_path)
        raise

    # preprocessing    
    X_cv, X_test, y_cv, y_test = run_preprocess(df, config)
    logger.info(
        "Entries: CV=%s | Test=%s",
        f"{len(X_cv):,}",
        f"{len(X_test):,}",
    )

    # data statistics plots
    plot_flag = config["output"].get("save_plots", False)
    plot_dir = Path(config["output"].get("plots_dir", "outputs/plots"))
    if plot_flag:
        plot_correlations(
            df         = df,
            output_dir = plot_dir,
        )
        plot_norm_stats(
            norm_stats    = compute_normalization_stats(X_cv),
            feature_names = X_cv.columns.tolist(),
            output_dir    = plot_dir,
            max_features  = len(X_cv.columns.tolist()),
        )

    # cross-validation parameters
    n_splits  = config["training"].get("n_splits", 5)
    n_repeats = config["training"].get("n_repeats", 10)
    n_features_to_select = config["training"].get("n_features_to_select", 30)
    alpha = config["cv"].get("alpha", 0.05)
    C     = config["cv"].get("C", 0.1)

    # cross-validation
    rfe_inclusion_prob, fdr_inclusion_prob, lasso_inclusion_prob = CrossValidation(
        X_cv, y_cv,
        n_splits   = n_splits,
        n_repeats  = n_repeats,
        parameters = [n_features_to_select, alpha, C],
        plot_dir   = plot_dir,
    )

    # evaluation
    evaluate(
        X_cv,   y_cv,
        X_test, y_test,
        n_features_to_select = n_features_to_select,
        fdr_inclusion_prob   = fdr_inclusion_prob,
        rfe_inclusion_prob   = rfe_inclusion_prob,
        lasso_inclusion_prob = lasso_inclusion_prob,
        plot_dir = plot_dir,
    )

    # esperimento di selection bias (Ambroise & McLachlan, 2002)
    if config.get("bias_experiment", {}).get("run", False):
        gene_grid = config["bias_experiment"].get(
            "gene_grid", [5, 10, 20, 50, 100, 200]
        )
        permute = config["bias_experiment"].get("permute_labels", False)

        logger.info("=== Esperimento selection bias (dati reali) ===")
        run_bias_experiment(
            X_cv, y_cv,
            n_splits   = n_splits,
            n_repeats  = n_repeats,
            gene_grid  = gene_grid,
            plot_dir   = plot_dir,
            permute_labels = False,
        )

        if permute:
            logger.info("=== Esperimento selection bias (etichette permutate / null model) ===")
            run_bias_experiment(
                X_cv, y_cv,
                n_splits   = n_splits,
                n_repeats  = n_repeats,
                gene_grid  = gene_grid,
                plot_dir   = plot_dir,
                permute_labels = True,
            )

if __name__ == "__main__":
    main()