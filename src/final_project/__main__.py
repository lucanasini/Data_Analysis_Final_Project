import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .bias_experiment import run_bias_experiment
from .evaluate import evaluate
from .plotting import plot_correlations, plot_norm_stats
from .preprocess import compute_normalization_stats, run_preprocess
from .utils import load_config_json

logging.basicConfig(
    level=logging.DEBUG,
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
    genes     = (config["training"].get("genes", 30)
                 if isinstance(config["training"].get("genes", 30), list)
                 else [config["training"].get("genes", 30)])
    alpha     = (config["training"].get("alpha", 0.05)
                 if isinstance(config["training"].get("alpha", 0.05), list)
                else [config["training"].get("alpha", 0.05)])
    C         = (config["training"].get("C", 0.1)
                 if isinstance(config["training"].get("C", 0.1), list)
                else [config["training"].get("C", 0.1)])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir  = Path(config["output"].get("runs_dir", "outputs/runs"))
    run_dir   = runs_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    # selection bias experiment (Ambroise & McLachlan, 2002)
    if not args.evaluate:
        logger.info("=== Selection bias experiment ===")
        results, inclusion_probs = run_bias_experiment(
            X_cv, y_cv,
            n_splits   = n_splits,
            n_repeats  = n_repeats,
            parameters = [genes, alpha, C],
            plot_dir   = plot_dir,
        )

        best_run_index = np.argmin(np.minimum(results["biased_rfe_val_loss"],
                                            results["unbiased_rfe_val_loss"]))
        best_inclusion_prob = inclusion_probs[best_run_index]
        logger.info("Best index: %d | Best parameters: [%s %s %s]",
                    best_run_index, results["n_genes"].iloc[best_run_index],
                    results["alpha"].iloc[best_run_index], results["C"].iloc[best_run_index])

        results.to_csv(run_dir / "bias_experiment_results.csv", index=False)

        feature_names = X_cv.columns.tolist()
        try:
                rfe_probs = best_inclusion_prob[0]
        except Exception:
            rfe_probs = [np.nan] * len(feature_names)
        try:
                lasso_probs = best_inclusion_prob[2]
        except Exception:
            lasso_probs = [np.nan] * len(feature_names)

        df_selected = pd.DataFrame({
            "feature": feature_names,
            "rfe_inclusion_prob": list(rfe_probs),
            "lasso_inclusion_prob": list(lasso_probs),
        })
        
        df_selected = df_selected[
            (df_selected["rfe_inclusion_prob"] != 0) | 
            (df_selected["lasso_inclusion_prob"] != 0)
        ]

        df_selected.to_csv(run_dir / "selected_features.csv", index=False)
    
    else:
        run_subdirs = list(run_dir.parent.glob("*/bias_experiment_results.csv"))
        if not run_subdirs:
            logger.error("No previous bias experiment results found in %s", run_dir.parent)
            raise FileNotFoundError(f"No bias_experiment_results.csv found in {run_dir.parent}")
        latest_csv = max(run_subdirs, key=lambda p: p.parent.stat().st_mtime)
        results = pd.read_csv(latest_csv)
        best_run_index = np.argmin(np.minimum(results["biased_rfe_val_loss"],
                                              results["unbiased_rfe_val_loss"]))
        best_inclusion_prob = pd.read_csv(latest_csv.parent / "selected_features.csv")

    # evaluation
    test_results = evaluate(
        X_cv,   y_cv,
        X_test, y_test,
        n_features_to_select = results["n_genes"].iloc[best_run_index],
        rfe_inclusion_prob   = (best_inclusion_prob["rfe_inclusion_prob"].values
                                if args.evaluate else best_inclusion_prob[0]),
        lasso_inclusion_prob = (best_inclusion_prob["lasso_inclusion_prob"].values
                                if args.evaluate else best_inclusion_prob[2]),
        plot_dir = plot_dir,
    )
    
    test_results.to_csv(run_dir / "test_results.csv", index=False)


if __name__ == "__main__":
    main()
