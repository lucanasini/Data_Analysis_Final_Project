import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from . import __version__
from .cross_validation import CrossValidation
from .plotting import plot_correlations, plot_norm_stats
from .preprocess import run_preprocess, compute_normalization_stats
from .utils import load_config_json

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
    if config["output"].get("save_plots", False):
        plot_dir = Path(config["output"].get("plots_dir", "outputs/plots"))

        plot_correlations(
            df         = df,
            output_dir = plot_dir,
        )

        plot_norm_stats(
            norm_stats=compute_normalization_stats(X_cv),
            feature_names=X_cv.columns.tolist(),
            output_dir=plot_dir,
            max_features=len(X_cv.columns.tolist()),
        )

    n_splits  = config["training"].get("n_splits", 5)
    n_repeats = config["training"].get("n_repeats", 10)
    n_features_to_select = config["training"].get("n_features_to_select", 30)

    rfe_selection_counts, fdr_selection_counts = CrossValidation(
        X_cv, y_cv,
        n_splits=n_splits,
        n_repeats=n_repeats,
        n_features_to_select=n_features_to_select,
    )

    # 4. CALCOLO STABILITÀ & INCLUSIONE (Task 1 & Task 2)
    total_runs = n_splits * n_repeats
    rfe_inclusion_prob = rfe_selection_counts / total_runs
    fdr_inclusion_prob = fdr_selection_counts / total_runs

    # Ricerca dei biomarcatori più robusti
    top_rfe_idx = np.argsort(rfe_inclusion_prob)[::-1][:10]
    print("\n=== TOP 10 BIOMARCATORI PER STABILITÀ (SVM-RFE) ===")
    for idx in top_rfe_idx:
        print(f"Gene: {X_cv.columns[idx]} | Probabilità Inclusione: {rfe_inclusion_prob[idx]:.2f}")
        
    # Confronto FDR vs Machine Learning
    overlap = np.sum((rfe_inclusion_prob > 0.5) & (fdr_inclusion_prob > 0.5))
    print(f"\nNumero di geni stabili (>50% CV) in comune tra FDR e SVM-RFE: {overlap}")

    # 5. VALUTAZIONE FINALE SUL HIDDEN TEST SET (Solo per verifica empirica)
    # Seleziona i geni che hanno una stabilità globale > 50% nel processo di CV
    stable_genes_mask = rfe_inclusion_prob > 0.5
    if np.sum(stable_genes_mask) == 0:  # Fallback se nessuno supera il 50%
        stable_genes_mask = np.argsort(rfe_inclusion_prob)[::-1][:n_features_to_select]
        
    scaler = StandardScaler()
    X_cv_scaled = scaler.fit_transform(X_cv)
    X_test_scaled = scaler.transform(X_test)

    final_model = SVC(kernel="linear", probability=True, random_state=42)
    final_model.fit(X_cv_scaled[:, stable_genes_mask], y_cv)
    test_preds = final_model.predict(X_test_scaled[:, stable_genes_mask])

    print(f"\nTest Set Accuracy (Sottogruppo Stabile): {accuracy_score(y_test, test_preds):.4f}")


















    # model = SVC(kernel="linear", probability=True,
    #             random_state=config["data"].get("split_seed", 42))
    # model.fit(X_train, y_train)
    # y_pred_prob = model.predict_proba(X_train)
    # y_pred = model.predict(X_train)
    # train_loss = log_loss(y_train, y_pred_prob)
    # train_acc = accuracy_score(y_train, y_pred)
    # y_val_pred_prob = model.predict_proba(X_val)
    # y_val_pred = model.predict(X_val)
    # val_loss = log_loss(y_val, y_val_pred_prob)
    # val_acc = accuracy_score(y_val, y_val_pred)

    # print(f"Train Loss = {train_loss} | Val Loss = {val_loss}")
    # print(f"Train Acc = {train_acc} | Val Acc = {val_acc}")


if __name__ == "__main__":
    main()