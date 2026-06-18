import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.svm import SVC

from . import __version__
from .plotting import plot_correlations, plot_norm_stats
from .preprocess import run_preprocess
from .utils import (
    load_config_json,
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
    (X, y), norm_stats, (train_indices, val_indices, test_indices) = run_preprocess(df, config)

    logger.info(
        "Train=%s, Val=%s, Test=%s",
        f"{len(train_indices):,}",
        f"{len(val_indices):,}",
        f"{len(test_indices):,}",
    )

    # data statistics plots
    if config["output"].get("save_plots", False):
        plot_dir = Path(config["output"].get("plots_dir", "outputs/plots"))

        plot_correlations(
            df         = df,
            output_dir = plot_dir,
        )

        plot_norm_stats(
            norm_stats=norm_stats,
            feature_names=X.columns.tolist(),
            output_dir=plot_dir,
            max_features=len(X.columns.tolist()),
        )


    X_train = X.iloc[train_indices].fillna(0).values
    X_val   = X.iloc[val_indices].fillna(0).values
    # X_test  = X.iloc[test_indices].fillna(0).values
    y_train = y.iloc[train_indices].to_numpy().argmax(axis=1)
    y_val   = y.iloc[val_indices].to_numpy().argmax(axis=1)
    # y_test  = y.iloc[test_indices].to_numpy().argmax(axis=1)

    model = SVC(kernel="linear", probability=True,
                random_state=config["data"].get("split_seed", 42))
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_train)
    y_pred = model.predict(X_train)
    train_loss = log_loss(y_train, y_pred_prob)
    train_acc = accuracy_score(y_train, y_pred)
    y_val_pred_prob = model.predict_proba(X_val)
    y_val_pred = model.predict(X_val)
    val_loss = log_loss(y_val, y_val_pred_prob)
    val_acc = accuracy_score(y_val, y_val_pred)

    print(f"Train Loss = {train_loss} | Val Loss = {val_loss}")
    print(f"Train Acc = {train_acc} | Val Acc = {val_acc}")










    # # datasets and dataloaders
    # common_kwargs = dict(
    #     file_path  = file_path,
    #     stats      = norm_stats,
    # )

    # loader_kwargs = dict(
    #     batch_size  = config["training"].get("batch_size", 1024),
    #     num_workers = config["training"].get("num_workers", 0),
    #     pin_memory  = config["training"].get("device", "auto") in ("gpu", "auto")
    #                   and torch.cuda.is_available(),
    #     drop_last   = config["data"].get("drop_last", False),
    # )

    # train_dataset = GN2Dataset(jet_indices=train_indices, **common_kwargs)
    # val_dataset   = GN2Dataset(jet_indices=val_indices,   **common_kwargs)
    # test_dataset  = GN2Dataset(jet_indices=test_indices,  **common_kwargs)

    # train_loader = gn2_dataloader(
    #     train_dataset, **loader_kwargs,
    #     shuffle=config["data"].get("shuffle", False))
    # val_loader   = gn2_dataloader(val_dataset,   **loader_kwargs, shuffle=False)
    # test_loader  = gn2_dataloader(test_dataset,  **loader_kwargs, shuffle=False)


    # device = get_device(config["training"].get("device", "auto"))
    # checkpoint_path = Path(config["output"].get("checkpoints_dir", "outputs/checkpoints"))
    # if not args.evaluate:

    #     # model
    #     model_config = config.get("model", {})
    #     model = GN2(
    #         n_vars           = X.shape[1],
    #         n_classes        = y.shape[1],
    #         init_hidden_dim  = model_config.get("initialiser_hidden_dim"),
    #         dropout          = model_config.get("transformer_dropout"),
    #         activation       = model_config.get("activation"),
    #     ).to(device)

    #     # training
    #     training_config = config.get("training", {})
    #     train(
    #         model        = model,
    #         train_loader = train_loader,
    #         val_loader   = val_loader,
    #         output_dir   = checkpoint_path,
    #         device       = device,
    #         optimizer    = training_config.get("optimizer"),
    #         max_epochs   = training_config.get("max_epochs"),
    #         warmup_frac  = training_config.get("warmup_frac"),
    #         weight_decay = training_config.get("weight_decay"),
    #         lr_initial   = training_config.get("lr_initial"),
    #         lr_peak      = training_config.get("lr_peak"),
    #         lr_final     = training_config.get("lr_final"),
    #         config       = config,
    #     )

    #     checkpoint_path = sorted(Path(checkpoint_path / "runs").glob("*/best_model.pt"))[-1]

    # else:
    #     checkpoint_path = checkpoint_path / "best_model/best_model.pt"

    # # evaluation
    # evaluate(
    #     test_loader     = test_loader,
    #     checkpoint_path = checkpoint_path,
    #     output_dir      = Path(config["output"].get("evaluate_dir", "outputs/eval")),
    #     device          = device,
    #     flavour_map     = config["data"]["flavour_map"],
    #     fc              = config["discriminant"]["fc_btag"],
    #     ftau_b          = config["discriminant"]["ftau_btag"],
    #     fb              = config["discriminant"]["fb_ctag"],
    #     ftau_c          = config["discriminant"]["ftau_ctag"],
    # )


if __name__ == "__main__":
    main()
