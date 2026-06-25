"""
bias_experiment.py
===================
Replica dell'esperimento di Ambroise & McLachlan (2002): confronto tra
selezione delle feature "biased" (fuori dalla CV) e "unbiased" (dentro
la CV), sia sui dati reali sia con etichette permutate casualmente
(null model), per evidenziare il selection bias.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .cross_validation import CrossValidationBiased, CrossValidationUnbiasedSimple
from .plotting import plot_bias_comparison

logger = logging.getLogger(f"{'bias-experiment':<16}")


def run_bias_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    n_repeats: int,
    gene_grid: list[int],
    plot_dir: str | Path = None,
    permute_labels: bool = False,
    random_state: int = 42,
):
    """
    Esegue il confronto biased vs unbiased su una griglia di numeri di geni.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        n_splits (int): Numero di split per la CV.
        n_repeats (int): Numero di ripetizioni della CV.
        gene_grid (list[int]): Lista dei numeri di geni da testare.
        plot_dir (str | Path, optional): Directory per salvare il plot.
        permute_labels (bool): Se True, permuta casualmente y prima di
            lanciare l'esperimento (null model: nessuna relazione vera
            tra X e y, l'accuratezza "vera" dovrebbe essere ~ 1/n_classi).
        random_state (int): Seed per la permutazione delle etichette.

    Returns:
        results (pd.DataFrame): colonne ["n_genes", "biased_acc", "unbiased_acc"]
    """
    if permute_labels:
        rng = np.random.default_rng(random_state)
        y = pd.Series(
            rng.permutation(y.values), index=y.index, name=y.name
        )
        logger.warning(
            "Etichette PERMUTATE: nessuna relazione reale tra X e y. "
            "Le accuratezze 'vere' attese sono circa pari al caso (chance level)."
        )

    n_classes = y.nunique()
    chance_level = 1.0 / n_classes
    logger.info("Chance level (random guessing): %.4f", chance_level)

    biased_accs = []
    unbiased_accs = []

    for n_genes in gene_grid:
        logger.info("=== n_genes = %d ===", n_genes)
        biased_acc = CrossValidationBiased(
            X, y,
            n_splits=n_splits,
            n_repeats=n_repeats,
            n_features_to_select=n_genes,
        )
        unbiased_acc = CrossValidationUnbiasedSimple(
            X, y,
            n_splits=n_splits,
            n_repeats=n_repeats,
            n_features_to_select=n_genes,
        )
        biased_accs.append(biased_acc)
        unbiased_accs.append(unbiased_acc)

    results = pd.DataFrame({
        "n_genes": gene_grid,
        "biased_acc": biased_accs,
        "unbiased_acc": unbiased_accs,
    })

    logger.info("\n%s", results.to_string(index=False))

    if plot_dir is not None:
        plot_bias_comparison(
            n_genes_list=gene_grid,
            biased_acc=biased_accs,
            unbiased_acc=unbiased_accs,
            chance_level=chance_level,
            output_dir=plot_dir,
            title_suffix=" (etichette permutate)" if permute_labels else "",
            filename="bias_comparison_permuted.pdf" if permute_labels else "bias_comparison.pdf",
        )

    return results
