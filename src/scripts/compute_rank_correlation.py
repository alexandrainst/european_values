"""Compute rank correlations between rankings.

Usage:
    uv run src/scripts/compute_rank_correlation.py
"""

import itertools as it
import logging

import click
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s ⋅ %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("compute_rank_correlation")


@click.command()
@click.option(
    "--file",
    "-f",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    multiple=True,
    help="Path to the CSV files containing rankings to compare.",
)
@click.option(
    "--column-name",
    "-c",
    type=str,
    default="question",
    help="Name of the column containing the rankings.",
)
def main(file: list[str], column_name: str) -> None:
    """Compute rank correlations between rankings in the provided CSV files.

    Args:
        file:
            Paths to the CSV files containing rankings to compare.
        column_name:
            Name of the column containing the rankings. Defaults to 'question'.
    """
    logger.info("Loading data from files...")
    rankings = [pd.read_csv(f)[column_name].tolist() for f in file]

    logger.info("Converting rankings to integer lists...")
    rankings[1:] = [
        [rankings[0].index(item) for item in ranking] for ranking in rankings[1:]
    ]
    rankings[0] = list(range(len(rankings[0])))

    logger.info("Computing rank correlations...")
    ranking_pairs = list(it.combinations(rankings, 2))
    rhos = [
        spearmanr(ranking1, ranking2).statistic for ranking1, ranking2 in ranking_pairs
    ]
    taus = [
        kendalltau(ranking1, ranking2).statistic for ranking1, ranking2 in ranking_pairs
    ]

    mean_rho = np.mean(rhos)
    stderr_rho = np.std(rhos, ddof=1) / np.sqrt(len(rhos))
    logger.info(
        f"Mean Spearman's rho: {mean_rho:.2%} ± {1.96 * stderr_rho:.2%} (95% CI)"
    )

    mean_tau = np.mean(taus)
    stderr_tau = np.std(taus, ddof=1) / np.sqrt(len(taus))
    logger.info(
        f"Mean Kendall's tau: {mean_tau:.2%} ± {1.96 * stderr_tau:.2%} (95% CI)"
    )


if __name__ == "__main__":
    main()
