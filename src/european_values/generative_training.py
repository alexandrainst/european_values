"""Training Gaussian Mixture Models on the dataset."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


def train_generative_model(
    survey_df: pd.DataFrame,
    model_type: str,
    n_cross_val: int,
    n_jobs: int,
    n_estimators: int,
    seed: int,
    bootstrap: bool,
    compute_importances: bool,
) -> None:
    """Train a Gaussian Mixture Model on EU survey data.

    Parameters are repurposed for GMM:
    - n_cross_val * 20 = samples per country for val/test
    - n_estimators = max n_components to try
    """
    if model_type != "gaussian_mixture":
        raise ValueError(
            f"This script only supports 'gaussian_mixture', got '{model_type}'"
        )

    # Filter for EU countries only
    logger.info("Filtering for EU countries only...")
    eu_df = survey_df[survey_df["country_group"] == "EU"].copy()
    logger.info(
        f"Found {len(eu_df)} samples from "
        f"{eu_df['country_code'].nunique()} EU countries"
    )

    # Get question columns
    question_columns = [col for col in eu_df.columns if col.startswith("question_")]
    logger.info(f"Training with {len(question_columns)} questions")

    # Split data by country
    logger.info("Splitting data into train/val/test sets...")
    samples_per_country = n_cross_val * 20  # 100 if n_cross_val=5

    train_dfs, val_dfs, test_dfs = [], [], []
    for country in eu_df["country_code"].unique():
        country_data = eu_df[eu_df["country_code"] == country].sample(
            frac=1, random_state=seed
        )
        n = len(country_data)

        # Take samples for val and test (or less if not enough data)
        n_val = min(samples_per_country, n // 5)
        n_test = min(samples_per_country, n // 5)

        test_dfs.append(country_data.iloc[:n_test])
        val_dfs.append(country_data.iloc[n_test : n_test + n_val])
        train_dfs.append(country_data.iloc[n_test + n_val :])

    train_matrix = pd.concat(train_dfs)[question_columns].values
    val_matrix = pd.concat(val_dfs)[question_columns].values
    test_matrix = pd.concat(test_dfs)[question_columns].values

    logger.info(
        f"Dataset sizes - Train: {len(train_matrix)}, "
        f"Val: {len(val_matrix)}, Test: {len(test_matrix)}"
    )

    # Model selection on validation set
    logger.info("Selecting optimal number of components...")
    n_components_range = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]
    n_components_range = [n for n in n_components_range if n <= n_estimators]

    # Create GMM instance once
    gmm = GaussianMixture(random_state=seed)

    bic_scores = {}
    for n_comp in n_components_range:
        gmm.n_components = n_comp
        gmm.fit(train_matrix)
        bic_scores[n_comp] = gmm.bic(val_matrix)
        logger.info(f"n_components={n_comp}: BIC={bic_scores[n_comp]:.2f}")

    best_n = min(bic_scores.keys(), key=lambda x: bic_scores[x])
    logger.info(f"Best n_components: {best_n}")

    # Evaluate on test set - reuse the same GMM instance
    logger.info("Evaluating on test set...")
    gmm.n_components = best_n
    gmm.fit(train_matrix)
    test_bic = gmm.bic(test_matrix)
    logger.info(f"Test BIC: {test_bic:.2f}")

    # Train final model on all data - reuse the same GMM instance
    logger.info("Training final model on entire EU dataset...")
    full_matrix = eu_df[question_columns].values
    gmm.fit(full_matrix)

    logger.info(
        f"Final model - BIC: {gmm.bic(full_matrix):.2f}, "
        f"Converged: {gmm.converged_}, Iterations: {gmm.n_iter_}"
    )

    # Save results
    save_results(bic_scores, best_n, test_bic, gmm, full_matrix, seed)


def save_results(
    bic_scores: dict,
    best_n: int,
    test_bic: float,
    model: GaussianMixture,
    full_matrix: np.ndarray,
    seed: int,
) -> None:
    """Save model results."""
    # Create directory
    Path("data/processed/gmm_model").mkdir(parents=True, exist_ok=True)

    # Save the complete model using joblib
    model_path = f"data/processed/gmm_model/gmm_n{best_n}_seed{seed}.pkl"
    joblib.dump(model, model_path)

    logger.info(f"Model saved to {model_path}")
