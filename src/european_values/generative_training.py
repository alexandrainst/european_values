"""Training Gaussian Mixture Models on the dataset."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def train_generative_model(
    survey_df: pd.DataFrame,
    max_components: int,
    samples_per_country_val_test: int,
    seed: int,
) -> None:
    """Train a Gaussian Mixture Model on EU survey data.

    Args:
        survey_df: A non-normalised dataframe containing the survey responses.
        max_components: Maximum number of components to try in model selection
        samples_per_country_val_test: Number of samples per country for
        validation/test sets
        seed: Random seed for reproducibility
    """
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
    train_dfs, val_dfs, test_dfs = [], [], []
    for country in eu_df["country_code"].unique():
        country_data = eu_df[eu_df["country_code"] == country].sample(
            frac=1, random_state=seed
        )
        n = len(country_data)

        # Take samples for val and test (or less if not enough data)
        n_val = min(samples_per_country_val_test, n // 5)
        n_test = min(samples_per_country_val_test, n // 5)

        test_dfs.append(country_data.iloc[:n_test])
        val_dfs.append(country_data.iloc[n_test : n_test + n_val])
        train_dfs.append(country_data.iloc[n_test + n_val :])

    # Set up the data as NumPy arrays
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
    n_components_range = [n for n in n_components_range if n <= max_components]

    bic_scores = {}
    for n_comp in n_components_range:
        # Create pipeline with scaler + GMM
        pipeline = make_pipeline(
            MinMaxScaler(feature_range=(0, 1)),
            GaussianMixture(n_components=n_comp, random_state=seed),
        )

        # Fit the entire pipeline on training data
        pipeline.fit(train_matrix)
        bic_scores[n_comp] = pipeline.named_steps["gaussianmixture"].bic(
            pipeline.named_steps["minmaxscaler"].transform(val_matrix)
        )
        logger.info(f"n_components={n_comp}: BIC={bic_scores[n_comp]:.2f}")

    best_n = min(bic_scores.keys(), key=lambda x: bic_scores[x])
    logger.info(f"Best n_components: {best_n}")

    # Create final pipeline with best number of components
    final_pipeline = make_pipeline(
        MinMaxScaler(feature_range=(0, 1)),
        GaussianMixture(n_components=best_n, random_state=seed),
    )

    # Train final model on all data
    logger.info("Training final model on entire EU dataset...")
    full_matrix = eu_df[question_columns].values
    final_pipeline.fit(full_matrix)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_bic = final_pipeline.named_steps["gaussianmixture"].bic(
        final_pipeline.named_steps["minmaxscaler"].transform(test_matrix)
    )
    logger.info(f"Test BIC: {test_bic:.2f}")

    gmm = final_pipeline.named_steps["gaussianmixture"]
    scaler = final_pipeline.named_steps["minmaxscaler"]
    scaled_full_matrix = scaler.transform(full_matrix)

    logger.info(
        f"Final model - BIC: {gmm.bic(scaled_full_matrix):.2f}, "
        f"Converged: {gmm.converged_}, Iterations: {gmm.n_iter_}"
    )

    logger.info("Testing probabilities on training data...")
    train_probabilities = gmm.predict_proba(scaled_full_matrix)
    print(f"Training data - Component weights: {gmm.weights_}")
    print(f"Training data - First 5 full probabilities:\n{train_probabilities[:5]}")

    # Try weighted approach
    weighted_probs = (
        np.sum(train_probabilities * gmm.weights_, axis=1) / gmm.n_components
    )
    max_probs = train_probabilities.max(axis=1)

    logger.info(f"Training data max average probability: {max_probs.mean():.4f}")
    logger.info(
        f"Training data max probability range: [{max_probs.min():.4f}, "
        f"{max_probs.max():.4f}]"
    )
    logger.info(f"Training data max probability std: {max_probs.std():.4f}")

    logger.info(
        f"Training data weighted average probability: {weighted_probs.mean():.4f}"
    )
    logger.info(
        f"Training data weighted probability range: [{weighted_probs.min():.4f}, "
        f"{weighted_probs.max():.4f}]"
    )
    logger.info(f"Training data weighted probability std: {weighted_probs.std():.4f}")

    # Save the complete pipeline (as requested)
    model_dir = Path("data/processed/gmm_model")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"gmm_n{best_n}_seed{seed}.pkl"
    joblib.dump(final_pipeline, model_path)
    logger.info(f"Pipeline saved to {model_path.resolve()!r}")
