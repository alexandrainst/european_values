"""Training Gaussian Mixture Models on the dataset."""

import logging
import typing as t
from pathlib import Path

import joblib
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def train_generative_model(
    eu_df: pd.DataFrame,
    max_components: int,
    samples_per_country_val_test: int,
    patience: int,
    covariance_type: t.Literal["full", "tied", "diag", "spherical"],
    seed: int,
) -> None:
    """Train a Gaussian Mixture Model on EU survey data.

    Args:
        eu_df:
            A non-normalised dataframe containing the survey responses from EU
            countries.
        max_components:
            Maximum number of components to try in model selection.
        samples_per_country_val_test:
            Number of samples per country for validation/test sets.
        patience:
            Number of iterations to wait for improvement in BIC before stopping.
        covariance_type:
            Type of covariance to use in the Gaussian Mixture Model.
        seed:
            Random seed for reproducibility.
    """
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
    best_bic = float("inf")
    best_n = 1
    patience_remaining = patience
    for n_comp in range(1, max_components + 1):
        pipeline = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                (
                    "gmm",
                    GaussianMixture(
                        n_components=n_comp,
                        covariance_type=covariance_type,
                        random_state=seed,
                    ),
                ),
            ]
        )
        pipeline.fit(train_matrix)
        bic_score = pipeline.named_steps["gmm"].bic(
            pipeline.named_steps["scaler"].transform(val_matrix)
        )
        if bic_score < best_bic:
            patience_remaining = patience
            best_bic = bic_score
            best_n = n_comp
            logger.info(
                f"New best BIC ({best_bic:.2f}) found at {n_comp:,} components."
            )
        elif patience_remaining > 1:
            patience_remaining -= 1
            logger.info(
                f"BIC ({bic_score:.2f}) did not improve at {n_comp:,} components. "
                f"Patience remaining: {patience_remaining}"
            )
        else:
            logger.info(
                f"BIC stopped improving, so the best BIC was found at {best_n:,} "
                "components."
            )
            break

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    pipeline = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            (
                "gmm",
                GaussianMixture(
                    n_components=best_n,
                    covariance_type=covariance_type,
                    random_state=seed,
                ),
            ),
        ]
    ).fit(train_matrix)
    scaler = pipeline.named_steps["scaler"]
    gmm = pipeline.named_steps["gmm"]
    test_bic = gmm.bic(scaler.transform(test_matrix))
    logger.info(f"Test BIC: {test_bic:.2f}")

    # Sanity check
    logger.info(
        f"Mean log likelihoods:\n"
        f"- train: {pipeline.score(train_matrix):.4f}\n"
        f"- val: {pipeline.score(val_matrix):.4f}\n"
        f"- test: {pipeline.score(test_matrix):.4f}"
    )

    # Train final model on all data
    logger.info("Training final model on entire EU dataset...")
    full_matrix = eu_df[question_columns].values
    pipeline.fit(full_matrix)

    # Save the complete pipeline
    model_path = Path("models", "gmm.pkl")
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info(f"Pipeline saved to {model_path.resolve()}")
