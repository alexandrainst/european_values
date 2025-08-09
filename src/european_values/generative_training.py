"""Training generative on the dataset."""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def train_generative_model(
    eu_df: pd.DataFrame,
    scaler: MinMaxScaler,
    test_samples_per_country: int,
    bandwidth: float,
    seed: int,
) -> None:
    """Train a generative model on EU survey data.

    Args:
        eu_df:
            A non-normalised dataframe containing the survey responses from EU
            countries.
        scaler:
            A data scaler that has been fitted on all of the data (not just the EU
            data).
        bandwidth:
            Bandwidth for the Kernel Density Estimation. It can be seen as a
            regularisation parameter: a smaller value will lead to more variance in the
            model, while a larger value will lead to more bias.
        test_samples_per_country:
            Number of samples per country for the test set.
        seed:
            Random seed for reproducibility.
    """
    # Get question columns
    question_columns = [col for col in eu_df.columns if col.startswith("question_")]
    logger.info(f"Training with {len(question_columns)} questions")

    # Split data by country
    logger.info("Splitting data into train/test sets...")
    train_dfs: list[pd.DataFrame] = []
    test_dfs: list[pd.DataFrame] = []
    for country in eu_df["country_code"].unique():
        country_data = eu_df[eu_df["country_code"] == country].sample(
            frac=1, random_state=seed
        )
        n_test = min(test_samples_per_country, len(country_data) // 5)
        test_dfs.append(country_data.iloc[:n_test])
        train_dfs.append(country_data.iloc[n_test:])

    # Set up the data as NumPy arrays
    train_matrix = scaler.transform(pd.concat(train_dfs)[question_columns].values)
    test_matrix = scaler.transform(pd.concat(test_dfs)[question_columns].values)
    logger.info(
        f"There are {len(train_matrix):,} training samples and {len(test_matrix):,} "
        "test samples."
    )

    # Initialise the model
    logger.info("Training the model on the training data...")
    model = KernelDensity(bandwidth=bandwidth, kernel="gaussian", algorithm="auto")
    model.fit(train_matrix)

    # Evaluate the model
    logger.info("Evaluating the model on the training and test data...")
    train_logprobs = model.score(train_matrix)
    test_logprobs = model.score(test_matrix)
    logger.info(
        f"Mean log-probs:\n- train: {train_logprobs:,.0f}\n- test: {test_logprobs:,.0f}"
    )

    # Train final model on all data
    logger.info("Training final model on entire EU dataset...")
    full_matrix = scaler.transform(eu_df[question_columns].values)
    model.fit(full_matrix)
    pipeline = Pipeline([("scaler", scaler), ("model", model)])

    # Save the complete pipeline
    model_path = Path("models", "model.pkl")
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info(f"Pipeline saved to {model_path.resolve()}")
