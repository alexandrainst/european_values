"""Training generative on the dataset."""

import logging
from pathlib import Path

import cloudpickle
import pandas as pd
import scipy.special
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from . import sigmoid_transformer

logger = logging.getLogger(__name__)


def train_generative_model(
    eu_df: pd.DataFrame, scaler: MinMaxScaler, test_samples_per_country: int, seed: int
) -> None:
    """Train a generative model on EU survey data.

    Args:
        eu_df:
            A non-normalised dataframe containing the survey responses from EU
            countries.
        scaler:
            A data scaler that has been fitted on all of the data (not just the EU
            data).
        test_samples_per_country:
            Number of samples per country for the test set.
        seed:
            Random seed for reproducibility.
    """
    # Get question columns
    question_columns = [col for col in eu_df.columns if col.startswith("question_")]
    logger.info(f"Training with {len(question_columns):,} questions")

    # Split data by country
    logger.info("Splitting data into train/test sets...")
    train_dfs: list[pd.DataFrame] = []
    val_dfs: list[pd.DataFrame] = []
    test_dfs: list[pd.DataFrame] = []
    for country in eu_df["country_code"].unique():
        country_data = eu_df.query("country_code == @country").sample(
            frac=1, random_state=seed
        )
        n_test = min(test_samples_per_country, len(country_data) // 5)
        test_dfs.append(country_data.iloc[:n_test])
        val_dfs.append(country_data.iloc[n_test : 2 * n_test])
        train_dfs.append(country_data.iloc[2 * n_test :])

    # Set up the data as NumPy arrays
    train_matrix = scaler.transform(pd.concat(train_dfs)[question_columns].values)
    val_matrix = scaler.transform(pd.concat(val_dfs)[question_columns].values)
    test_matrix = scaler.transform(pd.concat(test_dfs)[question_columns].values)
    logger.info(
        f"There are {len(train_matrix):,} training samples, "
        f"{len(val_matrix):,} validation samples, "
        f"and {len(test_matrix):,} test samples."
    )

    # Fit the model. We select a small bandwidth to ensure that the model fits the data
    # well (lower bandwidth means more sensitivity to the data, i.e., higher variance)
    logger.info("Training the model on the training data...")
    model = KernelDensity(bandwidth=0.1).fit(train_matrix)

    # Set the `transform` method of the model to the score_samples method, as this will
    # allow us to use the scaler, model and scorer in the same pipeline
    model.transform = model.score_samples.__get__(model)

    # logger.info("Computing the log-likelihoods for the training data...")
    logger.info("Computing the log-likelihoods for the training data...")
    train_log_likelihoods = model.transform(train_matrix)

    logger.info("Computing the log-likelihoods for the validation data...")
    val_log_likelihoods = model.transform(val_matrix)

    logger.info("Computing the log-likelihoods for the test data...")
    test_log_likelihoods = model.transform(test_matrix)

    # Fit the log-likelihood transform
    logger.info("Fitting the sigmoid transform on the validation data...")
    scorer = sigmoid_transformer.SigmoidTransformer().fit(val_log_likelihoods)

    logger.info("Evaluating the model on the training, validation and test data...")
    logger.info(
        f"Log-likelihoods for train:\n"
        f"\t- Mean: {train_log_likelihoods.mean():.4f}\n"
        f"\t- Std: {train_log_likelihoods.std():.4f}\n"
        f"\t- Min: {train_log_likelihoods.min():.4f}\n"
        f"\t- 10% quantile: {pd.Series(train_log_likelihoods).quantile(q=0.1):.4f}\n"
        f"\t- 90% quantile: {pd.Series(train_log_likelihoods).quantile(q=0.9):.4f}\n"
        f"\t- Max: {train_log_likelihoods.max():.4f}\n"
        f"Mean score for train: {scorer.transform(train_log_likelihoods).mean():.0%}"
    )
    logger.info(
        f"Log-likelihoods for validation:\n"
        f"\t- Mean: {val_log_likelihoods.mean():.4f}\n"
        f"\t- Std: {val_log_likelihoods.std():.4f}\n"
        f"\t- Min: {val_log_likelihoods.min():.4f}\n"
        f"\t- 10% quantile: {pd.Series(val_log_likelihoods).quantile(q=0.1):.4f}\n"
        f"\t- 90% quantile: {pd.Series(val_log_likelihoods).quantile(q=0.9):.4f}\n"
        f"\t- Max: {val_log_likelihoods.max():.4f}\n"
        f"Mean score for validation: {scorer.transform(val_log_likelihoods).mean():.0%}"
    )
    logger.info(
        f"Log-likelihoods for test:\n"
        f"\t- Mean: {test_log_likelihoods.mean():.4f}\n"
        f"\t- Std: {test_log_likelihoods.std():.4f}\n"
        f"\t- Min: {test_log_likelihoods.min():.4f}\n"
        f"\t- 10% quantile: {pd.Series(test_log_likelihoods).quantile(q=0.1):.4f}\n"
        f"\t- 90% quantile: {pd.Series(test_log_likelihoods).quantile(q=0.9):.4f}\n"
        f"\t- Max: {test_log_likelihoods.max():.4f}\n"
        f"Mean score for test: {scorer.transform(test_log_likelihoods).mean():.0%}"
    )

    # Train final model on all data
    logger.info("Training final model on entire EU dataset...")
    full_matrix = scaler.transform(eu_df[question_columns].values)
    model.fit(full_matrix)
    pipeline = Pipeline([("scaler", scaler), ("model", model), ("scorer", scorer)])

    # Save the complete pipeline
    model_path = Path("models", "pipeline.pkl")
    model_path.parent.mkdir(exist_ok=True)
    cloudpickle.register_pickle_by_value(module=sigmoid_transformer)
    cloudpickle.register_pickle_by_value(module=scipy.special)
    with model_path.open("wb") as f:
        cloudpickle.dump(obj=pipeline, file=f)
    logger.info(f"Pipeline saved to {model_path.as_posix()}")
