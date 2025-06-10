"""Data processing functions."""

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from feature_engine.imputation import RandomSampleImputer
from omegaconf import DictConfig
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

from .utils import group_country

logger = logging.getLogger(__name__)


def process_data(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """Process the survey data.

    Args:
        df:
            The survey data.
        config:
            The Hydra config.

    Returns:
        The processed DataFrame.

    Raises:
        ValueError:
            If any country or country group in the config is not present in the data.
    """
    # Group countries
    if config.use_country_groups:
        logger.info("Grouping countries into regions...")
        df["country_group"] = df.country_code.apply(group_country)
    else:
        logger.info("Using individual countries without grouping.")
        df["country_group"] = df.country_code

    # Filter countries
    if config.countries is not None:
        if any(country not in df.country_code.unique() for country in config.countries):
            raise ValueError(
                f"Some countries in the config ({config.countries}) are not present "
                f"in the data. Available countries: {df.country_code.unique().tolist()}"
            )
        logger.info(f"Filtering data for countries: {config.countries}")
        df = df.query("country_code in @config.countries").reset_index(drop=True)
        logger.info(f"Shape of the data after filtering: {df.shape}")

    # Filter country groups
    if config.country_groups is not None:
        if any(
            group not in df.country_group.unique() for group in config.country_groups
        ):
            raise ValueError(
                f"Some country groups in the config ({config.country_groups}) are not "
                f"present in the data. Available country groups: "
                f"{df.country_group.unique().tolist()}"
            )
        logger.info(f"Filtering data for country groups: {config.country_groups}")
        df = df.query("country_group in @config.country_groups").reset_index(drop=True)
        logger.info(f"Shape of the data after filtering: {df.shape}")

    # Remove questions for which a country exists that have not answered the question
    questions_with_missing_answers: dict[str, set[str]] = defaultdict(set)
    for country_group in df.country_group.unique():
        country_group_df = df.query("country_group == @country_group")
        na_df = country_group_df.isna().all(axis=0)
        assert isinstance(na_df, pd.Series)
        na_df = na_df[na_df]
        assert isinstance(na_df, pd.Series)
        questions = na_df.index.tolist()
        for question in questions:
            assert isinstance(question, str)
            questions_with_missing_answers[question].add(country_group)
    if questions_with_missing_answers:
        df = df.drop(columns=list(questions_with_missing_answers.keys()))
        questions_removed_str = "\n\t- ".join(
            [
                f"{question} (missing for {', '.join(country_groups)})"
                for question, country_groups in questions_with_missing_answers.items()
            ]
        )
        logger.info(
            f"Removed {len(questions_with_missing_answers)} questions where at least "
            f"one country group has not answered:\n\t- {questions_removed_str}"
        )
        logger.info(
            f"Shape of the data after removing questions with missing answers: "
            f"{df.shape}"
        )

    # Impute missing values
    logger.info("Imputing missing values...")
    question_columns = [col for col in df.columns if col.startswith("question_")]
    embedding_matrix = np.empty(shape=(df.shape[0], len(question_columns)))
    for country_group in tqdm(
        iterable=df.country_group.unique(),
        desc="Imputing missing values",
        unit="country_group",
    ):
        country_group_df = df.query("country_group == @country_group")
        country_group_df = country_group_df[question_columns].copy()
        assert isinstance(country_group_df, pd.DataFrame)
        country_embedding = (
            RandomSampleImputer(random_state=4242)
            .fit_transform(X=country_group_df)
            .values
        )
        assert isinstance(country_embedding, np.ndarray)
        embedding_matrix[country_group_df.index, :] = country_embedding

    # Normalize the data
    logger.info("Normalising the data...")
    embedding_matrix = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        X=embedding_matrix
    )

    # Update the survey DataFrame with the processed values
    df[question_columns] = embedding_matrix

    return df
