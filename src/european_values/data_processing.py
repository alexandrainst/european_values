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
    logger.info("Grouping countries into regions...")
    df["country_group"] = df.country_code.apply(group_country)

    # Filter countries
    if config.countries is not None:
        missing_countries = set(config.countries) - set(df.country_code.unique())
        if missing_countries:
            logger.warning(
                f"The countries {missing_countries} in the config are not present "
                "in the data. Ignoring them."
            )
        logger.info(f"Filtering data for countries: {config.countries}")
        df = df.query("country_code in @config.countries").reset_index(drop=True)
        logger.info(f"Shape of the data after filtering: {df.shape}")

    # Filter country groups
    if config.country_groups is not None:
        missing_groups = set(config.country_groups) - set(df.country_group.unique())
        if missing_groups:
            logger.warning(
                f"The country groups {missing_groups} in the config are not present "
                "in the data. Ignoring them."
            )
        logger.info(f"Filtering data for country groups: {config.country_groups}")
        df = df.query("country_group in @config.country_groups").reset_index(drop=True)
        logger.info(f"Shape of the data after filtering: {df.shape}")

    # Get the country groupings, which depends on whether we are working with countries
    # or country groups
    country_grouping_str = (
        "country_group" if config.use_country_groups else "country_code"
    )
    unique_country_groupings = (
        df.country_group.unique()
        if config.use_country_groups
        else df.country_code.unique()
    )

    # Remove questions for which a country or country group exists that have not
    # answered the question
    questions_with_missing_answers: dict[str, set[str]] = defaultdict(set)
    for country_grouping in unique_country_groupings:
        country_grouping_df = df.query(f"{country_grouping_str} == @country_grouping")
        na_df = country_grouping_df.isna().all(axis=0)
        assert isinstance(na_df, pd.Series)
        na_df = na_df[na_df]
        assert isinstance(na_df, pd.Series)
        questions = na_df.index.tolist()
        for question in questions:
            assert isinstance(question, str)
            questions_with_missing_answers[question].add(country_grouping)
    if questions_with_missing_answers:
        df = df.drop(columns=list(questions_with_missing_answers.keys()))
        questions_removed_str = "\n\t- ".join(
            [
                f"{question} (missing for {', '.join(groupings)})"
                for question, groupings in questions_with_missing_answers.items()
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
    for country_grouping in tqdm(
        iterable=unique_country_groupings,
        desc="Imputing missing values",
        unit=country_grouping_str,
    ):
        country_grouping_df = df.query(f"{country_grouping_str} == @country_grouping")
        country_grouping_df = country_grouping_df[question_columns].copy()
        assert isinstance(country_grouping_df, pd.DataFrame)
        country_embedding = (
            RandomSampleImputer(random_state=4242)
            .fit_transform(X=country_grouping_df)
            .values
        )
        assert isinstance(country_embedding, np.ndarray)
        embedding_matrix[country_grouping_df.index, :] = country_embedding

    # Normalize the data
    logger.info("Normalising the data...")
    embedding_matrix = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        X=embedding_matrix
    )

    # Update the survey DataFrame with the processed values
    df[question_columns] = embedding_matrix

    return df
