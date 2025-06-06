"""Data processing functions."""

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

from .utils import group_country

logger = logging.getLogger(__name__)


def process_data(df: pd.DataFrame, imputation_neighbours: int) -> pd.DataFrame:
    """Process the survey data.

    Args:
        df:
            The survey data.
        imputation_neighbours:
            The number of neighbours to use for the kNN imputation of missing values.

    Returns:
        The processed DataFrame.
    """
    # Group countries
    logger.info("Grouping countries into regions...")
    df["country_group"] = df.country_code.apply(group_country)

    # Remove questions for which a country group exists that have not answered the
    # question
    questions_with_missing_answers: dict[str, list[str]] = defaultdict(list)
    for country_group in df.country_group.unique():
        country_group_df = df.query("country_group == @country_group")
        na_df = country_group_df.isna().all(axis=0)
        assert isinstance(na_df, pd.Series)
        na_df = na_df[na_df]
        assert isinstance(na_df, pd.Series)
        questions = na_df.index.tolist()
        for question in questions:
            assert isinstance(question, str)
            questions_with_missing_answers[question].append(country_group)
    if questions_with_missing_answers:
        survey_df = df.drop(columns=list(questions_with_missing_answers.keys()))
        questions_removed_str = "\n\t- ".join(questions_with_missing_answers.keys())
        logger.info(
            f"Removed {len(questions_with_missing_answers)} questions where at least "
            f"one country group has not answered:\n\t- {questions_removed_str}"
        )
        logger.info(
            f"Shape of the data after removing questions with missing answers: "
            f"{survey_df.shape}"
        )

    # Impute missing values
    logger.info("Imputing missing values using kNN...")
    question_columns = [col for col in df.columns if col.startswith("question_")]
    embedding_matrix = np.empty(shape=(df.shape[0], len(question_columns)))
    imputer = KNNImputer(
        n_neighbors=imputation_neighbours, weights="distance", keep_empty_features=True
    )
    for country_group in tqdm(
        iterable=df.country_group.unique(),
        desc="Imputing missing values",
        unit="country group",
    ):
        country_df = df.query("country_group == @country_group")
        country_df = country_df[question_columns].copy()
        assert isinstance(country_df, pd.DataFrame)
        country_embedding = imputer.fit_transform(X=country_df)
        assert isinstance(country_embedding, np.ndarray)
        embedding_matrix[country_df.index, :] = country_embedding

    # Normalize the data
    logger.info("Normalising the data...")
    embedding_matrix = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        X=embedding_matrix
    )

    # Update the survey DataFrame with the processed values
    df[question_columns] = embedding_matrix

    return df
