"""Data processing functions."""

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process the survey data.

    Args:
        df:
            The survey data.

    Returns:
        The processed DataFrame.
    """
    # Remove questions for which a country exists that have not answered the question
    questions_with_missing_answers: dict[str, list[str]] = defaultdict(list)
    for country_code in df.country_code.unique():
        country_code_df = df.query("country_code == @country_code")
        na_df = country_code_df.isna().all(axis=0)
        assert isinstance(na_df, pd.Series)
        na_df = na_df[na_df]
        assert isinstance(na_df, pd.Series)
        questions = na_df.index.tolist()
        for question in questions:
            assert isinstance(question, str)
            questions_with_missing_answers[question].append(country_code)
    if questions_with_missing_answers:
        df = df.drop(columns=list(questions_with_missing_answers.keys()))
        questions_removed_str = "\n\t- ".join(
            [
                f"{question} (missing for {', '.join(countries)})"
                for question, countries in questions_with_missing_answers.items()
            ]
        )
        logger.info(
            f"Removed {len(questions_with_missing_answers)} questions where at least "
            f"one country has not answered:\n\t- {questions_removed_str}"
        )
        logger.info(
            f"Shape of the data after removing questions with missing answers: "
            f"{df.shape}"
        )

    # Impute missing values
    logger.info("Imputing missing values...")
    question_columns = [col for col in df.columns if col.startswith("question_")]
    embedding_matrix = np.empty(shape=(df.shape[0], len(question_columns)))
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    for country_code in tqdm(
        iterable=df.country_code.unique(),
        desc="Imputing missing values",
        unit="country",
    ):
        country_df = df.query("country_code == @country_code")
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
