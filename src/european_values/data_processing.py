"""Data processing functions."""

import logging

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
