"""Loading the data for the project."""

import logging
import re
from zipfile import ZipFile

import pandas as pd
from tqdm.auto import tqdm

from .constants import EVS_TREND_ANSWER_COLUMNS, EVS_TREND_CATEGORICAL_COLUMNS

logger = logging.getLogger(__name__)


def load_evs_trend_data() -> pd.DataFrame:
    """Load and process the EVS trend data from 1981-2017.

    Returns:
        The processed DataFrame.
    """
    logger.info("Loading EVS trend data from 1981-2017...")
    df = load_zipped_data(data_path="data/raw/evs-trend-1981-2017.zip")
    logger.info(f"Loaded {len(df):,} rows of data.")

    # Rename columns
    logger.info("Renaming columns...")
    metadata_mapping = dict(S007_01="respondent_id", S009="country_code", S020="year")
    df = df.rename(columns=metadata_mapping | EVS_TREND_ANSWER_COLUMNS)

    # Drop columns that are not needed
    logger.info("Dropping unnecessary columns...")
    columns_to_keep = list(metadata_mapping.values()) + sorted(
        EVS_TREND_ANSWER_COLUMNS.values()
    )
    df = df[columns_to_keep]

    # Convert all categorical columns to one-hot columns
    for column in tqdm(
        iterable=EVS_TREND_CATEGORICAL_COLUMNS,
        desc="One-hot encoding categorical columns",
    ):
        new_column = EVS_TREND_ANSWER_COLUMNS[column]
        one_hotted = pd.get_dummies(
            data=df[new_column], dtype=int, prefix=new_column, prefix_sep="_choice"
        )
        one_hotted = one_hotted.where(cond=one_hotted.sum(axis=1) != 0, other=None)
        df = pd.concat([df, one_hotted], axis=1)
    df = df.drop(
        columns=[
            EVS_TREND_ANSWER_COLUMNS[column] for column in EVS_TREND_CATEGORICAL_COLUMNS
        ]
    )

    # Convert metadata coded values to actual values
    logger.info("Converting metadata coded values to actual values...")
    df = df.assign(
        country_code=lambda series: [
            re.sub(pattern=r"\-.*", repl="", string=x) for x in series.country_code
        ],
        year=lambda s: [x if x > 0 else None for x in s.year],
    )

    # Non-answers are coded by negative values; we convert them to None
    logger.info("Converting non-answers to None...")
    df = df.map(lambda x: None if isinstance(x, int) and x < 0 else x)

    # Set datatypes
    logger.info("Setting datatypes...")
    df = df.convert_dtypes()

    logger.info(
        f"Data loaded and processed successfully. There are {len(df):,} rows and "
        f"{len(df.columns):,} columns."
    )

    assert isinstance(df, pd.DataFrame)
    return df


def load_evs_wvs_data() -> pd.DataFrame:
    """Load and process the joint EVS/WVS data from 2017-2022.

    Returns:
        The processed DataFrame.
    """
    load_zipped_data(data_path="data/raw/evs-wvs-2017-2022.zip")
    raise NotImplementedError


def load_zipped_data(data_path: str) -> pd.DataFrame:
    """Load the first file in a zipped file.

    This assumes that the zipped file contains a single Stata file.

    Args:
        data_path:
            The path to the zip file containing the data in Stata format.

    Returns:
        A pandas DataFrame containing the data.
    """
    with ZipFile(file=data_path) as zip_file:
        file_name = zip_file.namelist()[0]
        with zip_file.open(name=file_name) as file:
            df = pd.read_stata(file, convert_categoricals=False)
            assert isinstance(df, pd.DataFrame)
            return df
