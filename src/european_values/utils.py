"""Utility functions for the project."""

import logging

import pandas as pd

from .constants import COUNTRY_GROUPS

logger = logging.getLogger(__name__)


def group_country(country_code: str) -> str:
    """Group country codes into regions.

    Args:
        country_code:
            The country code to group.

    Returns:
        The group the country belongs to, or the country code if it does not belong to
        any group.
    """
    for group, countries in COUNTRY_GROUPS.items():
        if country_code in countries:
            return group
    else:
        logger.warning(
            f"Country code {country_code!r} does not belong to any group. Returning "
            f"the country code itself."
        )
        return country_code


def df_has_column_with_only_nans(df: pd.DataFrame) -> bool:
    """Check if a DataFrame has a column with only NaN values.

    Args:
        df:
            The DataFrame to check.

    Returns:
        True if there is at least one column with only NaN values, False otherwise.
    """
    return any(df[col].isna().all() for col in df.columns)
