"""Utility functions for the project."""

import logging

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
