"""Utility functions for the project."""

import logging

from .constants import (
    CARIBBEAN,
    CENTRAL_AMERICA,
    CENTRAL_ASIA,
    CENTRAL_EUROPE,
    EAST_ASIA,
    EASTERN_EUROPE,
    MIDDLE_EAST,
    NORTH_AFRICA,
    NORTH_AMERICA,
    NORTHERN_EUROPE,
    OCEANIA,
    SOUTH_AMERICA,
    SOUTH_ASIA,
    SOUTHEAST_ASIA,
    SOUTHERN_EUROPE,
    SUB_SAHARAN_AFRICA,
    WESTERN_EUROPE,
)

logger = logging.getLogger(__name__)


def group_country(country_code: str) -> str:
    """Group country codes into regions.

    Args:
        country_code:
            The country code to group.

    Returns:
        The group the country belongs to, or the country code if it does not belong to
        any group.

    Raises:
        ValueError:
            If the country code does not belong to any group.
    """
    if country_code in NORTH_AMERICA:
        return "North America"
    elif country_code in CENTRAL_AMERICA:
        return "Central America"
    elif country_code in CARIBBEAN:
        return "Caribbean"
    elif country_code in SOUTH_AMERICA:
        return "South America"
    elif country_code in NORTH_AFRICA:
        return "North Africa"
    elif country_code in SUB_SAHARAN_AFRICA:
        return "Sub Saharan Africa"
    elif country_code in WESTERN_EUROPE:
        return "Western Europe"
    elif country_code in NORTHERN_EUROPE:
        return "Northern Europe"
    elif country_code in SOUTHERN_EUROPE:
        return "Southern Europe"
    elif country_code in CENTRAL_EUROPE:
        return "Central Europe"
    elif country_code in EASTERN_EUROPE:
        return "Eastern Europe"
    elif country_code in SOUTH_ASIA:
        return "South Asia"
    elif country_code in SOUTHEAST_ASIA:
        return "Southeast Asia"
    elif country_code in EAST_ASIA:
        return "East Asia"
    elif country_code in CENTRAL_ASIA:
        return "Central Asia"
    elif country_code in OCEANIA:
        return "Oceania"
    elif country_code in MIDDLE_EAST:
        return "Middle East"
    else:
        raise ValueError(
            f"Country code {country_code!r} does not belong to any group. Please check "
            "the country code and try again."
        )
