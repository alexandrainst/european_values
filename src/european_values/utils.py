"""Utility functions for the project."""

from .constants import (
    AFRICAN_COUNTRY_CODES,
    ASIAN_COUNTRY_CODES,
    EUROPEAN_COUNTRY_CODES,
    MIDDLE_EASTERN_COUNTRY_CODES,
    NORTH_AMERICAN_COUNTRY_CODES,
    OCEANIA_COUNTRY_CODES,
    SOUTH_AMERICAN_COUNTRY_CODES,
)


def group_country(country_code: str) -> str:
    """Group countries into European Union and Non-European Union."""
    if country_code in EUROPEAN_COUNTRY_CODES:
        return "Europe"
    elif country_code in NORTH_AMERICAN_COUNTRY_CODES:
        return "North America"
    elif country_code in SOUTH_AMERICAN_COUNTRY_CODES:
        return "South America"
    elif country_code in MIDDLE_EASTERN_COUNTRY_CODES:
        return "Middle East"
    elif country_code in AFRICAN_COUNTRY_CODES:
        return "Africa"
    elif country_code in ASIAN_COUNTRY_CODES:
        return "Asia"
    elif country_code in OCEANIA_COUNTRY_CODES:
        return "Oceania"
    else:
        return country_code
