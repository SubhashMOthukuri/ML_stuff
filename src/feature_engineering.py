"""Feature Engineering"""

import pandas as pd
import numpy as np
import logging

from config_loader import get_config

logger = logging.getLogger(__name__)


def create_polynomial_features(df):
    """Create squared and interaction features."""
    df['age_squared'] = df['age'] ** 2
    df['rm_squared'] = df['rm'] ** 2
    df['dis_squared'] = df['dis'] ** 2
    logger.info("Polynomial features created: age_squared, rm_squared, dis_squared")
    return df


def create_interaction_features(df):
    """Create interaction features."""
    df['rm_lstat_interaction'] = df['rm'] * df['lstat']
    logger.info("Interaction feature created: rm_lstat_interaction")
    return df


def create_ratio_features(df):
    """Create ratio features."""
    df['tax_per_room'] = df['tax'] / df['rm']
    logger.info("Ratio feature created: tax_per_room")
    return df


ORIGINAL_FEATURES = [
    'crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis',
    'rad', 'tax', 'ptratio', 'b', 'lstat'
]
ENGINEERED_FEATURES = [
    'age_squared', 'rm_squared', 'rm_lstat_interaction',
    'tax_per_room', 'dis_squared'
]


def get_features_list():
    """Return list of all features."""
    return ORIGINAL_FEATURES + ENGINEERED_FEATURES


def engineer_features(df):
    """Complete feature engineering pipeline."""
    logger.info("Starting feature engineering pipeline")

    df = create_polynomial_features(df)
    df = create_interaction_features(df)
    df = create_ratio_features(df)

    logger.info(
        "Feature engineering complete — total: %d (original: %d, engineered: %d)",
        len(get_features_list()), len(ORIGINAL_FEATURES), len(ENGINEERED_FEATURES)
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = get_config()
    filepath = cfg['data']['raw_path']
    df = pd.read_csv(filepath)
    df = engineer_features(df)
