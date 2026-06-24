"""Data preparation and cleaning."""

import pandas as pd
import logging

from config_loader import get_config

logger = logging.getLogger(__name__)


def load_data(filepath):
    """Load the raw data."""
    df = pd.read_csv(filepath)
    logger.info("Loaded raw data from %s — shape: %s", filepath, df.shape)
    return df


def check_missing_values(df):
    """Check and log missing values."""
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.warning("Missing values found:\n%s", missing[missing > 0].to_string())
    else:
        logger.info("No missing values found")
    return missing


def remove_outliers(df, column=None, threshold=None):
    """Remove outliers above threshold in the target column."""
    cfg = get_config()
    if column is None:
        column = 'medv'
    if threshold is None:
        threshold = cfg['data']['outlier_threshold']

    original_size = len(df)
    df_clean = df[df[column] <= threshold].copy()
    removed = original_size - len(df_clean)
    logger.info(
        "Outlier removal on '%s' (threshold=%s): removed %d rows, %d remaining",
        column, threshold, removed, len(df_clean)
    )
    return df_clean


def check_skewness(df, column=None):
    """Check and log skewness of target column."""
    if column is None:
        column = 'medv'
    skew = df[column].skew()
    if skew >= 0.5:
        direction = "right-skewed"
    elif skew < -0.5:
        direction = "left-skewed"
    else:
        direction = "approximately normal"
    logger.info("Skewness of '%s': %.4f (%s)", column, skew, direction)
    return skew


def prepare_data(filepath=None, target_column=None, outlier_threshold=None):
    """Full data preparation pipeline driven by config.yaml."""
    cfg = get_config()

    if filepath is None:
        filepath = cfg['data']['raw_path']
    if target_column is None:
        target_column = 'medv'
    if outlier_threshold is None:
        outlier_threshold = cfg['data']['outlier_threshold']

    logger.info("Starting data preparation pipeline")

    df = load_data(filepath)
    check_missing_values(df)
    df_clean = remove_outliers(df, target_column, outlier_threshold)
    check_skewness(df_clean, target_column)

    logger.info("Data preparation complete — final shape: %s", df_clean.shape)
    return df_clean


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    df = prepare_data()
