"""
data_loader.py - Loads the official UNSW-NB15 training/testing CSV files
and guards against label leakage.
"""

import pandas as pd
import config


def load_csv(path):
    """Read a dataset CSV and validate its columns."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Download the official UNSW-NB15 training/testing files from "
            f"{config.DATASET_PAGE_URL}\n"
            f"Place them in: {config.DATA_DIR}"
        )

    # The official training CSV provides the column names.
    if path.name == "UNSW_NB15_training-set.csv":
        df = pd.read_csv(path)

    # The testing file currently appears to have no header.
    
    elif path.name == "UNSW_NB15_testing-set.csv":
        df = pd.read_csv(path)

    else:
        raise ValueError(f"Unsupported dataset file: {path.name}")

    if config.LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"'{config.LABEL_COLUMN}' column missing in {path.name}."
        )

    return df


def load_train_test():
    """Return the training and testing DataFrames."""
    return load_csv(config.TRAIN_FILE), load_csv(config.TEST_FILE)


def feature_columns(df):
    """Return columns safe for model input."""
    return [
        column for column in df.columns
        if column not in config.LEAKAGE_COLUMNS
    ]


def get_features(df):
    """Return features without label, attack category, or ID."""
    return df[feature_columns(df)].copy()


def get_labels(df):
    """Return binary labels as a NumPy array."""
    return df[config.LABEL_COLUMN].astype(int).to_numpy()


def assert_no_leakage(columns):
    """Raise an error if leakage columns are included."""
    bad = [
        column for column in columns
        if column in config.LEAKAGE_COLUMNS
    ]

    if bad:
        raise ValueError(
            f"Label leakage! These columns must not be used as input: {bad}"
        )