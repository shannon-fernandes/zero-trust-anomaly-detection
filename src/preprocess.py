"""Loads and cleans the UNSW-NB15 training/testing CSVs.

Interface (please do not rename):
    load_unsw(data_dir=DEFAULT_DATA_DIR, sample_train=None, random_state=42)
        -> X_train, X_test, y_train, y_test, info

info is a dictionary with:
    "report"          : dict of row counts and other facts for the report
    "cat_cols"        : names of the text columns that were converted to numbers
    "attack_cat_test" : attack type of each test row (for per-attack analysis)

Leakage prevention: every statistic (medians, category codes) is learned from
the TRAINING file only and then applied to the test file.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NON_FEATURE_COLUMNS = ["id", "attack_cat", "label"]  # id = row number, others = targets


def find_file(keyword, data_dir):
    matches = sorted(Path(data_dir).glob(f"*{keyword}*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No CSV containing '{keyword}' found in {data_dir}. See data/README.txt.")
    return matches[0]


def _read(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    if "label" not in df.columns:
        raise ValueError(f"No 'label' column in {path.name}. Run inspect_dataset.py.")
    return df.drop(columns=["id"], errors="ignore")


def load_unsw(data_dir=DEFAULT_DATA_DIR, sample_train=None, random_state=42):
    train_path = find_file("training", data_dir)
    test_path = find_file("testing", data_dir)
    train, test = _read(train_path), _read(test_path)

    report = {"train_file": train_path.name, "test_file": test_path.name,
              "train_rows_loaded": len(train), "test_rows_loaded": len(test)}

    # Remove exact duplicate rows from the TRAINING data only
    train = train.drop_duplicates()
    report["train_rows_after_duplicate_removal"] = len(train)
    if sample_train is not None and sample_train < len(train):
        train = train.sample(n=sample_train, random_state=random_state)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    report["train_rows_final"] = len(train)
    report["test_rows_final"] = len(test)

    # Targets
    y_train = train["label"].astype(int)
    y_test = test["label"].astype(int)
    if not set(y_train.unique()) <= {0, 1} or y_train.nunique() < 2:
        raise ValueError("'label' must contain both 0 (normal) and 1 (attack).")
    if "attack_cat" in test.columns:
        attack_cat_test = test["attack_cat"].astype(str).str.strip()
    else:
        attack_cat_test = pd.Series(["unknown"] * len(test))

    # Features
    features = [c for c in train.columns if c not in NON_FEATURE_COLUMNS]
    missing_in_test = [c for c in features if c not in test.columns]
    if missing_in_test:
        raise ValueError(f"Test file lacks columns: {missing_in_test}")
    X_train = train[features].copy()
    X_test = test[features].copy()

    cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(X_train[c])]
    num_cols = [c for c in features if c not in cat_cols]

    # Numeric: infinite -> missing -> fill with TRAIN median
    X_train[num_cols] = X_train[num_cols].replace([np.inf, -np.inf], np.nan)
    X_test[num_cols] = X_test[num_cols].replace([np.inf, -np.inf], np.nan)
    medians = X_train[num_cols].median()
    report["missing_values_filled_train"] = int(X_train[num_cols].isna().sum().sum())
    report["missing_values_filled_test"] = int(X_test[num_cols].isna().sum().sum())
    X_train[num_cols] = X_train[num_cols].fillna(medians)
    X_test[num_cols] = X_test[num_cols].fillna(medians)

    # Text columns -> numbers (codes learned from TRAIN; unseen categories -> -1)
    if cat_cols:
        for df in (X_train, X_test):
            for c in cat_cols:
                df[c] = df[c].fillna("unknown").astype(str).str.strip().str.lower()
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoded_train = encoder.fit_transform(X_train[cat_cols])
        encoded_test = encoder.transform(X_test[cat_cols])
        for i, c in enumerate(cat_cols):
            X_train[c] = encoded_train[:, i]
            X_test[c] = encoded_test[:, i]

    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    report["num_features"] = len(features)
    report["text_columns_encoded"] = cat_cols
    report["train_normal"] = int((y_train == 0).sum())
    report["train_attack"] = int((y_train == 1).sum())
    report["test_normal"] = int((y_test == 0).sum())
    report["test_attack"] = int((y_test == 1).sum())

    info = {"report": report, "cat_cols": cat_cols, "attack_cat_test": attack_cat_test}
    return X_train, X_test, y_train, y_test, info


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, info = load_unsw()
    print("=" * 60)
    print("PREPROCESSING REPORT")
    print("=" * 60)
    for key, value in info["report"].items():
        print(f"{key:40s}: {value}")
    print("\nAttack types in test file:")
    print(info["attack_cat_test"].value_counts().to_string())
    print("\nPREPROCESS OK")