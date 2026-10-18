"""Prints a diagnostic report for the UNSW-NB15 training and testing CSVs.

Run from the project's top folder:
    python src/inspect_dataset.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def find_file(keyword):
    matches = sorted(DATA_DIR.glob(f"*{keyword}*.csv"))
    if not matches:
        sys.exit(f"No CSV with '{keyword}' in its name found in {DATA_DIR}")
    return matches[0]


def describe(path):
    print("=" * 60)
    print("File:", path.name)
    print("=" * 60)
    df = pd.read_csv(path, low_memory=False)
    print("1) SHAPE (rows, columns):", df.shape)
    print("2) COLUMNS:")
    for i, col in enumerate(df.columns):
        print(f"   {i:3d}  {col!r}  ({df[col].dtype})")

    cols = {c.strip().lower(): c for c in df.columns}
    for name in ("label", "attack_cat"):
        if name in cols:
            print(f"\n3) COUNTS FOR {cols[name]!r}:")
            print(df[cols[name]].value_counts(dropna=False).to_string())
        else:
            print(f"\n3) No column called {name!r} found. Check the list above.")

    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    print("\n4) NON-NUMERIC COLUMNS:", text_cols if text_cols else "none")

    missing = df.isna().sum()
    missing = missing[missing > 0]
    print("\n5) MISSING VALUES:", "none" if missing.empty else "")
    if not missing.empty:
        print(missing.to_string())

    numeric = df.select_dtypes(include="number")
    inf = np.isinf(numeric).sum()
    inf = inf[inf > 0]
    print("\n6) INFINITE VALUES:", "none" if inf.empty else "")
    if not inf.empty:
        print(inf.to_string())

    print("\n7) DUPLICATE ROWS (ignoring 'id'):",
          int(df.drop(columns=[c for c in df.columns if c.lower() == "id"]).duplicated().sum()))
    return df


if __name__ == "__main__":
    train = describe(find_file("training"))
    test = describe(find_file("testing"))
    same = list(train.columns) == list(test.columns)
    print("\nTrain and test have identical columns:", same)