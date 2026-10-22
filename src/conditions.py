
"""Controlled test conditions for the UNSW-NB15 anomaly detection project.

Scenarios:
1. clean       - unchanged test data
2. noisy       - small perturbations to numeric feature values
3. missing     - simulated missing feature values, then imputed using
                 medians calculated from the training features
4. attack_10   - test subset with approximately 10% attacks
5. attack_30   - test subset with approximately 30% attacks

The original training and testing data are not modified.
"""

import numpy as np
import pandas as pd


def add_noise(X, noise_fraction=0.05, random_state=42):
    """Perturb a fraction of feature cells with small Gaussian noise."""
    if not 0 <= noise_fraction <= 1:
        raise ValueError("noise_fraction must be between 0 and 1.")

    rng = np.random.default_rng(random_state)
    result = X.copy().astype(float)

    if result.empty or noise_fraction == 0:
        return result

    values = result.to_numpy(copy=True)
    scales = result.std(axis=0).fillna(0).to_numpy().copy()
    scales = np.where(np.isfinite(scales), scales, 0)

    number_to_change = int(values.size * noise_fraction)
    if number_to_change == 0:
        return result

    flat_indices = rng.choice(
        values.size, size=number_to_change, replace=False
    )
    row_indices, col_indices = np.unravel_index(
        flat_indices, values.shape
    )

    values[row_indices, col_indices] += rng.normal(
        loc=0.0,
        scale=scales[col_indices] * 0.1,
    )

    return pd.DataFrame(
        values, columns=result.columns, index=result.index
    )


def add_missing_values(
    X_test, X_train, missing_fraction=0.05, random_state=42
):
    """Simulate missing values and impute using training-set medians."""
    if not 0 <= missing_fraction <= 1:
        raise ValueError("missing_fraction must be between 0 and 1.")

    rng = np.random.default_rng(random_state)
    result = X_test.copy().astype(float)

    if result.empty or missing_fraction == 0:
        return result

    values = result.to_numpy(copy=True)
    number_to_remove = int(values.size * missing_fraction)

    if number_to_remove:
        flat_indices = rng.choice(
            values.size, size=number_to_remove, replace=False
        )
        row_indices, col_indices = np.unravel_index(
            flat_indices, values.shape
        )
        values[row_indices, col_indices] = np.nan

    result = pd.DataFrame(
        values, columns=result.columns, index=result.index
    )

    # Use training data only to calculate imputation values.
    medians = X_train.median().reindex(result.columns).fillna(0)
    result = result.fillna(medians).fillna(0)

    return result


def set_attack_ratio(X, y, attack_ratio, random_state=42):
    """Return a subset with approximately the requested attack proportion.

    Label 0 = normal traffic; label 1 = attack traffic.
    Rows are sampled without replacement, and the original inputs
    remain unchanged.
    """
    if not 0 < attack_ratio < 1:
        raise ValueError("attack_ratio must be between 0 and 1.")

    labels = pd.Series(y, index=X.index)
    normal_indices = labels.index[labels == 0].to_numpy()
    attack_indices = labels.index[labels == 1].to_numpy()

    if len(normal_indices) == 0 or len(attack_indices) == 0:
        raise ValueError(
            "Both normal and attack examples are needed for this scenario."
        )

    # Keep the largest feasible subset for the requested ratio.
    total_size = min(
        len(labels),
        int(len(normal_indices) / (1 - attack_ratio)),
        int(len(attack_indices) / attack_ratio),
    )
    attack_count = round(total_size * attack_ratio)
    normal_count = total_size - attack_count

    # Ensure both classes are represented and counts are feasible.
    attack_count = max(1, min(attack_count, len(attack_indices)))
    normal_count = max(1, min(normal_count, len(normal_indices)))

    rng = np.random.default_rng(random_state)
    selected_normal = rng.choice(
        normal_indices, size=normal_count, replace=False
    )
    selected_attack = rng.choice(
        attack_indices, size=attack_count, replace=False
    )

    selected_indices = np.concatenate(
        [selected_normal, selected_attack]
    )
    rng.shuffle(selected_indices)

    X_subset = X.loc[selected_indices].copy()
    y_subset = labels.loc[selected_indices].copy()

    return X_subset, y_subset


def apply_scenario_to_raw_data(
    raw_df,
    scenario_name,
    noise_fraction=0.05,
    missing_fraction=0.05,
    random_state=42,
):
    """Apply noise or missing values to raw numeric network features."""
    result = raw_df.copy()

    if scenario_name not in {"noisy", "missing"}:
        return result

    numeric_columns = [
        col
        for col in result.select_dtypes(include=[np.number]).columns
        if col not in {"id", "label"}
    ]

    if not numeric_columns:
        return result

    # Convert to float BEFORE assigning decimals or NaN.
    result[numeric_columns] = result[numeric_columns].astype(float)
    values = result[numeric_columns].to_numpy(copy=True)
    rng = np.random.default_rng(random_state)

    if scenario_name == "noisy":
        scales = np.nanstd(values, axis=0)
        scales = np.where(np.isfinite(scales), scales, 0)
        count = int(values.size * noise_fraction)

        if count:
            indices = rng.choice(values.size, count, replace=False)
            rows, cols = np.unravel_index(indices, values.shape)
            values[rows, cols] += rng.normal(
                loc=0.0,
                scale=scales[cols] * 0.1,
            )

    elif scenario_name == "missing":
        count = int(values.size * missing_fraction)

        if count:
            indices = rng.choice(values.size, count, replace=False)
            rows, cols = np.unravel_index(indices, values.shape)
            values[rows, cols] = np.nan

        # Impute using medians from the selected raw scenario data.
        # Replace non-finite medians with zero as a safe fallback.
        medians = np.nanmedian(values, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0)

        missing_rows, missing_cols = np.where(np.isnan(values))
        values[missing_rows, missing_cols] = medians[missing_cols]

    # Assign a float DataFrame to avoid pandas integer-dtype errors.
    result[numeric_columns] = pd.DataFrame(
        values,
        columns=numeric_columns,
        index=result.index,
    )

    return result


def create_scenarios(
    X_train,
    X_test,
    y_test,
    noise_fraction=0.05,
    missing_fraction=0.05,
    attack_ratios=(0.10, 0.30),
    random_state=42,
):
    """Create all evaluation scenarios.

    Returns a dictionary where each value contains:
        "X": scenario features
        "y": corresponding true labels
    """
    X_clean = X_test.copy()
    y_clean = pd.Series(y_test, index=X_test.index).copy()

    scenarios = {
        "clean": {
            "X": X_clean,
            "y": y_clean,
        },
        "noisy": {
            "X": add_noise(
                X_clean,
                noise_fraction=noise_fraction,
                random_state=random_state,
            ),
            "y": y_clean.copy(),
        },
        "missing": {
            "X": add_missing_values(
                X_clean,
                X_train,
                missing_fraction=missing_fraction,
                random_state=random_state,
            ),
            "y": y_clean.copy(),
        },
    }

    for ratio in attack_ratios:
        X_ratio, y_ratio = set_attack_ratio(
            X_clean,
            y_clean,
            attack_ratio=ratio,
            random_state=random_state,
        )
        scenarios[f"attack_{int(ratio * 100)}"] = {
            "X": X_ratio,
            "y": y_ratio,
        }

    return scenarios


if __name__ == "__main__":
    # Basic standalone check using the project's existing preprocessing.
    from preprocess import load_unsw

    X_train, X_test, _, y_test, _ = load_unsw()
    scenarios = create_scenarios(X_train, X_test, y_test)

    for name, data in scenarios.items():
        attack_percentage = data["y"].mean() * 100
        print(
            f"{name:12s} | rows={len(data['X']):6d} | "
            f"features={data['X'].shape[1]:3d} | "
            f"attack rate={attack_percentage:5.1f}% | "
            f"missing values={int(data['X'].isna().sum().sum())}"
        )

    print("\nCONDITIONS OK")
