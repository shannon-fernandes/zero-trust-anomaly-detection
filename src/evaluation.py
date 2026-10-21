"""
evaluation.py - Accuracy, precision, recall, F1-score and confusion matrix.

Convention: label 1 = attack = "positive class".
Confusion matrix layout (same as scikit-learn):
              predicted 0   predicted 1
   actual 0 [     TN            FP     ]
   actual 1 [     FN            TP     ]
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

import config


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Length mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Cannot evaluate empty arrays.")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "n_samples": int(y_true.size),
    }


def per_category_detection(df, y_pred):
    """
    How often each attack_cat was flagged as attack. attack_cat is used HERE
    ONLY for reporting (never as a model input).
    """
    if config.ATTACK_CAT_COLUMN not in df.columns:
        raise ValueError(f"'{config.ATTACK_CAT_COLUMN}' column not found.")
    tmp = pd.DataFrame({"cat": df[config.ATTACK_CAT_COLUMN].to_numpy(),
                        "pred": np.asarray(y_pred).astype(int)})
    grouped = tmp.groupby("cat")["pred"].agg(["count", "mean"])
    return {cat: {"count": int(r["count"]), "flagged_as_attack_rate": float(r["mean"])}
            for cat, r in grouped.iterrows()}


def save_results(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return path


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_metrics_table(named_metrics):
    """Pretty text table for printing: {name: metrics_dict}."""
    lines = [f"{'Model':<26}{'Acc':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}"]
    for name, m in named_metrics.items():
        lines.append(f"{name:<26}{m['accuracy']:>8.4f}{m['precision']:>8.4f}"
                     f"{m['recall']:>8.4f}{m['f1']:>8.4f}")
    return "\n".join(lines)
