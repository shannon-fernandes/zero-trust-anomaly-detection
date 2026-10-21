
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_random_forest(X_train, y_train, random_state=42):
    """class_weight='balanced' gives the rarer class more importance."""
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                   n_jobs=-1, random_state=random_state)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    return model, time.perf_counter() - start


def train_logistic_regression(X_train, y_train, random_state=42):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced",
                           random_state=random_state))
    start = time.perf_counter()
    model.fit(X_train, y_train)
    return model, time.perf_counter() - start


def timed_predict(model, X):
    start = time.perf_counter()
    pred = np.asarray(model.predict(X))
    return pred, time.perf_counter() - start


def evaluate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "accuracy": accuracy_score(y_true, y_pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


if __name__ == "__main__":
    from preprocess import load_unsw

    X_train, X_test, y_train, y_test, _ = load_unsw()
    for name, trainer in (("Random Forest", train_random_forest),
                          ("Logistic Regression", train_logistic_regression)):
        model, t_train = trainer(X_train, y_train)
        pred, t_pred = timed_predict(model, X_test)
        print(f"\n{name}: train {t_train:.1f}s, predict {t_pred:.2f}s")
        for k, v in evaluate(y_test, pred).items():
            print(f"   {k:10s}: {v:.4f}" if isinstance(v, float) else f"   {k:10s}: {v}")
    print("\nTRAIN_MODEL OK")
