"""
baseline.py - Rule-based network traffic detection (no machine learning).

Idea: look at what NORMAL traffic in the training set looks like. For each
selected feature, a rule fires when a flow is unusually extreme compared with
normal traffic (above the 90th percentile for "high" rules, below the 10th for
"low" rules). The more rules fire, the higher the risk.

Works on RAW (unscaled) values, so use the raw dataframe from data_loader.py,
not a scaled matrix from a preprocessing step.
Labels are used ONLY inside fit() (to find normal rows). predict() and
risk_scores() never read the label column.
"""
import numpy as np
import pandas as pd

import config


class RuleBasedDetector:
    def __init__(self, rule_features=None, percentile=None,
                 min_rules=None, rules_for_full_risk=None):
        self.rule_features = dict(rule_features or config.RULE_FEATURES)
        self.percentile = config.RULE_PERCENTILE if percentile is None else percentile
        self.min_rules = config.MIN_RULES_FOR_ATTACK if min_rules is None else min_rules
        self.rules_for_full_risk = (config.RULES_FOR_FULL_RISK
                                    if rules_for_full_risk is None else rules_for_full_risk)
        if not 50 <= self.percentile < 100:
            raise ValueError("percentile must be in [50, 100)")
        if self.min_rules < 1 or self.rules_for_full_risk < 1:
            raise ValueError("min_rules and rules_for_full_risk must be >= 1")
        for feat, direction in self.rule_features.items():
            if direction not in ("high", "low"):
                raise ValueError(f"Direction for '{feat}' must be 'high' or 'low'")
        self.thresholds_ = None

    # ------------------------------------------------------------------ fit
    def fit(self, train_df):
        """Learn thresholds from normal (label == 0) rows of the TRAINING data."""
        missing = [f for f in self.rule_features if f not in train_df.columns]
        if missing:
            raise ValueError(f"Columns missing from training data: {missing}")
        normal = train_df[train_df[config.LABEL_COLUMN] == 0]
        if normal.empty:
            raise ValueError("Training data has no normal rows (label == 0).")
        self.thresholds_ = {}
        for feat, direction in self.rule_features.items():
            q = self.percentile if direction == "high" else 100 - self.percentile
            self.thresholds_[feat] = float(np.nanpercentile(normal[feat].astype(float), q))
        return self

    def _check_fitted(self):
        if self.thresholds_ is None:
            raise RuntimeError("Call fit(train_df) before predicting.")

    # ----------------------------------------------------------- inference
    def rule_matrix(self, df):
        """DataFrame of True/False: which rule fired for which row."""
        self._check_fitted()
        out = {}
        for feat, direction in self.rule_features.items():
            values = df[feat].astype(float)
            thr = self.thresholds_[feat]
            out[f"{feat}_{direction}"] = (values > thr) if direction == "high" else (values < thr)
        return pd.DataFrame(out, index=df.index)

    def rules_fired(self, df):
        """Number of rules fired per row."""
        return self.rule_matrix(df).sum(axis=1).to_numpy()

    def risk_scores(self, df):
        """Risk in [0, 1]: fired rules / rules_for_full_risk (capped at 1)."""
        return np.clip(self.rules_fired(df) / self.rules_for_full_risk, 0.0, 1.0)

    def predict(self, df):
        """Binary prediction: 1 = attack, 0 = normal."""
        return (self.rules_fired(df) >= self.min_rules).astype(int)
