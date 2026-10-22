"""
zero_trust.py - Turns a risk score (0..1) into ALLOW / FLAG / DENY.

Zero-trust idea: "never trust, always verify". Only clearly low-risk traffic is
allowed; uncertain traffic is FLAGged for extra checks; clearly risky traffic is
DENIED. A missing (NaN) risk score is FLAGged, never silently allowed.

The risk score can come from ANY source:
  * baseline.RuleBasedDetector.risk_scores(df)      (Person 3)
  * RandomForest.predict_proba(X)[:, 1]             (Person 2)
  * combine_scores(...) of both
"""
import numpy as np
import pandas as pd

import config
from evaluation import compute_metrics

ALLOW, FLAG, DENY = "ALLOW", "FLAG", "DENY"


class ZeroTrustPolicy:
    def __init__(self, allow_below=None, deny_at_or_above=None, flag_counts_as_attack=None):
        self.allow_below = config.ALLOW_BELOW if allow_below is None else allow_below
        self.deny_at_or_above = (config.DENY_AT_OR_ABOVE
                                 if deny_at_or_above is None else deny_at_or_above)
        self.flag_counts_as_attack = (config.FLAG_COUNTS_AS_ATTACK
                                      if flag_counts_as_attack is None else flag_counts_as_attack)
        if not 0.0 <= self.allow_below <= self.deny_at_or_above <= 1.0:
            raise ValueError("Need 0 <= allow_below <= deny_at_or_above <= 1 "
                             f"(got {self.allow_below}, {self.deny_at_or_above})")

    def decide_one(self, score):
        if np.isnan(score):
            return FLAG
        if score < 0 or score > 1:
            raise ValueError(f"Risk score must be in [0, 1], got {score}")
        if score >= self.deny_at_or_above:
            return DENY
        if score >= self.allow_below:
            return FLAG
        return ALLOW

    def decide(self, scores):
        """Vectorised decisions: numpy array of 'ALLOW'/'FLAG'/'DENY'."""
        s = np.asarray(scores, dtype=float)
        valid = s[~np.isnan(s)]
        if valid.size and (valid.min() < 0 or valid.max() > 1):
            raise ValueError("All risk scores must be within [0, 1].")
        out = np.full(s.shape, ALLOW, dtype=object)
        out[s >= self.allow_below] = FLAG
        out[s >= self.deny_at_or_above] = DENY
        out[np.isnan(s)] = FLAG
        return out.astype(str)

    def to_binary(self, decisions):
        """Decisions -> 0/1 for metrics. DENY=1, ALLOW=0, FLAG=1 or 0 (config)."""
        d = np.asarray(decisions)
        attack = (d == DENY) | ((d == FLAG) & self.flag_counts_as_attack)
        return attack.astype(int)

    def summarize(self, decisions, y_true=None):
        """Counts per decision (+ attack/normal breakdown if y_true is given)."""
        d = np.asarray(decisions)
        n = len(d)
        summary = {"total": int(n), "thresholds": {
            "allow_below": self.allow_below, "deny_at_or_above": self.deny_at_or_above,
            "flag_counts_as_attack": self.flag_counts_as_attack}, "decisions": {}}
        for name in (ALLOW, FLAG, DENY):
            mask = d == name
            entry = {"count": int(mask.sum()),
                     "percent": round(100 * mask.sum() / n, 2) if n else 0.0}
            if y_true is not None:
                y = np.asarray(y_true)
                entry["attacks"] = int((y[mask] == 1).sum())
                entry["normal"] = int((y[mask] == 0).sum())
            summary["decisions"][name] = entry
        return summary


def combine_scores(scores_a, scores_b, weight_a=0.5):
    """Weighted average of two risk scores (e.g. rule-based + Random Forest)."""
    if not 0.0 <= weight_a <= 1.0:
        raise ValueError("weight_a must be in [0, 1]")
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    if a.shape != b.shape:
        raise ValueError("Score arrays must have the same length.")
    return weight_a * a + (1 - weight_a) * b


def threshold_sweep(y_true, scores, allow_values=None, deny_values=None,
                    flag_counts_as_attack=None):
    """Try many (allow, deny) pairs; return a DataFrame of metrics per pair."""
    allow_values = config.SWEEP_ALLOW_VALUES if allow_values is None else allow_values
    deny_values = config.SWEEP_DENY_VALUES if deny_values is None else deny_values
    rows = []
    for a in allow_values:
        for d in deny_values:
            if a > d:
                continue
            policy = ZeroTrustPolicy(a, d, flag_counts_as_attack)
            decisions = policy.decide(scores)
            m = compute_metrics(y_true, policy.to_binary(decisions))
            rows.append({"allow_below": a, "deny_at_or_above": d,
                         "accuracy": m["accuracy"], "precision": m["precision"],
                         "recall": m["recall"], "f1": m["f1"],
                         "n_allow": int((decisions == ALLOW).sum()),
                         "n_flag": int((decisions == FLAG).sum()),
                         "n_deny": int((decisions == DENY).sum())})
    return pd.DataFrame(rows)
