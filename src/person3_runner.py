"""
person3_runner.py - Runs Person 3's pipeline end to end (NOT the project's
main.py; Person 4 owns main.py and can import run() from here).

    python person3_runner.py --source synthetic   # demo with FAKE data
    python person3_runner.py --source real        # needs the two official CSVs
    python person3_runner.py --source real --rf-scores rf_probs.npy
        (optional: Person 2's Random Forest attack probabilities for the test
         set, same row order as UNSW_NB15_testing-set.csv)

Outputs go to outputs/<source>/ so synthetic and real results never mix.
"""
import argparse

import numpy as np

import config
import data_loader
from baseline import RuleBasedDetector
from evaluation import (compute_metrics, format_metrics_table,
                        per_category_detection, save_results)
from synthetic_data import make_synthetic_unsw, SYNTHETIC_NOTICE
from visualization import (plot_category_detection, plot_confusion_matrix,
                           plot_decision_distribution, plot_metrics_comparison,
                           plot_risk_histogram, plot_threshold_sweep)
from zero_trust import ZeroTrustPolicy, threshold_sweep


def run(source="synthetic", output_dir=None, rf_scores=None,
        n_train=None, n_test=None):
    if source not in ("real", "synthetic"):
        raise ValueError("source must be 'real' or 'synthetic'")
    out = config.OUTPUT_DIR / source if output_dir is None else output_dir

    if source == "real":
        train_df, test_df = data_loader.load_train_test()
    else:
        train_df = make_synthetic_unsw(n_train or config.SYNTHETIC_ROWS_TRAIN, seed=1)
        test_df = make_synthetic_unsw(n_test or config.SYNTHETIC_ROWS_TEST, seed=2)

    y_test = data_loader.get_labels(test_df)
    named = {}
    results = {"data_source": source, "n_train": len(train_df), "n_test": len(test_df)}
    if source != "real":
        results["warning"] = SYNTHETIC_NOTICE

    # 1) Rule-based baseline
    detector = RuleBasedDetector().fit(train_df)
    base_scores = detector.risk_scores(test_df)
    base_pred = detector.predict(test_df)
    results["rule_thresholds"] = detector.thresholds_
    results["baseline"] = compute_metrics(y_test, base_pred)
    named["Rule-based baseline"] = results["baseline"]

    # 2) Zero-trust policy on the baseline risk score
    policy = ZeroTrustPolicy()
    decisions = policy.decide(base_scores)
    results["zero_trust_baseline"] = compute_metrics(y_test, policy.to_binary(decisions))
    results["zero_trust_baseline"]["summary"] = policy.summarize(decisions, y_test)
    named["Zero-trust (rules)"] = results["zero_trust_baseline"]

    # 3) Optional: zero-trust on Person 2's Random Forest probabilities
    if rf_scores is not None:
        rf_scores = np.asarray(rf_scores, dtype=float)
        if rf_scores.shape != y_test.shape:
            raise ValueError("rf_scores must have one value per test row.")
        rf_dec = policy.decide(rf_scores)
        results["zero_trust_rf"] = compute_metrics(y_test, policy.to_binary(rf_dec))
        results["zero_trust_rf"]["summary"] = policy.summarize(rf_dec, y_test)
        named["Zero-trust (RF scores)"] = results["zero_trust_rf"]

    # 4) Extra analysis
    results["per_category"] = per_category_detection(test_df, base_pred)
    sweep = threshold_sweep(y_test, base_scores)
    results["threshold_sweep"] = sweep.to_dict(orient="records")

    # 5) Save results + graphs
    save_results(results, out / "results.json")
    sweep.to_csv(out / "threshold_sweep.csv", index=False)
    fig = out / "figures"
    ds = source
    plot_confusion_matrix(results["baseline"], fig / "confusion_matrix_baseline.png",
                          "Confusion matrix - rule-based baseline", ds)
    plot_confusion_matrix(results["zero_trust_baseline"], fig / "confusion_matrix_zero_trust.png",
                          "Confusion matrix - zero-trust (rules)", ds)
    plot_metrics_comparison(named, fig / "metrics_comparison.png", "Metric comparison", ds)
    plot_decision_distribution(results["zero_trust_baseline"]["summary"],
                               fig / "decision_distribution.png",
                               "Zero-trust decisions (rule risk score)", ds)
    plot_risk_histogram(base_scores, y_test, fig / "risk_histogram.png",
                        "Rule-based risk scores", ds, policy_thresholds(policy))
    plot_threshold_sweep(sweep, fig / "threshold_sweep_f1.png",
                         "F1 vs ALLOW/DENY thresholds", "f1", ds)
    plot_category_detection(results["per_category"], fig / "category_detection.png",
                            "Detection by attack category (baseline)", ds)

    print(f"Data source: {source.upper()}" + ("" if source == "real" else f"  ({SYNTHETIC_NOTICE})"))
    print(format_metrics_table(named))
    print(f"Results and figures saved in: {out}")
    return results


def policy_thresholds(policy):
    return {"allow_below": policy.allow_below, "deny_at_or_above": policy.deny_at_or_above}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Person 3 pipeline")
    parser.add_argument("--source", choices=["real", "synthetic"], default="synthetic")
    parser.add_argument("--rf-scores", help=".npy file with RF attack probabilities for the test set")
    args = parser.parse_args()
    rf = np.load(args.rf_scores) if args.rf_scores else None
    run(args.source, rf_scores=rf)
