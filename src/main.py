
"""Main pipeline for AI-enabled Zero-Trust anomaly detection."""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config
import data_loader
from baseline import RuleBasedDetector
from evaluation import compute_metrics, save_results, format_metrics_table
from preprocess import load_unsw
from train_model import (
    train_random_forest,
    train_logistic_regression,
    timed_predict,
)
from zero_trust import ZeroTrustPolicy
from conditions import create_scenarios, apply_scenario_to_raw_data
from visualization import (
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_decision_distribution,
)


def main():
    print("=" * 65)
    print("ZERO-TRUST NETWORK ANOMALY DETECTION")
    print("=" * 65)

    # Load preprocessed data for machine-learning models.
    X_train, X_test, y_train, y_test, info = load_unsw()

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows:  {len(X_test)}")
    print(f"Features:      {X_train.shape[1]}")

    trained_models = {}
    named_metrics = {}

    results = {
        "data_source": "real",
        "dataset": "UNSW-NB15",
        "preprocessing_report": info["report"],
        "models": {},
    }

    # Train and evaluate machine-learning models.
    models = {
        "Random Forest": train_random_forest,
        "Logistic Regression": train_logistic_regression,
    }

    for name, trainer in models.items():
        print(f"\nTraining {name}...")
        model, train_seconds = trainer(X_train, y_train)
        trained_models[name] = model

        predictions, predict_seconds = timed_predict(model, X_test)
        metrics = compute_metrics(y_test, predictions)
        metrics["training_seconds"] = float(train_seconds)
        metrics["prediction_seconds"] = float(predict_seconds)

        results["models"][name] = metrics
        named_metrics[name] = metrics

    # Evaluate the rule-based baseline on the original raw test data.
    train_df, test_df = data_loader.load_train_test()
    y_raw = data_loader.get_labels(test_df)

    detector = RuleBasedDetector().fit(train_df)
    baseline_predictions = detector.predict(test_df)
    baseline_scores = detector.risk_scores(test_df)

    baseline_metrics = compute_metrics(y_raw, baseline_predictions)
    results["baseline"] = baseline_metrics
    named_metrics["Rule-based baseline"] = baseline_metrics

    # Evaluate Zero Trust on the original raw test data.
    policy = ZeroTrustPolicy()
    decisions = policy.decide(baseline_scores)
    decision_predictions = policy.to_binary(decisions)

    zero_trust_metrics = compute_metrics(y_raw, decision_predictions)
    zero_trust_metrics["summary"] = policy.summarize(decisions, y_raw)

    results["zero_trust"] = zero_trust_metrics
    named_metrics["Zero-trust (rules)"] = zero_trust_metrics

    # Create the five controlled evaluation scenarios.
    print("\n" + "=" * 65)
    print("EVALUATING CONTROLLED TEST CONDITIONS")
    print("=" * 65)

    scenarios = create_scenarios(
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        random_state=config.RANDOM_SEED,
    )

    results["scenarios"] = {}
    scenario_metrics = {}

    # Load raw data once for the rule-based detector.
    raw_train_df, raw_test_df = data_loader.load_train_test()

    for scenario_name, scenario_data in scenarios.items():
        print(f"\nEvaluating scenario: {scenario_name}")

        X_scenario = scenario_data["X"]
        y_scenario = scenario_data["y"].to_numpy()

        results["scenarios"][scenario_name] = {}
        scenario_metrics[scenario_name] = {}

        # Evaluate both already-trained machine-learning models.
        for model_name, model in trained_models.items():
            predictions, _ = timed_predict(model, X_scenario)
            metrics = compute_metrics(y_scenario, predictions)

            results["scenarios"][scenario_name][model_name] = metrics
            scenario_metrics[scenario_name][model_name] = metrics

        # Select matching raw test rows for the current scenario.
        raw_scenario_df = raw_test_df.loc[X_scenario.index].copy()
        raw_scenario_df = apply_scenario_to_raw_data(
            raw_scenario_df,
            scenario_name,
            random_state=config.RANDOM_SEED,
        )

        # Evaluate the rule-based baseline.
        scenario_detector = RuleBasedDetector().fit(raw_train_df)
        baseline_pred = scenario_detector.predict(raw_scenario_df)
        baseline_scores = scenario_detector.risk_scores(raw_scenario_df)

        baseline_result = compute_metrics(y_scenario, baseline_pred)
        results["scenarios"][scenario_name][
            "Rule-based baseline"
        ] = baseline_result
        scenario_metrics[scenario_name][
            "Rule-based baseline"
        ] = baseline_result

        # Evaluate Zero Trust using the rule-based risk scores.
        scenario_policy = ZeroTrustPolicy()
        scenario_decisions = scenario_policy.decide(baseline_scores)
        zero_trust_pred = scenario_policy.to_binary(scenario_decisions)

        zero_trust_result = compute_metrics(y_scenario, zero_trust_pred)
        results["scenarios"][scenario_name][
            "Zero-trust (rules)"
        ] = zero_trust_result
        scenario_metrics[scenario_name][
            "Zero-trust (rules)"
        ] = zero_trust_result

        # Print all four models' metrics for this scenario.
        for model_name, metrics in scenario_metrics[scenario_name].items():
            print(
                f"  {model_name:<24} "
                f"Accuracy={metrics['accuracy']:.4f}  "
                f"Precision={metrics['precision']:.4f}  "
                f"Recall={metrics['recall']:.4f}  "
                f"F1={metrics['f1']:.4f}"
            )

    # Save results and figures.
    output_dir = config.OUTPUT_DIR / "real"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    save_results(results, output_dir / "results.json")

    plot_confusion_matrix(
        baseline_metrics,
        figures_dir / "confusion_matrix_baseline.png",
        "Rule-based baseline confusion matrix",
        "real",
    )

    plot_confusion_matrix(
        zero_trust_metrics,
        figures_dir / "confusion_matrix_zero_trust.png",
        "Zero-trust confusion matrix",
        "real",
    )

    plot_metrics_comparison(
        named_metrics,
        figures_dir / "metrics_comparison.png",
        "Model performance comparison",
        "real",
    )

    plot_decision_distribution(
        zero_trust_metrics["summary"],
        figures_dir / "decision_distribution.png",
        "Zero-trust decisions",
        "real",
    )

    print("\n" + format_metrics_table(named_metrics))
    print(f"\nResults saved to: {output_dir.resolve()}")
    print("\nMAIN PIPELINE COMPLETED")


if __name__ == "__main__":
    main()
