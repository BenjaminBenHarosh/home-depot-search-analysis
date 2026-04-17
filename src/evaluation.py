"""Result aggregation and serialization helpers."""

import json
from datetime import datetime, timezone


def build_results_summary(
    best_model_name,
    best_feature_set,
    best_row,
    best_params,
    feature_results_path="feature_set_evaluation_results.csv",
):
    """Create a stable JSON-serializable summary payload."""
    return {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "best_model": best_model_name,
        "best_feature_set": best_feature_set,
        "metrics": {
            "rmse_split": float(best_row["RMSE (Split)"]),
            "rmse_cv_mean": float(best_row["RMSE (CV)"]),
            "rmse_cv_std": float(best_row["RMSE_STD"]),
            "num_features": int(best_row["Num Features"]),
        },
        "model_params": best_params,
        "artifacts": {
            "feature_evaluation_csv": feature_results_path,
            "submission_csv": "submission.csv",
            "feature_importance_plot": "feature_importance_barplot.png",
            "complexity_plot": "feature_count_vs_rmse.png",
        },
    }


def save_results(results_dict, output_path="results.json"):
    """Save run summary to JSON."""
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results_dict, handle, indent=2)

