"""Result aggregation and serialization helpers."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import validate


def build_results_summary(
    best_model_name,
    best_feature_set,
    best_row,
    best_params,
    feature_results_path="feature_set_evaluation_results.csv",
    run_context=None,
    run_id=None,
    schema_version="1.0.0",
):
    """Create a stable JSON-serializable summary payload."""
    context = run_context or {}
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_sha = "unknown"

    return {
        "schema_version": schema_version,
        "run_id": run_id,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
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
        "run_context": {
            "command": context.get("command"),
            "dataset_dir": context.get("dataset_dir"),
            "output_dir": context.get("output_dir"),
            "random_seed": context.get("random_seed"),
            "stem": context.get("stem"),
        },
    }


def validate_results_summary(results_dict, schema_path="schemas/results_summary.schema.json"):
    """Validate results summary against JSON schema."""
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    validate(instance=results_dict, schema=schema)


def save_results(results_dict, output_path="results.json", schema_path="schemas/results_summary.schema.json"):
    """Save run summary to JSON."""
    validate_results_summary(results_dict, schema_path=schema_path)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results_dict, handle, indent=2)

