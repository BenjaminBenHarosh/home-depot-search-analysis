import pandas as pd
import pytest

from src.evaluation import build_results_summary, validate_results_summary


def _make_summary():
    best_row = pd.Series({"RMSE (Split)": 0.48, "RMSE (CV)": 0.49, "RMSE_STD": 0.01, "Num Features": 7})
    return build_results_summary(
        best_model_name="GradientBoostingRegressor",
        best_feature_set=["query_length", "common_words"],
        best_row=best_row,
        best_params={"n_estimators": 100},
        run_id="test_run",
        run_context={
            "command": "run full-pipeline",
            "dataset_dir": "data",
            "output_dir": "outputs/runs/test_run",
            "random_seed": 42,
            "stem": True,
        },
    )


def test_valid_summary_passes_schema():
    summary = _make_summary()
    validate_results_summary(summary, schema_path="schemas/results_summary.schema.json")


def test_missing_required_field_fails_schema():
    summary = _make_summary()
    del summary["run_id"]
    with pytest.raises(Exception):
        validate_results_summary(summary, schema_path="schemas/results_summary.schema.json")

