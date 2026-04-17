import pandas as pd

from src.evaluation import build_results_summary


def test_build_results_summary_schema():
    best_row = pd.Series({"RMSE (Split)": 0.48, "RMSE (CV)": 0.49, "RMSE_STD": 0.01, "Num Features": 7})
    summary = build_results_summary(
        best_model_name="GradientBoostingRegressor",
        best_feature_set=["query_length", "common_words"],
        best_row=best_row,
        best_params={"n_estimators": 100},
        run_context={"command": "run full-pipeline", "dataset_dir": "data"},
    )
    assert "metrics" in summary
    assert "run_context" in summary
    assert summary["metrics"]["num_features"] == 7
