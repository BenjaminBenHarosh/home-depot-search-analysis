"""Lightweight pipeline stage tests with mocks (raises coverage for CLI-oriented paths)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src import pipeline as pl


class _FakeSearch:
    def __init__(self):
        self.best_params_ = {"n_estimators": 10, "max_depth": 5}
        self.best_score_ = -0.42
        self.cv_results_ = {
            "mean_test_score": [-0.4] * 5,
            **{f"split{i}_test_score": [-0.4] for i in range(5)},
        }


@pytest.fixture
def minimal_tables():
    df_train = pd.DataFrame({"id": [1], "product_uid": [1], "product_title": ["t"], "search_term": ["s"], "relevance": [2.0]})
    df_test = pd.DataFrame({"id": [2], "product_uid": [1], "product_title": ["t"], "search_term": ["s"]})
    df_attr = pd.DataFrame({"product_uid": [1], "name": ["x"], "value": ["y"]})
    df_desc = pd.DataFrame({"product_uid": [1], "product_description": ["d"]})
    return df_train, df_test, df_attr, df_desc


def test_run_compare_models_stage_writes_csv(tmp_path, monkeypatch, minimal_tables):
    df_train, df_test, df_attr, df_desc = minimal_tables

    monkeypatch.setattr(pl, "load_raw_datasets", lambda *_a, **_k: (df_train, df_test, df_attr, df_desc))
    monkeypatch.setattr(
        pl,
        "prepare_raw_data",
        lambda *_a, **_k: (pd.concat([df_train, df_test], ignore_index=True), 1),
    )
    monkeypatch.setattr(
        pl,
        "compare_models",
        lambda *_a, **_k: pd.DataFrame([{"Model": "M", "RMSE": 0.1, "Train time (s)": "1"}]),
    )

    out = pl.run_compare_models_stage(
        data_dir="ignored",
        output_dir=str(tmp_path),
        random_seed=1,
        log_level="INFO",
        run_id_explicit="unit_compare",
    )
    assert out["run_id"] == "unit_compare"
    csv_path = Path(out["output_dir"]) / "model_comparison.csv"
    assert csv_path.is_file()
    cfg = Path(out["output_dir"]) / "config_used.json"
    assert json.loads(cfg.read_text(encoding="utf-8"))["command"] == "run compare-models"


def test_run_tune_stage_writes_json(tmp_path, monkeypatch, minimal_tables):
    df_train, df_test, df_attr, df_desc = minimal_tables

    monkeypatch.setattr(pl, "load_raw_datasets", lambda *_a, **_k: (df_train, df_test, df_attr, df_desc))
    raw = pd.concat([df_train, df_test], ignore_index=True)
    monkeypatch.setattr(pl, "prepare_raw_data", lambda *_a, **_k: (raw, 1))

    feat = pd.DataFrame(
        {
            "id": [1],
            "relevance": [2.0],
            "query_length": [1],
            "common_words": [0],
            "tfidf_similarity": [0.0],
        }
    )

    def fake_build_feature_set(*_a, **_k):
        return feat

    monkeypatch.setattr("src.feature_engineering.build_feature_set", fake_build_feature_set)
    h, r = _FakeSearch(), _FakeSearch()
    r.best_score_ = -0.5
    monkeypatch.setattr(pl, "run_model_tuning_with_ttest", lambda *_a, **_k: (h, r))

    h_out, r_out, winner, name, meta = pl.run_tune_stage(
        data_dir="ignored",
        random_seed=1,
        n_iter=1,
        stem=True,
        output_dir=str(tmp_path),
        log_level="INFO",
        run_id_explicit="unit_tune",
    )
    assert h_out is h
    assert r_out is r
    assert winner is h
    assert name == "HistGradientBoostingRegressor"
    assert meta["run_id"] == "unit_tune"
    tune_json = Path(meta["output_dir"]) / "tune_best_params.json"
    payload = json.loads(tune_json.read_text(encoding="utf-8"))
    assert "hist_gradient_boosting_best_params" in payload
    assert "random_forest_best_params" in payload
    assert payload.get("selected_finalist") == "HistGradientBoostingRegressor"


def test_run_feature_search_stage_creates_run_folder(tmp_path, monkeypatch, minimal_tables):
    df_train, df_test, df_attr, df_desc = minimal_tables

    monkeypatch.setattr(pl, "load_raw_datasets", lambda *_a, **_k: (df_train, df_test, df_attr, df_desc))
    raw = pd.concat([df_train, df_test], ignore_index=True)
    monkeypatch.setattr(pl, "prepare_raw_data", lambda *_a, **_k: (raw, 1))

    feat = pd.DataFrame(
        {
            "id": [1],
            "relevance": [2.0],
            "query_length": [1],
            "common_words": [0],
            "tfidf_similarity": [0.0],
        }
    )

    def fake_build_feature_set(*_a, **_k):
        return feat

    monkeypatch.setattr("src.feature_engineering.build_feature_set", fake_build_feature_set)
    h, r = _FakeSearch(), _FakeSearch()
    r.best_score_ = -0.5
    monkeypatch.setattr(pl, "run_tune_stage", lambda **_k: (h, r, h, "HistGradientBoostingRegressor", {}))

    empty_fs = pd.DataFrame(columns=["Features", "Mean RMSE", "Std RMSE"])
    monkeypatch.setattr(pl, "run_feature_search", lambda *_a, **_k: empty_fs)

    out = pl.run_feature_search_stage(
        data_dir="ignored",
        feature_mode="presets",
        output_dir=str(tmp_path),
        random_seed=3,
        stem=True,
        log_level="INFO",
        run_id_explicit="unit_fs",
    )
    assert out["run_id"] == "unit_fs"
    assert Path(out["csv_path"]).name == "feature_search_results.csv"
