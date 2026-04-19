"""Shared orchestration functions for main entrypoint and CLI."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from src.data_loader import load_raw_datasets, prepare_raw_data
from src.evaluation import build_results_summary, save_results
from src.feature_engineering import feature_sets, generate_feature_combinations, load_feature_sets_from_yaml
from src.logging_config import configure_logging
from src.modeling import (
    benchmark_new_features,
    compare_models,
    finalist_estimator_cls,
    generate_submission_file,
    pick_finalist_winner,
    plot_overfitting_curve,
    run_baseline_evaluation,
    run_data_exploration,
    run_feature_search,
    run_feature_set_evaluation,
    run_full_feature_importance,
    run_model_tuning_with_ttest,
)


def ensure_output_dir(output_dir):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_run_id(random_seed):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_seed{random_seed}"


_RUN_ID_SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


def resolve_run_id(random_seed: int, explicit: str | None = None) -> str:
    """Use explicit run id if provided (must be filesystem-safe); else timestamp-based id."""
    if explicit is None or explicit == "":
        return _make_run_id(random_seed)
    if not _RUN_ID_SAFE.match(explicit) or ".." in explicit or "/" in explicit or "\\" in explicit:
        raise ValueError(
            f"Invalid --run-id {explicit!r}. Use only letters, digits, dot, underscore, or hyphen "
            "(no path separators)."
        )
    return explicit


def _json_safe(obj):
    return json.loads(json.dumps(obj, default=str))


def run_baseline_stage(data_dir, output_dir="outputs", random_seed=42, stem=True, log_level="INFO", run_id_explicit=None):
    output_path = ensure_output_dir(output_dir)
    run_id = resolve_run_id(random_seed, run_id_explicit)
    run_path = output_path / "runs" / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir=str(output_path), run_id=run_id, level=log_level)
    logger.info(f"Starting baseline run_id={run_id}")

    df_train, df_test, df_attr, df_desc = load_raw_datasets(data_dir)
    raw_data, num_train = prepare_raw_data(df_train, df_test, df_desc)
    run_data_exploration(df_train, df_attr, output_dir=str(run_path))
    run_baseline_evaluation(raw_data, df_attr, num_train)
    (run_path / "config_used.json").write_text(
        json.dumps(
            {
                "command": "run baseline",
                "dataset_dir": data_dir,
                "output_dir": str(run_path),
                "random_seed": random_seed,
                "stem": stem,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(f"Baseline run finished. Outputs written to {run_path}")
    return {"num_train": num_train, "stem": stem, "run_id": run_id, "output_dir": str(run_path)}


def run_compare_models_stage(
    data_dir,
    stem=True,
    include_hist_gradient=True,
    output_dir="outputs",
    random_seed=42,
    log_level="INFO",
    run_id_explicit=None,
):
    run_path = None
    run_id = None
    if output_dir:
        output_path = ensure_output_dir(output_dir)
        run_id = resolve_run_id(random_seed, run_id_explicit)
        run_path = output_path / "runs" / run_id
        run_path.mkdir(parents=True, exist_ok=True)
        configure_logging(output_dir=str(output_path), run_id=run_id, level=log_level)
        logger.info(f"Starting compare-models run_id={run_id}")

    df_train, df_test, df_attr, df_desc = load_raw_datasets(data_dir)
    raw_data, num_train = prepare_raw_data(df_train, df_test, df_desc)
    results_df = compare_models(
        raw_data,
        df_attr,
        ["query_length", "common_words", "tfidf_similarity"],
        num_train,
        stem=stem,
        include_hist_gradient=include_hist_gradient,
    )
    out: dict = {"results_df": results_df, "run_id": None, "output_dir": None}
    if output_dir and run_path is not None:
        results_df.to_csv(run_path / "model_comparison.csv", index=False)
        (run_path / "config_used.json").write_text(
            json.dumps(
                {
                    "command": "run compare-models",
                    "dataset_dir": data_dir,
                    "output_dir": str(run_path),
                    "random_seed": random_seed,
                    "stem": stem,
                    "include_hist_gradient": include_hist_gradient,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(f"Compare-models finished. Outputs written to {run_path}")
        out["run_id"] = run_id
        out["output_dir"] = str(run_path)
    return out


def run_tune_stage(data_dir, random_seed=42, n_iter=10, stem=True, output_dir=None, log_level="INFO", run_id_explicit=None):
    from src.feature_engineering import build_feature_set

    run_path = None
    run_id = None
    if output_dir:
        output_path = ensure_output_dir(output_dir)
        run_id = resolve_run_id(random_seed, run_id_explicit)
        run_path = output_path / "runs" / run_id
        run_path.mkdir(parents=True, exist_ok=True)
        configure_logging(output_dir=str(output_path), run_id=run_id, level=log_level)
        logger.info(f"Starting tune run_id={run_id}")

    df_train, df_test, df_attr, df_desc = load_raw_datasets(data_dir)
    raw_data, num_train = prepare_raw_data(df_train, df_test, df_desc)
    df_features = build_feature_set(raw_data, df_attr, ["query_length", "common_words", "tfidf_similarity"], stem=stem)
    df_train_features = df_features.iloc[:num_train]
    X = df_train_features.drop(columns=["id", "relevance"])
    y = df_train_features["relevance"]
    hist_search, rf_search = run_model_tuning_with_ttest(X, y, n_iter=n_iter, random_seed=random_seed)
    winner_search, selected_finalist = pick_finalist_winner(hist_search, rf_search)
    if output_dir and run_path is not None:
        tune_payload = {
            "hist_gradient_boosting_best_params": _json_safe(hist_search.best_params_),
            "hist_gradient_boosting_best_rmse": float(-hist_search.best_score_),
            "random_forest_best_params": _json_safe(rf_search.best_params_),
            "random_forest_best_rmse": float(-rf_search.best_score_),
            "selected_finalist": selected_finalist,
            "selection_metric": "neg_root_mean_squared_error",
        }
        (run_path / "tune_best_params.json").write_text(json.dumps(tune_payload, indent=2), encoding="utf-8")
        (run_path / "config_used.json").write_text(
            json.dumps(
                {
                    "command": "run tune",
                    "dataset_dir": data_dir,
                    "output_dir": str(run_path),
                    "random_seed": random_seed,
                    "n_iter": n_iter,
                    "stem": stem,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(f"Tune finished. Outputs written to {run_path}")
        return hist_search, rf_search, winner_search, selected_finalist, {"run_id": run_id, "output_dir": str(run_path)}
    return hist_search, rf_search, winner_search, selected_finalist, {}


def run_feature_search_stage(
    data_dir,
    feature_mode="presets",
    feature_config_path=None,
    output_dir="outputs",
    random_seed=42,
    stem=True,
    log_level="INFO",
    run_id_explicit=None,
):
    output_path_base = ensure_output_dir(output_dir)
    run_id = resolve_run_id(random_seed, run_id_explicit)
    run_path = output_path_base / "runs" / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir=str(output_path_base), run_id=run_id, level=log_level)
    logger.info(f"Starting feature-search run_id={run_id}")

    df_train, df_test, df_attr, df_desc = load_raw_datasets(data_dir)
    raw_data, num_train = prepare_raw_data(df_train, df_test, df_desc)
    _, _, winner_search, winner_name, _ = run_tune_stage(data_dir=data_dir, random_seed=random_seed, n_iter=10, stem=stem, output_dir=None)
    best_params = winner_search.best_params_.copy()
    best_params["random_state"] = random_seed
    estimator_cls = finalist_estimator_cls(winner_name)

    if feature_mode == "yaml":
        feature_candidates = load_feature_sets_from_yaml(feature_config_path or "configs/features.yaml")
        search_mode = "presets"
    elif feature_mode == "auto":
        feature_candidates = generate_feature_combinations(min_size=1, max_size=4)
        search_mode = "random"
    else:
        feature_candidates = feature_sets
        search_mode = "presets"

    output_csv = run_path / "feature_search_results.csv"
    results_df = run_feature_search(
        raw_data,
        df_attr,
        num_train,
        feature_candidates,
        model_params=best_params,
        estimator_cls=estimator_cls,
        search_mode=search_mode,
        sample_size=30,
        random_seed=random_seed,
        append_path=str(output_csv),
    )
    results_df.to_csv(output_csv, index=False)
    (run_path / "config_used.json").write_text(
        json.dumps(
            {
                "command": "run feature-search",
                "dataset_dir": data_dir,
                "output_dir": str(run_path),
                "random_seed": random_seed,
                "stem": stem,
                "feature_mode": feature_mode,
                "feature_config_path": feature_config_path,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(f"Feature search finished. Outputs written to {run_path}")
    return {"results_df": results_df, "run_id": run_id, "output_dir": str(run_path), "csv_path": str(output_csv)}


def run_full_pipeline(
    data_dir="home-depot-product-search-relevance",
    output_dir="outputs",
    random_seed=42,
    stem=True,
    log_level="INFO",
    run_id_explicit=None,
    run_model_comparison=False,
):
    output_path = ensure_output_dir(output_dir)
    run_id = resolve_run_id(random_seed, run_id_explicit)
    run_path = output_path / "runs" / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir=str(output_path), run_id=run_id, level=log_level)
    logger.info(f"Starting full pipeline run_id={run_id}")

    df_train, df_test, df_attr, df_desc = load_raw_datasets(data_dir)
    raw_data, num_train = prepare_raw_data(df_train, df_test, df_desc)

    run_data_exploration(df_train, df_attr, output_dir=str(run_path))
    run_baseline_evaluation(raw_data, df_attr, num_train)
    if run_model_comparison:
        compare_models(
            raw_data,
            df_attr,
            ["query_length", "common_words", "tfidf_similarity"],
            num_train,
            stem=stem,
            include_hist_gradient=True,
        )

    _, _, winner_search, winner_name, _ = run_tune_stage(data_dir=data_dir, random_seed=random_seed, stem=stem)
    best_params = winner_search.best_params_.copy()
    best_params["random_state"] = random_seed
    estimator_cls = finalist_estimator_cls(winner_name)

    feature_eval_csv = run_path / "feature_set_evaluation_results.csv"
    results_df = run_feature_set_evaluation(
        raw_data, df_attr, num_train, feature_sets, model_params=best_params, estimator_cls=estimator_cls
    )
    results_df.to_csv(feature_eval_csv, index=False)

    plot_overfitting_curve(results_df, save_path=str(run_path / "feature_count_vs_rmse.png"))

    best_feature_set = ["query_length", "initial_term_match", "jaccard", "common_words", "color_match", "fuzzy", "bigram_overlap"]
    row_match = results_df[results_df["Features"] == ", ".join(best_feature_set)]
    if row_match.empty:
        raise ValueError("Specified best feature set not found in results_df.")
    best_row = row_match.iloc[0]

    run_full_feature_importance(
        raw_data,
        df_attr,
        num_train,
        best_params,
        save_path=str(run_path / "feature_importance_barplot.png"),
        estimator_cls=estimator_cls,
    )
    generate_submission_file(
        raw_data,
        df_attr,
        num_train,
        best_feature_set,
        best_params,
        output_path=str(run_path / "submission.csv"),
        estimator_cls=estimator_cls,
    )

    benchmark_df = benchmark_new_features(raw_data, df_attr, num_train, model_params=best_params, estimator_cls=estimator_cls)
    benchmark_df.to_csv(run_path / "new_feature_benchmarks.csv", index=False)

    summary = build_results_summary(
        best_model_name=winner_name,
        best_feature_set=best_feature_set,
        best_row=best_row,
        best_params=best_params,
        feature_results_path=str(feature_eval_csv),
        run_id=run_id,
        run_context={
            "command": "run full-pipeline",
            "dataset_dir": data_dir,
            "output_dir": str(run_path),
            "random_seed": random_seed,
            "stem": stem,
        },
    )
    (run_path / "config_used.json").write_text(
        json.dumps(
            {
                "command": "run full-pipeline",
                "dataset_dir": data_dir,
                "output_dir": str(run_path),
                "random_seed": random_seed,
                "stem": stem,
                "run_model_comparison": run_model_comparison,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_results(
        summary,
        output_path=str(run_path / "results_summary.json"),
        schema_path="schemas/results_summary.schema.json",
    )
    logger.info(f"Run finished. Results written to {run_path}")
    return summary

