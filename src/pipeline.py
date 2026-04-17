"""Shared orchestration functions for main entrypoint and CLI."""

from __future__ import annotations

from pathlib import Path

from src.data_loader import load_raw_datasets, prepare_raw_data
from src.evaluation import build_results_summary, save_results
from src.feature_engineering import feature_sets, generate_feature_combinations, load_feature_sets_from_yaml
from src.modeling import (
    benchmark_new_features,
    compare_models,
    generate_submission_file,
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


def run_baseline_stage(data_dir, stem=True):
    df_train, df_test, df_attr, df_desc = load_raw_datasets(data_dir)
    raw_data, num_train = prepare_raw_data(df_train, df_test, df_desc)
    run_data_exploration(df_train, df_attr)
    run_baseline_evaluation(raw_data, df_attr, num_train)
    return {"num_train": num_train, "stem": stem}


def run_compare_models_stage(data_dir, stem=True, include_hist_gradient=True):
    df_train, df_test, df_attr, df_desc = load_raw_datasets(data_dir)
    raw_data, num_train = prepare_raw_data(df_train, df_test, df_desc)
    return compare_models(
        raw_data,
        df_attr,
        ["query_length", "common_words", "tfidf_similarity"],
        num_train,
        stem=stem,
        include_hist_gradient=include_hist_gradient,
    )


def run_tune_stage(data_dir, random_seed=42, n_iter=10, stem=True):
    from src.feature_engineering import build_feature_set

    df_train, df_test, df_attr, df_desc = load_raw_datasets(data_dir)
    raw_data, num_train = prepare_raw_data(df_train, df_test, df_desc)
    df_features = build_feature_set(raw_data, df_attr, ["query_length", "common_words", "tfidf_similarity"], stem=stem)
    df_train_features = df_features.iloc[:num_train]
    X = df_train_features.drop(columns=["id", "relevance"])
    y = df_train_features["relevance"]
    return run_model_tuning_with_ttest(X, y, n_iter=n_iter, random_seed=random_seed)


def run_feature_search_stage(data_dir, feature_mode="presets", feature_config_path=None, output_path=None, random_seed=42, stem=True):
    df_train, df_test, df_attr, df_desc = load_raw_datasets(data_dir)
    raw_data, num_train = prepare_raw_data(df_train, df_test, df_desc)
    _, gb_search = run_tune_stage(data_dir=data_dir, random_seed=random_seed, n_iter=10, stem=stem)
    best_params = gb_search.best_params_
    best_params["random_state"] = random_seed

    if feature_mode == "yaml":
        feature_candidates = load_feature_sets_from_yaml(feature_config_path or "configs/features.yaml")
        search_mode = "presets"
    elif feature_mode == "auto":
        feature_candidates = generate_feature_combinations(min_size=1, max_size=4)
        search_mode = "random"
    else:
        feature_candidates = feature_sets
        search_mode = "presets"

    results_df = run_feature_search(
        raw_data,
        df_attr,
        num_train,
        feature_candidates,
        model_params=best_params,
        search_mode=search_mode,
        sample_size=30,
        random_seed=random_seed,
        append_path=output_path,
    )
    if output_path:
        results_df.to_csv(output_path, index=False)
    return results_df


def run_full_pipeline(data_dir="home-depot-product-search-relevance", output_dir="outputs", random_seed=42, stem=True):
    output_path = ensure_output_dir(output_dir)

    df_train, df_test, df_attr, df_desc = load_raw_datasets(data_dir)
    raw_data, num_train = prepare_raw_data(df_train, df_test, df_desc)

    run_data_exploration(df_train, df_attr)
    run_baseline_evaluation(raw_data, df_attr, num_train)
    compare_models(raw_data, df_attr, ["query_length", "common_words", "tfidf_similarity"], num_train, stem=stem, include_hist_gradient=True)

    _, gb_search = run_tune_stage(data_dir=data_dir, random_seed=random_seed, stem=stem)
    best_params = gb_search.best_params_
    best_params["random_state"] = random_seed

    feature_eval_csv = output_path / "feature_set_evaluation_results.csv"
    results_df = run_feature_set_evaluation(raw_data, df_attr, num_train, feature_sets, model_params=best_params)
    results_df.to_csv(feature_eval_csv, index=False)

    plot_overfitting_curve(results_df, save_path=str(output_path / "feature_count_vs_rmse.png"))

    best_feature_set = ["query_length", "initial_term_match", "jaccard", "common_words", "color_match", "fuzzy", "bigram_overlap"]
    row_match = results_df[results_df["Features"] == ", ".join(best_feature_set)]
    if row_match.empty:
        raise ValueError("Specified best feature set not found in results_df.")
    best_row = row_match.iloc[0]

    run_full_feature_importance(raw_data, df_attr, num_train, best_params)
    generate_submission_file(raw_data, df_attr, num_train, best_feature_set, best_params, output_path=str(output_path / "submission.csv"))

    benchmark_df = benchmark_new_features(raw_data, df_attr, num_train, model_params=best_params)
    benchmark_df.to_csv(output_path / "new_feature_benchmarks.csv", index=False)

    summary = build_results_summary(
        best_model_name="GradientBoostingRegressor",
        best_feature_set=best_feature_set,
        best_row=best_row,
        best_params=best_params,
        feature_results_path=str(feature_eval_csv),
        run_context={
            "command": "run full-pipeline",
            "dataset_dir": data_dir,
            "output_dir": str(output_path),
            "random_seed": random_seed,
            "stem": stem,
        },
    )
    save_results(summary, output_path=str(output_path / "results.json"))
    return summary

