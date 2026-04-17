"""Main orchestration entrypoint."""

import warnings

from src.data_loader import load_raw_datasets, prepare_raw_data
from src.evaluation import build_results_summary, save_results
from src.feature_engineering import build_feature_set, feature_sets
from src.modeling import (
    compare_models,
    generate_submission_file,
    plot_overfitting_curve,
    run_baseline_evaluation,
    run_data_exploration,
    run_feature_set_evaluation,
    run_full_feature_importance,
    run_model_tuning_with_ttest,
)


def main():
    """Run the full training and evaluation pipeline."""
    warnings.filterwarnings("ignore")

    df_train, df_test, df_attr, df_pro_desc = load_raw_datasets()
    raw_data, num_train = prepare_raw_data(df_train, df_test, df_pro_desc)

    run_data_exploration(df_train, df_attr)
    run_baseline_evaluation(raw_data, df_attr, num_train)

    compare_models(raw_data, df_attr, ["query_length", "common_words", "tfidf_similarity"], num_train, stem=True)

    df_features = build_feature_set(raw_data, df_attr, ["query_length", "common_words", "tfidf_similarity"], stem=True)
    df_train_features = df_features.iloc[:num_train]
    X = df_train_features.drop(columns=["id", "relevance"])
    y = df_train_features["relevance"]

    _, gb_search = run_model_tuning_with_ttest(X, y)
    best_params = gb_search.best_params_
    best_params["random_state"] = 42

    results_df = run_feature_set_evaluation(raw_data, df_attr, num_train, feature_sets, model_params=best_params)
    results_df.to_csv("feature_set_evaluation_results.csv", index=False)
    plot_overfitting_curve(results_df)

    best_feature_set = ["query_length", "initial_term_match", "jaccard", "common_words", "color_match", "fuzzy", "bigram_overlap"]
    row_match = results_df[results_df["Features"] == ", ".join(best_feature_set)]
    if row_match.empty:
        raise ValueError("Specified best feature set not found in results_df.")

    best_row = row_match.iloc[0]
    print(f"\nBest Feature Set: {best_feature_set}")
    print(f"   CV RMSE: {best_row['RMSE (CV)']:.4f} ± {best_row['RMSE_STD']:.4f}")

    run_full_feature_importance(raw_data, df_attr, num_train, best_params)
    generate_submission_file(raw_data, df_attr, num_train, best_feature_set, best_params)

    summary = build_results_summary(
        best_model_name="GradientBoostingRegressor",
        best_feature_set=best_feature_set,
        best_row=best_row,
        best_params=best_params,
    )
    save_results(summary, output_path="results.json")

    print("\nSaved reproducible run summary to results.json")
    print(f"Final model: GradientBoostingRegressor | Random seed: {best_params['random_state']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"An error occurred: {error}")