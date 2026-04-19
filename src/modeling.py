"""Model training, evaluation, plotting, and experiment utilities."""

import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from loguru import logger
from scipy.stats import randint, ttest_rel, uniform
from sklearn.base import clone
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tqdm import tqdm

from src.feature_engineering import all_features, build_feature_set

FINALIST_HISTGB = "HistGradientBoostingRegressor"
FINALIST_RANDOM_FOREST = "RandomForestRegressor"


def pick_finalist_winner(hist_search, rf_search):
    """Pick the finalist with better CV score (maximized neg RMSE)."""
    if hist_search.best_score_ >= rf_search.best_score_:
        return hist_search, FINALIST_HISTGB
    return rf_search, FINALIST_RANDOM_FOREST


def finalist_estimator_cls(name: str):
    if name == FINALIST_HISTGB:
        return HistGradientBoostingRegressor
    if name == FINALIST_RANDOM_FOREST:
        return RandomForestRegressor
    raise ValueError(f"Unknown finalist model name: {name!r}")


def evaluate_model(model, X_train, y_train, name="Unnamed Model"):
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv_scores = cross_val_score(model, X_train, y_train, scoring="neg_root_mean_squared_error", cv=5)
    cv_rmse_mean = -np.mean(cv_scores)
    cv_rmse_std = np.std(cv_scores)
    logger.info(f"{name}: RMSE = {rmse:.4f}, CV RMSE = {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}, Train time = {train_time:.2f} sec")
    return name, rmse, train_time


def compare_models(df_all, df_attr, feature_set, num_train, stem=True, include_hist_gradient=False):
    logger.info("=== Model Comparison ===")
    df_features = build_feature_set(df_all.copy(), df_attr, feature_set, stem=stem)
    df_train = df_features.iloc[:num_train]
    X = df_train.drop(columns=["id", "relevance"])
    y = df_train["relevance"]

    models = [
        ("Random Forest", RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=15, max_depth=6, random_state=0)),
        ("Support Vector Regressor", make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0, epsilon=0.2))),
        ("KNN", KNeighborsRegressor(n_neighbors=5)),
    ]
    if include_hist_gradient:
        models.append(("HistGradientBoosting", HistGradientBoostingRegressor(max_depth=6, random_state=0)))

    results = []
    for name, model in tqdm(models, desc="Comparing models", unit="model"):
        model_name, rmse, train_time = evaluate_model(model, X, y, name)
        results.append((model_name, rmse, f"{train_time:.2f} sec"))

    results_df = pd.DataFrame(results, columns=["Model", "RMSE", "Train time (s)"])
    results_df.sort_values("RMSE", inplace=True)
    logger.info("Model Comparison Results:\n{}", results_df.to_string(index=False))
    return results_df


def _fold_neg_rmse_for_best(search, n_splits=5):
    idx = search.best_index_
    return np.array([-search.cv_results_[f"split{j}_test_score"][idx] for j in range(n_splits)])


def run_model_tuning_with_ttest(X, y, n_iter=10, random_seed=42):
    hist_param_dist = {
        "max_iter": randint(50, 200),
        "max_depth": randint(3, 15),
        "learning_rate": uniform(0.01, 0.3),
        "min_samples_leaf": randint(1, 20),
        "l2_regularization": uniform(0.0, 1.0),
    }
    rf_param_dist = {
        "n_estimators": randint(50, 200),
        "max_depth": randint(4, 15),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 10),
    }

    logger.info("--- Tuning HistGradientBoosting ---")
    hist_search = RandomizedSearchCV(
        HistGradientBoostingRegressor(random_state=0),
        hist_param_dist,
        n_iter=n_iter,
        cv=5,
        scoring="neg_root_mean_squared_error",
        random_state=random_seed,
        n_jobs=-1,
        verbose=0,
    )
    start_hist = time.time()
    hist_search.fit(X, y)
    hist_time = time.time() - start_hist
    hist_rmse_folds = _fold_neg_rmse_for_best(hist_search)
    hist_best_rmse = float(np.mean(hist_rmse_folds))
    hist_std = float(np.std(hist_rmse_folds))
    logger.info(f"HistGradientBoosting RMSE: {hist_best_rmse:.4f}")
    logger.info(f"Best HistGradientBoosting CV RMSE: {hist_best_rmse:.4f} ± {hist_std:.4f}")
    logger.info(f"Training time: {hist_time:.2f} seconds")
    logger.info("Best HistGradientBoosting Parameters:")
    for k, v in hist_search.best_params_.items():
        logger.info(f"  {k}: {v}")

    logger.info("--- Tuning Random Forest ---")
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=0, n_jobs=-1),
        rf_param_dist,
        n_iter=n_iter,
        cv=5,
        scoring="neg_root_mean_squared_error",
        random_state=random_seed,
        n_jobs=-1,
        verbose=0,
    )
    start_rf = time.time()
    rf_search.fit(X, y)
    rf_time = time.time() - start_rf
    rf_rmse_folds = _fold_neg_rmse_for_best(rf_search)
    rf_best_rmse = float(np.mean(rf_rmse_folds))
    rf_std = float(np.std(rf_rmse_folds))
    logger.info(f"Random Forest RMSE: {rf_best_rmse:.4f}")
    logger.info(f"Best Random Forest CV RMSE: {rf_best_rmse:.4f} ± {rf_std:.4f}")
    logger.info(f"Training time: {rf_time:.2f} seconds")
    logger.info("Best Random Forest Parameters:")
    for k, v in rf_search.best_params_.items():
        logger.info(f"  {k}: {v}")

    logger.info("Paired t-test result (best finalist, per-fold RMSE):")
    t_stat, p_val = ttest_rel(hist_rmse_folds, rf_rmse_folds)
    logger.info(f"t-stat = {t_stat:.4f}, p-value = {p_val:.4f}")
    if p_val < 0.05:
        logger.info("The difference is statistically significant.")
    else:
        logger.info("The difference is not statistically significant.")

    return hist_search, rf_search


def run_baseline_evaluation(raw_data, df_attr, num_train):
    logger.info("=== Baseline Evaluation ===")
    baseline_features = ["query_length", "common_words"]
    baseline_model = BaggingRegressor(
        estimator=HistGradientBoostingRegressor(max_depth=6, max_iter=100, random_state=42),
        n_estimators=45,
        random_state=42,
        n_jobs=-1,
        max_samples=0.1,
    )

    for stem_flag in tqdm([True, False], desc="Baseline stem variants", unit="variant"):
        label = "Stemmed" if stem_flag else "No Stem"
        logger.info(f"[Baseline Model - {label}]")
        df_baseline = build_feature_set(raw_data.copy(), df_attr, baseline_features, stem=stem_flag)
        df_train_baseline = df_baseline.iloc[:num_train]
        X = df_train_baseline.drop(columns=["id", "relevance"])
        y = df_train_baseline["relevance"]
        evaluate_model(baseline_model, X, y, name=f"Baseline ({label})")


def run_feature_set_evaluation(raw_data, df_attr, num_train, feature_sets, model_params=None, estimator_cls=GradientBoostingRegressor):
    logger.info("=== Feature Evaluation ===")
    results = []

    for stem_flag in tqdm([True, False], desc="Feature evaluation stem variants", unit="variant"):
        logger.info(f"--- {'With' if stem_flag else 'Without'} Stemming ---")
        for features in tqdm(feature_sets, desc="Feature sets", unit="set", leave=False):
            start_time = time.time()
            df_features = build_feature_set(raw_data.copy(), df_attr, features, stem=stem_flag)
            df_train = df_features.iloc[:num_train]
            X = df_train.drop(columns=["id", "relevance"])
            y = df_train["relevance"]

            if model_params is None:
                if estimator_cls is HistGradientBoostingRegressor:
                    model_params = {"max_iter": 100, "max_depth": 7, "random_state": 42}
                elif estimator_cls is RandomForestRegressor:
                    model_params = {"n_estimators": 100, "max_depth": 7, "random_state": 42, "n_jobs": -1}
                else:
                    model_params = {"n_estimators": 100, "max_depth": 7, "random_state": 42}

            base_model = estimator_cls(**model_params)
            model = BaggingRegressor(estimator=base_model, n_estimators=45, random_state=42, n_jobs=-1, max_samples=0.1)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse_split = np.sqrt(mean_squared_error(y_test, y_pred))

            cv_model = clone(model)
            cv_scores = cross_val_score(cv_model, X, y, scoring="neg_root_mean_squared_error", cv=5)
            rmse_cv = -np.mean(cv_scores)
            rmse_std = np.std(cv_scores)
            elapsed_time = time.time() - start_time

            results.append(
                {
                    "Features": ", ".join(features),
                    "Num Features": len(features),
                    "Stemming": stem_flag,
                    "RMSE (Split)": rmse_split,
                    "RMSE (CV)": rmse_cv,
                    "RMSE_STD": rmse_std,
                    "Train Time (s)": round(elapsed_time, 2),
                }
            )

            logger.info(
                f"Features: {features} --> Split RMSE: {rmse_split:.4f}, CV RMSE: {rmse_cv:.4f} ± {rmse_std:.4f} (train time: {elapsed_time:.2f}s)"
            )

    results_df = pd.DataFrame(results).sort_values(by="RMSE (CV)")
    display(results_df)
    return results_df


def run_feature_search(
    raw_data,
    df_attr,
    num_train,
    feature_sets,
    model_params=None,
    estimator_cls=GradientBoostingRegressor,
    search_mode="presets",
    sample_size=25,
    random_seed=42,
    append_path=None,
):
    """Run feature search with optional random sampling and resumable output."""
    candidate_sets = list(feature_sets)
    if search_mode == "random" and len(candidate_sets) > sample_size:
        rng = random.Random(random_seed)
        candidate_sets = rng.sample(candidate_sets, sample_size)

    results_df = run_feature_set_evaluation(
        raw_data, df_attr, num_train, candidate_sets, model_params=model_params, estimator_cls=estimator_cls
    )
    results_df["Search Mode"] = search_mode
    results_df["Random Seed"] = random_seed

    if append_path and os.path.exists(append_path):
        existing_df = pd.read_csv(append_path)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
        results_df = results_df.drop_duplicates(subset=["Features", "Stemming", "Search Mode", "Random Seed"], keep="last")

    return results_df.sort_values(by="RMSE (CV)")


def evaluate_on_test_set(df_all, df_attr, features, num_train, stem=True, model_params=None, estimator_cls=GradientBoostingRegressor):
    df_features = build_feature_set(df_all.copy(), df_attr, features, stem=stem)
    df_train = df_features.iloc[:num_train]
    df_test = df_features.iloc[num_train:]
    X_train = df_train.drop(columns=["id", "relevance"])
    y_train = df_train["relevance"]
    X_test = df_test.drop(columns=["id", "relevance"], errors="ignore")

    if model_params is None:
        if estimator_cls is HistGradientBoostingRegressor:
            model_params = {"max_iter": 100, "max_depth": 7, "random_state": 42}
        elif estimator_cls is RandomForestRegressor:
            model_params = {"n_estimators": 100, "max_depth": 7, "random_state": 42, "n_jobs": -1}
        else:
            model_params = {"n_estimators": 100, "max_depth": 7, "random_state": 42}

    model = estimator_cls(**model_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return pd.DataFrame({"id": df_test["id"].values, "relevance": y_pred})


def generate_submission_file(
    raw_data,
    df_attr,
    num_train,
    best_features,
    model_params,
    output_path="submission.csv",
    estimator_cls=GradientBoostingRegressor,
):
    logger.info("Generating final test predictions with best feature combination...")
    final_submission = evaluate_on_test_set(
        raw_data,
        df_attr,
        num_train=num_train,
        features=best_features,
        stem=True,
        model_params=model_params,
        estimator_cls=estimator_cls,
    )
    final_submission["relevance"] = final_submission["relevance"].clip(1.0, 3.0).round(2)
    final_submission.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}.")


def plot_relevance_histogram(df_train, save_path="relevance_histogram_annotated.png"):
    plt.figure(figsize=(6.5, 4.875))
    bins = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    counts, edges, _ = plt.hist(df_train["relevance"], bins=bins, edgecolor="black", rwidth=0.7)
    for count, left_edge in zip(counts, edges[:-1]):
        center = left_edge + (bins[1] - bins[0]) / 2
        plt.text(center, count + 300, f"{int(count)}", ha="center", fontsize=10)
    plt.xlabel("Relevance Score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(bins)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_relevance_boxplot(df_train, save_path="relevance_boxplot.png"):
    median_value = df_train["relevance"].median()
    mean_value = df_train["relevance"].mean()
    plt.figure(figsize=(3.75, 5))
    plt.boxplot(
        df_train["relevance"],
        vert=True,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(color="black"),
        medianprops=dict(color="#2CA02C", linewidth=2, linestyle="--"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )
    x_center = 1
    box_width = 0.075
    plt.hlines(y=mean_value, xmin=x_center - box_width, xmax=x_center + box_width, color="#2CA02C", linestyle="--", linewidth=2)
    plt.ylabel("Relevance score", fontsize=12)
    plt.xlabel("Relevance data", fontsize=12)
    plt.xticks([])
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.text(1.1, median_value - 0.03, f"Median = {median_value:.2f}", color="#2CA02C", fontsize=10, va="center")
    plt.text(1.1, mean_value, f"Mean = {mean_value:.2f}", color="#2CA02C", fontsize=10, va="center")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def run_data_exploration(df_train, df_attr, output_dir=None):
    logger.info("=== Data Exploration ===")
    total_pairs = df_train.shape[0]
    unique_products = df_train["product_uid"].nunique()
    top_products = df_train["product_uid"].value_counts().head(2)
    top1_uid, top2_uid = top_products.index
    top1_title = df_train[df_train["product_uid"] == top1_uid]["product_title"].iloc[0]
    top2_title = df_train[df_train["product_uid"] == top2_uid]["product_title"].iloc[0]
    relevance_stats = df_train["relevance"].describe()

    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Total product-query pairs",
                "Unique product count",
                "Top 1 product ID",
                "Top 1 product title",
                "Top 1 count",
                "Top 2 product ID",
                "Top 2 product title",
                "Top 2 count",
                "Relevance Mean",
                "Relevance Median",
                "Relevance Std Dev",
            ],
            "Value": [
                total_pairs,
                unique_products,
                top1_uid,
                top1_title,
                top_products.iloc[0],
                top2_uid,
                top2_title,
                top_products.iloc[1],
                f"{relevance_stats['mean']:.3f}",
                f"{df_train['relevance'].median():.3f}",
                f"{relevance_stats['std']:.3f}",
            ],
        }
    )

    logger.info("Data Description Summary")
    display(summary_df)
    logger.info("Top-5 Most Common Brand Names")
    top_brands = (
        df_attr[df_attr["name"] == "MFG Brand Name"]["value"].value_counts().head(6).rename_axis("Brand Name").reset_index(name="Count")
    )
    display(top_brands)
    if output_dir:
        plot_relevance_histogram(df_train, save_path=os.path.join(output_dir, "relevance_histogram_annotated.png"))
        plot_relevance_boxplot(df_train, save_path=os.path.join(output_dir, "relevance_boxplot.png"))
    else:
        plot_relevance_histogram(df_train)
        plot_relevance_boxplot(df_train)


def plot_overfitting_curve(results_df, save_path="feature_count_vs_rmse.png"):
    plt.figure(figsize=(8, 5))
    for stem_flag in [True, False]:
        subset = results_df[results_df["Stemming"] == stem_flag]
        grouped = subset.loc[subset.groupby("Num Features")["RMSE (CV)"].idxmin()].sort_values("Num Features").reset_index(drop=True)
        label = "With Stemming" if stem_flag else "Without Stemming"
        plt.errorbar(grouped["Num Features"], grouped["RMSE (CV)"], yerr=grouped["RMSE_STD"], fmt="-o", capsize=5, label=label)

    plt.xlabel("Number of Features")
    plt.ylabel("Cross-Validated RMSE (±1 std)")
    plt.title("Model Complexity vs Performance")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_feature_importance(model, X, y, feature_names, top_n=5, plot=True, save_path=None):
    model.fit(X, y)
    importances = model.feature_importances_
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(by="importance", ascending=False)
    logger.info(f"Top {top_n} most important features:\n{importance_df.head(top_n)}")

    if plot:
        plt.figure(figsize=(10, 6))
        bars = plt.barh(importance_df["feature"][:top_n][::-1], importance_df["importance"][:top_n][::-1], color="steelblue")
        plt.xlim(0, 1.075 * max(importance_df["importance"]))
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.005, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center")
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.xlabel("Feature importance")
        plt.title("Top Feature Importances")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    return importance_df


def run_full_feature_importance(
    raw_data,
    df_attr,
    num_train,
    best_params,
    save_path="feature_importance_barplot.png",
    estimator_cls=GradientBoostingRegressor,
):
    df_full = build_feature_set(raw_data, df_attr, all_features, stem=True)
    X_all = df_full.iloc[:num_train].drop(columns=["id", "relevance"], errors="ignore")
    y_all = df_full.iloc[:num_train]["relevance"]
    return plot_feature_importance(
        estimator_cls(**best_params),
        X_all,
        y_all,
        X_all.columns.tolist(),
        top_n=12,
        save_path=save_path,
    )


def benchmark_new_features(raw_data, df_attr, num_train, model_params, estimator_cls=GradientBoostingRegressor):
    """Benchmark domain-specific new features against the original best set."""
    baseline_best = ["query_length", "initial_term_match", "jaccard", "common_words", "color_match", "fuzzy", "bigram_overlap"]
    candidate_sets = [
        baseline_best,
        baseline_best + ["edit_distance"],
        baseline_best + ["numeric_unit_consistency"],
        baseline_best + ["query_category_match"],
    ]
    return run_feature_set_evaluation(
        raw_data, df_attr, num_train, candidate_sets, model_params=model_params, estimator_cls=estimator_cls
    )

