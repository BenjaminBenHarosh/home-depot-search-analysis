import math

import pandas as pd
import pytest

from src.feature_engineering import (
    all_features,
    build_default_feature_sets,
    build_feature_set,
    feature_sets,
    fuzzy_ratio,
    generate_feature_combinations,
    jaccard_similarity,
    load_feature_sets_from_yaml,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n_train=3, n_test=1):
    """Minimal DataFrame with n_train + n_test rows for feature testing."""
    rows = n_train + n_test
    return pd.DataFrame(
        {
            "id": list(range(1, rows + 1)),
            "product_uid": list(range(10, 10 + rows)),
            "search_term": ["red paint", "12 ft ladder", "power drill", "cabinet hinge"][:rows],
            "product_title": [
                "Exterior red paint",
                "Aluminum ladder 12 ft",
                "Cordless power drill",
                "Cabinet door hinge",
            ][:rows],
            "product_description": [
                "Weatherproof exterior paint",
                "12 ft multi use aluminum ladder",
                "Cordless brushless drill driver",
                "Heavy duty cabinet hinge for doors",
            ][:rows],
            "relevance": [2.5, 2.8, 3.0, 1.5][:rows],
        }
    )


def _make_attr():
    return pd.DataFrame({"product_uid": [], "name": [], "value": []})


# ---------------------------------------------------------------------------
# Existing feature tests
# ---------------------------------------------------------------------------

def test_jaccard_similarity_boundaries():
    assert jaccard_similarity("apple", "apple") == 1.0
    assert jaccard_similarity("apple", "banana") == 0.0


def test_fuzzy_ratio_perfect_match():
    assert fuzzy_ratio("drill", "drill") == 100


def test_build_feature_set_preserves_rows():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "product_uid": [10, 11],
            "search_term": ["red paint", "12 ft ladder"],
            "product_title": ["Exterior red paint", "Aluminum ladder"],
            "product_description": ["Weatherproof paint", "12 ft multi use ladder"],
            "relevance": [2.5, 2.8],
        }
    )
    df_attr = pd.DataFrame({"product_uid": [10, 11], "name": ["MFG Brand Name", "MFG Brand Name"], "value": ["Acme", "ToolCo"]})
    engineered = build_feature_set(df, df_attr, ["query_length", "common_words"], stem=True)
    assert engineered.shape[0] == 2
    assert "query_length" in engineered.columns


def test_generate_feature_combinations_not_empty():
    combos = generate_feature_combinations(features=["query_length", "common_words"], min_size=1, max_size=2)
    assert len(combos) == 3


def test_load_feature_sets_from_yaml():
    loaded = load_feature_sets_from_yaml("configs/features.yaml")
    assert len(loaded) >= 1


# ---------------------------------------------------------------------------
# build_default_feature_sets
# ---------------------------------------------------------------------------

def test_build_default_feature_sets_contains_all_singles():
    sets = build_default_feature_sets()
    singles = [[f] for f in all_features]
    for s in singles:
        assert s in sets, f"Single-feature set {s} missing from default presets"


def test_build_default_feature_sets_no_duplicates():
    sets = build_default_feature_sets()
    tuples = [tuple(s) for s in sets]
    assert len(tuples) == len(set(tuples)), "Duplicate feature sets found"


def test_feature_sets_module_level_is_valid():
    """The module-level feature_sets list must contain only known features."""
    from src.feature_engineering import validate_feature_list
    for fs in feature_sets:
        validate_feature_list(fs)  # raises ValueError on unknown feature


def test_build_default_feature_sets_missing_yaml_returns_singles_only(tmp_path):
    sets = build_default_feature_sets(config_path=str(tmp_path / "nonexistent.yaml"))
    assert sets == [[f] for f in all_features]


# ---------------------------------------------------------------------------
# TF-IDF leakage fix (num_train)
# ---------------------------------------------------------------------------

def test_tfidf_similarity_num_train_restricts_fit():
    """Vectorizer must fit on train rows only; full frame is transformed."""
    df = _make_df(n_train=3, n_test=1)
    result = build_feature_set(df.copy(), _make_attr(), ["tfidf_similarity"], stem=False, num_train=3)
    assert result.shape[0] == 4
    assert "tfidf_similarity" in result.columns
    assert result["tfidf_similarity"].between(0, 1).all()


# ---------------------------------------------------------------------------
# Phase 2 feature tests  (skipped until Phase 2 is implemented)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Requires Phase 2: add_tfidf_split not yet implemented")
def test_tfidf_split_produces_two_columns():
    df = _make_df()
    result = build_feature_set(df.copy(), _make_attr(), ["tfidf_split"], stem=False, num_train=3)
    assert "tfidf_title_sim" in result.columns
    assert "tfidf_desc_sim" in result.columns
    assert result["tfidf_title_sim"].between(0, 1).all()
    assert result["tfidf_desc_sim"].between(0, 1).all()


@pytest.mark.skip(reason="Requires Phase 2: add_lsa_similarity not yet implemented")
def test_lsa_similarity_values_finite():
    df = _make_df()
    result = build_feature_set(df.copy(), _make_attr(), ["lsa_similarity"], stem=False, num_train=3)
    assert "lsa_similarity" in result.columns
    assert result["lsa_similarity"].apply(math.isfinite).all()


@pytest.mark.skip(reason="Requires Phase 2: add_bm25_similarity not yet implemented")
def test_bm25_similarity_nonnegative():
    df = _make_df()
    result = build_feature_set(df.copy(), _make_attr(), ["bm25_similarity"], stem=False, num_train=3)
    assert "bm25_similarity" in result.columns
    assert (result["bm25_similarity"] >= 0).all()


@pytest.mark.skip(reason="Requires Phase 2: add_query_mean_relevance not yet implemented")
def test_query_mean_relevance_varies_across_queries():
    df = _make_df()
    result = build_feature_set(df.copy(), _make_attr(), ["query_mean_relevance"], stem=False, num_train=3)
    assert "query_mean_relevance" in result.columns
    # Each train query is unique so LOO estimates should not all be identical
    train_vals = result["query_mean_relevance"].iloc[:3]
    assert train_vals.nunique() > 1
