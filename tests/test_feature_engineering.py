import pandas as pd

from src.feature_engineering import (
    build_feature_set,
    fuzzy_ratio,
    generate_feature_combinations,
    jaccard_similarity,
    load_feature_sets_from_yaml,
)


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
    feature_sets = load_feature_sets_from_yaml("configs/features.yaml")
    assert len(feature_sets) >= 1
