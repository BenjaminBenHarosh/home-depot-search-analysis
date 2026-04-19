"""Feature engineering utilities and registry helpers."""

from __future__ import annotations

from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml
from fuzzywuzzy import fuzz
from nltk import ngrams
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

materials = {
    "wood",
    "metal",
    "steel",
    "aluminum",
    "plastic",
    "mdf",
    "copper",
    "brass",
    "glass",
    "rubber",
    "fiberglass",
    "vinyl",
    "ceramic",
    "stone",
    "polyester",
    "nylon",
    "composite",
    "bamboo",
}

units = {
    "in",
    "inch",
    "inches",
    "ft",
    "feet",
    "mm",
    "cm",
    "m",
    "oz",
    "lbs",
    "pound",
    "gallon",
    "ml",
    "liter",
    "litre",
    "quart",
    "yard",
}

colors = {
    "white",
    "black",
    "gray",
    "grey",
    "blue",
    "red",
    "green",
    "yellow",
    "brown",
    "silver",
    "gold",
    "beige",
    "ivory",
    "navy",
    "tan",
    "orange",
    "pink",
    "charcoal",
    "bronze",
    "teal",
    "maroon",
}

stemmer = SnowballStemmer("english")

all_features = [
    "query_length",
    "common_words",
    "brand_match",
    "tfidf_similarity",
    "query_has_number",
    "unit_match",
    "initial_term_match",
    "material_match",
    "color_match",
    "jaccard",
    "bigram_overlap",
    "fuzzy",
    "edit_distance",
    "numeric_unit_consistency",
    "query_category_match",
]

feature_sets = [
    ["tfidf_similarity"],
    ["initial_term_match"],
    ["query_length"],
    ["jaccard"],
    ["fuzzy"],
    ["query_has_number"],
    ["bigram_overlap"],
    ["color_match"],
    ["brand_match"],
    ["common_words"],
    ["unit_match"],
    ["material_match"],
]
feature_sets += [["query_length", "common_words"]]
feature_sets += [
    ["query_length", "common_words", "initial_term_match"],
    ["tfidf_similarity", "query_length", "common_words"],
    ["tfidf_similarity", "query_length", "initial_term_match"],
]
feature_sets += [["tfidf_similarity", "query_length", "initial_term_match", "common_words"]]
feature_sets += [["tfidf_similarity", "query_length", "initial_term_match", "common_words", "fuzzy"]]
feature_sets += [
    ["tfidf_similarity", "query_length", "jaccard", "query_has_number", "common_words", "color_match"],
    ["tfidf_similarity", "query_length", "initial_term_match", "common_words", "fuzzy", "jaccard"],
]
feature_sets += [
    ["query_length", "initial_term_match", "jaccard", "common_words", "color_match", "fuzzy", "bigram_overlap"],
    ["tfidf_similarity", "query_length", "initial_term_match", "common_words", "fuzzy", "jaccard", "query_has_number"],
]
feature_sets += [["tfidf_similarity", "query_length", "initial_term_match", "fuzzy", "jaccard", "common_words", "query_has_number", "bigram_overlap"]]
feature_sets += [["tfidf_similarity", "query_length", "initial_term_match", "fuzzy", "jaccard", "common_words", "query_has_number", "color_match", "unit_match"]]
feature_sets += [["tfidf_similarity", "query_length", "initial_term_match", "fuzzy", "jaccard", "common_words", "query_has_number", "color_match", "material_match", "brand_match"]]
feature_sets += [["tfidf_similarity", "query_length", "initial_term_match", "fuzzy", "jaccard", "common_words", "query_has_number", "color_match", "material_match", "brand_match", "unit_match"]]
feature_sets += [["tfidf_similarity", "query_length", "initial_term_match", "fuzzy", "jaccard", "common_words", "query_has_number", "color_match", "material_match", "brand_match", "unit_match", "bigram_overlap"]]

CATEGORY_KEYWORDS = {
    "paint": {"paint", "primer", "stain"},
    "power_tools": {"drill", "saw", "sander", "router"},
    "plumbing": {"faucet", "toilet", "pipe", "valve"},
    "lighting": {"light", "lamp", "bulb", "fixture"},
}


def str_stemmer(text):
    return " ".join([stemmer.stem(word) for word in text.lower().split()])


def str_common_word(str1, str2):
    return sum(int(str2.find(word) >= 0) for word in str1.split())


def jaccard_similarity(str1, str2):
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0


def bigram_overlap(str1, str2):
    query_ngrams = set(ngrams(str1.lower().split(), 2))
    text_ngrams = set(ngrams(str2.lower().split(), 2))
    return len(query_ngrams.intersection(text_ngrams))


def fuzzy_ratio(str1, str2):
    return fuzz.token_sort_ratio(str1, str2)


def edit_distance_ratio(str1, str2):
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def add_query_length(df):
    df["query_length"] = df["search_term"].map(lambda x: len(x.split())).astype(np.int64)
    return df


def add_jaccard_similarity(df):
    df["jaccard_title"] = df.apply(lambda row: jaccard_similarity(row["search_term"], row["product_title"]), axis=1)
    df["jaccard_description"] = df.apply(lambda row: jaccard_similarity(row["search_term"], row["product_description"]), axis=1)
    return df


def add_bigram_overlap(df):
    df["bigram_overlap_title"] = df.apply(lambda row: bigram_overlap(row["search_term"], row["product_title"]), axis=1)
    df["bigram_overlap_description"] = df.apply(lambda row: bigram_overlap(row["search_term"], row["product_description"]), axis=1)
    return df


def add_fuzzy_ratio(df):
    df["fuzzy_title"] = df.apply(lambda row: fuzzy_ratio(row["search_term"], row["product_title"]), axis=1)
    df["fuzzy_description"] = df.apply(lambda row: fuzzy_ratio(row["search_term"], row["product_description"]), axis=1)
    return df


def add_edit_distance(df):
    df["edit_distance_title"] = df.apply(lambda row: edit_distance_ratio(row["search_term"], row["product_title"]), axis=1)
    df["edit_distance_description"] = df.apply(lambda row: edit_distance_ratio(row["search_term"], row["product_description"]), axis=1)
    return df


def add_common_word_features(df):
    df["product_info"] = df["search_term"] + "\t" + df["product_title"] + "\t" + df["product_description"]
    df["title_overlap"] = df["product_info"].map(lambda x: str_common_word(x.split("\t")[0], x.split("\t")[1]))
    df["description_overlap"] = df["product_info"].map(lambda x: str_common_word(x.split("\t")[0], x.split("\t")[2]))
    df.drop("product_info", axis=1, inplace=True)
    return df


def add_brand_match(df, df_attr):
    df_brand = df_attr[df_attr["name"] == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
    df = pd.merge(df, df_brand, how="left", on="product_uid")
    df["brand"] = df["brand"].fillna("").map(str_stemmer)
    df["brand_match"] = df.apply(lambda x: str_common_word(x["search_term"], x["brand"]), axis=1)
    df.drop("brand", axis=1, inplace=True)
    return df


def add_tfidf_similarity(df, num_train=None):
    tfidf = TfidfVectorizer()
    product_text = df["product_title"] + " " + df["product_description"]
    train_text = product_text.iloc[:num_train] if num_train is not None else product_text
    tfidf.fit(train_text)
    tfidf_matrix = tfidf.transform(product_text)
    query_matrix = tfidf.transform(df["search_term"])
    df["tfidf_similarity"] = [cosine_similarity(query_matrix[i], tfidf_matrix[i])[0, 0] for i in range(len(df))]
    return df


def add_query_has_number(df):
    df["query_has_number"] = df["search_term"].map(lambda x: sum(any(char.isdigit() for char in token) for token in x.split()))
    return df


def add_numeric_unit_consistency(df):
    def numeric_unit_score(row):
        query_tokens = row["search_term"].split()
        text_tokens = (row["product_title"] + " " + row["product_description"]).split()
        query_nums = {token for token in query_tokens if any(ch.isdigit() for ch in token)}
        text_nums = {token for token in text_tokens if any(ch.isdigit() for ch in token)}
        query_units = {token for token in query_tokens if token in units}
        text_units = {token for token in text_tokens if token in units}
        num_overlap = len(query_nums.intersection(text_nums))
        unit_overlap = len(query_units.intersection(text_units))
        return float(num_overlap + unit_overlap)

    df["numeric_unit_consistency"] = df.apply(numeric_unit_score, axis=1)
    return df


def add_unit_match(df):
    def unit_overlap(row):
        query_tokens = set(row["search_term"].split())
        product_text = row["product_title"] + " " + row["product_description"]
        return int(any(unit in product_text for unit in query_tokens & units))

    df["unit_match"] = df.apply(unit_overlap, axis=1)
    return df


def add_initial_term_match(df):
    def count_early_hits(row):
        query_words = row["search_term"].split()[:2]
        return sum(word in row["product_title"] for word in query_words)

    df["initial_term_match"] = df.apply(count_early_hits, axis=1)
    return df


def add_material_match(df, df_attr):
    df_material = df_attr[df_attr["name"].str.contains("Material", case=False, na=False)]
    df_material = df_material[["product_uid", "value"]].rename(columns={"value": "material"})
    df = pd.merge(df, df_material, how="left", on="product_uid")
    df["material"] = df["material"].fillna("").map(str_stemmer)
    df["material_match"] = df.apply(
        lambda x: int(any(mat in x["material"] for mat in x["search_term"].split() if mat in materials)),
        axis=1,
    )
    df.drop("material", axis=1, inplace=True)
    return df


def add_color_match(df, df_attr):
    df_color = df_attr[df_attr["name"].str.contains("Color", case=False, na=False)]
    df_color = df_color[["product_uid", "value"]].rename(columns={"value": "color"})
    df = pd.merge(df, df_color, how="left", on="product_uid")
    df["color"] = df["color"].fillna("").map(str_stemmer)
    df["color_match"] = df.apply(
        lambda x: int(any(color in x["search_term"] and color in x["color"] for color in colors)),
        axis=1,
    )
    df.drop("color", axis=1, inplace=True)
    return df


def add_query_category_match(df):
    def category_flag(query):
        query_text = query.lower()
        for category_terms in CATEGORY_KEYWORDS.values():
            if any(term in query_text for term in category_terms):
                return 1
        return 0

    df["query_category_match"] = df["search_term"].map(category_flag)
    return df


def _feature_registry() -> dict[str, Callable]:
    return {
        "query_length": lambda df, _, __: add_query_length(df),
        "common_words": lambda df, _, __: add_common_word_features(df),
        "brand_match": lambda df, attr, __: add_brand_match(df, attr),
        "tfidf_similarity": lambda df, _, num_train: add_tfidf_similarity(df, num_train),
        "query_has_number": lambda df, _, __: add_query_has_number(df),
        "unit_match": lambda df, _, __: add_unit_match(df),
        "initial_term_match": lambda df, _, __: add_initial_term_match(df),
        "material_match": lambda df, attr, __: add_material_match(df, attr),
        "color_match": lambda df, attr, __: add_color_match(df, attr),
        "jaccard": lambda df, _, __: add_jaccard_similarity(df),
        "bigram_overlap": lambda df, _, __: add_bigram_overlap(df),
        "fuzzy": lambda df, _, __: add_fuzzy_ratio(df),
        "edit_distance": lambda df, _, __: add_edit_distance(df),
        "numeric_unit_consistency": lambda df, _, __: add_numeric_unit_consistency(df),
        "query_category_match": lambda df, _, __: add_query_category_match(df),
    }


FEATURE_REGISTRY = _feature_registry()


def validate_feature_list(features_to_include):
    seen = set()
    for feature in features_to_include:
        if feature in seen:
            raise ValueError(f"Duplicate feature requested: {feature}")
        if feature not in FEATURE_REGISTRY:
            raise ValueError(f"Unknown feature requested: {feature}")
        seen.add(feature)


def generate_feature_combinations(features=None, min_size=1, max_size=4):
    """Generate bounded feature combinations for automated search."""
    candidate_features = features or all_features
    validate_feature_list(candidate_features)
    combos = []
    for size in range(min_size, max_size + 1):
        combos.extend([list(combo) for combo in combinations(candidate_features, size)])
    return combos


def load_feature_sets_from_yaml(config_path):
    """Load named feature presets from YAML."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    presets = config.get("feature_sets", {})
    if not isinstance(presets, dict):
        raise ValueError("feature_sets must be a map of preset name -> feature list")
    loaded_sets = []
    for _, feature_list in presets.items():
        validate_feature_list(feature_list)
        loaded_sets.append(feature_list)
    return loaded_sets


def build_feature_set(df_raw, df_attr, features_to_include, stem=True, num_train=None):
    """Construct selected engineered features.

    Parameters
    ----------
    num_train:
        Number of rows that belong to the training split (``df_raw.iloc[:num_train]``).
        When provided, any feature that requires fitting (e.g. TF-IDF vectorizer) will
        be fit only on training rows to prevent data leakage into validation/test rows.
        Pass ``None`` (default) when building features for a pure test frame where no
        leakage risk exists.
    """
    df = df_raw.copy()
    validate_feature_list(features_to_include)
    if stem:
        cols = ["search_term", "product_title", "product_description"]
        df[cols] = df[cols].apply(lambda col: col.map(str_stemmer))

    for feature_name in features_to_include:
        transform = FEATURE_REGISTRY[feature_name]
        df = transform(df, df_attr, num_train)

    df.drop(["search_term", "product_title", "product_description", "product_uid"], axis=1, inplace=True)
    if df.columns.duplicated().any():
        duplicates = list(df.columns[df.columns.duplicated()])
        raise ValueError(f"Duplicate output columns detected: {duplicates}")
    return df

