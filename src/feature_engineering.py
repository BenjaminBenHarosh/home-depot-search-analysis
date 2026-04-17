"""Feature engineering utilities."""

import numpy as np
import pandas as pd
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


def add_tfidf_similarity(df):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df["product_title"] + " " + df["product_description"])
    query_matrix = tfidf.transform(df["search_term"])
    df["tfidf_similarity"] = [cosine_similarity(query_matrix[i], tfidf_matrix[i])[0, 0] for i in range(len(df))]
    return df


def add_query_has_number(df):
    df["query_has_number"] = df["search_term"].map(lambda x: sum(any(char.isdigit() for char in token) for token in x.split()))
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


def build_feature_set(df_raw, df_attr, features_to_include, stem=True):
    """Construct selected engineered features."""
    df = df_raw.copy()
    if stem:
        cols = ["search_term", "product_title", "product_description"]
        df[cols] = df[cols].apply(lambda col: col.map(str_stemmer))

    if "query_length" in features_to_include:
        df = add_query_length(df)
    if "common_words" in features_to_include:
        df = add_common_word_features(df)
    if "brand_match" in features_to_include:
        df = add_brand_match(df, df_attr)
    if "tfidf_similarity" in features_to_include:
        df = add_tfidf_similarity(df)
    if "query_has_number" in features_to_include:
        df = add_query_has_number(df)
    if "unit_match" in features_to_include:
        df = add_unit_match(df)
    if "initial_term_match" in features_to_include:
        df = add_initial_term_match(df)
    if "material_match" in features_to_include:
        df = add_material_match(df, df_attr)
    if "color_match" in features_to_include:
        df = add_color_match(df, df_attr)
    if "jaccard" in features_to_include:
        df = add_jaccard_similarity(df)
    if "bigram_overlap" in features_to_include:
        df = add_bigram_overlap(df)
    if "fuzzy" in features_to_include:
        df = add_fuzzy_ratio(df)

    df.drop(["search_term", "product_title", "product_description", "product_uid"], axis=1, inplace=True)
    return df

