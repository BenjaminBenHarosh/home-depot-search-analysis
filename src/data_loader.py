"""Data loading and merge utilities."""

import pandas as pd


def load_raw_datasets(data_dir="home-depot-product-search-relevance"):
    """Load Home Depot competition CSV files."""
    df_train = pd.read_csv(f"{data_dir}/train.csv", encoding="ISO-8859-1")
    df_test = pd.read_csv(f"{data_dir}/test.csv", encoding="ISO-8859-1")
    df_attr = pd.read_csv(f"{data_dir}/attributes.csv", encoding="ISO-8859-1")
    df_pro_desc = pd.read_csv(f"{data_dir}/product_descriptions.csv")
    return df_train, df_test, df_attr, df_pro_desc


def prepare_raw_data(df_train, df_test, df_pro_desc):
    """Combine train/test and merge product descriptions."""
    num_train = df_train.shape[0]
    raw_data = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    raw_data = pd.merge(raw_data, df_pro_desc, how="left", on="product_uid")
    return raw_data, num_train

