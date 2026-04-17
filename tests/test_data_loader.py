import pandas as pd

from src.data_loader import prepare_raw_data


def test_prepare_raw_data_merges_descriptions():
    df_train = pd.DataFrame({"id": [1], "product_uid": [100], "search_term": ["drill"], "relevance": [2.5]})
    df_test = pd.DataFrame({"id": [2], "product_uid": [101], "search_term": ["paint"]})
    df_desc = pd.DataFrame({"product_uid": [100, 101], "product_description": ["corded drill", "red paint"]})

    merged, num_train = prepare_raw_data(df_train, df_test, df_desc)
    assert num_train == 1
    assert merged.shape[0] == 2
    assert "product_description" in merged.columns
