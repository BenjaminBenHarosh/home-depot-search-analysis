import pandas as pd
import pytest

from src.data_loader import prepare_raw_data


def test_prepare_raw_data_merges_descriptions():
    df_train = pd.DataFrame({"id": [1], "product_uid": [100], "search_term": ["drill"], "relevance": [2.5]})
    df_test = pd.DataFrame({"id": [2], "product_uid": [101], "search_term": ["paint"]})
    df_desc = pd.DataFrame({"product_uid": [100, 101], "product_description": ["corded drill", "red paint"]})

    merged, num_train = prepare_raw_data(df_train, df_test, df_desc)
    assert num_train == 1
    assert merged.shape[0] == 2
    assert "product_description" in merged.columns


# ---------------------------------------------------------------------------
# Phase 2 spelling correction tests (skipped until Phase 2 is implemented)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Requires Phase 2: apply_spelling_correction not yet implemented")
def test_apply_spelling_correction_corrects_known_typo():
    from src.data_loader import apply_spelling_correction

    df = pd.DataFrame({"search_term": ["hammr drill", "paint brush"]})
    result = apply_spelling_correction(df)
    # "hammr" should be corrected to "hammer"; unchanged tokens stay put
    assert "hammr" not in result["search_term"].iloc[0]
    assert "paint" in result["search_term"].iloc[1]
    assert "brush" in result["search_term"].iloc[1]


@pytest.mark.skip(reason="Requires Phase 2: apply_spelling_correction not yet implemented")
def test_apply_spelling_correction_skips_numeric_tokens():
    from src.data_loader import apply_spelling_correction

    df = pd.DataFrame({"search_term": ["2x4 lumber", "12v battery"]})
    result = apply_spelling_correction(df)
    # numeric tokens must not be altered
    assert "2x4" in result["search_term"].iloc[0]
    assert "12v" in result["search_term"].iloc[1]


@pytest.mark.skip(reason="Requires Phase 2: apply_spelling_correction not yet implemented")
def test_apply_spelling_correction_does_not_mutate_input():
    from src.data_loader import apply_spelling_correction

    df = pd.DataFrame({"search_term": ["cabnit hinge"]})
    original = df["search_term"].iloc[0]
    apply_spelling_correction(df)
    assert df["search_term"].iloc[0] == original  # original unchanged (function returns copy)
