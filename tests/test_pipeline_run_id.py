import pytest

from src.pipeline import resolve_run_id


def test_resolve_run_id_explicit_ok():
    assert resolve_run_id(42, "my-run_v1") == "my-run_v1"


def test_resolve_run_id_explicit_invalid():
    with pytest.raises(ValueError, match="Invalid"):
        resolve_run_id(42, "../../etc")


def test_resolve_run_id_auto():
    rid = resolve_run_id(42, None)
    assert "seed42" in rid
