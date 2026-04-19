import pytest

from src.data_validation import REQUIRED_DATA_FILES, check_data_dir


def test_check_data_dir_resolves_and_validates(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    for name in REQUIRED_DATA_FILES:
        (d / name).write_text("h", encoding="utf-8")
    resolved = check_data_dir(str(d))
    assert resolved == d.resolve()


def test_check_data_dir_raises_when_missing(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(ValueError, match="Missing required file"):
        check_data_dir(str(d))


def test_check_data_dir_raises_when_not_dir(tmp_path):
    p = tmp_path / "not_a_dir.txt"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        check_data_dir(str(p))


def test_check_data_dir_raises_when_missing_dir(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        check_data_dir(str(tmp_path / "nonexistent"))
