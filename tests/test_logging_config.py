from pathlib import Path

from src.logging_config import configure_logging


def test_configure_logging_creates_run_log_path(tmp_path):
    output_dir = tmp_path / "outputs"
    run_id = "sample_run"
    log_path = configure_logging(str(output_dir), run_id, level="INFO")
    expected = output_dir / "runs" / run_id / "logs" / "run.log"
    assert Path(log_path) == expected
    assert expected.parent.exists()

