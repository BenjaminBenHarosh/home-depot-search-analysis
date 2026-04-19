from pathlib import Path

from loguru import logger

from src.logging_config import configure_logging


def test_configure_logging_creates_run_log_path(tmp_path):
    output_dir = tmp_path / "outputs"
    run_id = "sample_run"
    log_path = configure_logging(str(output_dir), run_id, level="INFO")
    expected = output_dir / "runs" / run_id / "logs" / "run.log"
    assert Path(log_path) == expected
    assert expected.parent.exists()


def test_configure_logging_tqdm_sink_accepts_log_line(tmp_path):
    output_dir = tmp_path / "outputs"
    configure_logging(str(output_dir), "tqdm_smoke", level="INFO")
    logger.info("ok after tqdm console sink")
    run_log = output_dir / "runs" / "tqdm_smoke" / "logs" / "run.log"
    text = run_log.read_text(encoding="utf-8")
    assert "ok after tqdm console sink" in text

