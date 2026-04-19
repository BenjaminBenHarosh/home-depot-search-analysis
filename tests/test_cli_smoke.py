from click.testing import CliRunner

from cli import cli


def test_cli_help_runs():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Run pipeline stages" in result.output


def test_cli_run_help_runs():
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "full-pipeline" in result.output
    assert "--log-level" in result.output
    assert "--quiet" in result.output


def test_cli_full_pipeline_help_lists_compare_models_flag():
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "full-pipeline", "--help"])
    assert result.exit_code == 0
    assert "--compare-models" in result.output


def test_cli_baseline_rejects_bad_data_dir():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "baseline", "--data-dir", "/nonexistent/path/that/does/not/exist"],
    )
    assert result.exit_code != 0
    assert "Data directory does not exist" in result.output
