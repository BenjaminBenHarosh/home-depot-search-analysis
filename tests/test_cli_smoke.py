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
