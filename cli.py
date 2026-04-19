"""Click CLI for Home Depot search analysis."""

from __future__ import annotations

from contextlib import contextmanager

import click

from src.data_validation import check_data_dir
from src.pipeline import (
    run_baseline_stage,
    run_compare_models_stage,
    run_feature_search_stage,
    run_full_pipeline,
    run_tune_stage,
)


def _resolve_log_level(log_level: str, quiet: bool) -> str:
    if quiet:
        return "WARNING"
    return log_level


def _validate_data_dir(data_dir: str) -> str:
    try:
        return str(check_data_dir(data_dir))
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


@click.group()
def cli():
    """Project command-line interface."""


@cli.group()
@click.option(
    "--log-level",
    type=click.Choice(list(_LOG_LEVELS), case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Minimum level for console and run log file.",
)
@click.option("--quiet", is_flag=True, help="Shortcut for --log-level WARNING.")
@click.pass_context
def run(ctx: click.Context, log_level: str, quiet: bool):
    """Run pipeline stages."""
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = _resolve_log_level(log_level.upper(), quiet)


@contextmanager
def _stage_errors():
    try:
        yield
    except click.ClickException:
        raise
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:
        raise click.ClickException(f"{type(exc).__name__}: {exc}") from exc


@run.command("baseline")
@click.pass_context
@click.option("--data-dir", default="home-depot-product-search-relevance", show_default=True)
@click.option("--output-dir", default="outputs", show_default=True)
@click.option("--random-seed", default=42, type=int, show_default=True)
@click.option("--stem/--no-stem", default=True, show_default=True)
@click.option(
    "--run-id",
    default=None,
    help="Optional fixed run id (letters, digits, ._- only). Default: timestamp_seed.",
)
def run_baseline(ctx: click.Context, data_dir: str, output_dir: str, random_seed: int, stem: bool, run_id: str | None):
    with _stage_errors():
        data_dir = _validate_data_dir(data_dir)
        log_level = ctx.obj["log_level"]
        summary = run_baseline_stage(
            data_dir=data_dir,
            output_dir=output_dir,
            random_seed=random_seed,
            stem=stem,
            log_level=log_level,
            run_id_explicit=run_id,
        )
    click.echo("Baseline stage complete.")
    click.echo(f"Baseline run_id: {summary['run_id']}")
    click.echo(f"Baseline outputs: {summary['output_dir']}")


@run.command("compare-models")
@click.pass_context
@click.option("--data-dir", default="home-depot-product-search-relevance", show_default=True)
@click.option("--output-dir", default="outputs", show_default=True)
@click.option("--random-seed", default=42, type=int, show_default=True)
@click.option("--stem/--no-stem", default=True, show_default=True)
@click.option("--include-hist-gradient/--no-include-hist-gradient", default=True, show_default=True)
@click.option("--run-id", default=None, help="Optional fixed run id (letters, digits, ._- only).")
def run_compare_models(
    ctx: click.Context,
    data_dir: str,
    output_dir: str,
    random_seed: int,
    stem: bool,
    include_hist_gradient: bool,
    run_id: str | None,
):
    with _stage_errors():
        data_dir = _validate_data_dir(data_dir)
        log_level = ctx.obj["log_level"]
        out = run_compare_models_stage(
            data_dir=data_dir,
            stem=stem,
            include_hist_gradient=include_hist_gradient,
            output_dir=output_dir,
            random_seed=random_seed,
            log_level=log_level,
            run_id_explicit=run_id,
        )
    click.echo(out["results_df"].to_string(index=False))
    if out.get("run_id"):
        click.echo(f"Compare-models run_id: {out['run_id']}")
        click.echo(f"Artifacts: {out['output_dir']}")


@run.command("tune")
@click.pass_context
@click.option("--data-dir", default="home-depot-product-search-relevance", show_default=True)
@click.option("--output-dir", default="outputs", show_default=True)
@click.option("--random-seed", default=42, type=int, show_default=True)
@click.option("--n-iter", default=10, type=int, show_default=True)
@click.option("--stem/--no-stem", default=True, show_default=True)
@click.option("--run-id", default=None, help="Optional fixed run id (letters, digits, ._- only).")
def run_tune(
    ctx: click.Context,
    data_dir: str,
    output_dir: str,
    random_seed: int,
    n_iter: int,
    stem: bool,
    run_id: str | None,
):
    with _stage_errors():
        data_dir = _validate_data_dir(data_dir)
        log_level = ctx.obj["log_level"]
        hist_search, rf_search, winner, winner_name, meta = run_tune_stage(
            data_dir=data_dir,
            random_seed=random_seed,
            n_iter=n_iter,
            stem=stem,
            output_dir=output_dir,
            log_level=log_level,
            run_id_explicit=run_id,
        )
    click.echo(f"HistGradientBoosting params: {hist_search.best_params_}")
    click.echo(f"Random Forest params: {rf_search.best_params_}")
    click.echo(f"Selected finalist ({winner_name}): {winner.best_params_}")
    if meta.get("run_id"):
        click.echo(f"Tune run_id: {meta['run_id']}")
        click.echo(f"Artifacts: {meta['output_dir']}")


@run.command("feature-search")
@click.pass_context
@click.option("--data-dir", default="home-depot-product-search-relevance", show_default=True)
@click.option("--feature-mode", type=click.Choice(["presets", "yaml", "auto"]), default="presets", show_default=True)
@click.option("--feature-config-path", default="configs/features.yaml", show_default=True)
@click.option("--output-dir", default="outputs", show_default=True)
@click.option("--random-seed", default=42, type=int, show_default=True)
@click.option("--stem/--no-stem", default=True, show_default=True)
@click.option("--run-id", default=None, help="Optional fixed run id (letters, digits, ._- only).")
def run_feature_search(
    ctx: click.Context,
    data_dir: str,
    feature_mode: str,
    feature_config_path: str,
    output_dir: str,
    random_seed: int,
    stem: bool,
    run_id: str | None,
):
    with _stage_errors():
        data_dir = _validate_data_dir(data_dir)
        log_level = ctx.obj["log_level"]
        out = run_feature_search_stage(
            data_dir=data_dir,
            feature_mode=feature_mode,
            feature_config_path=feature_config_path,
            output_dir=output_dir,
            random_seed=random_seed,
            stem=stem,
            log_level=log_level,
            run_id_explicit=run_id,
        )
    click.echo(f"Feature search run_id: {out['run_id']}")
    click.echo(f"Feature search complete. Saved to {out['csv_path']}.")
    click.echo(out["results_df"].head(10).to_string(index=False))


@run.command("full-pipeline")
@click.pass_context
@click.option("--data-dir", default="home-depot-product-search-relevance", show_default=True)
@click.option("--output-dir", default="outputs", show_default=True)
@click.option("--random-seed", default=42, type=int, show_default=True)
@click.option("--stem/--no-stem", default=True, show_default=True)
@click.option("--run-id", default=None, help="Optional fixed run id (letters, digits, ._- only).")
@click.option(
    "--compare-models/--no-compare-models",
    default=False,
    show_default=True,
    help="Run broad sklearn model comparison inside full pipeline (adds time). Default is off.",
)
def run_full(
    ctx: click.Context,
    data_dir: str,
    output_dir: str,
    random_seed: int,
    stem: bool,
    run_id: str | None,
    compare_models: bool,
):
    with _stage_errors():
        data_dir = _validate_data_dir(data_dir)
        log_level = ctx.obj["log_level"]
        summary = run_full_pipeline(
            data_dir=data_dir,
            output_dir=output_dir,
            random_seed=random_seed,
            stem=stem,
            log_level=log_level,
            run_id_explicit=run_id,
            run_model_comparison=compare_models,
        )
    click.echo(f"Pipeline complete. Best model: {summary['best_model']}")
    click.echo(f"Results run_id: {summary['run_id']}")
    click.echo(f"Results file: {output_dir}/runs/{summary['run_id']}/results_summary.json")


if __name__ == "__main__":
    cli()
