"""Click CLI for Home Depot search analysis."""

import click

from src.pipeline import (
    run_baseline_stage,
    run_compare_models_stage,
    run_feature_search_stage,
    run_full_pipeline,
    run_tune_stage,
)


@click.group()
def cli():
    """Project command-line interface."""


@cli.group()
def run():
    """Run pipeline stages."""


@run.command("baseline")
@click.option("--data-dir", default="home-depot-product-search-relevance", show_default=True)
@click.option("--stem/--no-stem", default=True, show_default=True)
def run_baseline(data_dir, stem):
    run_baseline_stage(data_dir=data_dir, stem=stem)
    click.echo("Baseline stage complete.")


@run.command("compare-models")
@click.option("--data-dir", default="home-depot-product-search-relevance", show_default=True)
@click.option("--stem/--no-stem", default=True, show_default=True)
@click.option("--include-hist-gradient/--no-include-hist-gradient", default=True, show_default=True)
def run_compare_models(data_dir, stem, include_hist_gradient):
    results = run_compare_models_stage(data_dir=data_dir, stem=stem, include_hist_gradient=include_hist_gradient)
    click.echo(results.to_string(index=False))


@run.command("tune")
@click.option("--data-dir", default="home-depot-product-search-relevance", show_default=True)
@click.option("--random-seed", default=42, type=int, show_default=True)
@click.option("--n-iter", default=10, type=int, show_default=True)
@click.option("--stem/--no-stem", default=True, show_default=True)
def run_tune(data_dir, random_seed, n_iter, stem):
    _, gb = run_tune_stage(data_dir=data_dir, random_seed=random_seed, n_iter=n_iter, stem=stem)
    click.echo(f"Best Gradient Boosting params: {gb.best_params_}")


@run.command("feature-search")
@click.option("--data-dir", default="home-depot-product-search-relevance", show_default=True)
@click.option("--feature-mode", type=click.Choice(["presets", "yaml", "auto"]), default="presets", show_default=True)
@click.option("--feature-config-path", default="configs/features.yaml", show_default=True)
@click.option("--output-dir", default="outputs", show_default=True)
@click.option("--random-seed", default=42, type=int, show_default=True)
@click.option("--stem/--no-stem", default=True, show_default=True)
def run_feature_search(data_dir, feature_mode, feature_config_path, output_dir, random_seed, stem):
    output_path = f"{output_dir}/feature_search_results.csv"
    results = run_feature_search_stage(
        data_dir=data_dir,
        feature_mode=feature_mode,
        feature_config_path=feature_config_path,
        output_path=output_path,
        random_seed=random_seed,
        stem=stem,
    )
    click.echo(f"Feature search complete. Saved to {output_path}.")
    click.echo(results.head(10).to_string(index=False))


@run.command("full-pipeline")
@click.option("--data-dir", default="home-depot-product-search-relevance", show_default=True)
@click.option("--output-dir", default="outputs", show_default=True)
@click.option("--random-seed", default=42, type=int, show_default=True)
@click.option("--stem/--no-stem", default=True, show_default=True)
def run_full(data_dir, output_dir, random_seed, stem):
    summary = run_full_pipeline(data_dir=data_dir, output_dir=output_dir, random_seed=random_seed, stem=stem)
    click.echo(f"Pipeline complete. Best model: {summary['best_model']}")
    click.echo(f"Results: {output_dir}/results.json")


if __name__ == "__main__":
    cli()

