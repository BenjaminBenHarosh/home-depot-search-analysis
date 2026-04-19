# Home Depot Product Search Relevance

Predicting search query-product relevance for the Kaggle Home Depot challenge using NLP feature engineering and ensemble regression models.

## Results

Current values below are placeholders from assignment notes and should be treated as provisional until refreshed from a new run.

| Model | RMSE (CV) | Features | Status |
|---|---:|---:|---|
| Gradient Boosting | 0.47XX | 7 | Best |
| Random Forest | 0.49XX | 7 | Competitive |
| Baseline (Bagging) | 0.55XX | 2 | Baseline |

- Best feature set: `query_length`, `initial_term_match`, `jaccard`, `common_words`, `color_match`, `fuzzy`, `bigram_overlap`
- Kaggle rank: `TBD`
- Training time: `TBD`

After running the pipeline, use **`outputs/runs/<run_id>/results_summary.json`** as the source of truth for metrics and artifact paths, and **`outputs/runs/<run_id>/feature_set_evaluation_results.csv`** for the feature-set sweep table when refreshing this section from a new run.

## Methodology

- Text preprocessing with stemming on query/title/description fields
- Handpicked and configurable NLP feature sets via registry + YAML presets
- Optional model comparison (Random Forest, Gradient Boosting, SVR, KNN, and optional HistGradientBoosting) via `run compare-models` or `run full-pipeline --compare-models`
- Two-finalist hyperparameter search: `HistGradientBoostingRegressor` and `RandomForestRegressor` with `RandomizedSearchCV` (5-fold CV); the better finalist by CV RMSE drives feature evaluation and submission
- Statistical comparison between the two finalists using a paired t-test on per-fold RMSE

## Project Structure

```text
src/
  __init__.py
  data_loader.py
  feature_engineering.py
  modeling.py
  evaluation.py
main.py
cli.py
home_depot.py
requirements.txt
```

- `cli.py`: canonical command-line interface using Click
- `main.py`: Python entrypoint delegating to shared pipeline orchestration
- `home_depot.py`: backward-compatible wrapper for legacy command usage
- `outputs/runs/<run_id>/results_summary.json`: schema-validated run summary
- `outputs/runs/<run_id>/config_used.json`: exact run configuration
- `outputs/runs/<run_id>/logs/run.log`: consolidated run log (INFO/WARNING/ERROR)

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

If you hit binary-extension errors with NumPy 2.x (for example `pyarrow`, `numexpr`), use a fresh venv or pin `numpy<2` until the rest of your stack catches up.

## Run

```bash
python cli.py run full-pipeline --data-dir home-depot-product-search-relevance --output-dir outputs
```

Full pipeline skips the broad model comparison by default (faster). After changing features or data, rerun a survey occasionally with **`--compare-models`**, or run **`python cli.py run compare-models`** as a separate stage.

Other useful commands (each stage that writes files uses **`outputs/runs/<run_id>/`** under `--output-dir`):

```bash
python cli.py run baseline
python cli.py run compare-models
python cli.py run tune --n-iter 20
python cli.py run feature-search --feature-mode yaml --feature-config-path configs/features.yaml
```

**Golden path** (baseline then full pipeline; open the printed `results_summary.json` path after full-pipeline):

```bash
python cli.py run baseline --data-dir home-depot-product-search-relevance --output-dir outputs
python cli.py run full-pipeline --data-dir home-depot-product-search-relevance --output-dir outputs
```

Legacy entrypoint still works: `python home_depot.py`

### Browse run results (optional)

`streamlit` is included in `requirements.txt`. After install, run:

```bash
streamlit run streamlit_app.py
```

The app only lists runs that include `results_summary.json` (for example from `run full-pipeline`). It does not upload data.

## Outputs Contract

Each full run writes into:

`outputs/runs/<run_id>/`

Key files:
- `results_summary.json` (validated against `schemas/results_summary.schema.json`)
- `config_used.json`
- `feature_set_evaluation_results.csv`
- `new_feature_benchmarks.csv`
- `submission.csv`
- `logs/run.log`

Stages may also write stage-specific files in the same run directory, for example `model_comparison.csv` (compare-models), `tune_best_params.json` (tune; includes both finalists and `selected_finalist`), or `feature_search_results.csv` (feature-search).

The `results_summary.json` includes `schema_version`, `run_id`, `git_sha`, metrics, artifact paths, and run context metadata.

## Data

Competition data is not committed due to size. Download it from [Kaggle](https://www.kaggle.com/c/home-depot-product-search-relevance/) and place files in:

`home-depot-product-search-relevance/`

Expected files:
- `train.csv`
- `test.csv`
- `attributes.csv`
- `product_descriptions.csv`
