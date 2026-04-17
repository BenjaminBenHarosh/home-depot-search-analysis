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

After running the pipeline, use `results.json` and `feature_set_evaluation_results.csv` as the source of truth for updating this section.

## Methodology

- Text preprocessing with stemming on query/title/description fields
- Handpicked and configurable NLP feature sets via registry + YAML presets
- Model comparison across Random Forest, Gradient Boosting, SVR, and KNN
- Hyperparameter tuning with `RandomizedSearchCV` and 5-fold cross-validation
- Statistical comparison using paired t-test

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

## Run

```bash
python cli.py run full-pipeline --data-dir home-depot-product-search-relevance --output-dir outputs
```

Other useful commands:

```bash
python cli.py run baseline
python cli.py run compare-models
python cli.py run tune --n-iter 20
python cli.py run feature-search --feature-mode yaml --feature-config-path configs/features.yaml
```

Legacy entrypoint still works: `python home_depot.py`

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

The `results_summary.json` includes `schema_version`, `run_id`, `git_sha`, metrics, artifact paths, and run context metadata.

## Data

Competition data is not committed due to size. Download it from [Kaggle](https://www.kaggle.com/c/home-depot-product-search-relevance/) and place files in:

`home-depot-product-search-relevance/`

Expected files:
- `train.csv`
- `test.csv`
- `attributes.csv`
- `product_descriptions.csv`
