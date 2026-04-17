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
- Handpicked NLP feature sets from 1 to 12 features
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
home_depot.py
requirements.txt
```

- `main.py`: canonical pipeline entrypoint
- `home_depot.py`: backward-compatible wrapper for legacy command usage
- `results.json`: machine-readable run summary (created after pipeline execution)
- `feature_set_evaluation_results.csv`: per-feature-set metrics (created after pipeline execution)

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Legacy entrypoint still works:

```bash
python home_depot.py
```

## Data

Competition data is not committed due to size. Download it from [Kaggle](https://www.kaggle.com/c/home-depot-product-search-relevance/) and place files in:

`home-depot-product-search-relevance/`

Expected files:
- `train.csv`
- `test.csv`
- `attributes.csv`
- `product_descriptions.csv`
