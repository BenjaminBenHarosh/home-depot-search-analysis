# Home Depot Product Search Relevance

This project builds a machine learning model to predict the relevance of a product to a user's search query on HomeDepot.com. It is based on the Kaggle competition: [Home Depot Product Search Relevance](https://www.kaggle.com/c/home-depot-product-search-relevance/).

The dataset is not included in this repository due to file size limitations.  
You can download it from the Kaggle competition page and place the files in the `home-depot-product-search-relevance/` directory.

## Project Structure

- `home_depot.py` — main script for data loading, preprocessing, feature engineering, model training, and evaluation.
- `home-depot-product-search-relevance/` — directory containing all input datasets:
  - `train.csv`
  - `test.csv`
  - `product_descriptions.csv`
  - `attributes.csv`
- `sample_submission.csv` — template for formatting prediction results for submission.
- `requirements.txt` — list of required packages.

## Setup and Usage

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the script

```bash
python home_depot.py
```
