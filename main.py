"""Main orchestration entrypoint."""

import warnings

from src.pipeline import run_full_pipeline


def main():
    """Run the full training and evaluation pipeline."""
    warnings.filterwarnings("ignore")
    summary = run_full_pipeline()
    print("\nSaved reproducible run summary to outputs/results.json")
    print(f"Final model: {summary['best_model']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"An error occurred: {error}")