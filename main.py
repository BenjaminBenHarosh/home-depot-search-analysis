"""Main orchestration entrypoint."""

import warnings

from loguru import logger

from src.pipeline import run_full_pipeline


def main():
    """Run the full training and evaluation pipeline."""
    warnings.filterwarnings("ignore")
    summary = run_full_pipeline()
    logger.info("Saved reproducible run summary under outputs/runs/<run_id>/results_summary.json")
    logger.info(f"Final model: {summary['best_model']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        logger.exception(f"An error occurred: {error}")