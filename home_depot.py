"""Backward-compatible entrypoint for the old script name."""

from loguru import logger

from main import main

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        logger.exception(f"An error occurred: {error}")