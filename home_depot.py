"""Backward-compatible entrypoint for the old script name."""

from main import main

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"An error occurred: {error}")