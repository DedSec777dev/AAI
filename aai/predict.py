"""Backward-compatible training entry point.

The original prototype trained models from this file. The workflow now lives in
train_model.py, but this wrapper keeps `python predict.py` working.
"""

from train_model import train_and_save_models


if __name__ == "__main__":
    results = train_and_save_models()
    print("Training complete.")
    print(results)
