"""Shared utilities for dataset loading, persistence, and validation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

import kagglehub
import pandas as pd


DATASET_SLUG = "dhairya903/flights-in-india"
SAVED_MODELS_DIR = Path("saved_models")
RANDOM_STATE = 42


def configure_logging() -> None:
    """Configure application logging once."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def download_dataset(dataset_slug: str = DATASET_SLUG) -> str:
    """Download the Kaggle dataset using the existing KaggleHub workflow."""
    logging.info("Downloading dataset from KaggleHub: %s", dataset_slug)
    return kagglehub.dataset_download(dataset_slug)


def find_csv_file(dataset_path: str | os.PathLike[str]) -> Path:
    """Return the first CSV file found inside the KaggleHub dataset directory."""
    path = Path(dataset_path)
    csv_files = sorted(path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in dataset directory: {path}")
    return csv_files[0]


def load_flights_dataset() -> pd.DataFrame:
    """Download and load the Indian flights dataset."""
    dataset_path = download_dataset()
    csv_path = find_csv_file(dataset_path)
    logging.info("Loading flight data from: %s", csv_path)
    return pd.read_csv(csv_path)


def validate_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Raise a clean error if required columns are missing."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def ensure_saved_models_dir() -> Path:
    """Create and return the model artifact directory."""
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return SAVED_MODELS_DIR
