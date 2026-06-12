"""Data cleaning and safe time preprocessing for flight delay modeling."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from utils import validate_columns


BASE_REQUIRED_COLUMNS = ["Company", "Origin", "Destination", "Departure Time"]


def parse_departure_hour(value: object) -> float:
    """Parse a departure time value and return hour, or NaN for malformed values."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    if not text:
        return np.nan

    for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M%p"):
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if not pd.isna(parsed):
            return float(parsed.hour)

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return np.nan
    return float(parsed.hour)


def get_time_block(hour: float | int) -> str:
    """Map a 24-hour departure hour to a readable time block."""
    hour_int = int(hour)
    if 5 <= hour_int < 12:
        return "Morning"
    if 12 <= hour_int < 17:
        return "Afternoon"
    if 17 <= hour_int < 21:
        return "Evening"
    return "Night"


def _coerce_operational_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure synthetic operational columns are numeric if already present."""
    for column in ("Weather_Severity", "Airspace_Congestion", "Delay_Duration", "Is_Delayed"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def preprocess_flight_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean flight records, validate time values, and add base time features."""
    validate_columns(df, BASE_REQUIRED_COLUMNS)

    clean_df = df.copy()
    before_dedup = len(clean_df)
    clean_df = clean_df.drop_duplicates().reset_index(drop=True)
    logging.info("Removed %s duplicate rows.", before_dedup - len(clean_df))

    for column in ("Company", "Origin", "Destination"):
        clean_df[column] = clean_df[column].astype("string").str.strip()
        clean_df[column] = clean_df[column].replace("", pd.NA).fillna("Unknown")

    clean_df["Departure Time"] = clean_df["Departure Time"].astype("string").str.strip()
    clean_df["Dep_Hour"] = clean_df["Departure Time"].map(parse_departure_hour)

    malformed_count = int(clean_df["Dep_Hour"].isna().sum())
    if malformed_count:
        logging.warning("Dropping %s rows with malformed departure times.", malformed_count)
        clean_df = clean_df.dropna(subset=["Dep_Hour"]).copy()

    clean_df["Dep_Hour"] = clean_df["Dep_Hour"].astype(int)
    clean_df["Time_Block"] = clean_df["Dep_Hour"].map(get_time_block)
    clean_df = _coerce_operational_columns(clean_df)

    return clean_df.reset_index(drop=True)
