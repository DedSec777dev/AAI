"""Feature engineering for delay classification and duration regression."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np
import pandas as pd

from utils import RANDOM_STATE


FEATURE_COLUMNS = [
    "Company",
    "Origin",
    "Destination",
    "Route",
    "Dep_Hour",
    "Time_Block",
    "Weather_Severity",
    "Airspace_Congestion",
    "Peak_Hour_Flag",
    "Weekend_Flag",
    "Flight_Density_Score",
    "Traffic_Risk",
    "Weather_Risk",
]

CATEGORICAL_FEATURES = ["Company", "Origin", "Destination", "Route", "Time_Block"]
NUMERIC_FEATURES = [
    "Dep_Hour",
    "Weather_Severity",
    "Airspace_Congestion",
    "Peak_Hour_Flag",
    "Weekend_Flag",
    "Flight_Density_Score",
    "Traffic_Risk",
    "Weather_Risk",
]


def route_key(origin: object, destination: object) -> str:
    """Build a stable route feature from origin and destination."""
    return f"{str(origin).strip()}_{str(destination).strip()}"


def build_route_density_map(df: pd.DataFrame) -> dict[str, float]:
    """Create normalized route-density scores from the training data."""
    route_counts = df["Route"].value_counts()
    max_count = float(route_counts.max()) if not route_counts.empty else 1.0
    return (route_counts / max_count).round(6).to_dict()


def _derive_weekend_flag(df: pd.DataFrame) -> pd.Series:
    """Use a date column when available; otherwise default to non-weekend."""
    for column in ("Date", "Flight Date", "Journey Date", "Departure Date"):
        if column in df.columns:
            parsed = pd.to_datetime(df[column], errors="coerce", dayfirst=True)
            return parsed.dt.dayofweek.isin([5, 6]).fillna(False).astype(int)
    return pd.Series(np.zeros(len(df), dtype=int), index=df.index)


def add_synthetic_operational_features(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve the prototype's weather/congestion simulation and delay targets."""
    engineered = df.copy()
    rng = np.random.default_rng(RANDOM_STATE)

    if "Weather_Severity" not in engineered.columns or engineered["Weather_Severity"].isna().any():
        engineered["Weather_Severity"] = rng.integers(1, 11, size=len(engineered))

    if "Airspace_Congestion" not in engineered.columns or engineered["Airspace_Congestion"].isna().any():
        engineered["Airspace_Congestion"] = rng.integers(1, 11, size=len(engineered))

    engineered["Weather_Severity"] = (
        pd.to_numeric(engineered["Weather_Severity"], errors="coerce").fillna(5).clip(1, 10)
    )
    engineered["Airspace_Congestion"] = (
        pd.to_numeric(engineered["Airspace_Congestion"], errors="coerce").fillna(5).clip(1, 10)
    )

    peak_hour_signal = engineered["Dep_Hour"].isin([7, 8, 9, 17, 18, 19, 20]).astype(float)
    evening_signal = (engineered["Dep_Hour"] >= 17).astype(float)
    weather_score = engineered["Weather_Severity"].astype(float) / 10.0
    congestion_score = engineered["Airspace_Congestion"].astype(float) / 10.0

    if "Is_Delayed" not in engineered.columns or engineered["Is_Delayed"].isna().any():
        delay_risk_score = (
            weather_score * 0.34
            + congestion_score * 0.38
            + peak_hour_signal * 0.18
            + evening_signal * 0.10
        )
        delay_probability_signal = 1.0 / (1.0 + np.exp(-(delay_risk_score - 0.48) / 0.14))
        delay_probability_signal = np.clip(delay_probability_signal, 0.05, 0.95)
        engineered["Is_Delayed"] = rng.binomial(1, delay_probability_signal).astype(int)

    if "Delay_Duration" not in engineered.columns or engineered["Delay_Duration"].isna().any():
        delayed_delay = (
            8.0
            + engineered["Weather_Severity"].astype(float) * 1.4
            + engineered["Airspace_Congestion"].astype(float) * 1.2
            + peak_hour_signal * 3.0
            + evening_signal * 4.0
            + rng.normal(0, 3.0, size=len(engineered))
        )
        delayed_delay = np.clip(delayed_delay, 15.0, 90.0)
        minor_operational_delay = rng.uniform(0.0, 12.0, size=len(engineered))
        engineered["Delay_Duration"] = np.where(
            engineered["Is_Delayed"].astype(int) == 1,
            delayed_delay,
            minor_operational_delay,
        ).round(2)

    return engineered


def engineer_features(
    df: pd.DataFrame,
    route_density_map: dict[str, float] | None = None,
    fit_density: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Add model features and return the engineered frame plus route-density map."""
    engineered = df.copy()
    engineered["Route"] = [
        route_key(origin, destination)
        for origin, destination in zip(engineered["Origin"], engineered["Destination"])
    ]
    engineered["Peak_Hour_Flag"] = engineered["Dep_Hour"].isin([7, 8, 9, 17, 18, 19, 20]).astype(int)
    engineered["Weekend_Flag"] = _derive_weekend_flag(engineered)

    if fit_density or route_density_map is None:
        route_density_map = build_route_density_map(engineered)

    default_density = float(np.mean(list(route_density_map.values()))) if route_density_map else 0.0
    engineered["Flight_Density_Score"] = (
        engineered["Route"].map(route_density_map).fillna(default_density).astype(float)
    )

    engineered["Traffic_Risk"] = (
        (engineered["Airspace_Congestion"].astype(float) / 10.0) * 0.7
        + engineered["Peak_Hour_Flag"].astype(float) * 0.2
        + engineered["Flight_Density_Score"].astype(float) * 0.1
    ).clip(0, 1)
    engineered["Weather_Risk"] = (engineered["Weather_Severity"].astype(float) / 10.0).clip(0, 1)

    logging.info("Engineered %s model features.", len(FEATURE_COLUMNS))
    return engineered, route_density_map


def build_single_flight_frame(
    company: str,
    origin: str,
    destination: str,
    dep_hour: int,
    time_block: str,
    weather_severity: float,
    congestion: float,
    route_density_map: dict[str, float],
) -> pd.DataFrame:
    """Create a one-row feature frame for inference."""
    row: dict[str, Any] = {
        "Company": company,
        "Origin": origin,
        "Destination": destination,
        "Departure Time": f"{dep_hour:02d}:00",
        "Dep_Hour": dep_hour,
        "Time_Block": time_block,
        "Weather_Severity": weather_severity,
        "Airspace_Congestion": congestion,
    }
    frame = pd.DataFrame([row])
    engineered, _ = engineer_features(frame, route_density_map=route_density_map, fit_density=False)
    return engineered[FEATURE_COLUMNS]


def stable_hash(value: str) -> int:
    """Return a deterministic integer hash for future feature extensions."""
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)
