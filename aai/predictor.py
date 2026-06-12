"""Reusable inference utility for flight delay predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from feature_engineering import build_single_flight_frame
from preprocess import get_time_block, parse_departure_hour
from utils import SAVED_MODELS_DIR


PIPELINE_PATH = SAVED_MODELS_DIR / "pipeline.pkl"


def _load_pipeline(path: Path = PIPELINE_PATH) -> dict[str, Any]:
    """Load the saved classifier/regressor pipeline bundle."""
    if not path.exists():
        raise FileNotFoundError(
            f"Pipeline artifact not found at {path}. Run `python train_model.py` first."
        )
    return joblib.load(path)


def _risk_level(probability: float) -> str:
    """Convert delay probability to a business-friendly risk label."""
    if probability < 0.30:
        return "Low"
    if probability < 0.70:
        return "Medium"
    return "High"


def _clip_score(value: float, name: str) -> float:
    """Validate and clip operational scores to the expected 1-10 range."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric value from 1 to 10.") from exc
    return max(1.0, min(10.0, numeric))


def predict_flight_delay(
    company: str,
    origin: str,
    destination: str,
    departure_time: str,
    weather_severity: float,
    congestion: float,
) -> dict[str, float | str]:
    """Predict delay probability, expected delay duration, and risk level."""
    dep_hour = parse_departure_hour(departure_time)
    if dep_hour != dep_hour:
        raise ValueError("Invalid departure_time. Use a format like '08:30' or '6:45 PM'.")

    bundle = _load_pipeline()
    classifier = bundle["classifier"]
    regressor = bundle["regressor"]
    route_density_map = bundle.get("route_density_map", {})

    feature_frame = build_single_flight_frame(
        company=company,
        origin=origin,
        destination=destination,
        dep_hour=int(dep_hour),
        time_block=get_time_block(int(dep_hour)),
        weather_severity=_clip_score(weather_severity, "weather_severity"),
        congestion=_clip_score(congestion, "congestion"),
        route_density_map=route_density_map,
    )

    if hasattr(classifier, "predict_proba"):
        delay_probability = float(classifier.predict_proba(feature_frame)[0][1])
    else:
        delay_probability = float(classifier.predict(feature_frame)[0])

    predicted_minutes = max(0.0, float(regressor.predict(feature_frame)[0]))
    return {
        "delay_probability": round(delay_probability, 4),
        "predicted_delay_minutes": round(predicted_minutes, 2),
        "risk_level": _risk_level(delay_probability),
    }
