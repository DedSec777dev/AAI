"""
SkyPulse Configuration — API Keys & Constants.

Environment variables take precedence over hardcoded defaults.
This file should NEVER be committed to version control.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# OpenSky Network API Credentials
# Docs: https://openskynetwork.github.io/opensky-api/rest.html
# ---------------------------------------------------------------------------
OPENSKY_USERNAME: str = os.getenv(
    "OPENSKY_USERNAME", "skyclear909@gmail.com-api-client"
)
OPENSKY_PASSWORD: str = os.getenv(
    "OPENSKY_PASSWORD", "fnP5KQPY35dZsPIMihCeC5QYpZf0CAuI"
)
OPENSKY_BASE_URL: str = "https://opensky-network.org/api"

# India bounding box (lat_min, lon_min, lat_max, lon_max)
OPENSKY_BBOX: dict[str, float] = {
    "lamin": 6.5,
    "lomin": 68.0,
    "lamax": 35.5,
    "lomax": 97.5,
}

# Polling interval in seconds (authenticated limit ~40 req/min)
OPENSKY_POLL_INTERVAL: int = int(os.getenv("OPENSKY_POLL_INTERVAL", "12"))

# ---------------------------------------------------------------------------
# SearchApi / SerpApi — Google Flights Engine
# SearchApi docs: https://www.searchapi.io/docs/google-flights
# SerpApi docs: https://serpapi.com/google-flights-api
# ---------------------------------------------------------------------------
FLIGHT_SEARCH_PROVIDER: str = os.getenv("FLIGHT_SEARCH_PROVIDER", "serpapi").lower()
FLIGHT_SEARCH_API_KEY: str = os.getenv(
    "SERPAPI_KEY", "8af16aa45781657d8e9153856ce458c6a3b26723e2e33c38c13ac167e770b8e5"
)
FLIGHT_SEARCH_BASE_URL: str = os.getenv(
    "FLIGHT_SEARCH_BASE_URL",
    "https://serpapi.com/search.json",
)

# Backwards-compatible aliases for older imports/env names.
SERPAPI_KEY: str = FLIGHT_SEARCH_API_KEY
SERPAPI_BASE_URL: str = FLIGHT_SEARCH_BASE_URL

# Cache TTL for flight search results (seconds)
SERPAPI_CACHE_TTL: int = int(os.getenv("SERPAPI_CACHE_TTL", "600"))

# ---------------------------------------------------------------------------
# OpenWeatherMap
# ---------------------------------------------------------------------------
OWM_API_KEY: str = os.getenv("OWM_API_KEY", "7e8fcb860df6eabb0057440ef100c59c")

# ---------------------------------------------------------------------------
# IATA-to-city mapping for SerpApi (SerpApi uses IATA codes)
# ---------------------------------------------------------------------------
AIRPORT_IATA_MAP: dict[str, str] = {
    "DEL": "DEL",
    "BOM": "BOM",
    "BLR": "BLR",
    "MAA": "MAA",
    "CCU": "CCU",
    "HYD": "HYD",
    "COK": "COK",
    "GOI": "GOI",
    "PNQ": "PNQ",
    "AMD": "AMD",
    "JAI": "JAI",
    "LKO": "LKO",
    "GAU": "GAU",
    "IXC": "IXC",
    "PAT": "PAT",
    "VNS": "VNS",
    "IXB": "IXB",
    "SXR": "SXR",
    "IDR": "IDR",
    "BBI": "BBI",
}
