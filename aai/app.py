"""
Flask backend for the SkyPulse Flight Operations Dashboard.

Serves a FlightAware-style frontend with:
  - Live ADS-B tracking via OpenSky Network API
  - Real-time flight pricing via SearchApi/SerpApi Google Flights
  - Historical reliability scoring (Pandas-based synthetic dataset)
  - Live weather overlay via OpenWeatherMap API proxy
  - ML-powered delay prediction (existing pipeline)
"""

from __future__ import annotations

import math
import random
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request

from config import (
    FLIGHT_SEARCH_API_KEY,
    FLIGHT_SEARCH_BASE_URL,
    FLIGHT_SEARCH_PROVIDER,
    OWM_API_KEY,
    OPENSKY_BASE_URL,
    OPENSKY_BBOX,
    OPENSKY_PASSWORD,
    OPENSKY_POLL_INTERVAL,
    OPENSKY_USERNAME,
    SERPAPI_CACHE_TTL,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)

# ---------------------------------------------------------------------------
# Airport reference data with geographic coordinates (Indian domestic network)
# ---------------------------------------------------------------------------
AIRPORTS: list[dict] = [
    {"code": "DEL", "name": "Indira Gandhi International",              "city": "New Delhi",   "lat": 28.5562, "lon": 77.1000},
    {"code": "BOM", "name": "Chhatrapati Shivaji Maharaj International", "city": "Mumbai",      "lat": 19.0896, "lon": 72.8656},
    {"code": "BLR", "name": "Kempegowda International",                  "city": "Bengaluru",   "lat": 13.1986, "lon": 77.7066},
    {"code": "MAA", "name": "Chennai International",                     "city": "Chennai",     "lat": 12.9941, "lon": 80.1709},
    {"code": "CCU", "name": "Netaji Subhas Chandra Bose International",  "city": "Kolkata",     "lat": 22.6547, "lon": 88.4467},
    {"code": "HYD", "name": "Rajiv Gandhi International",                "city": "Hyderabad",   "lat": 17.2403, "lon": 78.4294},
    {"code": "COK", "name": "Cochin International",                      "city": "Kochi",       "lat": 10.1520, "lon": 76.4019},
    {"code": "GOI", "name": "Manohar International",                     "city": "Goa",         "lat": 15.3808, "lon": 73.8314},
    {"code": "PNQ", "name": "Pune Airport",                              "city": "Pune",        "lat": 18.5822, "lon": 73.9197},
    {"code": "AMD", "name": "Sardar Vallabhbhai Patel International",    "city": "Ahmedabad",   "lat": 23.0772, "lon": 72.6347},
    {"code": "JAI", "name": "Jaipur International",                      "city": "Jaipur",      "lat": 26.8242, "lon": 75.8122},
    {"code": "LKO", "name": "Chaudhary Charan Singh International",      "city": "Lucknow",     "lat": 26.7606, "lon": 80.8893},
    {"code": "GAU", "name": "Lokpriya Gopinath Bordoloi International",  "city": "Guwahati",    "lat": 26.1061, "lon": 91.5859},
    {"code": "IXC", "name": "Chandigarh Airport",                        "city": "Chandigarh",  "lat": 30.6735, "lon": 76.7885},
    {"code": "PAT", "name": "Jay Prakash Narayan International",         "city": "Patna",       "lat": 25.5913, "lon": 85.0880},
    {"code": "VNS", "name": "Lal Bahadur Shastri International",         "city": "Varanasi",    "lat": 25.4524, "lon": 82.8593},
    {"code": "IXB", "name": "Bagdogra Airport",                          "city": "Bagdogra",    "lat": 26.6812, "lon": 88.3286},
    {"code": "SXR", "name": "Sheikh ul-Alam International",              "city": "Srinagar",    "lat": 33.9871, "lon": 74.7742},
    {"code": "IDR", "name": "Devi Ahilyabai Holkar Airport",             "city": "Indore",      "lat": 22.7217, "lon": 75.8011},
    {"code": "BBI", "name": "Biju Patnaik International",                "city": "Bhubaneswar", "lat": 20.2444, "lon": 85.8178},
]

# Quick lookup by airport code
_AIRPORT_MAP: dict[str, dict] = {a["code"]: a for a in AIRPORTS}

# ---------------------------------------------------------------------------
# Route connectivity — defines which airports connect to each other.
# ---------------------------------------------------------------------------
ROUTE_CONNECTIONS: dict[str, list[str]] = {
    "DEL": ["BOM", "BLR", "MAA", "CCU", "HYD", "COK", "GOI", "PNQ", "AMD", "JAI", "LKO", "GAU", "IXC", "PAT", "VNS", "IXB", "SXR", "IDR", "BBI"],
    "BOM": ["DEL", "BLR", "MAA", "CCU", "HYD", "COK", "GOI", "PNQ", "AMD", "JAI", "LKO", "GAU", "IDR", "PAT", "BBI"],
    "BLR": ["DEL", "BOM", "MAA", "CCU", "HYD", "COK", "GOI", "PNQ", "AMD", "JAI", "LKO", "GAU", "IDR", "BBI"],
    "MAA": ["DEL", "BOM", "BLR", "CCU", "HYD", "COK", "PNQ", "BBI"],
    "CCU": ["DEL", "BOM", "BLR", "MAA", "HYD", "GAU", "PAT", "IXB", "BBI", "VNS"],
    "HYD": ["DEL", "BOM", "BLR", "MAA", "CCU", "COK", "GOI", "PNQ", "AMD", "BBI", "IDR"],
    "COK": ["DEL", "BOM", "BLR", "MAA", "HYD", "GOI"],
    "GOI": ["DEL", "BOM", "BLR", "HYD", "COK", "PNQ"],
    "PNQ": ["DEL", "BOM", "BLR", "HYD", "GOI"],
    "AMD": ["DEL", "BOM", "BLR", "HYD", "JAI"],
    "JAI": ["DEL", "BOM", "BLR", "AMD", "LKO"],
    "LKO": ["DEL", "BOM", "BLR", "JAI", "PAT", "VNS"],
    "GAU": ["DEL", "BOM", "BLR", "CCU", "IXB"],
    "IXC": ["DEL", "BOM", "SXR"],
    "PAT": ["DEL", "BOM", "CCU", "LKO", "VNS"],
    "VNS": ["DEL", "CCU", "LKO", "PAT"],
    "IXB": ["DEL", "CCU", "GAU"],
    "SXR": ["DEL", "IXC", "JAI"],
    "IDR": ["DEL", "BOM", "BLR", "HYD"],
    "BBI": ["DEL", "BOM", "BLR", "MAA", "CCU", "HYD"],
}

# ---------------------------------------------------------------------------
# Airline data
# ---------------------------------------------------------------------------
AIRLINES: list[dict[str, str]] = [
    {"code": "6E", "name": "IndiGo"},
    {"code": "AI", "name": "Air India"},
    {"code": "SG", "name": "SpiceJet"},
    {"code": "UK", "name": "Vistara"},
    {"code": "G8", "name": "Go First"},
    {"code": "QP", "name": "Akasa Air"},
    {"code": "IX", "name": "Air India Express"},
    {"code": "I5", "name": "AirAsia India"},
]

# Airline-specific reliability profiles (base on-time percentage)
AIRLINE_RELIABILITY: dict[str, float] = {
    "IndiGo": 0.78,
    "Air India": 0.70,
    "SpiceJet": 0.62,
    "Vistara": 0.82,
    "Go First": 0.65,
    "Akasa Air": 0.80,
    "Air India Express": 0.68,
    "AirAsia India": 0.66,
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. LIVE ADS-B TRACKING — OpenSky Network API
# ═══════════════════════════════════════════════════════════════════════════

_adsb_lock = threading.Lock()
_live_aircraft: list[dict] = []
_adsb_stale: bool = False
_adsb_last_success: float = 0.0
_adsb_error_message: str = ""


def fetch_live_airspace() -> tuple[list[dict], bool]:
    """
    Fetch real-time aircraft state vectors from OpenSky Network
    for the India bounding box.

    Returns
    -------
    tuple[list[dict], bool]
        (aircraft_list, is_stale) — is_stale is True if data is from cache
        due to an API error.
    """
    global _live_aircraft, _adsb_stale, _adsb_last_success, _adsb_error_message

    try:
        resp = requests.get(
            f"{OPENSKY_BASE_URL}/states/all",
            params={
                "lamin": OPENSKY_BBOX["lamin"],
                "lomin": OPENSKY_BBOX["lomin"],
                "lamax": OPENSKY_BBOX["lamax"],
                "lomax": OPENSKY_BBOX["lomax"],
            },
            auth=(OPENSKY_USERNAME, OPENSKY_PASSWORD),
            timeout=10,
        )

        if resp.status_code == 429:
            app.logger.warning("OpenSky rate-limited (429). Serving cached data.")
            _adsb_stale = True
            _adsb_error_message = "Rate-limited by OpenSky API"
            return _live_aircraft, True

        if resp.status_code != 200:
            app.logger.warning("OpenSky error %s: %s", resp.status_code, resp.text[:200])
            _adsb_stale = True
            _adsb_error_message = f"OpenSky returned HTTP {resp.status_code}"
            return _live_aircraft, True

        data = resp.json()
        states = data.get("states") or []

        aircraft = []
        for sv in states:
            # OpenSky state vector indices:
            # 0=icao24, 1=callsign, 2=origin_country, 3=time_position,
            # 4=last_contact, 5=longitude, 6=latitude, 7=baro_altitude,
            # 8=on_ground, 9=velocity, 10=true_track, 11=vertical_rate,
            # 12=sensors, 13=geo_altitude, 14=squawk, 15=spi, 16=position_source

            icao24 = sv[0]
            callsign = (sv[1] or "").strip()
            longitude = sv[5]
            latitude = sv[6]
            baro_alt = sv[7]       # meters
            on_ground = sv[8]
            velocity = sv[9]       # m/s
            true_track = sv[10]    # degrees from north
            vertical_rate = sv[11]

            # Skip aircraft on ground, or with missing position
            if on_ground or latitude is None or longitude is None:
                continue

            # Convert velocity m/s → knots
            speed_knots = round(velocity * 1.94384, 0) if velocity else 0
            # Convert altitude meters → feet
            alt_feet = round(baro_alt * 3.28084) if baro_alt else 0

            aircraft.append({
                "icao24": icao24,
                "callsign": callsign or icao24.upper(),
                "lat": round(latitude, 4),
                "lng": round(longitude, 4),
                "altitude": alt_feet,
                "speed_knots": int(speed_knots),
                "heading_degrees": round(true_track, 1) if true_track is not None else 0,
                "vertical_rate": round(vertical_rate, 1) if vertical_rate else 0,
                "on_ground": False,
            })

        with _adsb_lock:
            _live_aircraft = aircraft
            _adsb_stale = False
            _adsb_last_success = time.time()
            _adsb_error_message = ""

        app.logger.info("[OpenSky] Fetched %d airborne aircraft over India.", len(aircraft))
        return aircraft, False

    except requests.exceptions.Timeout:
        app.logger.warning("OpenSky API timeout. Serving cached data.")
        _adsb_stale = True
        _adsb_error_message = "OpenSky API timeout"
        return _live_aircraft, True

    except Exception as exc:
        app.logger.warning("OpenSky fetch error: %s", exc)
        _adsb_stale = True
        _adsb_error_message = str(exc)
        return _live_aircraft, True


def _opensky_polling_loop() -> None:
    """Background thread: poll OpenSky every OPENSKY_POLL_INTERVAL seconds."""
    # Initial delay to let Flask start up
    time.sleep(3)
    while True:
        try:
            fetch_live_airspace()
        except Exception as exc:
            app.logger.error("OpenSky polling loop error: %s", exc)
        time.sleep(OPENSKY_POLL_INTERVAL)


# Start OpenSky polling thread
_opensky_thread = threading.Thread(target=_opensky_polling_loop, daemon=True)
_opensky_thread.start()
print(f"[SkyPulse] OpenSky polling thread started (every {OPENSKY_POLL_INTERVAL}s)")


# ═══════════════════════════════════════════════════════════════════════════
# 2. FLIGHT PRICING — SearchApi/SerpApi Google Flights
# ═══════════════════════════════════════════════════════════════════════════

# Simple in-memory cache: key=(origin, dest, date) → (timestamp, data)
_serpapi_cache: dict[tuple[str, str, str], tuple[float, list[dict]]] = {}
_serpapi_cache_lock = threading.Lock()
_flight_search_last_error: str = ""


def _mask_api_key(api_key: str) -> str:
    """Return a short masked key for diagnostics."""
    if not api_key:
        return "missing"
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return f"{api_key[:4]}...{api_key[-4:]}"


def _extract_flight_search_error(data: dict) -> str:
    """Normalize common SearchApi/SerpApi error response shapes."""
    error = data.get("error") or data.get("message")
    if isinstance(error, str) and error.strip():
        return error.strip()

    errors = data.get("errors")
    if isinstance(errors, list) and errors:
        return "; ".join(str(item) for item in errors[:3])
    if isinstance(errors, dict) and errors:
        return "; ".join(f"{key}: {value}" for key, value in list(errors.items())[:3])

    return ""


def _sanitize_flight_search_error(message: str) -> str:
    """Remove API keys from provider errors before logging or returning them."""
    if not message:
        return ""
    return re.sub(r"([?&]api_key=)[^&\s)]+", r"\1***", message)


def _parse_price(value) -> int | None:
    """Normalize provider prices like 5120, "₹5,120", or {"value": 5120}."""
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("value", "price", "extracted_price"):
            parsed = _parse_price(value.get(key))
            if parsed is not None:
                return parsed
        return None
    if isinstance(value, (int, float)):
        return int(value) if value > 0 else None
    if isinstance(value, str):
        digits = re.sub(r"[^\d.]", "", value)
        if not digits:
            return None
        try:
            return int(float(digits))
        except ValueError:
            return None
    return None


def fetch_flight_offers(
    origin: str,
    destination: str,
    date: str,
) -> list[dict]:
    """
    Fetch real flight offers from a Google Flights search API engine.

    Parameters
    ----------
    origin : str
        IATA code (e.g. "DEL")
    destination : str
        IATA code (e.g. "BOM")
    date : str
        Outbound date in YYYY-MM-DD format

    Returns
    -------
    list[dict]
        Each dict: airline, flight_number, departure_time, arrival_time,
        price, duration, stops
    """
    global _flight_search_last_error

    cache_key = (origin, destination, date)

    # Check cache
    with _serpapi_cache_lock:
        cached = _serpapi_cache.get(cache_key)
        if cached:
            ts, data = cached
            if time.time() - ts < SERPAPI_CACHE_TTL:
                app.logger.info("[%s] Cache hit for %s→%s on %s", FLIGHT_SEARCH_PROVIDER, origin, destination, date)
                _flight_search_last_error = ""
                return data

    try:
        if not FLIGHT_SEARCH_API_KEY:
            _flight_search_last_error = "Flight search API key is missing."
            app.logger.warning("[%s] %s", FLIGHT_SEARCH_PROVIDER, _flight_search_last_error)
            return []

        params = {
            "engine": "google_flights",
            "departure_id": origin,
            "arrival_id": destination,
            "outbound_date": date,
            "currency": "INR",
            "hl": "en",
            "gl": "in",
            "api_key": FLIGHT_SEARCH_API_KEY,
        }
        if FLIGHT_SEARCH_PROVIDER == "searchapi":
            params["flight_type"] = "one_way"
        else:
            params["type"] = "2"  # SerpApi one-way

        resp = requests.get(FLIGHT_SEARCH_BASE_URL, params=params, timeout=15)

        try:
            data = resp.json()
        except ValueError:
            data = {}

        if resp.status_code != 200:
            provider_error = _extract_flight_search_error(data)
            _flight_search_last_error = _sanitize_flight_search_error(
                provider_error or f"HTTP {resp.status_code}: {resp.text[:200]}"
            )
            app.logger.warning(
                "[%s] Flight search error %s: %s",
                FLIGHT_SEARCH_PROVIDER,
                resp.status_code,
                _flight_search_last_error,
            )
            return []

        provider_error = _extract_flight_search_error(data)
        if provider_error:
            _flight_search_last_error = _sanitize_flight_search_error(provider_error)
            app.logger.warning("[%s] Flight search error: %s", FLIGHT_SEARCH_PROVIDER, _flight_search_last_error)
            return []

        flights = []

        # Parse "best_flights" and "other_flights" from the provider response.
        for section_key in ("best_flights", "other_flights"):
            section = data.get(section_key, [])
            for flight_group in section:
                flight_legs = flight_group.get("flights", [])
                price = _parse_price(
                    flight_group.get("price")
                    or flight_group.get("extracted_price")
                    or flight_group.get("price_info")
                )
                total_duration = flight_group.get("total_duration")
                stops = len(flight_legs) - 1

                if not flight_legs:
                    continue

                # Use first leg for primary flight info
                first_leg = flight_legs[0]
                departure_airport = first_leg.get("departure_airport", {})
                arrival_airport = flight_legs[-1].get("arrival_airport", {})

                airline_name = first_leg.get("airline", "Unknown")
                flight_number = first_leg.get("flight_number", "")

                dep_time = departure_airport.get("time", "")
                arr_time = arrival_airport.get("time", "")

                # Extract airline code from flight_number (e.g. "6E 2045" → "6E")
                airline_code = ""
                if flight_number:
                    parts = flight_number.split()
                    if parts:
                        airline_code = parts[0]

                flights.append({
                    "airline": airline_name,
                    "airline_code": airline_code,
                    "flight_number": flight_number.replace(" ", "-") if flight_number else f"--",
                    "departure_time": dep_time,
                    "arrival_time": arr_time,
                    "price": price,
                    "duration_minutes": total_duration,
                    "stops": stops,
                    "is_best": section_key == "best_flights",
                })

        # Store in cache
        with _serpapi_cache_lock:
            _serpapi_cache[cache_key] = (time.time(), flights)

        _flight_search_last_error = "" if flights else "No flight offers returned by provider."
        app.logger.info(
            "[%s] Fetched %d flight offers for %s→%s on %s",
            FLIGHT_SEARCH_PROVIDER, len(flights), origin, destination, date,
        )
        return flights

    except requests.exceptions.Timeout:
        _flight_search_last_error = "Flight search API timeout."
        app.logger.warning("[%s] timeout for %s→%s", FLIGHT_SEARCH_PROVIDER, origin, destination)
        return []

    except Exception as exc:
        _flight_search_last_error = _sanitize_flight_search_error(str(exc))
        app.logger.warning("[%s] error: %s", FLIGHT_SEARCH_PROVIDER, _flight_search_last_error)
        return []


# ═══════════════════════════════════════════════════════════════════════════
# 3. HISTORICAL RELIABILITY SCORE ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def generate_reliability_scores() -> pd.DataFrame:
    """
    Build a synthetic historical flight dataset covering the last 365 days.

    Generates ~50,000 flight records with airline-specific delay patterns
    to compute reliability scores (probability of on-time arrival).

    Returns a pre-processed DataFrame with per-airline and per-route scores.
    """
    np.random.seed(42)
    records = []

    airlines_list = [
        {"code": "6E", "name": "IndiGo"},
        {"code": "AI", "name": "Air India"},
        {"code": "SG", "name": "SpiceJet"},
        {"code": "UK", "name": "Vistara"},
    ]

    base_date = datetime.now() - timedelta(days=365)
    hours = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

    for airline in airlines_list:
        base_reliability = AIRLINE_RELIABILITY.get(airline["name"], 0.70)

        for origin_code, dest_codes in ROUTE_CONNECTIONS.items():
            for dest_code in dest_codes:
                # Generate 8-15 flights per route per airline over the year
                n_flights = random.randint(8, 15)
                for _ in range(n_flights):
                    day_offset = random.randint(0, 364)
                    flight_date = base_date + timedelta(days=day_offset)
                    dep_hour = random.choice(hours)
                    dep_minute = random.choice([0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])

                    scheduled = flight_date.replace(
                        hour=dep_hour, minute=dep_minute, second=0, microsecond=0
                    )

                    # Route-based difficulty factor
                    origin = _AIRPORT_MAP[origin_code]
                    dest = _AIRPORT_MAP[dest_code]
                    dist = math.sqrt(
                        (origin["lat"] - dest["lat"])**2 +
                        (origin["lon"] - dest["lon"])**2
                    )
                    route_difficulty = min(1.0, dist / 20.0)  # normalize

                    # Time-of-day effect (evening flights more delayed)
                    time_factor = 0.0
                    if dep_hour >= 18:
                        time_factor = 0.12
                    elif dep_hour >= 14:
                        time_factor = 0.06

                    # Season effect (monsoon: Jun-Sep more delayed)
                    month = flight_date.month
                    season_factor = 0.10 if 6 <= month <= 9 else 0.0

                    # Compute delay probability for this specific flight
                    delay_prob = (1 - base_reliability) + route_difficulty * 0.05 + time_factor + season_factor
                    delay_prob = min(0.95, max(0.02, delay_prob))

                    is_delayed = np.random.random() < delay_prob

                    if is_delayed:
                        # Delay follows a skewed distribution
                        delay_minutes = max(1, int(np.random.exponential(scale=25)))
                        delay_minutes = min(180, delay_minutes)
                    else:
                        # On-time or very slight delay
                        delay_minutes = max(0, int(np.random.normal(loc=2, scale=3)))

                    actual = scheduled + timedelta(minutes=delay_minutes)

                    records.append({
                        "Airline": airline["name"],
                        "AirlineCode": airline["code"],
                        "FlightNumber": f"{airline['code']}-{random.randint(100, 999)}",
                        "Origin": origin_code,
                        "Destination": dest_code,
                        "ScheduledDepartureTime": scheduled.strftime("%H:%M"),
                        "ActualDepartureTime": actual.strftime("%H:%M"),
                        "DelayMinutes": delay_minutes,
                        "Date": flight_date.strftime("%Y-%m-%d"),
                        "IsDelayed15": 1 if delay_minutes > 15 else 0,
                    })

    df = pd.DataFrame(records)
    return df


def _compute_reliability_lookup(df: pd.DataFrame) -> dict:
    """
    Pre-compute reliability scores as nested lookup:
      reliability_lookup[airline] -> overall score
      reliability_lookup[(airline, origin, dest)] -> route-specific score
    """
    lookup = {}

    # Per-airline overall
    for airline, group in df.groupby("Airline"):
        on_time_rate = 1 - group["IsDelayed15"].mean()
        lookup[airline] = round(on_time_rate * 100, 1)

    # Per-route per-airline
    for (airline, origin, dest), group in df.groupby(["Airline", "Origin", "Destination"]):
        on_time_rate = 1 - group["IsDelayed15"].mean()
        lookup[(airline, origin, dest)] = round(on_time_rate * 100, 1)

    return lookup


# Generate reliability data at startup
print("[SkyPulse] Generating historical reliability dataset...")
_reliability_df = generate_reliability_scores()
_reliability_lookup = _compute_reliability_lookup(_reliability_df)
print(f"[SkyPulse] Reliability engine ready: {len(_reliability_df)} historical records, {len(_reliability_lookup)} score entries.")


def get_reliability_score(airline_name: str, origin: str = None, dest: str = None) -> float:
    """
    Get reliability score for an airline, optionally route-specific.

    Returns a percentage (0-100). Falls back to airline-level if route not found.
    """
    if origin and dest:
        route_score = _reliability_lookup.get((airline_name, origin, dest))
        if route_score is not None:
            return route_score

    airline_score = _reliability_lookup.get(airline_name)
    if airline_score is not None:
        return airline_score

    # Fallback based on base reliability profile
    base = AIRLINE_RELIABILITY.get(airline_name, 0.70)
    return round(base * 100, 1)


# ═══════════════════════════════════════════════════════════════════════════
# 4. PREDICTION HELPER — wraps the real ML model
# ═══════════════════════════════════════════════════════════════════════════

def _generate_flight_number(airline_code: str) -> str:
    """Create a realistic flight number like '6E-2045'."""
    return f"{airline_code}-{random.randint(1000, 9999)}"


def _generate_departure_time(base_hour: int | None = None) -> str:
    """Return a random but realistic departure time string (HH:MM)."""
    if base_hour is None:
        base_hour = random.randint(5, 23)
    minute = random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
    return f"{base_hour:02d}:{minute:02d}"


def predict_delay(
    departure: str,
    destination: str,
    airline_name: str,
    departure_time: str,
) -> dict:
    """
    Call the trained ML pipeline and return prediction results.

    This function wraps `predictor.predict_flight_delay()`. Weather severity
    and airspace congestion are randomized to simulate real-time sensor data
    (replace with live feeds when available).

    Parameters
    ----------
    departure : str
        Origin airport city name (e.g. "New Delhi").
    destination : str
        Destination airport city name (e.g. "Mumbai").
    airline_name : str
        Airline company name (e.g. "IndiGo").
    departure_time : str
        Scheduled departure in "HH:MM" format.

    Returns
    -------
    dict
        Keys: delay_probability, predicted_delay_minutes, risk_level
    """
    try:
        from predictor import predict_flight_delay

        # Simulate real-time operational data (replace with live API feeds)
        weather_severity = round(random.uniform(1.0, 10.0), 1)
        congestion = round(random.uniform(1.0, 10.0), 1)

        result = predict_flight_delay(
            company=airline_name,
            origin=departure,
            destination=destination,
            departure_time=departure_time,
            weather_severity=weather_severity,
            congestion=congestion,
        )
        return result

    except FileNotFoundError:
        # Model not trained yet — return simulated prediction
        return _simulate_prediction()

    except Exception as exc:
        app.logger.warning("Model prediction failed, using simulation: %s", exc)
        return _simulate_prediction()


def _simulate_prediction() -> dict:
    """Fallback simulated prediction when the model is unavailable."""
    probability = round(random.uniform(0.05, 0.95), 4)
    minutes = round(probability * 60, 2) if probability > 0.3 else 0.0
    if probability < 0.30:
        risk = "Low"
    elif probability < 0.70:
        risk = "Medium"
    else:
        risk = "High"
    return {
        "delay_probability": probability,
        "predicted_delay_minutes": minutes,
        "risk_level": risk,
    }


def _build_flight_entry(
    airline: dict,
    source: dict,
    dest: dict,
    serpapi_flight: dict | None = None,
) -> dict:
    """
    Build a single flight record with ML-powered delay prediction,
    historical reliability score, and optional live pricing.
    """
    # Use live provider data if available, otherwise generate.
    if serpapi_flight:
        dep_time = serpapi_flight.get("departure_time", _generate_departure_time())
        # Extract just HH:MM from possible datetime strings
        if dep_time and len(dep_time) > 5:
            # Providers may return "2026-06-18 08:30" format.
            parts = dep_time.split()
            dep_time = parts[-1] if parts else dep_time
        dep_time = dep_time[:5] if dep_time else _generate_departure_time()
        arr_time = serpapi_flight.get("arrival_time", "")
        if arr_time and len(arr_time) > 5:
            parts = arr_time.split()
            arr_time = parts[-1] if parts else arr_time
        arr_time = arr_time[:5] if arr_time else ""
        flight_number = serpapi_flight.get("flight_number", _generate_flight_number(airline["code"]))
        price = serpapi_flight.get("price")
        duration = serpapi_flight.get("duration_minutes")
        stops = serpapi_flight.get("stops", 0)
        is_best = serpapi_flight.get("is_best", False)
    else:
        dep_time = _generate_departure_time()
        arr_time = ""
        flight_number = _generate_flight_number(airline["code"])
        price = None
        duration = None
        stops = 0
        is_best = False

    prediction = predict_delay(
        source["city"], dest["city"], airline["name"], dep_time
    )

    risk = prediction.get("risk_level", "Low")
    prob = prediction.get("delay_probability", 0)
    minutes = prediction.get("predicted_delay_minutes", 0)

    delay_minutes = max(1, int(round(minutes)))

    if risk == "Low":
        status = f"Low Risk (~{delay_minutes} min delay)"
        status_color = "green"
    elif risk == "Medium":
        status = f"Medium Risk (~{delay_minutes} min delay)"
        status_color = "amber"
    else:
        status = f"High Risk (~{delay_minutes} min delay)"
        status_color = "red"

    # Get reliability score — weighted composite
    historical_rel = get_reliability_score(airline["name"], source["code"], dest["code"])
    ml_ontime = (1 - prob) * 100  # convert probability to on-time percentage
    # Weighted: 60% historical, 40% ML
    composite_reliability = round(0.6 * historical_rel + 0.4 * ml_ontime, 1)
    composite_reliability = max(0, min(100, composite_reliability))

    # Determine reliability color
    if composite_reliability >= 85:
        rel_color = "green"
    elif composite_reliability >= 65:
        rel_color = "amber"
    else:
        rel_color = "red"

    return {
        "flight_number": flight_number,
        "airline": airline["name"],
        "airline_code": airline["code"],
        "departure_time": dep_time,
        "arrival_time": arr_time,
        "destination_code": dest["code"],
        "destination_city": dest["city"],
        "destination_name": dest["name"],
        "destination_lat": dest["lat"],
        "destination_lon": dest["lon"],
        "delay_probability": prob,
        "predicted_delay_minutes": minutes,
        "risk_level": risk,
        "status": status,
        "status_color": status_color,
        "reliability_score": composite_reliability,
        "reliability_color": rel_color,
        "historical_reliability": historical_rel,
        "price": price,
        "duration_minutes": duration,
        "stops": stops,
        "is_best": is_best,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4b. ROUTE INFERENCE — Match live aircraft to airport network
# ═══════════════════════════════════════════════════════════════════════════

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute bearing from point 1 to point 2 in degrees (0-360)."""
    dlon = math.radians(lon2 - lon1)
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2_r)
    y = (math.cos(lat1_r) * math.sin(lat2_r) -
         math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _angle_diff(a: float, b: float) -> float:
    """Absolute angular difference in degrees (0-180)."""
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def _infer_route(
    lat: float, lng: float, heading: float
) -> tuple[dict, dict, float] | None:
    """
    Infer the most likely origin and destination airports for an aircraft.

    Uses heading to determine which airports are "behind" (origin) and
    "ahead" (destination). Only returns a result if a plausible connected
    route exists in our ROUTE_CONNECTIONS network.

    Returns (origin_airport, dest_airport, progress_pct) or None.
    """
    if heading is None or heading == 0:
        return None

    reverse_heading = (heading + 180) % 360

    # Score each airport: how well does it match as origin (behind) or dest (ahead)?
    candidates_behind = []
    candidates_ahead = []

    for ap in AIRPORTS:
        dist = _haversine_km(lat, lng, ap["lat"], ap["lon"])
        if dist < 15:  # Skip airports very close (likely on the ground there)
            continue
        if dist > 2500:  # Skip airports too far away
            continue

        bearing_to_ap = _bearing_deg(lat, lng, ap["lat"], ap["lon"])

        # Airport is "ahead" if bearing to it is close to our heading
        ahead_diff = _angle_diff(bearing_to_ap, heading)
        # Airport is "behind" if bearing to it is close to reverse heading
        behind_diff = _angle_diff(bearing_to_ap, reverse_heading)

        if behind_diff < 60:  # Within 60° cone behind
            candidates_behind.append((behind_diff, dist, ap))
        if ahead_diff < 60:   # Within 60° cone ahead
            candidates_ahead.append((ahead_diff, dist, ap))

    if not candidates_behind or not candidates_ahead:
        return None

    # Sort by angular accuracy, then by distance
    candidates_behind.sort(key=lambda x: (x[0], x[1]))
    candidates_ahead.sort(key=lambda x: (x[0], x[1]))

    # Try to find a connected pair (origin → destination exists in our routes)
    for _, _, origin_ap in candidates_behind[:5]:
        for _, _, dest_ap in candidates_ahead[:5]:
            if origin_ap["code"] == dest_ap["code"]:
                continue
            connected = ROUTE_CONNECTIONS.get(origin_ap["code"], [])
            if dest_ap["code"] in connected:
                # Found a valid route — compute progress
                total_dist = _haversine_km(
                    origin_ap["lat"], origin_ap["lon"],
                    dest_ap["lat"], dest_ap["lon"],
                )
                from_origin = _haversine_km(
                    origin_ap["lat"], origin_ap["lon"],
                    lat, lng,
                )
                progress = min(99.0, max(1.0, (from_origin / max(total_dist, 1)) * 100))
                return origin_ap, dest_ap, progress

    return None


def _generate_arc_path(
    start: list[float], end: list[float], num_points: int = 40
) -> list[list[float]]:
    """Generate a curved arc path between two [lat, lon] points."""
    mid_lat = (start[0] + end[0]) / 2
    mid_lon = (start[1] + end[1]) / 2
    d_lat = end[0] - start[0]
    d_lon = end[1] - start[1]
    dist = math.sqrt(d_lat ** 2 + d_lon ** 2)
    if dist < 0.01:
        return [start, end]

    offset = dist * 0.12
    ctrl_lat = mid_lat + (-d_lon / dist) * offset
    ctrl_lon = mid_lon + (d_lat / dist) * offset

    points = []
    for i in range(num_points + 1):
        t = i / num_points
        u = 1 - t
        lat = u * u * start[0] + 2 * u * t * ctrl_lat + t * t * end[0]
        lon = u * u * start[1] + 2 * u * t * ctrl_lon + t * t * end[1]
        points.append([round(lat, 4), round(lon, 4)])

    return points


# ═══════════════════════════════════════════════════════════════════════════
# 5. ROUTES — API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the main frontend page."""
    return render_template("index.html")


@app.route("/api/airports")
def get_airports():
    """Return the full list of airports with coordinates for map markers."""
    return jsonify(AIRPORTS)


@app.route("/api/flight_search_status")
def get_flight_search_status():
    """Return non-secret flight search API configuration diagnostics."""
    return jsonify({
        "provider": FLIGHT_SEARCH_PROVIDER,
        "base_url": FLIGHT_SEARCH_BASE_URL,
        "api_key": _mask_api_key(FLIGHT_SEARCH_API_KEY),
        "api_key_configured": bool(FLIGHT_SEARCH_API_KEY),
        "last_error": _flight_search_last_error,
    })


@app.route("/api/get_connections", methods=["GET"])
def get_connections():
    """
    Return all outbound flights from a source airport with delay predictions
    and reliability scores.
    """
    source_code = request.args.get("source_airport", "").strip().upper()

    if not source_code:
        return jsonify({"error": "source_airport parameter is required."}), 400

    source = _AIRPORT_MAP.get(source_code)
    if not source:
        return jsonify({"error": f"Unknown airport code: {source_code}"}), 404

    # Get connected destination codes
    dest_codes = ROUTE_CONNECTIONS.get(source_code, [])
    connections = [_AIRPORT_MAP[c] for c in dest_codes if c in _AIRPORT_MAP]

    # Generate 1-3 flights per destination with ML predictions
    flights = []
    for dest in connections:
        num_flights = random.randint(1, 3)
        chosen_airlines = random.sample(
            AIRLINES, k=min(num_flights, len(AIRLINES))
        )
        for airline in chosen_airlines:
            flights.append(_build_flight_entry(airline, source, dest))

    # Sort by departure time
    flights.sort(key=lambda f: f["departure_time"])

    return jsonify({
        "source": source,
        "connections": connections,
        "flights": flights,
    })


@app.route("/api/flights", methods=["POST"])
def api_flights():
    """
    Fetch real flight offers from the configured Google Flights provider.

    Expected JSON body:
        { "origin": "DEL", "destination": "BOM", "date": "2026-06-18" }

    If date is omitted, defaults to tomorrow.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid request body."}), 400

    origin = data.get("origin", "").strip().upper()
    destination = data.get("destination", "").strip().upper()
    date = data.get("date", "").strip()

    if not origin or not destination:
        return jsonify({"error": "Both origin and destination are required."}), 400

    if origin == destination:
        return jsonify({"error": "Origin and destination cannot be the same."}), 400

    source = _AIRPORT_MAP.get(origin)
    dest = _AIRPORT_MAP.get(destination)
    if not source or not dest:
        return jsonify({"error": "Unknown airport code."}), 404

    # Default to tomorrow if no date provided
    if not date:
        date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    # Fetch from the configured flight search provider.
    serpapi_flights = fetch_flight_offers(origin, destination, date)

    flights = []
    if serpapi_flights:
        for sf in serpapi_flights:
            # Try to match airline to our known airlines
            airline_match = None
            for a in AIRLINES:
                if (a["name"].lower() in sf.get("airline", "").lower() or
                        a["code"].lower() == sf.get("airline_code", "").lower()):
                    airline_match = a
                    break

            if not airline_match:
                airline_match = {
                    "code": sf.get("airline_code", "??"),
                    "name": sf.get("airline", "Unknown"),
                }

            flights.append(_build_flight_entry(airline_match, source, dest, sf))
    else:
        # Provider returned nothing — generate ML-predicted entries as fallback.
        num_flights = random.randint(4, 7)
        selected_airlines = random.choices(AIRLINES, k=num_flights)
        flights = [
            _build_flight_entry(airline, source, dest)
            for airline in selected_airlines
        ]

    flights.sort(key=lambda f: f["departure_time"])

    # Find best value (lowest price with decent reliability) and most reliable
    if any(f.get("price") for f in flights):
        priced = [f for f in flights if f.get("price")]
        if priced:
            best_value = min(priced, key=lambda f: f["price"] / max(f["reliability_score"], 1))
            best_value["best_value"] = True
        most_reliable = max(flights, key=lambda f: f["reliability_score"])
        most_reliable["most_reliable"] = True

    return jsonify({
        "flights": flights,
        "route": f"{origin} → {destination}",
        "route_name": f"{source['city']} → {dest['city']}",
        "date": date,
        "pricing_available": any(f.get("price") for f in flights),
        "flight_search_provider": FLIGHT_SEARCH_PROVIDER,
        "flight_search_error": _flight_search_last_error,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Point-to-point prediction (legacy endpoint, kept for compatibility).

    Expected JSON body:
        { "departure": "DEL", "destination": "BOM" }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid request body."}), 400

    dep_code = data.get("departure", "").strip().upper()
    dest_code = data.get("destination", "").strip().upper()

    if not dep_code or not dest_code:
        return jsonify({"error": "Both departure and destination are required."}), 400
    if dep_code == dest_code:
        return jsonify({"error": "Departure and destination cannot be the same."}), 400

    source = _AIRPORT_MAP.get(dep_code)
    dest = _AIRPORT_MAP.get(dest_code)
    if not source or not dest:
        return jsonify({"error": "Unknown airport code."}), 404

    num_flights = random.randint(4, 7)
    selected_airlines = random.choices(AIRLINES, k=num_flights)

    flights = [
        _build_flight_entry(airline, source, dest)
        for airline in selected_airlines
    ]
    flights.sort(key=lambda f: f["departure_time"])

    return jsonify({
        "flights": flights,
        "route": f"{dep_code} → {dest_code}",
        "route_name": f"{source['city']} → {dest['city']}",
    })


# ---------------------------------------------------------------------------
# ADS-B Endpoint — returns LIVE aircraft positions from OpenSky
# ---------------------------------------------------------------------------
@app.route("/api/adsb")
def get_adsb():
    """Return current live aircraft state from OpenSky Network."""
    with _adsb_lock:
        planes = list(_live_aircraft)
        stale = _adsb_stale
        error_msg = _adsb_error_message

    return jsonify({
        "aircraft": planes,
        "count": len(planes),
        "stale": stale,
        "error": error_msg if stale else "",
        "last_update": _adsb_last_success,
    })


@app.route("/api/adsb/<callsign>")
def get_adsb_detail(callsign: str):
    """Return detailed info for a specific aircraft by callsign or ICAO24.

    If the aircraft is between two of our designated airports, also returns
    `route_path` (arc waypoints), `origin`, `destination`, and `progress_pct`.
    """
    with _adsb_lock:
        ac = None
        for a in _live_aircraft:
            if (a["callsign"].upper() == callsign.upper() or
                    a["icao24"].upper() == callsign.upper()):
                ac = a
                break

    if not ac:
        return jsonify({"error": f"Aircraft {callsign} not found"}), 404

    result = {
        "icao24": ac["icao24"],
        "callsign": ac["callsign"],
        "lat": ac["lat"],
        "lng": ac["lng"],
        "altitude": ac["altitude"],
        "speed_knots": ac["speed_knots"],
        "heading_degrees": ac["heading_degrees"],
        "vertical_rate": ac["vertical_rate"],
        "on_ground": ac["on_ground"],
    }

    # Try to infer origin/destination among our airports
    route = _infer_route(ac["lat"], ac["lng"], ac["heading_degrees"])
    if route:
        origin_ap, dest_ap, progress = route
        # Generate arc waypoints for the route path
        route_path = _generate_arc_path(
            [origin_ap["lat"], origin_ap["lon"]],
            [dest_ap["lat"], dest_ap["lon"]],
            40,
        )
        result["origin_code"] = origin_ap["code"]
        result["origin_city"] = origin_ap["city"]
        result["destination_code"] = dest_ap["code"]
        result["destination_city"] = dest_ap["city"]
        result["route_path"] = route_path
        result["progress_pct"] = round(progress, 1)

    return jsonify(result)


# ---------------------------------------------------------------------------
# Reliability Endpoint
# ---------------------------------------------------------------------------
@app.route("/api/reliability")
def get_reliability():
    """
    Return reliability score for an airline (optionally route-specific).

    Query params: airline, origin (optional), destination (optional)
    """
    airline = request.args.get("airline", "").strip()
    origin = request.args.get("origin", "").strip().upper() or None
    dest = request.args.get("destination", "").strip().upper() or None

    if not airline:
        # Return all airline scores
        scores = {}
        for a_name in AIRLINE_RELIABILITY:
            scores[a_name] = get_reliability_score(a_name)
        return jsonify({"scores": scores})

    score = get_reliability_score(airline, origin, dest)
    return jsonify({
        "airline": airline,
        "origin": origin,
        "destination": dest,
        "reliability_score": score,
        "rating": "Excellent" if score >= 85 else "Good" if score >= 75 else "Fair" if score >= 65 else "Poor",
    })


# ---------------------------------------------------------------------------
# Weather Proxy Endpoint (OpenWeatherMap)
# ---------------------------------------------------------------------------
@app.route("/api/weather")
def get_weather():
    """
    Proxy weather data from OpenWeatherMap Current Weather API.

    Query params: lat, lon
    Returns: temperature, visibility, wind, conditions, alerts
    """
    lat = request.args.get("lat", "")
    lon = request.args.get("lon", "")

    if not lat or not lon:
        return jsonify({"error": "lat and lon parameters are required."}), 400

    try:
        # Current weather
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": OWM_API_KEY,
            "units": "metric",
        }
        resp = requests.get(url, params=params, timeout=8)
        data = resp.json()

        if resp.status_code != 200:
            return jsonify({"error": data.get("message", "Weather API error")}), resp.status_code

        # Parse response into clean format
        weather_main = data.get("weather", [{}])[0]
        main = data.get("main", {})
        wind = data.get("wind", {})
        visibility = data.get("visibility", 10000)  # meters

        result = {
            "temp_c": round(main.get("temp", 0), 1),
            "feels_like_c": round(main.get("feels_like", 0), 1),
            "humidity": main.get("humidity", 0),
            "pressure_hpa": main.get("pressure", 0),
            "visibility_km": round(visibility / 1000, 1),
            "wind_speed_kmh": round(wind.get("speed", 0) * 3.6, 1),
            "wind_deg": wind.get("deg", 0),
            "wind_gust_kmh": round(wind.get("gust", 0) * 3.6, 1) if "gust" in wind else None,
            "condition": weather_main.get("main", "Clear"),
            "description": weather_main.get("description", "clear sky"),
            "icon": weather_main.get("icon", "01d"),
            "clouds_pct": data.get("clouds", {}).get("all", 0),
            "location_name": data.get("name", ""),
            # Severe weather flag
            "is_severe": weather_main.get("id", 800) < 600,  # IDs < 600 = rain/storm/snow
        }

        return jsonify(result)

    except requests.exceptions.Timeout:
        return jsonify({"error": "Weather API timeout"}), 504
    except Exception as exc:
        app.logger.warning("Weather API error: %s", exc)
        return jsonify({"error": "Failed to fetch weather data"}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[SkyPulse] Starting Flight Operations Dashboard on http://127.0.0.1:5001")
    app.run(debug=True, host="127.0.0.1", port=5001, use_reloader=False)
