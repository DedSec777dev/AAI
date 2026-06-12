"""
Flask backend for the SkyPulse Flight Operations Dashboard.

Serves a FlightAware-style frontend with:
  - Simulated real-time ADS-B plane tracking (background thread)
  - Historical reliability scoring (Pandas-based synthetic dataset)
  - Live weather overlay via OpenWeatherMap API proxy
  - ML-powered delay prediction (existing pipeline)
"""

from __future__ import annotations

import math
import random
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)

# ---------------------------------------------------------------------------
# OpenWeatherMap API Key
# ---------------------------------------------------------------------------
OWM_API_KEY = "7e8fcb860df6eabb0057440ef100c59c"

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
# 1. SIMULATED ADS-B TRACKING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

# Indian registration prefix callsigns
_CALLSIGNS = [
    "VT-IFA", "VT-ALK", "VT-JBK", "VT-INH", "VT-WGA",
    "VT-SCH", "VT-AXD", "VT-KFA", "VT-DHN", "VT-SPJ",
    "VT-NAK", "VT-GHR", "VT-BLQ", "VT-MPN", "VT-EXP",
    "VT-RJN", "VT-KOC", "VT-PLB", "VT-AKS", "VT-VIS",
    "VT-IDG", "VT-CRW", "VT-SRN", "VT-LKN", "VT-JAP",
]

# Thread-safe aircraft state
_adsb_lock = threading.Lock()
_aircraft_state: list[dict] = []


def _compute_heading(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute bearing from point 1 to point 2 in degrees (0-360)."""
    dlon = math.radians(lon2 - lon1)
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def _generate_waypoints(origin: dict, destination: dict, num_points: int = 40) -> list[tuple[float, float]]:
    """Generate a curved flight path with realistic waypoints between two airports."""
    lat1, lon1 = origin["lat"], origin["lon"]
    lat2, lon2 = destination["lat"], destination["lon"]

    # Create a slight arc (quadratic bezier) for realism
    mid_lat = (lat1 + lat2) / 2
    mid_lon = (lon1 + lon2) / 2
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    dist = math.sqrt(d_lat**2 + d_lon**2)

    # Offset the midpoint perpendicular to the route for a curve
    offset = dist * random.uniform(0.05, 0.15) * random.choice([-1, 1])
    ctrl_lat = mid_lat + (-d_lon / max(dist, 0.01)) * offset
    ctrl_lon = mid_lon + (d_lat / max(dist, 0.01)) * offset

    waypoints = []
    for i in range(num_points + 1):
        t = i / num_points
        u = 1 - t
        lat = u * u * lat1 + 2 * u * t * ctrl_lat + t * t * lat2
        lon = u * u * lon1 + 2 * u * t * ctrl_lon + t * t * lon2
        # Add tiny random perturbation for realism
        lat += random.gauss(0, 0.005)
        lon += random.gauss(0, 0.005)
        waypoints.append((lat, lon))

    return waypoints


def _create_aircraft(index: int) -> dict:
    """Create a single simulated aircraft with a random route."""
    callsign = _CALLSIGNS[index]
    airline = random.choice(AIRLINES)

    # Pick a random valid route
    origin_code = random.choice(list(ROUTE_CONNECTIONS.keys()))
    dest_codes = ROUTE_CONNECTIONS[origin_code]
    dest_code = random.choice(dest_codes)

    origin = _AIRPORT_MAP[origin_code]
    destination = _AIRPORT_MAP[dest_code]

    waypoints = _generate_waypoints(origin, destination)
    # Start at a random progress along the route
    progress = random.randint(0, len(waypoints) - 2)
    current = waypoints[progress]

    # Compute heading toward next waypoint
    nxt = waypoints[min(progress + 1, len(waypoints) - 1)]
    heading = _compute_heading(current[0], current[1], nxt[0], nxt[1])

    return {
        "callsign": callsign,
        "airline_code": airline["code"],
        "airline_name": airline["name"],
        "origin_code": origin_code,
        "origin_city": origin["city"],
        "destination_code": dest_code,
        "destination_city": destination["city"],
        "lat": current[0],
        "lng": current[1],
        "altitude": random.randint(25000, 40000),
        "speed_knots": random.randint(350, 500),
        "heading_degrees": round(heading, 1),
        "waypoints": waypoints,
        "waypoint_index": progress,
        "flight_number": f"{airline['code']}-{random.randint(100, 999)}",
    }


def _reassign_route(aircraft: dict) -> None:
    """Give an aircraft a new random route when it reaches its destination."""
    origin_code = aircraft["destination_code"]
    dest_codes = ROUTE_CONNECTIONS.get(origin_code, [])
    if not dest_codes:
        origin_code = random.choice(list(ROUTE_CONNECTIONS.keys()))
        dest_codes = ROUTE_CONNECTIONS[origin_code]

    dest_code = random.choice(dest_codes)
    origin = _AIRPORT_MAP[origin_code]
    destination = _AIRPORT_MAP[dest_code]

    aircraft["origin_code"] = origin_code
    aircraft["origin_city"] = origin["city"]
    aircraft["destination_code"] = dest_code
    aircraft["destination_city"] = destination["city"]
    aircraft["waypoints"] = _generate_waypoints(origin, destination)
    aircraft["waypoint_index"] = 0
    aircraft["altitude"] = random.randint(25000, 40000)
    aircraft["speed_knots"] = random.randint(350, 500)


def _adsb_simulation_loop() -> None:
    """Background loop: advance all aircraft along their routes every 2 seconds."""
    global _aircraft_state

    # Initialize 25 aircraft
    with _adsb_lock:
        _aircraft_state = [_create_aircraft(i) for i in range(25)]

    while True:
        time.sleep(2)
        with _adsb_lock:
            for ac in _aircraft_state:
                wp = ac["waypoints"]
                idx = ac["waypoint_index"]

                # Advance 1-2 waypoints per tick
                advance = random.randint(1, 2)
                new_idx = min(idx + advance, len(wp) - 1)
                ac["waypoint_index"] = new_idx
                ac["lat"] = wp[new_idx][0]
                ac["lng"] = wp[new_idx][1]

                # Compute heading toward next waypoint
                if new_idx < len(wp) - 1:
                    nxt = wp[new_idx + 1]
                    ac["heading_degrees"] = round(
                        _compute_heading(ac["lat"], ac["lng"], nxt[0], nxt[1]), 1
                    )

                # Altitude variation during flight
                progress = new_idx / max(len(wp) - 1, 1)
                if progress < 0.15:
                    # Climbing
                    ac["altitude"] = int(15000 + progress / 0.15 * 20000)
                elif progress > 0.85:
                    # Descending
                    descent = (progress - 0.85) / 0.15
                    ac["altitude"] = int(35000 - descent * 20000)
                else:
                    # Cruise with slight variation
                    ac["altitude"] = ac["altitude"] + random.randint(-200, 200)
                    ac["altitude"] = max(28000, min(42000, ac["altitude"]))

                # Speed variation
                ac["speed_knots"] = max(320, min(520, ac["speed_knots"] + random.randint(-10, 10)))

                # If reached destination, reassign new route
                if new_idx >= len(wp) - 1:
                    _reassign_route(ac)


# Start ADS-B simulation thread
_adsb_thread = threading.Thread(target=_adsb_simulation_loop, daemon=True)
_adsb_thread.start()


# ═══════════════════════════════════════════════════════════════════════════
# 2. HISTORICAL RELIABILITY SCORE ENGINE
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
# 3. PREDICTION HELPER — wraps the real ML model
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
) -> dict:
    """
    Build a single flight record with ML-powered delay prediction
    and historical reliability score.
    """
    dep_time = _generate_departure_time()
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
        "flight_number": _generate_flight_number(airline["code"]),
        "airline": airline["name"],
        "airline_code": airline["code"],
        "departure_time": dep_time,
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
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. ROUTES — API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the main frontend page."""
    return render_template("index.html")


@app.route("/api/airports")
def get_airports():
    """Return the full list of airports with coordinates for map markers."""
    return jsonify(AIRPORTS)


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
# ADS-B Endpoint — returns simulated live aircraft positions
# ---------------------------------------------------------------------------
@app.route("/api/adsb")
def get_adsb():
    """Return current state of all 25 simulated aircraft."""
    with _adsb_lock:
        # Return a copy without internal waypoint data
        planes = []
        for ac in _aircraft_state:
            planes.append({
                "callsign": ac["callsign"],
                "flight_number": ac["flight_number"],
                "airline_code": ac["airline_code"],
                "airline_name": ac["airline_name"],
                "origin_code": ac["origin_code"],
                "origin_city": ac["origin_city"],
                "destination_code": ac["destination_code"],
                "destination_city": ac["destination_city"],
                "lat": round(ac["lat"], 4),
                "lng": round(ac["lng"], 4),
                "altitude": ac["altitude"],
                "speed_knots": ac["speed_knots"],
                "heading_degrees": ac["heading_degrees"],
            })
    return jsonify(planes)


@app.route("/api/adsb/<callsign>")
def get_adsb_detail(callsign: str):
    """Return detailed info + route path for a specific aircraft."""
    with _adsb_lock:
        ac = None
        for a in _aircraft_state:
            if a["callsign"] == callsign.upper():
                ac = a
                break
        if not ac:
            return jsonify({"error": f"Aircraft {callsign} not found"}), 404

        # Build route path from waypoints
        route_path = [(round(w[0], 4), round(w[1], 4)) for w in ac["waypoints"]]

        return jsonify({
            "callsign": ac["callsign"],
            "flight_number": ac["flight_number"],
            "airline_code": ac["airline_code"],
            "airline_name": ac["airline_name"],
            "origin_code": ac["origin_code"],
            "origin_city": ac["origin_city"],
            "destination_code": ac["destination_code"],
            "destination_city": ac["destination_city"],
            "lat": round(ac["lat"], 4),
            "lng": round(ac["lng"], 4),
            "altitude": ac["altitude"],
            "speed_knots": ac["speed_knots"],
            "heading_degrees": ac["heading_degrees"],
            "route_path": route_path,
            "progress_pct": round(
                ac["waypoint_index"] / max(len(ac["waypoints"]) - 1, 1) * 100, 1
            ),
        })


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
