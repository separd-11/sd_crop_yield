"""Download daily ERA5 weather for every raion centre via Open-Meteo.

No API key needed. Output: data/raw/coords.json, data/raw/weather.json
"""
from __future__ import annotations
import json
import time
import urllib.parse
import urllib.request

from utils import (GEOCODE_OVERRIDES, RAW, get_logger, load_config,
                   read_json, territories, write_json)

log = get_logger("fetch_weather")

DAILY_VARS = "temperature_2m_max,temperature_2m_min,precipitation_sum"
MAX_RETRIES = 5


def raion_names() -> list[str]:
    """Territory labels from the yields file, aggregate rows removed."""
    meta = read_json(RAW / "yields.json")
    return territories(
        meta["dimension"]["Raioane/Regiuni"]["category"]["label"].values())


def geocode(name: str, api: str) -> tuple[float, float] | None:
    query = GEOCODE_OVERRIDES.get(
        name, name.split()[-1] if name.startswith("Municipiul") else name)
    url = f"{api}?name={urllib.parse.quote(query)}&count=5&country=MD"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            hits = json.load(resp).get("results", [])
    except Exception as exc:
        log.warning("geocoding failed for %s: %s", name, exc)
        return None
    if not hits:
        return None
    return hits[0]["latitude"], hits[0]["longitude"]


def fetch_one(lat: float, lon: float, cfg: dict) -> dict:
    url = (f"{cfg['weather_api']}?latitude={lat}&longitude={lon}"
           f"&start_date={cfg['start_date']}&end_date={cfg['end_date']}"
           f"&daily={DAILY_VARS}&timezone=UTC")
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                return json.load(resp)["daily"]
        except Exception as exc:
            last = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"weather download failed after {MAX_RETRIES} tries: {last}")


def main() -> None:
    cfg = load_config()["data"]
    names = raion_names()

    coords_path = RAW / "coords.json"
    coords = read_json(coords_path) if coords_path.exists() else {}
    for name in names:
        if name not in coords:
            found = geocode(name, cfg["geocode_api"])
            if found:
                coords[name] = found
            time.sleep(0.25)
    write_json(coords, coords_path)
    log.info("geocoded %d/%d raions", len(coords), len(names))

    weather_path = RAW / "weather.json"
    weather = read_json(weather_path) if weather_path.exists() else {}
    missing = [n for n in coords if n not in weather]
    for i, name in enumerate(missing, 1):
        lat, lon = coords[name]
        weather[name] = fetch_one(lat, lon, cfg)
        log.info("  [%d/%d] %s", i, len(missing), name)
        time.sleep(1.5)
        if i % 10 == 0:
            write_json(weather, weather_path)

    write_json(weather, weather_path)
    days = len(next(iter(weather.values()))["time"]) if weather else 0
    log.info("weather for %d raions, %d days each", len(weather), days)


if __name__ == "__main__":
    main()
