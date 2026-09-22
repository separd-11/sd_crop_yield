"""Download raion-level crop yields from the BNS PxWeb API.

Output: data/raw/yields.json. Cached; delete the file to refresh.
"""
from __future__ import annotations
import json
import urllib.request

from utils import RAW, get_logger, load_config, write_json

log = get_logger("fetch_yields")


def fetch_metadata(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def fetch_yields(url: str, crop_codes: list[str], indicator: str) -> dict:
    query = {
        "query": [
            {"code": "Culturi agricole",
             "selection": {"filter": "item", "values": crop_codes}},
            {"code": "Indicatori",
             "selection": {"filter": "item", "values": [indicator]}},
        ],
        "response": {"format": "json-stat2"},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(query).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def main() -> None:
    cfg = load_config()["data"]
    out = RAW / "yields.json"
    if out.exists():
        log.info("%s already present, skipping download", out.name)
        return

    log.info("fetching table metadata")
    write_json(fetch_metadata(cfg["bns_table"]), RAW / "yields_meta.json")

    log.info("fetching yields for %d crops", len(cfg["crop_codes"]))
    data = fetch_yields(cfg["bns_table"], cfg["crop_codes"], cfg["indicator_code"])
    write_json(data, out)

    n_total = len(data["value"])
    n_obs = sum(1 for v in data["value"] if v is not None)
    log.info("saved %s: %d cells, %d non-missing (%.0f%%)",
             out.name, n_total, n_obs, 100 * n_obs / n_total)


if __name__ == "__main__":
    main()
