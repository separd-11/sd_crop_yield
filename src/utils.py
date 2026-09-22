"""Paths, config loading, logging."""
from __future__ import annotations
import json
import logging
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"

for _d in (RAW, PROCESSED, FIGURES, TABLES):
    _d.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-22s  %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def write_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)


def read_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


CROP_LABELS = {
    "Griu - total (de toamna si de primavara)": "griu",
    "Orz -  total (de toamna si de primavara)": "orz",
    "Porumb boabe": "porumb",
    "Floarea soarelui": "floarea_soarelui",
    "Soia": "soia",
    "Sfecla de zahar": "sfecla",
    "Cartofi": "cartofi",
    "Legume de cimp": "legume",
}

WINTER_CROPS = {"griu", "orz"}
