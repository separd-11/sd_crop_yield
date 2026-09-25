"""Invariants the panel and the models must satisfy before any result is trusted."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from models import ML_FEATURES, model_zoo  # noqa: E402
from utils import load_config, region_map  # noqa: E402

PANEL = ROOT / "data" / "processed" / "panel.csv"
pytestmark = pytest.mark.skipif(not PANEL.exists(), reason="run `make panel` first")


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return pd.read_csv(PANEL)


def test_one_row_per_raion_crop_year(panel):
    assert not panel.duplicated(["raion", "crop", "year"]).any()


def test_yields_are_positive(panel):
    assert (panel["yield_cwt_ha"] > 0).all()
    assert np.isfinite(panel["log_yield"]).all()


def test_every_raion_has_a_region(panel):
    assert panel["region"].notna().all()


def test_killing_degree_days_never_negative(panel):
    assert (panel["kdd"] >= 0).all()
    assert (panel["gdd"] >= 0).all()


def test_aggregate_rows_were_excluded(panel):
    assert not {"Total pe tara", "Nord", "Centru", "Sud"} & set(panel["raion"])


def test_region_map_handles_the_two_standalone_territories():
    labels = ["Total pe tara", "Municipiul Chisinau", "Nord", "..Briceni",
              "Sud", "..Cahul", "U.T.A Gagauzia"]
    mapping = region_map(labels)
    assert mapping["Briceni"] == "Nord"
    assert mapping["Cahul"] == "Sud"
    assert mapping["Municipiul Chisinau"] == "Municipiul Chisinau"
    assert mapping["U.T.A Gagauzia"] == "U.T.A Gagauzia"
    assert "Total pe tara" not in mapping


def test_every_model_honours_the_fit_predict_contract(panel):
    cfg = load_config()
    sub = panel[panel["crop"] == "griu"].dropna(subset=ML_FEATURES + ["log_yield"])
    train, test = sub[sub["year"] < 2020], sub[sub["year"] >= 2020]
    raions = sorted(panel["raion"].unique())
    for name, factory in model_zoo(cfg, raions).items():
        pred = factory().fit(train, train["log_yield"].to_numpy()).predict(test)
        assert len(pred) == len(test), name
        assert np.isfinite(pred).all(), name
