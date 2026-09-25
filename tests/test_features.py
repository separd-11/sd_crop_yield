"""Degree days are the one place where a silent sign or unit error would
change every number downstream, so they get checked against hand computation."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import degree_days, killing_degree_days, longest_dry_spell  # noqa: E402


def test_flat_day_between_base_and_cap():
    # A day held at 20 C with base 10 contributes exactly 10 degree days.
    out = degree_days(np.array([20.0]), np.array([20.0]), base=10.0, cap=30.0)
    assert out[0] == pytest.approx(10.0)


def test_flat_day_below_base_contributes_nothing():
    out = degree_days(np.array([4.0]), np.array([4.0]), base=10.0, cap=30.0)
    assert out[0] == pytest.approx(0.0)


def test_cap_truncates_the_hot_part():
    # Held at 40 C, everything above the 30 C cap is discarded: 30 - 10 = 20.
    out = degree_days(np.array([40.0]), np.array([40.0]), base=10.0, cap=30.0)
    assert out[0] == pytest.approx(20.0)


def test_killing_degree_days_count_only_the_excess():
    out = killing_degree_days(np.array([35.0]), np.array([35.0]), thr=30.0)
    assert out[0] == pytest.approx(5.0)


def test_no_killing_degree_days_below_the_threshold():
    out = killing_degree_days(np.array([25.0]), np.array([25.0]), thr=30.0)
    assert out[0] == pytest.approx(0.0)


def test_swinging_day_earns_killing_degrees_the_mean_would_hide():
    """Mean 30 C looks harmless; half the day above 30 C is not."""
    out = killing_degree_days(np.array([20.0]), np.array([40.0]), thr=30.0)
    assert out[0] > 2.0


def test_dry_spell_finds_the_longest_run():
    rain = np.array([5.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 2.0])
    assert longest_dry_spell(rain, dry_mm=1.0) == 3


def test_dry_spell_is_zero_when_it_rains_every_day():
    assert longest_dry_spell(np.array([2.0, 3.0, 4.0]), dry_mm=1.0) == 0
