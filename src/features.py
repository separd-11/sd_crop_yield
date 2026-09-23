"""Weather features shared by the pipeline and the app.

Heat that helps (growing degree days) is kept separate from heat that hurts
(killing degree days); a mean temperature hides the difference.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def degree_days(tmin: np.ndarray, tmax: np.ndarray, base: float, cap: float) -> np.ndarray:
    """Degree days from daily min/max, integrating a sine-shaped day.

    The (tmean - base) shortcut discards the hours spent above the cap, which
    is the part that matters.
    """
    hours = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    mean = (tmax + tmin) / 2.0
    amp = (tmax - tmin) / 2.0
    curve = mean[:, None] + amp[:, None] * np.sin(hours - np.pi / 2)[None, :]
    eff = np.clip(curve, base, cap) - base
    return eff.mean(axis=1)


def killing_degree_days(tmin: np.ndarray, tmax: np.ndarray, thr: float) -> np.ndarray:
    hours = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    mean = (tmax + tmin) / 2.0
    amp = (tmax - tmin) / 2.0
    curve = mean[:, None] + amp[:, None] * np.sin(hours - np.pi / 2)[None, :]
    return np.clip(curve - thr, 0, None).mean(axis=1)


def longest_dry_spell(precip: np.ndarray, dry_mm: float) -> int:
    best = run = 0
    for p in precip:
        run = run + 1 if (p is not None and p < dry_mm) else 0
        best = max(best, run)
    return best


def weather_features(daily: dict, fcfg: dict) -> pd.DataFrame:
    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df = df.dropna(subset=["temperature_2m_max", "temperature_2m_min"])

    tmin = df["temperature_2m_min"].to_numpy(float)
    tmax = df["temperature_2m_max"].to_numpy(float)
    df["gdd_d"] = degree_days(tmin, tmax, fcfg["gdd_base"], fcfg["gdd_cap"])
    df["kdd_d"] = killing_degree_days(tmin, tmax, fcfg["kdd_threshold"])
    df["hot_d"] = (tmax > fcfg["hot_day_threshold"]).astype(int)

    season = df[df["month"].isin(fcfg["season_months"])]
    out = season.groupby("year").agg(
        gdd=("gdd_d", "sum"),
        kdd=("kdd_d", "sum"),
        hot_days=("hot_d", "sum"),
        prec_season=("precipitation_sum", "sum"),
        tmax_mean=("temperature_2m_max", "mean"),
    )

    # Timing effects the season totals hide.
    for m in fcfg["season_months"]:
        sub = df[df["month"] == m].groupby("year")
        out[f"prec_m{m}"] = sub["precipitation_sum"].sum()
        out[f"tmax_m{m}"] = sub["temperature_2m_max"].mean()

    crit = df[df["month"].isin([6, 7])]
    out["dry_spell"] = crit.groupby("year")["precipitation_sum"].apply(
        lambda s: longest_dry_spell(s.to_numpy(float), fcfg["dry_day_mm"])
    )

    # Autumn and winter recharge matters for winter cereals.
    df["hydro_year"] = np.where(df["month"] >= 10, df["year"] + 1, df["year"])
    winter = df[df["month"].isin([10, 11, 12, 1, 2, 3])]
    out["prec_winter"] = winter.groupby("hydro_year")["precipitation_sum"].sum()

    return out.reset_index()
