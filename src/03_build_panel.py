"""Join yields and weather into the raion x crop x year panel.

Output: data/processed/panel.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from features import weather_features
from utils import (CROP_LABELS, PROCESSED, RAW, WINTER_CROPS, get_logger,
                   load_config, read_json)

log = get_logger("build_panel")


def main() -> None:
    cfg = load_config()
    fcfg = cfg["features"]

    yields = read_json(RAW / "yields.json")
    weather = read_json(RAW / "weather.json")

    dim = yields["dimension"]
    geo = list(dim["Raioane/Regiuni"]["category"]["label"].values())
    crops = list(dim["Culturi agricole"]["category"]["label"].values())
    years = [int(y) for y in dim["Ani"]["category"]["label"].values()]
    values = yields["value"]
    n_crop, n_year = len(crops), len(years)

    wx = {}
    for name, daily in weather.items():
        wx[name] = weather_features(daily, fcfg).set_index("year")

    rows = []
    for gi, label in enumerate(geo):
        if not label.startswith(".."):
            continue
        raion = label.replace("..", "").strip()
        if raion not in wx:
            log.warning("no weather for %s, dropped", raion)
            continue
        for ci, crop in enumerate(crops):
            crop_short = CROP_LABELS.get(crop, crop[:16])
            for yi, year in enumerate(years):
                value = values[(gi * n_crop + ci) * n_year + yi]
                if value is None or year not in wx[raion].index:
                    continue
                row = {"raion": raion, "crop": crop_short, "year": year,
                       "yield_cwt_ha": float(value)}
                row.update(wx[raion].loc[year].to_dict())
                rows.append(row)

    panel = pd.DataFrame(rows)
    panel["log_yield"] = np.log(panel["yield_cwt_ha"].clip(lower=0.1))
    panel["prec_season_sq"] = panel["prec_season"] ** 2
    panel["is_winter_crop"] = panel["crop"].isin(WINTER_CROPS).astype(int)
    panel["trend"] = panel["year"] - panel["year"].min()

    out = PROCESSED / "panel.csv"
    panel.to_csv(out, index=False)
    log.info("panel: %d rows | %d raions | %d crops | %d-%d",
             len(panel), panel.raion.nunique(), panel.crop.nunique(),
             panel.year.min(), panel.year.max())
    log.info("columns: %s", ", ".join(panel.columns))
    log.info("written to %s", out.relative_to(out.parents[2]))


if __name__ == "__main__":
    main()
