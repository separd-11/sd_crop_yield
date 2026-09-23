"""Where the two models disagree about a drought, drawn on the map.

Output: results/figures/map_drought.png
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geo import choropleth
from models import BaselineFE, ML_FEATURES, gbm_matrix, make_gbm
from utils import FIGURES, PROCESSED, get_logger, load_config

log = get_logger("map")
CROP, YEAR = "porumb", 2012


def main() -> None:
    cfg = load_config()
    base_feats = cfg["model"]["baseline_features"]
    need = sorted(set(ML_FEATURES + base_feats + ["log_yield"]))
    panel = pd.read_csv(PROCESSED / "panel.csv")
    raions = sorted(panel["raion"].unique())
    sub = panel[panel["crop"] == CROP].dropna(subset=need)
    test, train = sub[sub["year"] == YEAR], sub[sub["year"] != YEAR]
    y_tr = train["log_yield"].to_numpy()

    pred_b = np.exp(BaselineFE(base_feats).fit(train, y_tr).predict(test))
    pred_g = np.exp(make_gbm(cfg["model"]["gbm"], cfg["model"]["seed"]).fit(
        gbm_matrix(train, raions), y_tr).predict(gbm_matrix(test, raions)))
    actual = test["yield_cwt_ha"].to_numpy()
    names = test["raion"].to_numpy()

    panels = [("Actual", dict(zip(names, actual))),
              ("Parametric model", dict(zip(names, pred_b))),
              ("Gradient boosting", dict(zip(names, pred_g)))]
    hi = max(max(v.values()) for _, v in panels)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (title, values) in zip(axes, panels):
        choropleth(ax, values, vmin=0, vmax=hi, label=title)
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, hi))
    fig.colorbar(sm, ax=axes, shrink=0.75, label="Yield, centners per hectare")
    fig.suptitle(f"{CROP}, {YEAR}: a drought neither model was trained on")
    fig.savefig(FIGURES / "map_drought.png", dpi=150, bbox_inches="tight")
    log.info("actual %.1f | parametric %.1f | boosting %.1f",
             actual.mean(), pred_b.mean(), pred_g.mean())
    log.info("figure written: results/figures/map_drought.png")


if __name__ == "__main__":
    main()
