"""Forecasting, not interpolation: train on the past, predict the next year.

Leave-one-year-out lets a model predicting 2012 learn from 2020. That answers
"can it handle a year it has not seen", which is not the same as "can it be
used". Here the window only ever expands forward, which is the setting anyone
applying this would actually face.

Output: results/tables/rolling_origin.csv, results/figures/rolling_origin.png
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import (BaselineFE, GlobalMean, ML_FEATURES, RaionMean, gbm_matrix,
                    make_gbm, rmse)
from utils import FIGURES, PROCESSED, TABLES, get_logger, load_config

log = get_logger("rolling")

MIN_TRAIN_YEARS = 7
ORDER = ["raion mean + trend", "baseline FE", "gradient boosting"]


def main() -> None:
    cfg = load_config()
    seed = cfg["model"]["seed"]
    base_feats = cfg["model"]["baseline_features"]
    panel = pd.read_csv(PROCESSED / "panel.csv")
    raions = sorted(panel["raion"].unique())
    need = sorted(set(ML_FEATURES + base_feats + ["log_yield"]))

    rows = []
    for crop, sub in panel.groupby("crop"):
        sub = sub.dropna(subset=need).reset_index(drop=True)
        if len(sub) < 200:
            continue
        years = sorted(sub["year"].unique())
        for target in years[MIN_TRAIN_YEARS:]:
            train = sub[sub["year"] < target]
            test = sub[sub["year"] == target]
            if len(test) < 5:
                continue
            y_tr, y_te = train["log_yield"].to_numpy(), test["log_yield"].to_numpy()
            preds = {
                "raion mean + trend": RaionMean(with_trend=True).fit(train, y_tr).predict(test),
                "baseline FE": BaselineFE(base_feats).fit(train, y_tr).predict(test),
                "gradient boosting": make_gbm(cfg["model"]["gbm"], seed).fit(
                    gbm_matrix(train, raions), y_tr).predict(gbm_matrix(test, raions)),
            }
            for name, pred in preds.items():
                rows.append({"crop": crop, "year": target, "model": name,
                             "n_train_years": len(train["year"].unique()),
                             "rmse": rmse(y_te, pred)})
        log.info("%s done", crop)

    res = pd.DataFrame(rows)
    res.round(4).to_csv(TABLES / "rolling_origin.csv", index=False)

    pooled = (res.groupby(["crop", "model"])
              .apply(lambda d: np.sqrt((d["rmse"] ** 2).mean()), include_groups=False)
              .unstack()[ORDER])
    log.info("\nRMSE over all forecast years, train on the past only\n%s",
             pooled.round(3).to_string())
    log.info("\nbest model per crop:\n%s", pooled.idxmin(axis=1).to_string())

    overall = (res.groupby("model")
               .apply(lambda d: np.sqrt((d["rmse"] ** 2).mean()), include_groups=False))
    log.info("\npooled over every crop and year:\n%s", overall[ORDER].round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    crops = sorted(pooled.index)
    pos = np.arange(len(crops))
    for i, name in enumerate(ORDER):
        axes[0].bar(pos + (i - 1) * 0.27, pooled.loc[crops, name], 0.27, label=name)
    axes[0].set_xticks(pos, crops, rotation=35, ha="right")
    axes[0].set_ylabel("RMSE, log yield")
    axes[0].set_title("Forecasting the next year")
    axes[0].legend(frameon=False, fontsize=8)

    by_year = (res.groupby(["year", "model"])
               .apply(lambda d: np.sqrt((d["rmse"] ** 2).mean()), include_groups=False)
               .unstack()[ORDER])
    for name in ORDER:
        axes[1].plot(by_year.index, by_year[name], marker="o", markersize=4, label=name)
    axes[1].set_xlabel("Forecast year")
    axes[1].set_ylabel("RMSE, log yield")
    axes[1].set_title("The training window only grows")
    axes[1].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "rolling_origin.png", dpi=150)
    log.info("figure written: results/figures/rolling_origin.png")


if __name__ == "__main__":
    main()
