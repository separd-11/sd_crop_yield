"""Is the weather worth modelling at all?

Before comparing two weather models it is worth knowing what a model that
ignores the weather achieves. If "this raion, as usual, plus a trend" is close,
then the whole exercise is measuring something small.

Output: results/tables/naive_baselines.csv, results/figures/naive_baselines.png
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from models import (BaselineFE, GlobalMean, ML_FEATURES, Persistence, RaionMean,
                    gbm_matrix, make_gbm, rmse)
from utils import FIGURES, PROCESSED, TABLES, get_logger, load_config

log = get_logger("naive")

SCHEMES = ("random 5-fold", "leave-one-year-out", "leave-one-raion-out")
ORDER = ["global mean", "raion mean", "raion mean + trend", "last year",
         "baseline FE", "gradient boosting"]


def folds(df: pd.DataFrame, scheme: str, seed: int):
    if scheme == "random 5-fold":
        yield from KFold(5, shuffle=True, random_state=seed).split(df)
    elif scheme == "leave-one-year-out":
        for year in sorted(df["year"].unique()):
            m = (df["year"] == year).to_numpy()
            yield np.where(~m)[0], np.where(m)[0]
    else:
        for raion in sorted(df["raion"].unique()):
            m = (df["raion"] == raion).to_numpy()
            if m.sum() and (~m).sum() > 50:
                yield np.where(~m)[0], np.where(m)[0]


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
        for scheme in SCHEMES:
            preds = {name: [] for name in ORDER}
            truth = []
            for tr, te in folds(sub, scheme, seed):
                train, test = sub.iloc[tr], sub.iloc[te]
                y_tr = train["log_yield"].to_numpy()
                preds["global mean"].append(GlobalMean().fit(train, y_tr).predict(test))
                preds["raion mean"].append(RaionMean().fit(train, y_tr).predict(test))
                preds["raion mean + trend"].append(
                    RaionMean(with_trend=True).fit(train, y_tr).predict(test))
                preds["last year"].append(Persistence().fit(train, y_tr).predict(test))
                preds["baseline FE"].append(
                    BaselineFE(base_feats).fit(train, y_tr).predict(test))
                preds["gradient boosting"].append(
                    make_gbm(cfg["model"]["gbm"], seed).fit(
                        gbm_matrix(train, raions), y_tr).predict(gbm_matrix(test, raions)))
                truth.append(test["log_yield"].to_numpy())
            y = np.concatenate(truth)
            for name in ORDER:
                rows.append({"crop": crop, "scheme": scheme, "model": name,
                             "rmse": rmse(y, np.concatenate(preds[name]))})
        log.info("%s done", crop)

    res = pd.DataFrame(rows)
    res.round(4).to_csv(TABLES / "naive_baselines.csv", index=False)

    for scheme in SCHEMES:
        table = (res[res.scheme == scheme]
                 .pivot(index="crop", columns="model", values="rmse")[ORDER])
        log.info("\n%s\n%s", scheme, table.round(3).to_string())

    loyo = (res[res.scheme == "leave-one-year-out"]
            .pivot(index="crop", columns="model", values="rmse")[ORDER])
    gain = 100 * (1 - loyo["baseline FE"] / loyo["raion mean + trend"])
    log.info("\nwhat the weather buys over 'this raion as usual, plus a trend',\n"
             "leave-one-year-out, %% of RMSE removed\n%s",
             gain.round(1).sort_values(ascending=False).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    crops = sorted(res.crop.unique())
    width = 0.13
    for ax, scheme in zip(axes, SCHEMES):
        table = (res[res.scheme == scheme]
                 .pivot(index="crop", columns="model", values="rmse")[ORDER])
        pos = np.arange(len(crops))
        for i, name in enumerate(ORDER):
            ax.bar(pos + (i - len(ORDER) / 2 + 0.5) * width,
                   table.loc[crops, name], width, label=name)
        ax.set_xticks(pos, crops, rotation=35, ha="right")
        ax.set_title(scheme)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("RMSE, log yield")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.suptitle("Weather models against models that ignore the weather")
    fig.tight_layout()
    fig.savefig(FIGURES / "naive_baselines.png", dpi=150)
    log.info("figure written: results/figures/naive_baselines.png")


if __name__ == "__main__":
    main()
