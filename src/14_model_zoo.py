"""Nine models, four ways of checking them.

The question is not only which model wins but whether the winner depends on how
you check. Rolling origin is included because it is the setting anyone applying
this would face: train on the past, predict the next season.

Output: results/tables/model_zoo.csv, results/figures/model_zoo.png
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from models import ML_FEATURES, model_zoo, rmse
from utils import FIGURES, PROCESSED, TABLES, get_logger, load_config

log = get_logger("zoo")

SCHEMES = ("random 5-fold", "leave-one-year-out", "leave-one-raion-out",
           "rolling origin")
MIN_TRAIN_YEARS = 7


def folds(df: pd.DataFrame, scheme: str, seed: int):
    if scheme == "random 5-fold":
        yield from KFold(5, shuffle=True, random_state=seed).split(df)
    elif scheme == "leave-one-year-out":
        for year in sorted(df["year"].unique()):
            m = (df["year"] == year).to_numpy()
            yield np.where(~m)[0], np.where(m)[0]
    elif scheme == "leave-one-raion-out":
        for raion in sorted(df["raion"].unique()):
            m = (df["raion"] == raion).to_numpy()
            if m.sum() and (~m).sum() > 50:
                yield np.where(~m)[0], np.where(m)[0]
    else:
        years = sorted(df["year"].unique())
        for target in years[MIN_TRAIN_YEARS:]:
            tr = np.where((df["year"] < target).to_numpy())[0]
            te = np.where((df["year"] == target).to_numpy())[0]
            if len(te) >= 5:
                yield tr, te


def main() -> None:
    cfg = load_config()
    seed = cfg["model"]["seed"]
    panel = pd.read_csv(PROCESSED / "panel.csv")
    raions = sorted(panel["raion"].unique())
    zoo = model_zoo(cfg, raions)
    need = sorted(set(ML_FEATURES + cfg["model"]["baseline_features"] + ["log_yield"]))

    rows = []
    for crop, sub in panel.groupby("crop"):
        sub = sub.dropna(subset=need).reset_index(drop=True)
        if len(sub) < 200:
            continue
        for scheme in SCHEMES:
            splits = list(folds(sub, scheme, seed))
            preds = {name: [] for name in zoo}
            truth = []
            for tr, te in splits:
                train, test = sub.iloc[tr], sub.iloc[te]
                y_tr = train["log_yield"].to_numpy()
                for name, factory in zoo.items():
                    preds[name].append(factory().fit(train, y_tr).predict(test))
                truth.append(test["log_yield"].to_numpy())
            y = np.concatenate(truth)
            for name in zoo:
                rows.append({"crop": crop, "scheme": scheme, "model": name,
                             "rmse": rmse(y, np.concatenate(preds[name]))})
        log.info("%s done", crop)

    res = pd.DataFrame(rows)
    res.round(4).to_csv(TABLES / "model_zoo.csv", index=False)

    pooled = res.pivot_table(index="model", columns="scheme", values="rmse")[list(SCHEMES)]
    log.info("\nRMSE averaged over crops\n%s", pooled.round(3).to_string())

    ranks = (res.groupby(["crop", "scheme"])["rmse"]
             .rank().to_frame("rank").join(res[["model", "scheme"]])
             .pivot_table(index="model", columns="scheme", values="rank"))[list(SCHEMES)]
    log.info("\naverage rank across crops (1 = best)\n%s",
             ranks.round(2).sort_values("rolling origin").to_string())
    ranks.round(3).to_csv(TABLES / "model_zoo_ranks.csv")

    for scheme in SCHEMES:
        winners = (res[res.scheme == scheme]
                   .loc[lambda d: d.groupby("crop")["rmse"].idxmin()]
                   .groupby("model").size().sort_values(ascending=False))
        log.info("\n%s, crops won\n%s", scheme, winners.to_string())

    order = ranks["rolling origin"].sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pos = np.arange(len(order))
    width = 0.2
    for i, scheme in enumerate(SCHEMES):
        ax.barh(pos + (i - 1.5) * width, pooled.loc[order, scheme], width, label=scheme)
    ax.set_yticks(pos, order)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE, log yield, averaged over crops")
    ax.set_title("Which model is best depends on how you check")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_zoo.png", dpi=150)
    log.info("figure written: results/figures/model_zoo.png")


if __name__ == "__main__":
    main()
