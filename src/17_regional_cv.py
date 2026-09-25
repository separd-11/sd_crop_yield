"""Leaving out one raion is a weak spatial test; neighbours stay in training.

Held-out raions sit next to raions the model has seen, sharing soil, climate
and often the same weather front. Blocking a whole development region breaks
that. If the spatial result is real it should survive the coarser block.

Output: results/tables/regional_cv.csv, results/figures/regional_cv.png
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import ML_FEATURES, model_zoo, rmse
from utils import FIGURES, PROCESSED, TABLES, get_logger, load_config

log = get_logger("regional")

SCHEMES = ("leave-one-raion-out", "leave-one-region-out")


def folds(df: pd.DataFrame, scheme: str):
    column = "raion" if scheme == "leave-one-raion-out" else "region"
    for value in sorted(df[column].unique()):
        m = (df[column] == value).to_numpy()
        if m.sum() >= 5 and (~m).sum() > 50:
            yield np.where(~m)[0], np.where(m)[0], value


def main() -> None:
    cfg = load_config()
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
            preds = {name: [] for name in zoo}
            truth = []
            for tr, te, _ in folds(sub, scheme):
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
    res.round(4).to_csv(TABLES / "regional_cv.csv", index=False)
    table = res.pivot_table(index="model", columns="scheme", values="rmse")[list(SCHEMES)]
    table["penalty %"] = 100 * (table[SCHEMES[1]] / table[SCHEMES[0]] - 1)
    log.info("\nRMSE averaged over crops\n%s",
             table.round(3).sort_values("penalty %").to_string())

    for scheme in SCHEMES:
        won = (res[res.scheme == scheme]
               .loc[lambda d: d.groupby("crop")["rmse"].idxmin()]
               .groupby("model").size().sort_values(ascending=False))
        log.info("\n%s, crops won\n%s", scheme, won.to_string())

    order = table.sort_values(SCHEMES[1]).index.tolist()
    fig, ax = plt.subplots(figsize=(9.5, 5))
    pos = np.arange(len(order))
    for i, scheme in enumerate(SCHEMES):
        ax.barh(pos + (i - 0.5) * 0.36, table.loc[order, scheme], 0.36, label=scheme)
    ax.set_yticks(pos, order)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE, log yield, averaged over crops")
    ax.set_title("Blocking a whole region instead of a single raion")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "regional_cv.png", dpi=150)
    log.info("figure written: results/figures/regional_cv.png")


if __name__ == "__main__":
    main()
