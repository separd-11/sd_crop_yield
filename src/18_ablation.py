"""Which features earn their place?

The panel carries eighteen weather features, several of them near-duplicates of
each other. Dropping a group and re-scoring says whether it was doing work or
adding noise the flexible models could overfit.

Output: results/tables/ablation.csv, results/figures/ablation.png
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import models as M
from models import SklearnAdapter, make_gbm, rmse
from utils import FIGURES, PROCESSED, TABLES, get_logger, load_config

log = get_logger("ablation")

MONTHLY = [f"prec_m{m}" for m in range(4, 9)] + [f"tmax_m{m}" for m in range(4, 9)]
SETS = {
    "all features": M.ML_FEATURES,
    "without monthly detail": [f for f in M.ML_FEATURES if f not in MONTHLY],
    "without dry spell": [f for f in M.ML_FEATURES if f != "dry_spell"],
    "without winter rain": [f for f in M.ML_FEATURES if f != "prec_winter"],
    "degree days and rain only": ["gdd", "kdd", "prec_season", "trend"],
}
SCHEMES = ("leave-one-year-out", "rolling origin")
MIN_TRAIN_YEARS = 7


def folds(df: pd.DataFrame, scheme: str):
    years = sorted(df["year"].unique())
    if scheme == "leave-one-year-out":
        for year in years:
            m = (df["year"] == year).to_numpy()
            yield np.where(~m)[0], np.where(m)[0]
    else:
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
    original = list(M.ML_FEATURES)

    rows = []
    for name, features in SETS.items():
        M.ML_FEATURES[:] = features          # the adapters read this list
        need = sorted(set(features + cfg["model"]["baseline_features"] + ["log_yield"]))
        for crop, sub in panel.groupby("crop"):
            sub = sub.dropna(subset=need).reset_index(drop=True)
            if len(sub) < 200:
                continue
            for scheme in SCHEMES:
                preds = {"gradient boosting": [], "ridge": []}
                truth = []
                for tr, te in folds(sub, scheme):
                    train, test = sub.iloc[tr], sub.iloc[te]
                    y_tr = train["log_yield"].to_numpy()
                    preds["gradient boosting"].append(SklearnAdapter(
                        make_gbm(cfg["model"]["gbm"], seed), raions
                    ).fit(train, y_tr).predict(test))
                    preds["ridge"].append(SklearnAdapter(
                        make_pipeline(StandardScaler(),
                                      RidgeCV(alphas=np.logspace(-2, 3, 20))),
                        raions, one_hot=True).fit(train, y_tr).predict(test))
                    truth.append(test["log_yield"].to_numpy())
                y = np.concatenate(truth)
                for model, chunks in preds.items():
                    rows.append({"features": name, "n_features": len(features),
                                 "crop": crop, "scheme": scheme, "model": model,
                                 "rmse": rmse(y, np.concatenate(chunks))})
        log.info("%s done (%d features)", name, len(features))
    M.ML_FEATURES[:] = original

    res = pd.DataFrame(rows)
    res.round(4).to_csv(TABLES / "ablation.csv", index=False)
    for scheme in SCHEMES:
        table = (res[res.scheme == scheme]
                 .pivot_table(index="features", columns="model", values="rmse")
                 .reindex(list(SETS)))
        table["n"] = [len(SETS[k]) for k in table.index]
        log.info("\n%s, RMSE averaged over crops\n%s", scheme, table.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    for ax, scheme in zip(axes, SCHEMES):
        table = (res[res.scheme == scheme]
                 .pivot_table(index="features", columns="model", values="rmse")
                 .reindex(list(SETS)))
        pos = np.arange(len(table))
        for i, model in enumerate(table.columns):
            ax.barh(pos + (i - 0.5) * 0.36, table[model], 0.36, label=model)
        ax.set_yticks(pos, table.index)
        ax.invert_yaxis()
        ax.set_xlabel("RMSE, log yield")
        ax.set_title(scheme)
        ax.grid(axis="x", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Dropping feature groups")
    fig.tight_layout()
    fig.savefig(FIGURES / "ablation.png", dpi=150)
    log.info("figure written: results/figures/ablation.png")


if __name__ == "__main__":
    main()
