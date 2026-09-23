"""How much of the year-to-year swing does each model actually capture?

Regress the predicted year mean on the actual year mean, out of sample. A slope
of one means the model tracks good and bad years fully; a slope near zero means
it always answers with the average year.

This is the mechanism behind the validation result: boosting does not fail
randomly on unseen years, it shrinks them toward the middle.

Output: results/tables/shrinkage.csv, results/figures/shrinkage.png
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import BaselineFE, ML_FEATURES, gbm_matrix, make_gbm
from utils import FIGURES, PROCESSED, TABLES, get_logger, load_config

log = get_logger("shrinkage")
HIGHLIGHT = ["porumb", "griu", "floarea_soarelui", "soia"]


def loyo_year_means(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    raions = sorted(panel["raion"].unique())
    base_feats = cfg["model"]["baseline_features"]
    need = sorted(set(ML_FEATURES + base_feats + ["log_yield"]))
    rows = []
    for crop, sub in panel.groupby("crop"):
        sub = sub.dropna(subset=need)
        if len(sub) < 200:
            continue
        for year in sorted(sub["year"].unique()):
            test, train = sub[sub["year"] == year], sub[sub["year"] != year]
            if len(test) < 5:
                continue
            y_tr = train["log_yield"].to_numpy()
            pred_b = BaselineFE(base_feats).fit(train, y_tr).predict(test)
            pred_g = make_gbm(cfg["model"]["gbm"], cfg["model"]["seed"]).fit(
                gbm_matrix(train, raions), y_tr).predict(gbm_matrix(test, raions))
            rows.append({"crop": crop, "year": year,
                         "actual": test["log_yield"].mean(),
                         "baseline": pred_b.mean(), "boosting": pred_g.mean()})
        log.info("%s done", crop)
    return pd.DataFrame(rows)


def slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(x - x.mean(), y - y.mean(), 1)[0])


def main() -> None:
    cfg = load_config()
    panel = pd.read_csv(PROCESSED / "panel.csv")
    means = loyo_year_means(panel, cfg)
    means.to_csv(TABLES / "shrinkage.csv", index=False)

    rows = []
    for crop, s in means.groupby("crop"):
        rows.append({"crop": crop,
                     "baseline": slope(s["actual"].to_numpy(), s["baseline"].to_numpy()),
                     "boosting": slope(s["actual"].to_numpy(), s["boosting"].to_numpy())})
    table = pd.DataFrame(rows).set_index("crop")

    centred = means.copy()
    for col in ("actual", "baseline", "boosting"):
        centred[col] -= centred.groupby("crop")[col].transform("mean")
    overall = {"baseline": slope(centred["actual"].to_numpy(), centred["baseline"].to_numpy()),
               "boosting": slope(centred["actual"].to_numpy(), centred["boosting"].to_numpy())}
    table.loc["ALL CROPS"] = overall
    table.round(3).to_csv(TABLES / "shrinkage_slopes.csv")
    log.info("\nslope of predicted year mean on actual year mean\n%s",
             table.round(2).to_string())
    log.info("\nBoosting captures %.0f%% of the swing between years, "
             "the parametric model %.0f%%.",
             100 * overall["boosting"], 100 * overall["baseline"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    lim = [centred["actual"].min() - 0.1, centred["actual"].max() + 0.1]
    axes[0].plot(lim, lim, color="black", linewidth=0.9, label="perfect tracking")
    for col, marker, name in (("baseline", "o", "baseline FE"),
                              ("boosting", "s", "gradient boosting")):
        axes[0].scatter(centred["actual"], centred[col], s=18, marker=marker,
                        alpha=0.65, label=f"{name} (slope {overall[col]:.2f})")
        fit = np.poly1d(np.polyfit(centred["actual"], centred[col], 1))
        axes[0].plot(lim, fit(lim), linewidth=1.6)
    axes[0].set_xlabel("Actual year mean, log yield (centred by crop)")
    axes[0].set_ylabel("Predicted year mean")
    axes[0].set_title("Both models flatten the good and bad years.\n"
                      "Boosting flattens them twice as hard.")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.25)

    crops = table.drop(index="ALL CROPS").sort_values("boosting")
    y = np.arange(len(crops))
    axes[1].barh(y - 0.2, crops["baseline"], 0.4, label="baseline FE")
    axes[1].barh(y + 0.2, crops["boosting"], 0.4, label="gradient boosting")
    axes[1].axvline(1.0, color="black", linewidth=0.9)
    axes[1].set_yticks(y, crops.index)
    axes[1].set_xlabel("Share of the between-year swing captured")
    axes[1].set_title("Every crop, same direction")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGURES / "shrinkage.png", dpi=150)
    log.info("figure written: results/figures/shrinkage.png")


if __name__ == "__main__":
    main()
