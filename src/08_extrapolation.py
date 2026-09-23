"""Why does gradient boosting lose on unseen years?

Three candidate explanations, tested in order:

1. Extrapolation. A held-out year sits outside the weather the model saw, and
   trees can only answer with their most extreme leaf.
2. Atypicality. The further a year is from the average year, the worse the
   ensemble does.
3. Year level. Under random splitting the model sees some raions from year y
   and can infer that year's overall level; blocking the year takes that away.

Output: results/tables/extrapolation*.csv, results/figures/extrapolation.png
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import BaselineFE, ML_FEATURES, gbm_matrix, make_gbm
from utils import FIGURES, PROCESSED, TABLES, get_logger, load_config

log = get_logger("extrapolation")

# A distance over all 18 features would be dominated by the monthly duplicates.
AXES = ["gdd", "kdd", "prec_season"]


def distances(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mu = train[AXES].mean().to_numpy()
    sd = train[AXES].std().to_numpy().copy()
    sd[sd == 0] = 1.0
    tr, te = (train[AXES].to_numpy() - mu) / sd, (test[AXES].to_numpy() - mu) / sd

    beyond = np.maximum(tr.min(axis=0) - te, te - tr.max(axis=0))
    out_of_range = np.maximum(beyond, 0).max(axis=1)
    d = np.linalg.norm(te[:, None, :] - tr[None, :, :], axis=2)
    return out_of_range, np.sort(d, axis=1)[:, :5].mean(axis=1)


def year_atypicality(panel: pd.DataFrame) -> pd.Series:
    """Distance of each year from the centre of the other years."""
    means = panel.groupby("year")[AXES].mean()
    out = {}
    for year in means.index:
        others = means.drop(index=year)
        z = (means.loc[year] - others.mean()) / others.std()
        out[year] = float(np.sqrt((z ** 2).sum()))
    return pd.Series(out)


def run_folds(panel: pd.DataFrame, cfg: dict, raions: list[str]) -> pd.DataFrame:
    seed, base_feats = cfg["model"]["seed"], cfg["model"]["baseline_features"]
    need = sorted(set(ML_FEATURES + base_feats + ["log_yield"]))
    rows = []
    for crop, sub in panel.groupby("crop"):
        sub = sub.dropna(subset=need).reset_index(drop=True)
        if len(sub) < 200:
            continue
        for year in sorted(sub["year"].unique()):
            test, train = sub[sub["year"] == year], sub[sub["year"] != year]
            if len(test) < 5 or len(train) < 50:
                continue
            y_tr, y_te = train["log_yield"].to_numpy(), test["log_yield"].to_numpy()
            pred_b = BaselineFE(base_feats).fit(train, y_tr).predict(test)
            pred_g = make_gbm(cfg["model"]["gbm"], seed).fit(
                gbm_matrix(train, raions), y_tr).predict(gbm_matrix(test, raions))
            oor, knn = distances(train, test)
            rows.append(pd.DataFrame({
                "crop": crop, "year": year, "out_of_range": oor, "knn_dist": knn,
                "err_baseline": np.abs(y_te - pred_b),
                "err_gbm": np.abs(y_te - pred_g),
                "centred_base": (y_te - y_te.mean()) - (pred_b - pred_b.mean()),
                "centred_gbm": (y_te - y_te.mean()) - (pred_g - pred_g.mean()),
                "bias_base": pred_b.mean() - y_te.mean(),
                "bias_gbm": pred_g.mean() - y_te.mean()}))
        log.info("%s done", crop)
    return pd.concat(rows, ignore_index=True)


def rms(s) -> float:
    return float(np.sqrt((np.asarray(s, float) ** 2).mean()))


def main() -> None:
    cfg = load_config()
    panel = pd.read_csv(PROCESSED / "panel.csv")
    res = run_folds(panel, cfg, sorted(panel["raion"].unique()))
    res.to_csv(TABLES / "extrapolation.csv", index=False)

    log.info("\n%s\n1. OUTSIDE THE TRAINING WEATHER RANGE", "=" * 70)
    inside = res["out_of_range"] == 0
    for name, col in (("baseline FE", "err_baseline"), ("gradient boosting", "err_gbm")):
        log.info("  %-20s inside %.3f  outside %.3f", name,
                 rms(res.loc[inside, col]), rms(res.loc[~inside, col]))
    log.info("  only %.1f%% of observations fall outside", 100 * (~inside).mean())

    log.info("\n2. ATYPICAL YEARS")
    atyp = year_atypicality(panel)
    by_year = res.groupby("year").agg(base=("err_baseline", rms), gbm=("err_gbm", rms))
    by_year["atypicality"] = atyp.reindex(by_year.index)
    by_year["handicap"] = 100 * (by_year["gbm"] / by_year["base"] - 1)
    by_year.round(3).to_csv(TABLES / "extrapolation_years.csv")
    split = by_year["atypicality"].median()
    log.info("  typical years   handicap %+.1f%%",
             by_year.loc[by_year.atypicality <= split, "handicap"].mean())
    log.info("  atypical years  handicap %+.1f%%",
             by_year.loc[by_year.atypicality > split, "handicap"].mean())
    log.info("  correlation with atypicality: %.2f",
             by_year[["atypicality", "handicap"]].corr().iloc[0, 1])

    log.info("\n3. GETTING THE YEAR LEVEL RIGHT")
    folds = res.drop_duplicates(["crop", "year"])
    b_base, b_gbm = rms(folds.bias_base), rms(folds.bias_gbm)
    w_base, w_gbm = rms(res.centred_base), rms(res.centred_gbm)
    log.info("  baseline FE        year level %.3f   within year %.3f", b_base, w_base)
    log.info("  gradient boosting  year level %.3f   within year %.3f", b_gbm, w_gbm)
    log.info("  boosting is %+.0f%% worse on the level, %+.0f%% worse within the year",
             100 * (b_gbm / b_base - 1), 100 * (w_gbm / w_base - 1))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    edges = np.quantile(res["knn_dist"], np.linspace(0, 1, 9))
    binned = res.groupby(pd.cut(res["knn_dist"], edges, include_lowest=True),
                         observed=True).agg(x=("knn_dist", "mean"),
                                            base=("err_baseline", rms),
                                            gbm=("err_gbm", rms))
    axes[0].plot(binned.x, binned.base, "o-", label="baseline FE")
    axes[0].plot(binned.x, binned.gbm, "s-", label="gradient boosting")
    axes[0].set_xlabel("Distance to nearest training weather")
    axes[0].set_ylabel("RMSE, log yield")
    axes[0].set_title("1. Unfamiliar weather")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].scatter(by_year.atypicality, by_year.handicap, s=35)
    for yr, row in by_year.iterrows():
        if row.handicap > 30 or row.atypicality > 2.5:
            axes[1].annotate(str(yr), (row.atypicality, row.handicap),
                             fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("How unusual the year was")
    axes[1].set_ylabel("Boosting handicap, %")
    axes[1].set_title("2. Atypical years")

    x = np.arange(2)
    axes[2].bar(x - 0.18, [b_base, w_base], 0.36, label="baseline FE")
    axes[2].bar(x + 0.18, [b_gbm, w_gbm], 0.36, label="gradient boosting")
    axes[2].set_xticks(x, ["year level", "within year"])
    axes[2].set_ylabel("RMSE, log yield")
    axes[2].set_title("3. Where the error sits")
    axes[2].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "extrapolation.png", dpi=150)
    log.info("\nfigure written: results/figures/extrapolation.png")


if __name__ == "__main__":
    main()
