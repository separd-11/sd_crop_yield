"""One model over all crops at once, to buy back statistical power.

Nineteen years per crop is not many, and part of the width of the intervals in
11_uncertainty is simply that. Pooling keeps a separate weather response for
every crop but estimates the raion effects and the noise on 4500 rows instead
of six hundred, so if the wide intervals were a power problem they should
narrow here.

Output: results/tables/pooled.csv, results/figures/pooled.png
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models import (ML_FEATURES, PooledFE, RaionMean, SklearnAdapter, SPLINE_ON,
                    make_gbm, rmse)
from utils import FIGURES, PROCESSED, TABLES, get_logger, load_config

log = get_logger("pooled")

SCHEMES = ("random 5-fold", "leave-one-year-out", "leave-one-raion-out",
           "rolling origin")
N_BOOT = 2000
MIN_TRAIN_YEARS = 7


def folds(df: pd.DataFrame, scheme: str, seed: int):
    from sklearn.model_selection import KFold
    if scheme == "random 5-fold":
        yield from KFold(5, shuffle=True, random_state=seed).split(df)
    elif scheme == "leave-one-year-out":
        for year in sorted(df["year"].unique()):
            m = (df["year"] == year).to_numpy()
            yield np.where(~m)[0], np.where(m)[0]
    elif scheme == "leave-one-raion-out":
        for raion in sorted(df["raion"].unique()):
            m = (df["raion"] == raion).to_numpy()
            if m.sum() and (~m).sum() > 200:
                yield np.where(~m)[0], np.where(m)[0]
    else:
        years = sorted(df["year"].unique())
        for target in years[MIN_TRAIN_YEARS:]:
            tr = np.where((df["year"] < target).to_numpy())[0]
            te = np.where((df["year"] == target).to_numpy())[0]
            if len(te) >= 5:
                yield tr, te


def build_zoo(cfg: dict, raions: list[str], crops: list[str]) -> dict:
    seed = cfg["model"]["seed"]
    ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
    return {
        "raion x crop mean + trend": lambda: RaionMean(with_trend=True, by="raion_crop"),
        "pooled FE": lambda: PooledFE(cfg["model"]["baseline_features"]),
        "ridge, all features": lambda: SklearnAdapter(ridge(), raions, one_hot=True,
                                                      crops=crops),
        "splines + ridge": lambda: SklearnAdapter(ridge(), raions, one_hot=True,
                                                  spline_on=SPLINE_ON, crops=crops),
        "extra trees": lambda: SklearnAdapter(
            ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                                random_state=seed), raions, crops=crops),
        "gradient boosting": lambda: SklearnAdapter(
            make_gbm(cfg["model"]["gbm"], seed), raions, crops=crops),
    }


def bootstrap_gap(oof: pd.DataFrame, a: str, b: str,
                  rng: np.random.Generator) -> tuple[float, float]:
    """Interval on rmse(a) - rmse(b), drawing whole years with replacement."""
    years = oof["year"].unique()
    by_year = {y: oof.index[oof["year"] == y].to_numpy() for y in years}
    err_a = (oof["y"] - oof[a]).to_numpy() ** 2
    err_b = (oof["y"] - oof[b]).to_numpy() ** 2
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = np.concatenate([by_year[y] for y in rng.choice(years, len(years), True)])
        draws[i] = np.sqrt(err_a[idx].mean()) - np.sqrt(err_b[idx].mean())
    return tuple(np.percentile(draws, [2.5, 97.5]))


def main() -> None:
    cfg = load_config()
    seed = cfg["model"]["seed"]
    panel = pd.read_csv(PROCESSED / "panel.csv")
    need = sorted(set(ML_FEATURES + cfg["model"]["baseline_features"] + ["log_yield"]))
    df = panel.dropna(subset=need).reset_index(drop=True)
    df["raion_crop"] = df["raion"] + "|" + df["crop"]
    raions, crops = sorted(df["raion"].unique()), sorted(df["crop"].unique())
    zoo = build_zoo(cfg, raions, crops)
    rng = np.random.default_rng(seed)

    rows, intervals = [], []
    for scheme in SCHEMES:
        preds = {name: [] for name in zoo}
        keep = []
        for tr, te in folds(df, scheme, seed):
            train, test = df.iloc[tr], df.iloc[te]
            y_tr = train["log_yield"].to_numpy()
            for name, factory in zoo.items():
                preds[name].append(factory().fit(train, y_tr).predict(test))
            keep.append(test[["year", "log_yield"]])
        base = pd.concat(keep, ignore_index=True).rename(columns={"log_yield": "y"})
        oof = base.assign(**{n: np.concatenate(v) for n, v in preds.items()})
        for name in zoo:
            rows.append({"scheme": scheme, "model": name,
                         "rmse": rmse(oof["y"], oof[name])})
        for flexible in ("gradient boosting", "extra trees"):
            lo, hi = bootstrap_gap(oof, flexible, "pooled FE", rng)
            gap = rmse(oof["y"], oof[flexible]) - rmse(oof["y"], oof["pooled FE"])
            intervals.append({"scheme": scheme, "model": flexible, "gap": gap,
                              "lo": lo, "hi": hi, "width": hi - lo})
        log.info("%s done", scheme)

    res = pd.DataFrame(rows)
    res.round(4).to_csv(TABLES / "pooled.csv", index=False)
    log.info("\nRMSE, one model over all crops\n%s",
             res.pivot(index="model", columns="scheme", values="rmse")[list(SCHEMES)]
             .round(3).to_string())

    iv = pd.DataFrame(intervals)
    iv.round(4).to_csv(TABLES / "pooled_intervals.csv", index=False)
    log.info("\ngap against the pooled parametric model, positive = flexible worse\n%s",
             iv.assign(interval=lambda d: d.apply(
                 lambda r: f"[{r.lo:+.3f}, {r.hi:+.3f}]", axis=1))
             [["scheme", "model", "gap", "interval", "width"]]
             .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    per_crop = pd.read_csv(TABLES / "uncertainty.csv")
    per_crop["width"] = per_crop["hi"] - per_crop["lo"]
    old = per_crop[per_crop.scheme == "leave-one-year-out"]["width"].mean()
    new = iv[(iv.scheme == "leave-one-year-out") & (iv.model == "gradient boosting")
             ]["width"].iloc[0]
    log.info("\ninterval width, leave-one-year-out: per crop %.3f, pooled %.3f (%.0f%% of it)",
             old, new, 100 * new / old)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    table = res.pivot(index="model", columns="scheme", values="rmse")[list(SCHEMES)]
    order = table["rolling origin"].sort_values().index.tolist()
    pos = np.arange(len(order))
    for i, scheme in enumerate(SCHEMES):
        axes[0].barh(pos + (i - 1.5) * 0.2, table.loc[order, scheme], 0.2, label=scheme)
    axes[0].set_yticks(pos, order)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("RMSE, log yield")
    axes[0].set_title("All crops in one model")
    axes[0].legend(frameon=False, fontsize=8)

    sel = iv[iv.model == "gradient boosting"]
    pos = np.arange(len(sel))
    axes[1].errorbar(sel["gap"], pos, xerr=[sel["gap"] - sel["lo"], sel["hi"] - sel["gap"]],
                     fmt="o", capsize=3, color="tab:red")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_yticks(pos, sel["scheme"])
    axes[1].set_xlabel("Boosting minus parametric, pooled")
    axes[1].set_title("Narrower on 4500 rows, but the year-blocked\n"
                      "gap still spans zero")

    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "pooled.png", dpi=150)
    log.info("figure written: results/figures/pooled.png")


if __name__ == "__main__":
    main()
