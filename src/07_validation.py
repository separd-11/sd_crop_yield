"""The central experiment: does the validation scheme change the verdict?

Neighbouring raions share the same drought, so a random split puts 2012 on
both sides and rewards a model for having already seen that year. Blocking by
year, or by raion, asks the question we care about: how well does this predict
a year, or a place, it has never seen.

Output: results/tables/validation.csv, results/figures/validation.png
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from models import BaselineFE, ML_FEATURES, gbm_matrix, make_gbm, rmse
from utils import FIGURES, PROCESSED, TABLES, get_logger, load_config

log = get_logger("validation")


def folds_random(df: pd.DataFrame, k: int, seed: int):
    for tr, te in KFold(n_splits=k, shuffle=True, random_state=seed).split(df):
        yield df.index[tr], df.index[te]


def folds_by_year(df: pd.DataFrame, *_):
    for year in sorted(df["year"].unique()):
        te = df.index[df["year"] == year]
        tr = df.index[df["year"] != year]
        if len(te) and len(tr):
            yield tr, te


def folds_by_raion(df: pd.DataFrame, *_):
    for raion in sorted(df["raion"].unique()):
        te = df.index[df["raion"] == raion]
        tr = df.index[df["raion"] != raion]
        if len(te) and len(tr):
            yield tr, te


SCHEMES = {
    "random 5-fold": folds_random,
    "leave-one-year-out": folds_by_year,
    "leave-one-raion-out": folds_by_raion,
}


def evaluate(df: pd.DataFrame, cfg: dict, raions: list[str]) -> list[dict]:
    seed = cfg["model"]["seed"]
    base_feats = cfg["model"]["baseline_features"]
    k = cfg["validation"]["random_folds"]
    out = []

    for scheme, splitter in SCHEMES.items():
        preds = {"baseline FE": [], "gradient boosting": [], "GBM monotonic": []}
        truth = []
        for tr_idx, te_idx in splitter(df, k, seed):
            tr, te = df.loc[tr_idx], df.loc[te_idx]
            y_tr = tr["log_yield"].to_numpy()

            base = BaselineFE(base_feats).fit(tr, y_tr)
            preds["baseline FE"].append(base.predict(te))

            X_tr, X_te = gbm_matrix(tr, raions), gbm_matrix(te, raions)
            for name, constraints in (("gradient boosting", None),
                                      ("GBM monotonic", cfg["model"]["monotonic"])):
                gbm = make_gbm(cfg["model"]["gbm"], seed, constraints).fit(X_tr, y_tr)
                preds[name].append(gbm.predict(X_te))

            truth.append(te["log_yield"].to_numpy())

        y = np.concatenate(truth)
        for name, chunks in preds.items():
            out.append({"scheme": scheme, "model": name,
                        "rmse": rmse(y, np.concatenate(chunks))})
    return out


def main() -> None:
    cfg = load_config()
    panel = pd.read_csv(PROCESSED / "panel.csv")
    raions = sorted(panel["raion"].unique())
    need = sorted(set(ML_FEATURES + cfg["model"]["baseline_features"] + ["log_yield"]))

    rows = []
    for crop, sub in panel.groupby("crop"):
        sub = sub.dropna(subset=need).reset_index(drop=True)
        if len(sub) < 200:
            log.info("%s skipped (%d rows)", crop, len(sub))
            continue
        log.info("validating %s (n=%d)", crop, len(sub))
        for rec in evaluate(sub, cfg, raions):
            rec["crop"] = crop
            rec["n"] = len(sub)
            rows.append(rec)

    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "validation.csv", index=False)

    table = res.pivot_table(index=["crop", "model"], columns="scheme", values="rmse")
    table = table[["random 5-fold", "leave-one-year-out", "leave-one-raion-out"]]
    table["year-block penalty %"] = 100 * (
        table["leave-one-year-out"] / table["random 5-fold"] - 1)
    log.info("\nRMSE on log yield by validation scheme\n%s",
             table.round(3).to_string())
    table.round(4).to_csv(TABLES / "validation_summary.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    crops = sorted(res["crop"].unique())
    width = 0.09
    positions = np.arange(len(crops))
    combos = [(s, m) for s in SCHEMES
              for m in ("baseline FE", "gradient boosting", "GBM monotonic")]
    for i, (scheme, model) in enumerate(combos):
        vals = [res[(res.crop == c) & (res.scheme == scheme) &
                    (res.model == model)]["rmse"].mean() for c in crops]
        ax.bar(positions + (i - len(combos) / 2 + 0.5) * width, vals, width,
               label=f"{scheme} / {model.split()[-1]}")
    ax.set_xticks(positions)
    ax.set_xticklabels(crops, rotation=30, ha="right")
    ax.set_ylabel("RMSE, log yield")
    ax.set_title("Random splitting flatters the model on a spatio-temporal panel")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "validation.png", dpi=150)
    log.info("figure written: results/figures/validation.png")


if __name__ == "__main__":
    main()
