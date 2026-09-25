"""Tuning changes the answer too, and it can be got wrong in its own way.

Hyper-parameters have to be chosen somehow, and the choice is usually made by
an inner cross-validation. If that inner loop splits at random while the outer
one blocks the year, the search is rewarded for picking a model that exploits
the leak - and then that model is graded honestly and fails.

Three configurations on the pooled panel:

  naive        random outer, random inner
  mismatched   year-blocked outer, random inner
  consistent   year-blocked outer, year-blocked inner

Output: results/tables/nested_tuning.csv, results/figures/nested_tuning.png
"""
from __future__ import annotations
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from models import ML_FEATURES, SklearnAdapter, make_gbm, rmse
from utils import FIGURES, PROCESSED, TABLES, get_logger, load_config

log = get_logger("nested")

GRID = [{"max_depth": d, "min_samples_leaf": leaf}
        for d, leaf in itertools.product([2, 3, 4, 6], [5, 20, 50])]

CONFIGS = {
    "naive": ("random", "random"),
    "mismatched": ("year", "random"),
    "consistent": ("year", "year"),
}


def split_indices(df: pd.DataFrame, kind: str, seed: int, n_blocks: int = 5):
    if kind == "random":
        yield from KFold(n_blocks, shuffle=True, random_state=seed).split(df)
        return
    years = np.array(sorted(df["year"].unique()))
    blocks = np.array_split(years, min(n_blocks, len(years)))
    for block in blocks:
        m = df["year"].isin(block).to_numpy()
        if m.sum() and (~m).sum():
            yield np.where(~m)[0], np.where(m)[0]


def outer_splits(df: pd.DataFrame, kind: str, seed: int):
    if kind == "random":
        yield from KFold(5, shuffle=True, random_state=seed).split(df)
        return
    for year in sorted(df["year"].unique()):
        m = (df["year"] == year).to_numpy()
        yield np.where(~m)[0], np.where(m)[0]


def tune(train: pd.DataFrame, cfg: dict, raions, crops, inner: str, seed: int) -> dict:
    """Pick hyper-parameters by an inner cross-validation of the given kind."""
    scores = np.zeros(len(GRID))
    counts = np.zeros(len(GRID))
    for tr, te in split_indices(train.reset_index(drop=True), inner, seed):
        sub_tr = train.iloc[tr]
        sub_te = train.iloc[te]
        y_tr = sub_tr["log_yield"].to_numpy()
        y_te = sub_te["log_yield"].to_numpy()
        for i, params in enumerate(GRID):
            gbm_cfg = {**cfg["model"]["gbm"], **params}
            model = SklearnAdapter(make_gbm(gbm_cfg, seed), raions, crops=crops)
            pred = model.fit(sub_tr, y_tr).predict(sub_te)
            scores[i] += rmse(y_te, pred) ** 2 * len(y_te)
            counts[i] += len(y_te)
    return GRID[int(np.argmin(scores / counts))]


def main() -> None:
    cfg = load_config()
    seed = cfg["model"]["seed"]
    panel = pd.read_csv(PROCESSED / "panel.csv")
    need = sorted(set(ML_FEATURES + cfg["model"]["baseline_features"] + ["log_yield"]))
    df = panel.dropna(subset=need).reset_index(drop=True)
    raions, crops = sorted(df["raion"].unique()), sorted(df["crop"].unique())

    rows = []
    for name, (outer, inner) in CONFIGS.items():
        preds, truth = [], []
        for fold, (tr, te) in enumerate(outer_splits(df, outer, seed)):
            train, test = df.iloc[tr], df.iloc[te]
            chosen = tune(train, cfg, raions, crops, inner, seed)
            gbm_cfg = {**cfg["model"]["gbm"], **chosen}
            model = SklearnAdapter(make_gbm(gbm_cfg, seed), raions, crops=crops)
            y_tr = train["log_yield"].to_numpy()
            preds.append(model.fit(train, y_tr).predict(test))
            truth.append(test["log_yield"].to_numpy())
            rows.append({"config": name, "fold": fold, **chosen})
        y = np.concatenate(truth)
        score = rmse(y, np.concatenate(preds))
        for r in rows:
            if r["config"] == name:
                r["rmse"] = score
        log.info("%-12s outer=%-6s inner=%-6s  RMSE %.3f", name, outer, inner, score)

    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "nested_tuning.csv", index=False)

    summary = res.groupby("config").agg(
        rmse=("rmse", "first"),
        depth=("max_depth", "mean"),
        leaf=("min_samples_leaf", "mean"),
        folds=("fold", "size"))
    log.info("\nwhat each configuration picked, averaged over outer folds\n%s",
             summary.round(3).to_string())

    for name in CONFIGS:
        sel = res[res.config == name]
        log.info("%s: depth %s | leaf %s", name,
                 dict(sel.max_depth.value_counts().sort_index()),
                 dict(sel.min_samples_leaf.value_counts().sort_index()))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    order = list(CONFIGS)
    axes[0].bar(order, [summary.loc[c, "rmse"] for c in order],
                color=["tab:blue", "tab:orange", "tab:green"])
    axes[0].set_ylabel("RMSE, log yield")
    axes[0].set_title("Honest score of the tuned model")
    axes[0].tick_params(axis="x", rotation=15)

    for ax, column, label in ((axes[1], "max_depth", "tree depth chosen"),
                              (axes[2], "min_samples_leaf", "minimum leaf size chosen")):
        values = sorted(res[column].unique())
        width = 0.26
        for i, name in enumerate(order):
            counts = res[res.config == name][column].value_counts(normalize=True)
            ax.bar(np.arange(len(values)) + (i - 1) * width,
                   [counts.get(v, 0) for v in values], width, label=name)
        ax.set_xticks(range(len(values)), values)
        ax.set_ylabel("share of outer folds")
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("A random inner loop buys a bigger model, then pays for it")
    fig.tight_layout()
    fig.savefig(FIGURES / "nested_tuning.png", dpi=150)
    log.info("figure written: results/figures/nested_tuning.png")


if __name__ == "__main__":
    main()
