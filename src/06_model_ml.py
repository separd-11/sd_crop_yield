"""Gradient boosting per crop, with and without monotonic constraints.

If a tree ensemble given no functional form reproduces the shape the
parametric model assumes, the assumed shape is probably right.

Output: results/tables/ml_partial_dependence.csv, results/figures/response_kdd.png
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

from models import ML_FEATURES, gbm_matrix, make_gbm
from utils import FIGURES, PROCESSED, TABLES, get_logger, load_config

log = get_logger("model_ml")
FOCUS = ["porumb", "soia", "floarea_soarelui", "griu", "orz", "sfecla"]
WINTER = {"griu", "orz"}


def kdd_curve(model, X) -> tuple:
    result = partial_dependence(model, X, [ML_FEATURES.index("kdd")],
                                grid_resolution=40, percentiles=(0.02, 0.98),
                                kind="average")
    grid, avg = result["grid_values"][0], result["average"][0]
    return grid, avg - avg[0]


def averaged_curve(X, y, gbm_cfg, constraints, seeds):
    """Average over seeds to soften the staircase trees produce."""
    curves = []
    for seed in range(seeds):
        model = make_gbm(gbm_cfg, seed, constraints).fit(X, y)
        grid, delta = kdd_curve(model, X)
        curves.append(delta)
    return grid, np.mean(curves, axis=0)


def main() -> None:
    cfg = load_config()
    mcfg, seed = cfg["model"], cfg["model"]["seed"]
    panel = pd.read_csv(PROCESSED / "panel.csv")
    raions = sorted(panel["raion"].unique())

    records, curves = [], {}
    for crop, sub in panel.groupby("crop"):
        sub = sub.dropna(subset=ML_FEATURES + ["log_yield"])
        if len(sub) < 150:
            continue
        X, y = gbm_matrix(sub, raions), sub["log_yield"].to_numpy()

        free = make_gbm(mcfg["gbm"], seed).fit(X, y)
        mono = make_gbm(mcfg["gbm"], seed, mcfg["monotonic"]).fit(X, y)

        for label, constraints in (("unconstrained", None),
                                   ("monotonic", mcfg["monotonic"])):
            grid, delta = averaged_curve(X, y, mcfg["gbm"], constraints,
                                         mcfg["curve_seeds"])
            if label == "monotonic":
                curves[crop] = (grid, delta)
            for g, d in zip(grid, delta):
                records.append({"crop": crop, "model": label,
                                "kdd": g, "delta_log_yield": d})
        log.info("%-18s n=%4d  R2 free=%.3f  R2 monotonic=%.3f",
                 crop, len(sub), free.score(X, y), mono.score(X, y))

    pd.DataFrame(records).to_csv(TABLES / "ml_partial_dependence.csv", index=False)

    # Past the 90th percentile only a tenth of the panel remains.
    x_max = float(panel["kdd"].quantile(0.90))

    fig, ax = plt.subplots(figsize=(8, 5))
    for crop in FOCUS:
        if crop not in curves:
            continue
        grid, delta = curves[crop]
        keep = grid <= x_max
        ax.plot(grid[keep], 100 * delta[keep], "--" if crop in WINTER else "-",
                linewidth=2, label=crop)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Killing degree days, April to August (degrees above 30 C)")
    ax.set_ylabel("Change in yield, %")
    ax.set_title("Heat response under monotonic constraints\n"
                 "dashed = winter cereals, solid = summer crops")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "response_kdd.png", dpi=150)
    log.info("figure written: results/figures/response_kdd.png")


if __name__ == "__main__":
    main()
