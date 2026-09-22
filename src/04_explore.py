"""Exploratory analysis of the panel.

Output: results/figures/eda_*.png, results/tables/panel_summary.csv
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import FIGURES, PROCESSED, TABLES, get_logger

log = get_logger("explore")
FOCUS = ["porumb", "griu", "floarea_soarelui", "soia"]


def shock_residuals(panel: pd.DataFrame) -> pd.DataFrame:
    """Log yield with the raion level and linear trend removed; what is left
    is mostly weather."""
    out = {}
    for crop, sub in panel.groupby("crop"):
        sub = sub.dropna(subset=["log_yield"])
        X = pd.get_dummies(sub["raion"], prefix="r").astype(float)
        X["trend"] = sub["trend"].to_numpy(float)
        X.insert(0, "const", 1.0)
        beta, *_ = np.linalg.lstsq(X.to_numpy(), sub["log_yield"].to_numpy(), rcond=None)
        resid = sub["log_yield"].to_numpy() - X.to_numpy() @ beta
        out[crop] = pd.Series(resid, index=pd.MultiIndex.from_frame(sub[["raion", "year"]]))
    return pd.DataFrame(out)


def main() -> None:
    panel = pd.read_csv(PROCESSED / "panel.csv")

    summary = panel.groupby("crop").agg(
        n=("yield_cwt_ha", "size"),
        raions=("raion", "nunique"),
        years=("year", "nunique"),
        yield_min=("yield_cwt_ha", "min"),
        yield_median=("yield_cwt_ha", "median"),
        yield_max=("yield_cwt_ha", "max"),
        shock_sd=("log_yield", "std"),
    ).round(2)
    summary.to_csv(TABLES / "panel_summary.csv")
    log.info("\n%s", summary.to_string())

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for ax, crop in zip(axes.ravel(), FOCUS):
        sub = panel[panel.crop == crop].groupby("year")["yield_cwt_ha"].mean()
        ax.plot(sub.index, sub.values, marker="o", markersize=3.5, linewidth=1.6)
        for drought in (2007, 2012, 2015, 2020, 2022):
            ax.axvline(drought, color="firebrick", alpha=0.25, linewidth=3)
        ax.set_title(crop)
        ax.grid(alpha=0.25)
    fig.supylabel("Average yield across raions, centners per hectare")
    fig.suptitle("Drought years shaded: 2007, 2012, 2015, 2020, 2022")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_yields.png", dpi=150)

    # How much a crop mix can spread risk depends on this.
    shocks = shock_residuals(panel)
    order = [c for c in ["griu", "orz", "porumb", "floarea_soarelui", "soia",
                         "sfecla", "cartofi", "legume"] if c in shocks.columns]
    corr = shocks[order].corr()
    corr.round(3).to_csv(TABLES / "shock_correlation.csv")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.to_numpy(), vmin=0, vmax=1, cmap="YlOrRd")
    ax.set_xticks(range(len(order)), order, rotation=45, ha="right")
    ax.set_yticks(range(len(order)), order)
    for i in range(len(order)):
        for j in range(len(order)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Correlation of yield shocks across crops")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_shock_corr.png", dpi=150)

    upper = corr.to_numpy()[np.triu_indices(len(order), 1)]
    log.info("mean pairwise shock correlation: %.2f", upper.mean())
    log.info("figures written to results/figures/")


if __name__ == "__main__":
    main()
