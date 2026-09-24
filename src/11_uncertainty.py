"""Is the reversal real, or could a different draw of years undo it?

The validation result so far is two point estimates. Here each scheme is run
once to collect out-of-fold predictions, and the evaluation is then resampled:
years are drawn with replacement, because a year is the block inside which
errors are correlated - that is what the mechanism analysis established.

Reported for every crop and scheme: the gap in RMSE between boosting and the
parametric model, with a percentile interval. A positive gap means boosting is
worse. The claim survives only if the interval sits clear of zero and flips
sign between the schemes.

Output: results/tables/uncertainty.csv, results/figures/uncertainty.png
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

log = get_logger("uncertainty")

N_BOOT = 2000
N_REPEATS = 20          # different random fold assignments
SCHEMES = ("random 5-fold", "leave-one-year-out", "leave-one-raion-out")
MAIN_CROPS = ("porumb", "griu", "orz", "floarea_soarelui")


def folds(df: pd.DataFrame, scheme: str, seed: int):
    if scheme == "random 5-fold":
        yield from KFold(5, shuffle=True, random_state=seed).split(df)
    elif scheme == "leave-one-year-out":
        for year in sorted(df["year"].unique()):
            mask = (df["year"] == year).to_numpy()
            yield np.where(~mask)[0], np.where(mask)[0]
    else:
        for raion in sorted(df["raion"].unique()):
            mask = (df["raion"] == raion).to_numpy()
            if mask.sum() and (~mask).sum() > 50:
                yield np.where(~mask)[0], np.where(mask)[0]


def out_of_fold(df: pd.DataFrame, cfg: dict, raions: list[str],
                scheme: str, seed: int) -> pd.DataFrame:
    """Predictions for every row, made by a model that never saw that row."""
    base_feats = cfg["model"]["baseline_features"]
    rows = []
    for tr, te in folds(df, scheme, seed):
        train, test = df.iloc[tr], df.iloc[te]
        y_tr = train["log_yield"].to_numpy()
        pred_b = BaselineFE(base_feats).fit(train, y_tr).predict(test)
        pred_g = make_gbm(cfg["model"]["gbm"], cfg["model"]["seed"]).fit(
            gbm_matrix(train, raions), y_tr).predict(gbm_matrix(test, raions))
        rows.append(pd.DataFrame({"year": test["year"].to_numpy(),
                                  "y": test["log_yield"].to_numpy(),
                                  "base": pred_b, "gbm": pred_g}))
    return pd.concat(rows, ignore_index=True)


def bootstrap_gaps(oofs: dict[str, pd.DataFrame],
                   rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Resample whole years once per draw and score every scheme on that draw.

    Sharing the draw across schemes is what makes the comparison between them
    paired, so the interval on the shift is not inflated by year-to-year noise
    that both schemes see alike.
    """
    prepared = {}
    for scheme, oof in oofs.items():
        prepared[scheme] = (
            {y: oof.index[oof["year"] == y].to_numpy() for y in oof["year"].unique()},
            ((oof["y"] - oof["base"]).to_numpy() ** 2),
            ((oof["y"] - oof["gbm"]).to_numpy() ** 2))

    years = np.unique(next(iter(oofs.values()))["year"])
    out = {scheme: np.empty(N_BOOT) for scheme in oofs}
    for b in range(N_BOOT):
        picked = rng.choice(years, size=len(years), replace=True)
        for scheme, (by_year, err_b, err_g) in prepared.items():
            idx = np.concatenate([by_year[y] for y in picked if y in by_year])
            out[scheme][b] = np.sqrt(err_g[idx].mean()) - np.sqrt(err_b[idx].mean())
    return out


def main() -> None:
    cfg = load_config()
    seed = cfg["model"]["seed"]
    panel = pd.read_csv(PROCESSED / "panel.csv")
    raions = sorted(panel["raion"].unique())
    need = sorted(set(ML_FEATURES + cfg["model"]["baseline_features"] + ["log_yield"]))
    rng = np.random.default_rng(seed)

    rows, shifts = [], []
    for crop, sub in panel.groupby("crop"):
        sub = sub.dropna(subset=need).reset_index(drop=True)
        if len(sub) < 200:
            continue
        oofs = {s: out_of_fold(sub, cfg, raions, s, seed) for s in SCHEMES}
        draws = bootstrap_gaps(oofs, rng)
        for scheme in SCHEMES:
            oof = oofs[scheme]
            gap = rmse(oof["y"], oof["gbm"]) - rmse(oof["y"], oof["base"])
            lo, hi = np.percentile(draws[scheme], [2.5, 97.5])
            rows.append({"crop": crop, "scheme": scheme, "gap": gap,
                         "lo": lo, "hi": hi,
                         "p_boosting_worse": float((draws[scheme] > 0).mean())})

        # The claim of the project: blocking the year removes the advantage.
        shift = draws["leave-one-year-out"] - draws["random 5-fold"]
        oof_r, oof_y = oofs["random 5-fold"], oofs["leave-one-year-out"]
        point = ((rmse(oof_y["y"], oof_y["gbm"]) - rmse(oof_y["y"], oof_y["base"]))
                 - (rmse(oof_r["y"], oof_r["gbm"]) - rmse(oof_r["y"], oof_r["base"])))
        shifts.append({"crop": crop, "shift": point,
                       "lo": np.percentile(shift, 2.5),
                       "hi": np.percentile(shift, 97.5),
                       "p_positive": float((shift > 0).mean())})
        log.info("%s done", crop)

    res = pd.DataFrame(rows)

    # How much does the random-split verdict depend on which split you drew?
    spread = []
    for crop in MAIN_CROPS:
        sub = panel[panel["crop"] == crop].dropna(subset=need).reset_index(drop=True)
        gaps = []
        for r in range(N_REPEATS):
            oof = out_of_fold(sub, cfg, raions, "random 5-fold", seed + r)
            gaps.append(rmse(oof["y"], oof["gbm"]) - rmse(oof["y"], oof["base"]))
        spread.append({"crop": crop, "mean_gap": np.mean(gaps),
                       "sd_gap": np.std(gaps, ddof=1),
                       "min": np.min(gaps), "max": np.max(gaps),
                       "share_boosting_wins": float(np.mean(np.array(gaps) < 0))})
        log.info("repeated random splits: %s done", crop)
    pd.DataFrame(spread).round(4).to_csv(TABLES / "uncertainty_repeats.csv", index=False)

    res.round(4).to_csv(TABLES / "uncertainty.csv", index=False)
    shift_tab = pd.DataFrame(shifts)
    shift_tab.round(4).to_csv(TABLES / "uncertainty_shift.csv", index=False)
    log.info("\nhow much the gap moves when the year is blocked\n"
             "(positive = boosting loses ground; paired draws)\n%s",
             shift_tab.assign(interval=lambda d: d.apply(
                 lambda r: f"[{r.lo:+.3f}, {r.hi:+.3f}]", axis=1))
             [["crop", "shift", "interval", "p_positive"]]
             .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    log.info("\nRMSE gap, boosting minus parametric (positive = boosting worse)\n"
             "95%% interval from resampling whole years\n%s",
             res[res.crop.isin(MAIN_CROPS)]
             .assign(interval=lambda d: d.apply(
                 lambda r: f"[{r.lo:+.3f}, {r.hi:+.3f}]", axis=1))
             [["crop", "scheme", "gap", "interval", "p_boosting_worse"]]
             .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    log.info("\nspread of the random-split verdict over %d fold draws\n%s",
             N_REPEATS, pd.DataFrame(spread).round(3).to_string(index=False))

    crops = [c for c in MAIN_CROPS if c in set(res.crop)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    offsets = {s: o for s, o in zip(SCHEMES, (-0.26, 0.0, 0.26))}
    colours = dict(zip(SCHEMES, ("tab:blue", "tab:red", "tab:green")))
    for scheme in SCHEMES:
        sel = res[(res.scheme == scheme) & (res.crop.isin(crops))]
        pos = [crops.index(c) + offsets[scheme] for c in sel.crop]
        axes[0].errorbar(sel.gap, pos,
                         xerr=[sel.gap - sel.lo, sel.hi - sel.gap],
                         fmt="o", capsize=3, color=colours[scheme], label=scheme)
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_yticks(range(len(crops)), crops)
    axes[0].set_xlabel("RMSE gap, boosting minus parametric")
    axes[0].set_title("Under random and raion blocking boosting is\n"
                      "clearly ahead. Under year blocking it is not.")
    axes[0].legend(frameon=False, fontsize=8)

    sel = shift_tab[shift_tab.crop.isin(crops)].set_index("crop").loc[crops]
    pos = np.arange(len(crops))
    axes[1].errorbar(sel["shift"], pos,
                     xerr=[sel["shift"] - sel["lo"], sel["hi"] - sel["shift"]],
                     fmt="s", capsize=3, color="tab:purple")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_yticks(pos, crops)
    axes[1].set_xlabel("How far the gap moves when the year is blocked")
    axes[1].set_title("The move itself is unambiguous.\n"
                      "Paired draws, so year noise cancels.")

    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("95% intervals from drawing whole years with replacement", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "uncertainty.png", dpi=150, bbox_inches="tight")
    log.info("figure written: results/figures/uncertainty.png")


if __name__ == "__main__":
    main()
