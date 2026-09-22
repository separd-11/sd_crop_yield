"""Parametric response function per crop.

Output: results/tables/baseline_coefficients.csv
"""
from __future__ import annotations
import pandas as pd

from models import BaselineFE
from utils import PROCESSED, TABLES, get_logger, load_config

log = get_logger("model_baseline")


def main() -> None:
    cfg = load_config()
    feats = cfg["model"]["baseline_features"]
    panel = pd.read_csv(PROCESSED / "panel.csv")

    rows = []
    for crop, sub in panel.groupby("crop"):
        sub = sub.dropna(subset=feats + ["log_yield"])
        if len(sub) < 100:
            log.warning("%s has only %d rows, skipped", crop, len(sub))
            continue
        model = BaselineFE(feats).fit(sub, sub["log_yield"].to_numpy())
        summary = model.summary(sub, sub["log_yield"].to_numpy())
        summary.insert(0, "crop", crop)
        summary["n"] = len(sub)
        rows.append(summary)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(TABLES / "baseline_coefficients.csv", index=False)

    log.info("effect of killing degree days (heat above 30C), %% per KDD unit:")
    kdd = out[out.term == "kdd"].sort_values("coef")
    for _, r in kdd.iterrows():
        log.info("  %-18s %7.3f%%  (se %.3f, t %5.1f, n=%d)",
                 r.crop, 100 * r.coef, 100 * r.se, r.t, r.n)


if __name__ == "__main__":
    main()
