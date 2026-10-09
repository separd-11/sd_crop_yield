"""What a model does with a year it has never seen.

Hold one year out, train both models on the rest, and look at what they say
about it. The parametric model tracks good and bad years; the tree ensemble
answers with something close to an average year. That shrinkage is the finding
of the project, made touchable.

A warming-scenario tab was dropped on purpose: neither model is validated
outside the weather it was trained on, so offering the slider would promise
what the work itself refuses.

Run: streamlit run app/app.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geo import choropleth                                       # noqa: E402
from models import BaselineFE, ML_FEATURES, gbm_matrix, make_gbm  # noqa: E402
from utils import PROCESSED, TABLES, load_config                  # noqa: E402

CROPS = {
    "porumb": "Кукуруза", "griu": "Пшеница", "orz": "Ячмень",
    "floarea_soarelui": "Подсолнечник", "soia": "Соя",
    "sfecla": "Сахарная свёкла", "cartofi": "Картофель",
    "legume": "Овощи открытого грунта",
}
DROUGHTS = {2007, 2012, 2015, 2020, 2022, 2024}
FACT, PARAM, BOOST = "Факт", "Параметрическая", "Бустинг"

st.set_page_config(page_title="Урожайность и невиданные годы", layout="wide")


@st.cache_data(show_spinner=False)
def panel() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "panel.csv")


@st.cache_data(show_spinner=False)
def hold_out(crop: str, year: int) -> pd.DataFrame:
    """Train both models without this year, predict it, return per raion."""
    cfg = load_config()
    base_feats = cfg["model"]["baseline_features"]
    need = sorted(set(ML_FEATURES + base_feats + ["log_yield"]))
    sub = panel()
    raions = sorted(sub["raion"].unique())
    sub = sub[sub["crop"] == crop].dropna(subset=need)
    test, train = sub[sub["year"] == year], sub[sub["year"] != year]
    y_tr = train["log_yield"].to_numpy()

    pred_b = BaselineFE(base_feats).fit(train, y_tr).predict(test)
    pred_g = make_gbm(cfg["model"]["gbm"], cfg["model"]["seed"]).fit(
        gbm_matrix(train, raions), y_tr).predict(gbm_matrix(test, raions))
    return pd.DataFrame({
        "Район": test["raion"].to_numpy(),
        FACT: test["yield_cwt_ha"].to_numpy(),
        PARAM: np.exp(pred_b),
        BOOST: np.exp(pred_g)})


def maps(res: pd.DataFrame):
    """Three choropleths on one colour scale: truth and what each model said."""
    panels = [(FACT, FACT), ("Параметрическая", PARAM), ("Бустинг", BOOST)]
    hi = float(res[[FACT, PARAM, BOOST]].to_numpy().max())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    for ax, (title, col) in zip(axes, panels):
        choropleth(ax, dict(zip(res["Район"], res[col])), vmin=0, vmax=hi, label=title)
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, hi))
    fig.colorbar(sm, ax=axes, shrink=0.75, label="Урожайность, ц/га")
    return fig


st.title("Что модель делает с годом, которого не видела")

data = panel()
with st.sidebar:
    st.header("Выбор случая")
    crop = st.selectbox("Культура", list(CROPS), format_func=CROPS.get, index=0)
    years = sorted(data.loc[data["crop"] == crop, "year"].unique())
    year = st.select_slider(
        "Год, отложенный из обучения", years,
        value=2012 if 2012 in years else years[0])
    if year in DROUGHTS:
        st.warning(f"{year} год был засушливым.")
    st.divider()
    st.caption("Обе модели переобучаются с нуля без выбранного года, "
               "а затем предсказывают его.")

tab1, tab2 = st.tabs(["Выбранный год", "Все годы"])

with tab1:
    res = hold_out(crop, year)
    a, b, g = res[FACT].mean(), res[PARAM].mean(), res[BOOST].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Факт, {year}", f"{a:.1f} ц/га")
    c2.metric("Параметрическая", f"{b:.1f}", f"{100 * (b / a - 1):+.0f}%")
    c3.metric("Бустинг", f"{g:.1f}", f"{100 * (g / a - 1):+.0f}%")

    st.pyplot(maps(res), use_container_width=True)
    st.caption(
        "Возьми засушливый год — и увидишь, что ансамбль отказывается опускаться "
        "достаточно низко. Возьми рекордный 2021-й — он так же откажется "
        "подниматься. Серым показаны Приднестровье и Бендеры, которых нет в "
        "статистике БНС.")
    with st.expander("Показать по районам таблицей и столбиками"):
        st.bar_chart(res.set_index("Район")[[FACT, PARAM, BOOST]],
                     height=340, stack=False, y_label="Урожайность, ц/га")
        st.dataframe(res.round(1), hide_index=True, width="stretch")

with tab2:
    slopes = pd.read_csv(TABLES / "shrinkage.csv")
    s = slopes[slopes["crop"] == crop].set_index("year")
    chart = pd.DataFrame({FACT: np.exp(s["actual"]),
                          PARAM: np.exp(s["baseline"]),
                          BOOST: np.exp(s["boosting"])})
    chart.index = chart.index.astype(str)
    st.subheader(f"{CROPS[crop]}: каждый год предсказан без самого себя")
    st.line_chart(chart, height=380, y_label="Среднее по году, ц/га")
    fit = pd.read_csv(TABLES / "shrinkage_slopes.csv").set_index("crop")
    c1, c2 = st.columns(2)
    c1.metric("Размах, сохранённый параметрической моделью",
              f"{100 * fit.loc[crop, 'baseline']:.0f}%")
    c2.metric("Размах, сохранённый бустингом",
              f"{100 * fit.loc[crop, 'boosting']:.0f}%")
    st.caption(
        "По всем культурам параметрическая модель удерживает 60% размаха между "
        "годами, бустинг — 38%. До ста процентов не дотягивает никто, потому "
        "что погода объясняет не всё, но ансамбль отдаёт почти вдвое больше.")
