"""The competing models behind one fit/predict interface, so the validation
script trains exactly the same objects the reporting scripts do."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

ML_FEATURES = [
    "gdd", "kdd", "hot_days", "prec_season", "tmax_mean", "dry_spell", "prec_winter",
    "prec_m4", "prec_m5", "prec_m6", "prec_m7", "prec_m8",
    "tmax_m4", "tmax_m5", "tmax_m6", "tmax_m7", "tmax_m8",
    "trend",
]


class BaselineFE:
    """log(yield) = a_raion + b*trend + f(weather) + e

    The specification the crop-yield literature uses, and the thing gradient
    boosting has to beat.
    """

    def __init__(self, weather_features: list[str]):
        self.weather_features = list(weather_features)
        self.coef_: np.ndarray | None = None
        self.columns_: list[str] | None = None

    def _design(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        dummies = pd.get_dummies(df["raion"], prefix="r").astype(float)
        X = pd.concat([dummies, df[["trend"] + self.weather_features].astype(float)], axis=1)
        X.insert(0, "const", 1.0)
        if columns is not None:
            # A fold may miss a raion; align to the training design.
            X = X.reindex(columns=columns, fill_value=0.0)
        return X

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "BaselineFE":
        X = self._design(df)
        self.columns_ = list(X.columns)
        self.coef_, *_ = np.linalg.lstsq(X.to_numpy(), np.asarray(y, float), rcond=None)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._design(df, self.columns_)
        return X.to_numpy() @ self.coef_

    def summary(self, df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Coefficients and standard errors for the weather terms."""
        X = self._design(df).to_numpy()
        resid = np.asarray(y, float) - X @ self.coef_
        n, k = X.shape
        sigma2 = resid @ resid / (n - k)
        se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * sigma2)
        idx = {c: i for i, c in enumerate(self.columns_)}
        rows = []
        for name in self.weather_features:
            i = idx[name]
            rows.append({"term": name, "coef": self.coef_[i], "se": se[i],
                         "t": self.coef_[i] / se[i] if se[i] else np.nan})
        return pd.DataFrame(rows)


def monotonic_vector(constraints: dict | None) -> np.ndarray | None:
    """Per-feature constraint array; the trailing slot is the raion code."""
    if not constraints:
        return None
    vec = np.zeros(len(ML_FEATURES) + 1, dtype=int)
    for name, sign in constraints.items():
        vec[ML_FEATURES.index(name)] = sign
    return vec


class GlobalMean:
    """Everything is the average. The floor any model has to clear."""

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "GlobalMean":
        self.mu_ = float(np.mean(y))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.mu_)


class RaionMean:
    """Each raion keeps its own long-run average, and nothing else happens."""

    def __init__(self, with_trend: bool = False):
        self.with_trend = with_trend

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "RaionMean":
        y = np.asarray(y, float)
        self.slope_ = 0.0
        if self.with_trend:
            t = df["trend"].to_numpy(float)
            self.slope_ = float(np.polyfit(t, y, 1)[0])
            y = y - self.slope_ * t
        self.mu_ = float(np.mean(y))
        self.by_raion_ = pd.Series(y, index=df["raion"].to_numpy()).groupby(level=0).mean()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = df["raion"].map(self.by_raion_).fillna(self.mu_).to_numpy(float)
        return out + self.slope_ * df["trend"].to_numpy(float)


class Persistence:
    """Last year's yield in the same raion. What a farmer would guess."""

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "Persistence":
        self.table_ = {(r, yr): v for r, yr, v
                       in zip(df["raion"], df["year"], np.asarray(y, float))}
        self.by_raion_ = pd.Series(np.asarray(y, float),
                                   index=df["raion"].to_numpy()).groupby(level=0).mean()
        self.mu_ = float(np.mean(y))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = []
        for raion, year in zip(df["raion"], df["year"]):
            value = self.table_.get((raion, year - 1))
            if value is None:
                value = self.by_raion_.get(raion, self.mu_)
            out.append(value)
        return np.asarray(out, float)


def make_gbm(cfg: dict, seed: int,
             constraints: dict | None = None) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=cfg["max_iter"],
        learning_rate=cfg["learning_rate"],
        max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_samples_leaf"],
        l2_regularization=cfg["l2_regularization"],
        monotonic_cst=monotonic_vector(constraints),
        random_state=seed,
    )


def gbm_matrix(df: pd.DataFrame, raion_levels: list[str]) -> np.ndarray:
    """Features plus raion as an integer code."""
    codes = pd.Categorical(df["raion"], categories=raion_levels).codes
    return np.column_stack([df[ML_FEATURES].to_numpy(float), codes.astype(float)])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


class SklearnAdapter:
    """Wraps a scikit-learn estimator behind the fit(df, y) / predict(df) interface.

    Tree ensembles take the raion as a single integer code; everything else gets
    one-hot columns, because a code would be read as an ordering.
    """

    def __init__(self, estimator, raions: list[str], one_hot: bool = False,
                 spline_on: list[str] | None = None):
        self.estimator = estimator
        self.raions = list(raions)
        self.one_hot = one_hot
        self.spline_on = spline_on

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        X = df[ML_FEATURES].to_numpy(float)
        if self.spline_on:
            idx = [ML_FEATURES.index(c) for c in self.spline_on]
            X = np.column_stack([X, self.spline_.transform(X[:, idx])])
        if self.one_hot:
            codes = pd.Categorical(df["raion"], categories=self.raions).codes
            dummies = np.zeros((len(df), len(self.raions)))
            ok = codes >= 0
            dummies[np.arange(len(df))[ok], codes[ok]] = 1.0
            return np.column_stack([X, dummies])
        codes = pd.Categorical(df["raion"], categories=self.raions).codes
        return np.column_stack([X, codes.astype(float)])

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "SklearnAdapter":
        if self.spline_on:
            idx = [ML_FEATURES.index(c) for c in self.spline_on]
            self.spline_ = SplineTransformer(n_knots=6, degree=3).fit(
                df[ML_FEATURES].to_numpy(float)[:, idx])
        self.estimator.fit(self._matrix(df), np.asarray(y, float))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(self._matrix(df))


SPLINE_ON = ["gdd", "kdd", "prec_season", "dry_spell"]


def model_zoo(cfg: dict, raions: list[str]) -> dict:
    """Every model the comparison runs, as factories so each fold gets a fresh one."""
    seed = cfg["model"]["seed"]
    ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
    return {
        "raion mean + trend": lambda: RaionMean(with_trend=True),
        "baseline FE": lambda: BaselineFE(cfg["model"]["baseline_features"]),
        "ridge, all features": lambda: SklearnAdapter(ridge(), raions, one_hot=True),
        "splines + ridge": lambda: SklearnAdapter(ridge(), raions, one_hot=True,
                                                  spline_on=SPLINE_ON),
        "random forest": lambda: SklearnAdapter(
            RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                  n_jobs=-1, random_state=seed), raions),
        "extra trees": lambda: SklearnAdapter(
            ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3,
                                n_jobs=-1, random_state=seed), raions),
        "gradient boosting": lambda: SklearnAdapter(
            make_gbm(cfg["model"]["gbm"], seed), raions),
        "k nearest neighbours": lambda: SklearnAdapter(
            make_pipeline(StandardScaler(),
                          KNeighborsRegressor(n_neighbors=10, weights="distance")),
            raions, one_hot=True),
        "neural net": lambda: SklearnAdapter(
            make_pipeline(StandardScaler(),
                          MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1500,
                                       early_stopping=True, random_state=seed)),
            raions, one_hot=True),
    }
