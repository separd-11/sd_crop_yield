"""The competing models behind one fit/predict interface, so the validation
script trains exactly the same objects the reporting scripts do."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

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
