"""Models compared in the notebook. Every model predicts log(price)."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

from .data import CATEGORICAL, NUMERIC

SEED = 0


class GroupMedian(BaseEstimator, RegressorMixin):
    """Baseline: median log price of cars with the same make and model year."""

    def fit(self, X, y):
        frame = X[["manufacturer", "age"]].assign(y=np.asarray(y))
        self.table_ = frame.groupby(["manufacturer", "age"])["y"].median()
        self.make_ = frame.groupby("manufacturer")["y"].median()
        self.global_ = float(np.median(y))
        return self

    def predict(self, X):
        keys = pd.MultiIndex.from_frame(X[["manufacturer", "age"]])
        pred = self.table_.reindex(keys).to_numpy()
        fallback = X["manufacturer"].map(self.make_).fillna(self.global_).to_numpy()
        return np.where(np.isnan(pred), fallback, pred)


def _numeric():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler())


def ridge():
    pre = ColumnTransformer([
        ("num", _numeric(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), CATEGORICAL),
    ])
    return make_pipeline(pre, Ridge(alpha=1.0))


def _ordinal():
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1), CATEGORICAL),
    ])


def random_forest():
    # max_samples keeps training time reasonable on ~200k rows
    return make_pipeline(_ordinal(), RandomForestRegressor(
        n_estimators=150, min_samples_leaf=2, max_samples=0.3, max_features=0.5,
        n_jobs=-1, random_state=SEED))


def hist_gradient_boosting():
    # HistGradientBoosting allows at most 255 categories per feature; "model"
    # has ~10k, so it is target-encoded (with internal cross-fitting to avoid
    # leaking the target) and the other categoricals are handled natively.
    low_card = [c for c in CATEGORICAL if c != "model"]
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("model", TargetEncoder(cv=KFold(5, shuffle=True, random_state=SEED)), ["model"]),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1), low_card),
    ])
    cat_mask = [False] * (len(NUMERIC) + 1) + [True] * len(low_card)
    return make_pipeline(pre, HistGradientBoostingRegressor(
        max_iter=800, learning_rate=0.08, max_leaf_nodes=63, categorical_features=cat_mask,
        early_stopping=True, random_state=SEED))


def lightgbm(**params):
    defaults = dict(n_estimators=2000, learning_rate=0.05, num_leaves=127, min_child_samples=20,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                    cat_smooth=10, random_state=SEED, verbose=-1)
    return lgb.LGBMRegressor(**{**defaults, **params})


def as_lgb_frame(X: pd.DataFrame, categories: dict | None = None):
    """LightGBM uses pandas categoricals natively; reuse training categories at test time."""
    X = X.copy()
    categories = categories or {c: pd.CategoricalDtype(X[c].unique()) for c in CATEGORICAL}
    for c in CATEGORICAL:
        known = categories[c].categories
        # categories not seen in training become missing
        X[c] = pd.Categorical(X[c].where(X[c].isin(known)), dtype=categories[c])
    return X, categories


@dataclass
class Scores:
    r2_log: float
    r2: float
    rmse: float
    mae: float
    median_ape: float

    @classmethod
    def compute(cls, y_true_log, y_pred_log) -> "Scores":
        yt, yp = np.exp(y_true_log), np.exp(y_pred_log)
        return cls(
            r2_log=r2_score(y_true_log, y_pred_log),
            r2=r2_score(yt, yp),
            rmse=float(np.sqrt(mean_squared_error(yt, yp))),
            mae=mean_absolute_error(yt, yp),
            median_ape=float(np.median(np.abs(yp - yt) / yt) * 100),
        )
