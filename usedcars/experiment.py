"""Train every model on one split and score it on a held-out test set."""

from __future__ import annotations

import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import models
from .data import CATEGORICAL, NUMERIC, TARGET, log_price


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = models.SEED):
    X, y = df[NUMERIC + CATEGORICAL], log_price(df[TARGET])
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def run(df: pd.DataFrame, include_forest: bool = True, verbose: bool = True):
    """Returns (results table, fitted models, test split)."""
    X_train, X_test, y_train, y_test = split(df)
    candidates = {
        "Baseline: median by make & year": models.GroupMedian(),
        "Ridge regression (one-hot)": models.ridge(),
        "Random forest": models.random_forest() if include_forest else None,
        "Histogram gradient boosting": models.hist_gradient_boosting(),
    }
    rows, fitted = [], {}
    for name, model in candidates.items():
        if model is None:
            continue
        t = time.time()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({"model": name, **vars(models.Scores.compute(y_test, pred)),
                     "train_seconds": time.time() - t})
        fitted[name] = model
        if verbose:
            print(f"{name:34s} R2(log)={rows[-1]['r2_log']:.3f} MAE=${rows[-1]['mae']:,.0f}")

    # LightGBM with native categoricals and early stopping on a validation slice
    t = time.time()
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.1,
                                                  random_state=models.SEED)
    X_fit, cats = models.as_lgb_frame(X_fit)
    X_val, _ = models.as_lgb_frame(X_val, cats)
    X_te, _ = models.as_lgb_frame(X_test, cats)
    gbm = models.lightgbm()
    with warnings.catch_warnings():  # eval_set is renamed in newer LightGBM releases
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*eval_set.*")
        gbm.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(100, verbose=False)])
    pred = gbm.predict(X_te)
    name = "LightGBM (native categoricals)"
    rows.append({"model": name, **vars(models.Scores.compute(y_test, pred)),
                 "train_seconds": time.time() - t})
    fitted[name] = (gbm, cats)
    if verbose:
        print(f"{name:34s} R2(log)={rows[-1]['r2_log']:.3f} MAE=${rows[-1]['mae']:,.0f} "
              f"({gbm.best_iteration_} trees)")

    table = pd.DataFrame(rows).set_index("model").sort_values("mae")
    return table, fitted, (X_train, X_test, y_train, y_test)
