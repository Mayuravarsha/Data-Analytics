"""Loading and cleaning the Craigslist listings.

The raw file (Kaggle "Used Cars Dataset", scraped April-May 2021) has
426,880 listings. Cleaning rules and why:

* price between $1,000 and $150,000: about 10 % of listings are $0-$999
  (placeholders such as "call for price" or lease deposits) and a few are
  absurd ($3.7 billion). The original notebook kept all of these, and the
  squared error on them dominated every metric.
* model year 1990-2022: listings were posted in 2021, so 2021 and 2022
  model years are legitimate and are kept.
* odometer between 1 and 400,000 miles.
* one row per VIN: dealers re-post the same car many times (55 % of VINs
  repeat). Keeping the copies puts the same car in both the training and the
  test set and inflates the scores.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "data" / "vehicles.csv"
SAMPLE = ROOT / "data" / "vehicles_sample.csv.gz"

RAW_COLUMNS = ["price", "year", "manufacturer", "model", "condition", "cylinders", "fuel",
               "odometer", "title_status", "transmission", "VIN", "drive", "type",
               "paint_color", "state", "posting_date"]
CATEGORICAL = ["manufacturer", "model", "condition", "fuel", "title_status", "transmission",
               "drive", "type", "paint_color", "state"]
NUMERIC = ["age", "odometer", "cylinders"]
TARGET = "price"

PRICE_RANGE = (1_000, 150_000)
YEAR_RANGE = (1990, 2022)
ODOMETER_RANGE = (1, 400_000)


def read_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Read the full CSV if available, else the bundled 30k-row sample.

    Looks at ``path``, then $VEHICLES_CSV, then data/vehicles.csv (pulled from Git LFS).
    """
    if path is None:
        env = os.environ.get("VEHICLES_CSV")
        path = env if env else (FULL if _is_real_csv(FULL) else SAMPLE)
    return pd.read_csv(path, usecols=lambda c: c in RAW_COLUMNS, low_memory=False)


def _is_real_csv(path: Path) -> bool:
    # an un-pulled LFS file is a 130-byte text pointer
    return path.exists() and path.stat().st_size > 10_000


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df = df[df["price"].between(*PRICE_RANGE)]
    df = df[df["year"].between(*YEAR_RANGE)]
    df = df[df["odometer"].between(*ODOMETER_RANGE)]
    has_vin = df["VIN"].notna()
    df = pd.concat([df[has_vin].drop_duplicates("VIN"), df[~has_vin]])
    # a VIN-less car posted twice looks identical on these columns
    df = df.drop_duplicates(["price", "year", "manufacturer", "model", "odometer", "state"])

    posted = pd.to_datetime(df["posting_date"], utc=True, errors="coerce")
    df["age"] = (posted.dt.year - df["year"]).clip(lower=0)
    df["cylinders"] = pd.to_numeric(
        df["cylinders"].astype("string").str.extract(r"(\d+)")[0], errors="coerce").astype(float)
    df["age"] = df["age"].astype(float)
    df["odometer"] = df["odometer"].astype(float)
    df["model"] = normalise_model(df["model"])
    for col in CATEGORICAL:
        df[col] = df[col].astype("string").str.strip().str.lower().fillna("missing").astype(object)
    return df[[TARGET, *NUMERIC, *CATEGORICAL]].reset_index(drop=True)


def normalise_model(model: pd.Series) -> pd.Series:
    """Keep the first two words of the model name: 'silverado 1500 crew cab lt' -> 'silverado 1500'."""
    words = (model.astype("string").str.lower()
             .str.replace(r"[^a-z0-9\- ]", " ", regex=True)
             .str.split().str[:2].str.join(" "))
    return words.replace("", pd.NA)


def load(path: Path | str | None = None) -> pd.DataFrame:
    return clean(read_raw(path))


def log_price(y) -> np.ndarray:
    return np.log(np.asarray(y, dtype=float))
