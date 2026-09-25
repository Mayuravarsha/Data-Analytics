import numpy as np
import pandas as pd
import pytest

from usedcars import models
from usedcars.data import SAMPLE, clean, normalise_model, read_raw
from usedcars.experiment import run


def listing(**kw):
    base = dict(price=15000, year=2015, manufacturer="Ford", model="F-150 XLT", condition="good",
                cylinders="6 cylinders", fuel="gas", odometer=80000, title_status="clean",
                transmission="automatic", VIN=None, drive="4wd", type="truck",
                paint_color="white", state="tx", posting_date="2021-04-20T10:00:00-0500")
    return {**base, **kw}


def test_price_year_and_odometer_filters():
    raw = pd.DataFrame([
        listing(price=0), listing(price=999), listing(price=3_736_928_711),
        listing(year=1985), listing(odometer=0), listing(odometer=1_000_000),
        listing(price=5000, model="civic"), listing(year=2022, model="bronco"),
    ])
    df = clean(raw)
    assert sorted(df["price"]) == [5000, 15000]
    assert (df["age"] >= 0).all()


def test_vin_duplicates_keep_one_row():
    raw = pd.DataFrame([listing(VIN="1FT", price=20000), listing(VIN="1FT", price=19500, state="ok"),
                        listing(VIN="2HG", price=9000, model="accord")])
    assert len(clean(raw)) == 2


def test_features():
    df = clean(pd.DataFrame([listing(cylinders="other"), listing(model="Civic", condition=None)]))
    assert df.loc[0, "cylinders"] != df.loc[0, "cylinders"]           # NaN for "other"
    assert df.loc[1, "condition"] == "missing"
    assert df.loc[0, "age"] == 6 and df["age"].dtype == float
    assert df.loc[0, "model"] == "f-150 xlt"


def test_normalise_model():
    s = pd.Series(["Silverado 1500 Crew Cab LT", "  camry  ", "f-150!", None, "***"])
    assert list(normalise_model(s).fillna("NA")) == ["silverado 1500", "camry", "f-150", "NA", "NA"]


def test_group_median_falls_back_to_make_then_global():
    X = pd.DataFrame({"manufacturer": ["ford", "ford", "honda"], "age": [1.0, 1.0, 5.0]})
    m = models.GroupMedian().fit(X, np.log([20000, 30000, 8000]))
    test = pd.DataFrame({"manufacturer": ["ford", "ford", "tesla"], "age": [1.0, 9.0, 1.0]})
    pred = np.exp(m.predict(test))
    assert pred[0] == pytest.approx(np.sqrt(20000 * 30000))
    assert pred[1] == pytest.approx(pred[0])                  # unseen age -> make median
    assert pred[2] == pytest.approx(20000)                    # unseen make -> global median


def test_scores_on_perfect_and_biased_predictions():
    y = np.log([10000, 20000, 40000])
    perfect = models.Scores.compute(y, y)
    assert perfect.mae == 0 and perfect.r2 == 1
    off = models.Scores.compute(y, y + np.log(1.1))
    assert off.median_ape == pytest.approx(10)


@pytest.mark.slow
def test_models_beat_baseline_on_sample():
    assert SAMPLE.exists()
    df = clean(read_raw(SAMPLE))
    assert len(df) > 10_000
    table, _, _ = run(df, include_forest=False, verbose=False)
    base = table.loc["Baseline: median by make & year", "mae"]
    assert table.loc["LightGBM (native categoricals)", "mae"] < 0.8 * base
    assert table["r2_log"].max() > 0.7
