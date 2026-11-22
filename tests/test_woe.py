import numpy as np
import pandas as pd
import pytest
from conftest import DATA

from scoring.data import load
from scoring.woe import WoEBinner


def test_woe_formula_on_toy_example():
    x = pd.DataFrame({"c": ["a"] * 100 + ["b"] * 100})
    y = pd.Series([1] * 20 + [0] * 80 + [1] * 50 + [0] * 50)
    b = WoEBinner(min_share=0.01).fit(x, y)
    t = b.tables_["c"].set_index("bin")
    G, B = 130, 70
    expected_a = np.log(((80 + 0.5) / (G + 0.5)) / ((20 + 0.5) / (B + 0.5)))
    assert t.loc["a", "woe"] == pytest.approx(expected_a)
    assert b.iv_["c"] > 0


def test_bin_without_bads_is_finite():
    x = pd.DataFrame({"c": ["a"] * 50 + ["b"] * 50})
    y = pd.Series([0] * 50 + [1] * 25 + [0] * 25)
    b = WoEBinner(min_share=0.01).fit(x, y)
    assert np.isfinite(b.tables_["c"]["woe"]).all()


@pytest.fixture(scope="module")
def fitted():
    df = load(DATA)
    X = df[["pay_1", "limit_bal", "education", "age"]]
    return WoEBinner().fit(X, df["default"]), X


def test_pay_1_bad_rate_monotone(fitted):
    b, _ = fitted
    br = b.tables_["pay_1"]["bad_rate"].to_numpy()
    assert (np.diff(br) >= 0).all() or (np.diff(br) <= 0).all()
    assert len(br) <= 6


def test_iv_non_negative_and_pay_strongest(fitted):
    b, _ = fitted
    assert (b.iv_ >= 0).all()
    assert b.iv_.idxmax() == "pay_1"


def test_transform_out_of_range_and_unseen(fitted):
    b, X = fitted
    new = pd.DataFrame({"pay_1": [99, -99], "limit_bal": [1e9, -1], "education": ["unknown", "graduate"],
                        "age": [150, 0]})
    w = b.transform(new)
    assert w.notna().all().all()
    assert w.shape == (2, 4)
