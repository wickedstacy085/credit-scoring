import pytest
from conftest import DATA

from scoring.data import load


def test_load():
    df = load(DATA)
    assert len(df) == 30_000
    assert df["default"].mean() == pytest.approx(0.2212)
    assert "pay_1" in df and "pay_0" not in df
    assert set(df["education"].unique()) <= {"graduate", "university", "high_school", "other"}
    # производные признаки
    assert df["max_delay_6m"].between(-2, 8).all()
    assert df["util_1"].notna().all()
