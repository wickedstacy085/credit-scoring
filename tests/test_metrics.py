import numpy as np
import pandas as pd
import pytest

from scoring.metrics import calibration_table, cutoff_table, gini, ks, psi

rng = np.random.default_rng(0)


def test_gini_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert gini(y, np.array([.1, .2, .8, .9])) == pytest.approx(1)
    y = rng.integers(0, 2, 20_000)
    assert abs(gini(y, rng.random(20_000))) < 0.03


def test_ks_separable():
    y = np.array([0] * 50 + [1] * 50)
    p = np.r_[np.linspace(0, .4, 50), np.linspace(.6, 1, 50)]
    assert ks(y, p) == pytest.approx(1)


def test_psi():
    a = rng.normal(600, 50, 10_000)
    assert psi(a, a) == pytest.approx(0, abs=1e-9)
    assert psi(a, rng.normal(650, 50, 10_000)) > 0.25


def test_calibration_sums():
    p = rng.uniform(0.05, 0.6, 50_000)
    y = (rng.random(50_000) < p).astype(int)
    t = calibration_table(y, p, bins=10)
    assert len(t) == 10
    assert t["predicted"].sum() == pytest.approx(t["actual"].sum(), rel=0.02)


def test_cutoff_table():
    scores = np.array([500, 550, 600, 650, 700])
    y = np.array([1, 1, 0, 0, 0])
    ead = np.full(5, 100.0)
    t = cutoff_table(scores, y, ead, lgd=0.5, margin=0.1, cutoffs=[0, 575])
    all_, strict = t.iloc[0], t.iloc[1]
    assert all_["approval_rate"] == 1 and all_["bad_rate"] == pytest.approx(0.4)
    assert all_["profit"] == pytest.approx(3 * 10 - 2 * 50)
    assert strict["approval_rate"] == pytest.approx(0.6) and strict["profit"] == pytest.approx(30)
