import numpy as np
import pandas as pd
import pytest
from conftest import DATA
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from scoring.data import load
from scoring.model import fit_logit, predict_pd, select_features
from scoring.scorecard import Scorecard
from scoring.woe import WoEBinner


@pytest.fixture(scope="module")
def setup():
    df = load(DATA)
    feats = [c for c in df.columns if c not in ("default", "sex", "marriage")]
    tr, te = train_test_split(df, test_size=0.3, stratify=df["default"], random_state=42)
    b = WoEBinner().fit(tr[feats], tr["default"])
    wtr = b.transform(tr[feats])
    sel = select_features(b.iv_, wtr)
    res = fit_logit(wtr[sel], tr["default"])
    return df, tr, te, b, sel, res


def test_selection_rules(setup):
    _, _, _, b, sel, _ = setup
    assert all(b.iv_[f] >= 0.02 for f in sel)
    assert "sex" not in sel and "marriage" not in sel


def test_coefficients_negative(setup):
    *_, res = setup
    assert (res.params.drop("const") < 0).all()


def test_scale_anchor_and_pdo():
    card = Scorecard(base=600, odds=50, pdo=20)
    assert card.from_log_odds_good(np.log(50)) == pytest.approx(600)
    assert card.from_log_odds_good(np.log(100)) - card.from_log_odds_good(np.log(50)) == pytest.approx(20)


def test_score_monotone_in_pd(setup):
    _, _, te, b, sel, res = setup
    card = Scorecard()
    w = b.transform(te[sel])
    pd_ = predict_pd(res, w)
    s = card.score(res, w)
    assert spearmanr(s, pd_).statistic == pytest.approx(-1)


def test_points_sum_to_score(setup):
    _, _, te, b, sel, res = setup
    card = Scorecard()
    pts = card.points_table(b, res, sel)
    row = te[sel].iloc[[0]]
    w = b.transform(row)
    used = [c for c in res.params.index if c != "const"]
    manual = sum(pts.loc[(pts.feature == f) & np.isclose(pts.woe, w[f].iloc[0]), "points"].iloc[0] for f in used)
    assert manual == pytest.approx(card.score(res, w).iloc[0])


def test_model_compact_and_significant(setup):
    *_, res = setup
    p = res.pvalues.drop("const")
    assert (p < 0.05).all()
    assert len(p) <= 12
