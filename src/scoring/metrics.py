"""Метрики валидации скоринговой модели и выбор порога одобрения."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score


def gini(y, p) -> float:
    """Gini = 2·AUC − 1."""
    return 2 * roc_auc_score(y, p) - 1


def ks(y, p) -> float:
    """Максимальное расстояние между распределениями прогноза у «плохих» и «хороших»."""
    y, p = np.asarray(y), np.asarray(p)
    return ks_2samp(p[y == 1], p[y == 0]).statistic


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index по децилям ожидаемого распределения. < 0,1 — стабильно, > 0,25 — сдвиг."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)[1:-1]))
    e = np.bincount(np.searchsorted(edges, expected), minlength=len(edges) + 1) / len(expected)
    a = np.bincount(np.searchsorted(edges, actual), minlength=len(edges) + 1) / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Децили прогноза: средний прогноз PD против фактической доли дефолтов."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d["p"], bins, labels=False, duplicates="drop")
    t = d.groupby("bin").agg(n=("y", "size"), predicted=("p", "sum"), actual=("y", "sum"))
    t["predicted_rate"] = t["predicted"] / t["n"]
    t["actual_rate"] = t["actual"] / t["n"]
    return t.reset_index()


def cutoff_table(scores, y, ead, lgd: float = 0.45, margin: float = 0.15, cutoffs=None) -> pd.DataFrame:
    """Для каждого порога: одобряем score ≥ cutoff.
    Прибыль = margin·EAD по вернувшим − LGD·EAD по дефолтным (упрощённая годовая модель)."""
    scores, y, ead = np.asarray(scores, float), np.asarray(y), np.asarray(ead, float)
    if cutoffs is None:
        cutoffs = np.unique(np.round(np.quantile(scores, np.linspace(0, 0.9, 19))))
    rows = []
    for c in cutoffs:
        ok = scores >= c
        good, bad = ok & (y == 0), ok & (y == 1)
        rows.append({"cutoff": c, "approval_rate": ok.mean(),
                     "bad_rate": y[ok].mean() if ok.any() else np.nan,
                     "profit": margin * ead[good].sum() - lgd * ead[bad].sum()})
    return pd.DataFrame(rows)
