"""Отбор признаков и логистическая регрессия на WoE."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

EXCLUDED = {"sex", "marriage"}   # защищённые характеристики в скоринг не входят


def select_features(iv: pd.Series, woe_train: pd.DataFrame, threshold: float = 0.02,
                    max_corr: float = 0.6) -> list[str]:
    """По убыванию IV: берём признак, если IV ≥ порога и он не коррелирует (|ρ| ≥ max_corr) с уже взятыми."""
    corr = woe_train.corr().abs()
    chosen: list[str] = []
    for f in iv.sort_values(ascending=False).index:
        if f in EXCLUDED or iv[f] < threshold or f not in woe_train:
            continue
        if all(corr.loc[f, c] < max_corr for c in chosen):
            chosen.append(f)
    return chosen


def fit_logit(Xw: pd.DataFrame, y: pd.Series, alpha: float = 0.05):
    """Логит на WoE с пошаговым исключением.

    При WoE = ln(%good/%bad) коэффициенты должны быть отрицательными: признак с положительным
    коэффициентом (эффект мультиколлинеарности) убираем первым; затем по одному убираем признак
    с наибольшим p-value, пока все не станут значимы на уровне alpha."""
    feats = list(Xw.columns)
    while True:
        res = sm.Logit(np.asarray(y, float), sm.add_constant(Xw[feats], has_constant="add")).fit(disp=0)
        coefs, pvals = res.params.drop("const"), res.pvalues.drop("const")
        if (coefs >= 0).any():
            feats.remove(coefs.idxmax())
        elif (pvals >= alpha).any():
            feats.remove(pvals.idxmax())
        else:
            return res


def predict_pd(res, Xw: pd.DataFrame) -> pd.Series:
    cols = [c for c in res.params.index if c != "const"]
    return pd.Series(res.predict(sm.add_constant(Xw[cols], has_constant="add")), index=Xw.index)
