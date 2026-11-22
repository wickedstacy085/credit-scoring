"""Перевод логита в баллы: score = offset + factor · ln(odds good:bad)."""
import numpy as np
import pandas as pd


class Scorecard:
    def __init__(self, base: float = 600, odds: float = 50, pdo: float = 20):
        self.factor = pdo / np.log(2)                       # +pdo баллов при удвоении шансов
        self.offset = base - self.factor * np.log(odds)     # base баллов при шансах odds:1

    def from_log_odds_good(self, log_odds_good):
        return self.offset + self.factor * log_odds_good

    def _coefs(self, res):
        return res.params["const"], res.params.drop("const")

    def score(self, res, Xw: pd.DataFrame) -> pd.Series:
        b0, b = self._coefs(res)
        log_odds_bad = b0 + Xw[b.index].to_numpy() @ b.to_numpy()
        return pd.Series(self.from_log_odds_good(-log_odds_bad), index=Xw.index)

    def points_table(self, binner, res, features: list[str]) -> pd.DataFrame:
        """Баллы за каждый бин; константа и offset поровну распределены между признаками."""
        b0, b = self._coefs(res)
        k = len(b)
        rows = []
        for f in b.index:
            t = binner.tables_[f]
            for _, r in t.iterrows():
                pts = -(b[f] * r["woe"] + b0 / k) * self.factor + self.offset / k
                rows.append({"feature": f, "bin": r["label"], "n": r["n"], "bad_rate": r["bad_rate"],
                             "woe": r["woe"], "points": pts})
        return pd.DataFrame(rows)
