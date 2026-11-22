"""Биннинг и Weight of Evidence. WoE = ln(%good / %bad): чем выше, тем ниже риск."""
import numpy as np
import pandas as pd

SMOOTH = 0.5  # сглаживание, чтобы бин без «плохих» или «хороших» не давал ±∞


class WoEBinner:
    def __init__(self, max_bins: int = 6, min_share: float = 0.05, init_bins: int = 20):
        self.max_bins, self.min_share, self.init_bins = max_bins, min_share, init_bins

    # ---------- обучение ----------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEBinner":
        y = pd.Series(np.asarray(y), index=X.index)
        self.G_, self.B_ = int((y == 0).sum()), int((y == 1).sum())
        self.edges_, self.maps_, self.tables_, iv = {}, {}, {}, {}
        for col in X.columns:
            x = X[col]
            if x.dtype.kind in "biuf":
                edges = self._fit_numeric(x, y)
                self.edges_[col] = edges
                labels = self._bin_numeric(x, edges)
                names = self._interval_names(edges)
            else:
                mapping = self._fit_categorical(x, y)
                self.maps_[col] = mapping
                labels = x.map(mapping).fillna("other")
                names = None
            t = self._table(labels, y, names)
            self.tables_[col] = t
            iv[col] = t["iv"].sum()
        self.iv_ = pd.Series(iv).sort_values(ascending=False)
        return self

    def _stats(self, codes: np.ndarray, y: np.ndarray, k: int):
        n = np.bincount(codes, minlength=k).astype(float)
        bad = np.bincount(codes, weights=y, minlength=k)
        return n, bad

    def _fit_numeric(self, x: pd.Series, y: pd.Series) -> list[float]:
        xs = x.to_numpy(float)
        yv = y.to_numpy(float)
        uniq = np.unique(xs)
        if len(uniq) <= self.init_bins:
            cuts = list((uniq[:-1] + uniq[1:]) / 2)
        else:
            cuts = list(np.unique(np.quantile(xs, np.linspace(0, 1, self.init_bins + 1)[1:-1])))
        # направление монотонности — по знаку ранговой корреляции признака и дефолта
        direction = np.sign(pd.Series(xs).corr(pd.Series(yv), method="spearman")) or 1.0

        def state(c):
            codes = np.searchsorted(c, xs, side="left")
            n, bad = self._stats(codes, yv, len(c) + 1)
            return n, bad / np.maximum(n, 1)

        changed = True
        while changed and cuts:
            changed = False
            n, br = state(cuts)
            # 1) слишком маленькие бины — сливаем с соседом, ближайшим по уровню риска
            small = np.where(n / len(xs) < self.min_share)[0]
            if len(small):
                i = small[0]
                if i == 0:
                    j = 0
                elif i == len(n) - 1:
                    j = i - 1
                else:
                    j = i - 1 if abs(br[i] - br[i - 1]) <= abs(br[i] - br[i + 1]) else i
                cuts.pop(j)
                changed = True
                continue
            # 2) монотонность: первое нарушение — сливаем пару
            d = np.diff(br) * direction
            bad_pairs = np.where(d < 0)[0]
            if len(bad_pairs):
                cuts.pop(bad_pairs[0])
                changed = True
                continue
            # 3) лишние бины — сливаем пару с самым близким риском
            if len(n) > self.max_bins:
                cuts.pop(int(np.argmin(np.abs(np.diff(br)))))
                changed = True
        return cuts

    def _fit_categorical(self, x: pd.Series, y: pd.Series) -> dict:
        share = x.value_counts(normalize=True)
        return {v: (v if s >= self.min_share else "other") for v, s in share.items()}

    @staticmethod
    def _bin_numeric(x: pd.Series, edges: list[float]) -> pd.Series:
        return pd.Series(np.searchsorted(edges, x.to_numpy(float), side="left"), index=x.index)

    @staticmethod
    def _interval_names(edges: list[float]) -> list[str]:
        b = [-np.inf, *edges, np.inf]
        def fmt(v):
            if v == -np.inf:
                return "−∞"
            if v == np.inf:
                return "+∞"
            if abs(v) >= 1000:
                return f"{v:,.0f}".replace(",", " ")
            return f"{v:.2f}".rstrip("0").rstrip(".").replace("-", "−")
        return [f"({fmt(b[i])}; {fmt(b[i + 1])}]" for i in range(len(b) - 1)]

    def _table(self, labels: pd.Series, y: pd.Series, names) -> pd.DataFrame:
        g = pd.DataFrame({"bin": labels, "y": y}).groupby("bin")["y"].agg(n="size", bad="sum")
        if names is not None:
            g = g.reindex(range(len(names)), fill_value=0)
        g["good"] = g["n"] - g["bad"]
        g["bad_rate"] = g["bad"] / g["n"].where(g["n"] > 0)
        pg = (g["good"] + SMOOTH) / (self.G_ + SMOOTH)
        pb = (g["bad"] + SMOOTH) / (self.B_ + SMOOTH)
        g["woe"] = np.log(pg / pb)
        g["iv"] = (pg - pb) * g["woe"]
        g = g.reset_index()
        if names is not None:
            g["label"] = names
        else:
            g["label"] = g["bin"]
        return g

    # ---------- применение ----------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in X.columns:
            woe = self.tables_[col].set_index("bin")["woe"]
            if col in self.edges_:
                codes = self._bin_numeric(X[col], self.edges_[col])
                out[col] = codes.map(woe).to_numpy()
            else:
                labels = X[col].map(self.maps_[col]).fillna("other")
                out[col] = labels.map(woe).fillna(0.0).to_numpy()   # невиданная категория — нейтральный WoE
        return pd.DataFrame(out, index=X.index)
