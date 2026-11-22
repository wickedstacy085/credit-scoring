"""Загрузка данных UCI «Default of credit card clients» и производные признаки."""
from pathlib import Path

import numpy as np
import pandas as pd

PAY = [f"pay_{i}" for i in range(1, 7)]
BILL = [f"bill_amt{i}" for i in range(1, 7)]
PAYAMT = [f"pay_amt{i}" for i in range(1, 7)]


def load(path: str | Path = "data/credit_default.parquet") -> pd.DataFrame:
    """Чистые названия колонок, расшифрованные категории и признаки поведения за 6 месяцев.

    pay_k — статус платежа k месяцев назад (−2/−1/0 — без просрочки, 1–8 — месяцев просрочки);
    в исходнике первая колонка названа PAY_0, здесь — pay_1, как bill_amt1 и pay_amt1."""
    df = pd.read_parquet(path)
    df.columns = df.columns.str.lower()
    df = df.rename(columns={"pay_0": "pay_1", "default payment next month": "default"}).drop(columns="id")
    df["sex"] = df["sex"].map({1: "male", 2: "female"})
    df["education"] = df["education"].map({1: "graduate", 2: "university", 3: "high_school"}).fillna("other")
    df["marriage"] = df["marriage"].map({1: "married", 2: "single"}).fillna("other")

    df["max_delay_6m"] = df[PAY].max(axis=1)
    df["months_delayed_6m"] = (df[PAY] > 0).sum(axis=1)
    df["util_1"] = df["bill_amt1"] / df["limit_bal"]                      # загрузка лимита сейчас
    df["util_avg_6m"] = df[BILL].mean(axis=1) / df["limit_bal"]
    prev_bill = df["bill_amt2"]
    df["pay_ratio_1"] = np.where(prev_bill > 0, df["pay_amt1"] / prev_bill.where(prev_bill > 0), 1.0)
    df["pay_ratio_1"] = df["pay_ratio_1"].clip(0, 2)                     # доля прошлого счёта, оплаченная в месяце
    return df
