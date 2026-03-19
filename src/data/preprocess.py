from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd

CHINESE_TO_ENGLISH = {
    "日期": "date",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "收盘价": "close",
    "成交量(股)": "volume",
    "涨跌额": "price_change",
    "涨跌幅(%)": "pct_change",
}

REQUIRED_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "price_change",
    "pct_change",
]


def preprocess_daily_prices(
    df: pd.DataFrame,
    rename_map: Mapping[str, str] = CHINESE_TO_ENGLISH,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:

    out = df.rename(columns=rename_map).copy()

    missing = set(REQUIRED_COLUMNS) - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns after rename: {sorted(missing)}")

    out = out[REQUIRED_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    numeric_cols = ["open", "high", "low", "close", "volume", "price_change", "pct_change"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["date", "close"])
    out = out.drop_duplicates(subset=["date"], keep="last")
    out = out.sort_values("date").reset_index(drop=True)

    if start_date is not None:
        out = out[out["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        out = out[out["date"] <= pd.to_datetime(end_date)]

    # --- compute returns ---
    out["log_return"] = 100.0 * np.log(out["close"]).diff()

    # --- drop first NaN ---
    out = out.dropna(subset=["log_return"])

    # --- 🔥 FIX: remove extreme data errors ---
    out = out[(out["log_return"] > -50) & (out["log_return"] < 50)]

    out = out.reset_index(drop=True)

    return out