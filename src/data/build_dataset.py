from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from config import ETF_FILES
from src.data.load_data import load_raw_csv
from src.data.preprocess import preprocess_daily_prices


def load_and_preprocess_one(
    csv_path: str | Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    raw = load_raw_csv(csv_path)
    clean = preprocess_daily_prices(raw, start_date=start_date, end_date=end_date)
    return clean


def build_market_dataset(
    file_map: Optional[Dict[str, str | Path]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a merged log-return dataset with columns:
        date | SPY | QQQ | DIA | IWN
    """
    if file_map is None:
        file_map = ETF_FILES

    merged: pd.DataFrame | None = None

    for ticker, path in file_map.items():
        df = load_and_preprocess_one(path, start_date=start_date, end_date=end_date)
        df = df[["date", "log_return"]].rename(columns={"log_return": ticker})

        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="date", how="inner")

    if merged is None:
        raise ValueError("No files provided.")

    merged = merged.sort_values("date").dropna().reset_index(drop=True)
    return merged