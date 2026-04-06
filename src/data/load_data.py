from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

COMMON_CSV_ENCODINGS = ("utf-8", "utf-8-sig", "gbk", "gb18030", "big5")


def read_csv_with_fallback_encodings(csv_path: Path) -> pd.DataFrame:
    """Read a CSV using common UTF/Chinese encodings."""
    last_error: UnicodeDecodeError | None = None

    for encoding in COMMON_CSV_ENCODINGS:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError as err:
            last_error = err

    if last_error is None:
        return pd.read_csv(csv_path)

    raise UnicodeDecodeError(
        last_error.encoding,
        last_error.object,
        last_error.start,
        last_error.end,
        f"Unable to decode {csv_path} with tried encodings: {', '.join(COMMON_CSV_ENCODINGS)}",
    )


def load_raw_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load one raw CSV file without preprocessing."""
    csv_path = Path(csv_path)
    return read_csv_with_fallback_encodings(csv_path)