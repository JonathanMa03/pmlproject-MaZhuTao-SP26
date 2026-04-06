from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class VARDesign:
    """
    Container for the lagged VAR design matrices.

    Attributes
    ----------
    X : np.ndarray
        Design matrix of shape (T - p, k), where
        k = 1 + d * p if intercept=True, else d * p.
    Y : np.ndarray
        Target matrix of shape (T - p, d).
    dates : pd.Series | None
        Dates aligned with Y, length T - p.
    feature_names : list[str]
        Names of columns in X.
    p : int
        VAR lag order.
    d : int
        Number of series.
    intercept : bool
        Whether an intercept was included.
    """
    X: np.ndarray
    Y: np.ndarray
    dates: Optional[pd.Series]
    feature_names: list[str]
    p: int
    d: int
    intercept: bool


class BaseVAR:
    """
    Shared utilities for VAR-style models.

    This class does not fit a model by itself.
    It only builds the lagged design matrices that all later models will use.
    """

    def __init__(self, p: int = 1, intercept: bool = True):
        if p < 1:
            raise ValueError("p must be at least 1.")
        self.p = p
        self.intercept = intercept

    def build_design(
        self,
        data: pd.DataFrame,
        date_col: str = "date",
    ) -> VARDesign:
        """
        Build lagged design matrix X and aligned target matrix Y for a VAR(p).

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with one date column and the remaining columns as numeric series.
        date_col : str
            Name of the date column.

        Returns
        -------
        VARDesign
            Lagged design object containing X, Y, aligned dates, and metadata.
        """
        if date_col not in data.columns:
            raise ValueError(f"'{date_col}' not found in DataFrame.")

        value_cols = [c for c in data.columns if c != date_col]
        if len(value_cols) == 0:
            raise ValueError("No value columns found.")

        df = data.copy()
        df = df.sort_values(date_col).reset_index(drop=True)

        # numeric matrix
        Y_all = df[value_cols].to_numpy(dtype=float)
        dates_all = df[date_col].reset_index(drop=True)

        T, d = Y_all.shape
        if T <= self.p:
            raise ValueError(
                f"Need more observations than lags: got T={T}, p={self.p}."
            )

        X_rows = []
        Y_rows = []
        aligned_dates = []

        for t in range(self.p, T):
            row = []

            if self.intercept:
                row.append(1.0)

            for lag in range(1, self.p + 1):
                row.extend(Y_all[t - lag, :].tolist())

            X_rows.append(row)
            Y_rows.append(Y_all[t, :])
            aligned_dates.append(dates_all.iloc[t])

        X = np.asarray(X_rows, dtype=float)
        Y = np.asarray(Y_rows, dtype=float)
        dates = pd.Series(aligned_dates, name=date_col)

        feature_names = self._make_feature_names(value_cols)

        return VARDesign(
            X=X,
            Y=Y,
            dates=dates,
            feature_names=feature_names,
            p=self.p,
            d=d,
            intercept=self.intercept,
        )

    def _make_feature_names(self, value_cols: list[str]) -> list[str]:
        names: list[str] = []

        if self.intercept:
            names.append("intercept")

        for lag in range(1, self.p + 1):
            for col in value_cols:
                names.append(f"{col}_lag{lag}")

        return names

    @staticmethod
    def to_numpy(data: pd.DataFrame, date_col: str = "date") -> np.ndarray:
        """
        Return only the numeric series as a NumPy array.
        """
        value_cols = [c for c in data.columns if c != date_col]
        return data[value_cols].to_numpy(dtype=float)