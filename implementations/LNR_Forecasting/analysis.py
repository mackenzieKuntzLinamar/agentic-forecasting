"""Small analysis helpers for the LNR forecasting experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_target_frame(frame: pd.DataFrame) -> pd.Series:
    """Return compact descriptive statistics for a return target frame."""
    values = pd.to_numeric(frame["value"], errors="coerce").dropna()
    if values.empty:
        return pd.Series(dtype=float)
    return pd.Series(
        {
            "observations": int(values.size),
            "mean_log_return": float(values.mean()),
            "annualized_volatility": float(values.std(ddof=1) * np.sqrt(252)),
            "positive_share": float((values > 0).mean()),
            "latest_value": float(values.iloc[-1]),
        }
    )


__all__ = ["summarize_target_frame"]
