"""Leak-safe data service for Linamar Corporation (``LNR.TO``) forecasting.

The target is adjusted-close cumulative log return over 1, 5, or 21 business
sessions. Yahoo market covariates are lagged by one business day before they are
registered, so a forecast origin cannot see same-session covariate values.
"""

from __future__ import annotations

from datetime import datetime, timezone

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters.yfinance import YFinanceDailyAdapter
from aieng.forecasting.data.features import (
    StaticFrameAdapter,
    apply_one_business_day_feature_lag,
    business_daily_ffill,
    to_level_feature_from_daily,
    to_log_return_feature,
)


LNR_TICKER = "LNR.TO"
LNR_SERIES_ID = "lnr_close_adj_cad"
LNR_RETURN_WINDOWS: tuple[int, ...] = (1, 5, 21)
LNR_RETURN_TARGETS: dict[int, str] = {window: f"lnr_logret_{window}b" for window in LNR_RETURN_WINDOWS}

LNR_WINDOW_LABELS: dict[int, str] = {
    1: "next-session",
    5: "forward 1-week (5 business days)",
    21: "forward 1-month (21 business days)",
}

SERIES_ID_VIX_LEVEL = "vix_level_l1b"
SERIES_ID_OIL_RETURN = "oil_log_ret_1b_l1b"
SERIES_ID_NASDAQ_RETURN = "nasdaq_log_ret_1b_l1b"
SERIES_ID_GOLD_RETURN = "gold_log_ret_1b_l1b"
SERIES_ID_DOLLAR_RETURN = "dollar_index_log_ret_1b_l1b"

DEFAULT_COVARIATE_SERIES_IDS: list[str] = [
    SERIES_ID_VIX_LEVEL,
    SERIES_ID_OIL_RETURN,
    SERIES_ID_NASDAQ_RETURN,
    SERIES_ID_GOLD_RETURN,
    SERIES_ID_DOLLAR_RETURN,
]

_COVARIATES: dict[str, tuple[str, str]] = {
    SERIES_ID_VIX_LEVEL: ("^VIX", "level"),
    SERIES_ID_OIL_RETURN: ("CL=F", "return"),
    SERIES_ID_NASDAQ_RETURN: ("^IXIC", "return"),
    SERIES_ID_GOLD_RETURN: ("GC=F", "return"),
    SERIES_ID_DOLLAR_RETURN: ("DX-Y.NYB", "return"),
}

DEFAULT_LNR_COVARIATE_SERIES_IDS = DEFAULT_COVARIATE_SERIES_IDS


def naive_utc_now() -> datetime:
    """Return the current UTC time without timezone information."""
    return datetime.now(tz=timezone.utc).replace(tzinfo=None)


def build_lnr_service(cache_dir: Path | None = None) -> DataService:
    """Build the LNR target service used by the mirrored notebooks."""
    return build_lnr_multivariate_service(
        include_covariates=False,
        cache_dir=cache_dir,
    )


def lnr_logret_series_id(window: int) -> str:
    """Return the target id for an N-business-day LNR return."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}.")
    return f"lnr_logret_{window}b"


def _return_frame(price_df: pd.DataFrame, window: int) -> pd.DataFrame:
    frame = price_df[["timestamp", "value"]].copy().sort_values("timestamp")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame[frame["value"] > 0].dropna(subset=["value"]).reset_index(drop=True)
    frame["value"] = np.log(frame["value"] / frame["value"].shift(window))
    frame = frame.dropna(subset=["value"]).reset_index(drop=True)
    frame["released_at"] = pd.to_datetime(frame["timestamp"])
    return frame[["timestamp", "value", "released_at"]]


def _load_close_frame(
    ticker: str,
    *,
    start: str,
    end: str | None,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    raw = YFinanceDailyAdapter(
        ticker,
        field="Adj Close",
        start=start,
        end=end,
        cache_dir=cache_dir,
        refresh=refresh,
    ).fetch()
    return raw[["timestamp", "value"]].sort_values("timestamp").reset_index(drop=True)


def _register_target_service(
    price_df: pd.DataFrame,
    *,
    windows: tuple[int, ...],
) -> DataService:
    service = DataService()
    service.register(
        LNR_SERIES_ID,
        StaticFrameAdapter(price_df.assign(released_at=price_df["timestamp"])),
        SeriesMetadata(
            series_id=LNR_SERIES_ID,
            description="Linamar Corporation adjusted close (Yahoo Finance LNR.TO)",
            source=f"Yahoo Finance ({LNR_TICKER})",
            units="CAD/share",
            frequency="B",
            table_id="yahoo:LNR.TO:adj-close",
        ),
    )
    for window in windows:
        series_id = lnr_logret_series_id(window)
        label = LNR_WINDOW_LABELS.get(window, f"{window} business days")
        service.register(
            series_id,
            StaticFrameAdapter(_return_frame(price_df, window)),
            SeriesMetadata(
                series_id=series_id,
                description=f"LNR.TO adjusted-close cumulative log return over {window} business day(s) ({label})",
                source=f"Yahoo Finance ({LNR_TICKER}), derived",
                units="log-return",
                frequency="B",
                table_id=f"yahoo:LNR.TO:logret-{window}b",
            ),
        )
    return service


def build_lnr_multivariate_service(
    *,
    windows: tuple[int, ...] = LNR_RETURN_WINDOWS,
    include_covariates: bool = True,
    covariate_series_ids: list[str] | None = None,
    strict_covariates: bool = False,
    refresh: bool = False,
    start: str = "2015-01-01",
    end: str | None = None,
    cache_dir: Path | None = None,
) -> DataService:
    """Build LNR return targets and optional lagged Yahoo market covariates."""
    resolved_cache = cache_dir or Path("data/yfinance")
    resolved_cache.mkdir(parents=True, exist_ok=True)
    prices = _load_close_frame(LNR_TICKER, start=start, end=end, cache_dir=resolved_cache, refresh=refresh)
    service = _register_target_service(prices, windows=windows)
    if not include_covariates:
        return service

    desired = covariate_series_ids or DEFAULT_COVARIATE_SERIES_IDS
    for series_id in desired:
        ticker, feature_kind = _COVARIATES[series_id]
        try:
            close = _load_close_frame(ticker, start=start, end=end, cache_dir=resolved_cache, refresh=refresh)
            feature = to_level_feature_from_daily(close) if feature_kind == "level" else to_log_return_feature(close)
            feature = business_daily_ffill(feature)
            feature = apply_one_business_day_feature_lag(feature)
            service.register(
                series_id,
                StaticFrameAdapter(feature),
                SeriesMetadata(
                    series_id=series_id,
                    description=f"{ticker} {feature_kind}, lagged one business day",
                    source=f"Yahoo Finance ({ticker}), derived",
                    units="level" if feature_kind == "level" else "log-return",
                    frequency="B",
                    table_id=f"yahoo:{ticker}:{feature_kind}:lag1b",
                ),
            )
        except (RuntimeError, ValueError, KeyError) as exc:
            if strict_covariates:
                raise RuntimeError(f"Failed to build required covariate {series_id!r}.") from exc
            warnings.warn(f"Skipping unavailable covariate {series_id!r}: {exc}", stacklevel=2)
    return service


__all__ = [
    "DEFAULT_COVARIATE_SERIES_IDS",
    "DEFAULT_LNR_COVARIATE_SERIES_IDS",
    "LNR_RETURN_TARGETS",
    "LNR_RETURN_WINDOWS",
    "LNR_SERIES_ID",
    "LNR_TICKER",
    "SERIES_ID_DOLLAR_RETURN",
    "SERIES_ID_GOLD_RETURN",
    "SERIES_ID_NASDAQ_RETURN",
    "SERIES_ID_OIL_RETURN",
    "SERIES_ID_VIX_LEVEL",
    "build_lnr_multivariate_service",
    "build_lnr_service",
    "lnr_logret_series_id",
    "naive_utc_now",
]
