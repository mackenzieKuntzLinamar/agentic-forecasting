"""LNR stock-price forecasting implementation."""

from .data import (
    DEFAULT_COVARIATE_SERIES_IDS,
    LNR_RETURN_TARGETS,
    LNR_RETURN_WINDOWS,
    LNR_SERIES_ID,
    LNR_TICKER,
    build_lnr_multivariate_service,
    lnr_logret_series_id,
)


__all__ = [
    "DEFAULT_COVARIATE_SERIES_IDS",
    "LNR_RETURN_TARGETS",
    "LNR_RETURN_WINDOWS",
    "LNR_SERIES_ID",
    "LNR_TICKER",
    "build_lnr_multivariate_service",
    "lnr_logret_series_id",
]
