"""Full energy/oil forecasting curriculum adapted to Linamar Corporation."""

from .data import (
	DEFAULT_COVARIATE_SERIES_IDS,
	DEFAULT_LNR_COVARIATE_SERIES_IDS,
	LNR_RETURN_TARGETS,
	LNR_RETURN_WINDOWS,
	LNR_SERIES_ID,
	LNR_TICKER,
	build_lnr_multivariate_service,
	build_lnr_service,
	lnr_logret_series_id,
	naive_utc_now,
)

__all__ = [
	"DEFAULT_COVARIATE_SERIES_IDS",
	"DEFAULT_LNR_COVARIATE_SERIES_IDS",
	"LNR_RETURN_TARGETS",
	"LNR_RETURN_WINDOWS",
	"LNR_SERIES_ID",
	"LNR_TICKER",
	"build_lnr_multivariate_service",
	"build_lnr_service",
	"lnr_logret_series_id",
	"naive_utc_now",
]
