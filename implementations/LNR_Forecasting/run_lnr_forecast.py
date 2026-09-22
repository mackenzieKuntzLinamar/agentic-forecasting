"""Build the LNR service and print a recent target summary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from LNR_Forecasting.analysis import summarize_target_frame
from LNR_Forecasting.data import LNR_RETURN_TARGETS, build_lnr_multivariate_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Load LNR.TO targets and Yahoo market covariates.")
    parser.add_argument("--refresh", action="store_true", help="Refresh Yahoo Finance caches.")
    parser.add_argument("--no-covariates", action="store_true", help="Load LNR targets only.")
    parser.add_argument("--start", default="2015-01-01", help="Earliest history date.")
    args = parser.parse_args()

    service = build_lnr_multivariate_service(
        include_covariates=not args.no_covariates,
        refresh=args.refresh,
        start=args.start,
    )
    as_of = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    print("LNR FORECASTING DATA")
    print("--------------------")
    print(f"As of: {as_of:%Y-%m-%d}")
    print(f"Registered series: {', '.join(service.series_ids)}")
    for window, series_id in LNR_RETURN_TARGETS.items():
        frame = service.get_series(series_id, as_of=as_of)
        summary = summarize_target_frame(frame)
        print(
            f"{window:>2} business days: {summary['observations']:.0f} rows, latest return {summary['latest_value']:.6f}"
        )


if __name__ == "__main__":
    main()
