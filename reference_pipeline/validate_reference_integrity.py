from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from reference.config import DEFAULT_PATHS, DEFAULT_THRESHOLDS
from reference.utils import PipelineError, configure_logging, rows_to_dicts, utc_now_iso, write_json


LOGGER = logging.getLogger("quintic.validate_reference_integrity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate research prices and ticker reference integrity before research use."
    )
    parser.add_argument(
        "--research-prices",
        default=str(DEFAULT_PATHS.research_dir / "research_prices_all.parquet"),
    )
    parser.add_argument(
        "--reference-master",
        default=str(DEFAULT_PATHS.reference_dir / "ticker_reference_master.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_PATHS.reference_dir),
    )
    parser.add_argument("--as-of", default=None)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    research_prices_path = Path(args.research_prices)
    reference_master_path = Path(args.reference_master)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not research_prices_path.exists():
        raise PipelineError(f"Research prices file not found: {research_prices_path}")
    if not reference_master_path.exists():
        raise PipelineError(f"Reference master file not found: {reference_master_path}")

    prices = pd.read_parquet(research_prices_path).copy()
    reference = pd.read_parquet(reference_master_path).copy()

    failures: list[dict[str, object]] = []
    warnings_list: list[dict[str, object]] = []

    duplicate_rows = prices.loc[prices.duplicated(subset=["ticker", "date"], keep=False)]
    if not duplicate_rows.empty:
        failures.append(
            {
                "check": "duplicate_ticker_date",
                "count": int(len(duplicate_rows)),
                "examples": rows_to_dicts(duplicate_rows[["ticker", "date"]]),
            }
        )

    max_allowed = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.utcnow().normalize()
    future_rows = prices.loc[pd.to_datetime(prices["date"], errors="coerce") > max_allowed]
    if not future_rows.empty:
        failures.append(
            {
                "check": "future_price_dates",
                "count": int(len(future_rows)),
                "examples": rows_to_dicts(future_rows[["ticker", "date"]]),
            }
        )

    price_tickers = set(prices["ticker"].astype(str).unique())
    reference_tickers = set(reference["ticker"].astype(str).unique())
    missing_reference = sorted(price_tickers - reference_tickers)

    if missing_reference:
        LOGGER.warning(
            "Dropping %s tickers missing from reference. Example: %s",
            len(missing_reference),
            missing_reference[:10],
        )
        warnings_list.append(
            {
                "check": "tickers_missing_from_reference",
                "count": len(missing_reference),
                "examples": [{"ticker": t} for t in missing_reference[:50]],
            }
        )
        prices = prices.loc[~prices["ticker"].isin(missing_reference)].copy()

    ok_rows = reference.loc[reference["reference_status"] == "ok"].copy()

    bad_ok_market_cap = ok_rows.loc[ok_rows["market_cap"].isna()]
    if not bad_ok_market_cap.empty:
        failures.append(
            {
                "check": "ok_rows_missing_market_cap",
                "count": int(len(bad_ok_market_cap)),
                "examples": rows_to_dicts(bad_ok_market_cap[["ticker", "market_cap"]]),
            }
        )

    bad_ok_common = ok_rows.loc[~ok_rows["is_common_stock"].fillna(False)]
    if not bad_ok_common.empty:
        failures.append(
            {
                "check": "ok_rows_not_common_stock",
                "count": int(len(bad_ok_common)),
                "examples": rows_to_dicts(
                    bad_ok_common[["ticker", "type_description", "reference_status"]]
                ),
            }
        )

    unreasonable_caps = reference.loc[
        (reference["market_cap"].notna()) & (reference["market_cap"] <= 0)
    ]
    if not unreasonable_caps.empty:
        failures.append(
            {
                "check": "non_positive_market_caps",
                "count": int(len(unreasonable_caps)),
                "examples": rows_to_dicts(unreasonable_caps[["ticker", "market_cap"]]),
            }
        )

    status_counts = reference["reference_status"].value_counts(dropna=False).to_dict()
    if status_counts.get("missing_sector", 0) > 0 or status_counts.get("missing_industry", 0) > 0:
        failures.append(
            {
                "check": "classification_is_hard_rejecting",
                "count": int(
                    status_counts.get("missing_sector", 0)
                    + status_counts.get("missing_industry", 0)
                ),
                "examples": [{"status_counts": status_counts}],
            }
        )

    report = {
        "run_timestamp_utc": utc_now_iso(),
        "research_prices": str(research_prices_path),
        "reference_master": str(reference_master_path),
        "as_of": args.as_of,
        "thresholds": {
            "min_price": DEFAULT_THRESHOLDS.min_price,
            "min_avg_volume": DEFAULT_THRESHOLDS.min_avg_volume,
            "min_market_cap": DEFAULT_THRESHOLDS.min_market_cap,
        },
        "passed": len(failures) == 0,
        "failure_count": len(failures),
        "warning_count": len(warnings_list),
        "failures": failures,
        "warnings": warnings_list,
        "price_rows": int(len(prices)),
        "reference_rows": int(len(reference)),
        "reference_ok_rows": int((reference["reference_status"] == "ok").sum()),
    }

    write_json(output_dir / "reference_integrity_report.json", report)
    pd.DataFrame(failures).to_csv(output_dir / "reference_integrity_failures.csv", index=False)
    pd.DataFrame(warnings_list).to_csv(output_dir / "reference_integrity_warnings.csv", index=False)

    if failures:
        raise PipelineError(f"Reference integrity validation failed with {len(failures)} issue(s).")


if __name__ == "__main__":
    main()