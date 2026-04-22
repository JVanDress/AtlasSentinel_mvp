from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from reference.config import DEFAULT_PATHS, RuntimeConfig
from reference.polygon_client import fetch_reference_rows
from reference.utils import (
    PipelineError,
    configure_logging,
    ensure_dir,
    stable_reference_industry,
    utc_now_iso,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Polygon reference data for research tickers and build the reference master.")
    parser.add_argument("--research-prices", default=str(DEFAULT_PATHS.research_dir / "research_prices_all.parquet"))
    parser.add_argument("--output-dir", default=str(DEFAULT_PATHS.reference_dir))
    parser.add_argument("--api-key", default=None, help="Polygon API key. Falls back to POLYGON_API_KEY if omitted.")
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--include-inactive", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    research_prices_path = Path(args.research_prices)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    if not research_prices_path.exists():
        raise PipelineError(f"Research prices file not found: {research_prices_path}")

    prices = pd.read_parquet(research_prices_path, columns=["ticker"])
    tickers = sorted(t for t in prices["ticker"].astype(str).str.upper().str.strip().dropna().unique() if t)
    if not tickers:
        raise PipelineError("No tickers found in research prices file.")

    config = RuntimeConfig(
        polygon_api_key=args.api_key or __import__("os").getenv("POLYGON_API_KEY"),
        as_of=args.as_of,
        include_inactive=args.include_inactive,
        sleep_seconds_between_calls=args.sleep_seconds,
        request_timeout_seconds=args.timeout_seconds,
    )

    reference, fetch_audit = fetch_reference_rows(tickers, config)
    if reference.empty:
        raise PipelineError("Polygon reference fetch returned no rows.")

    reference["sector_missing"] = reference["sector"].isna() | (reference["sector"].astype(str).str.strip() == "")
    reference["industry_missing"] = reference["industry"].isna() | (reference["industry"].astype(str).str.strip() == "")
    reference["sub_industry_missing"] = reference["sub_industry"].isna() | (reference["sub_industry"].astype(str).str.strip() == "")
    reference["reference_industry"] = reference.apply(
        lambda row: stable_reference_industry(
            row.get("sub_industry"),
            row.get("industry"),
            row.get("sic_description"),
            row.get("type_description"),
        ),
        axis=1,
    )

    if not args.include_inactive:
        reference = reference.loc[reference["active"] != False].copy()

    reference = reference.sort_values(["reference_status", "ticker"]).reset_index(drop=True)

    parquet_path = output_dir / "ticker_reference_master.parquet"
    csv_path = output_dir / "ticker_reference_master.csv"
    fetch_audit_path = output_dir / "polygon_reference_fetch_audit.csv"
    summary_path = output_dir / "ticker_reference_summary.json"

    reference.to_parquet(parquet_path, index=False)
    reference.to_csv(csv_path, index=False)
    fetch_audit.to_csv(fetch_audit_path, index=False)

    summary = {
        "run_timestamp_utc": utc_now_iso(),
        "research_prices": str(research_prices_path),
        "output_file": str(parquet_path),
        "csv_file": str(csv_path),
        "fetch_audit_file": str(fetch_audit_path),
        "row_count": int(len(reference)),
        "ticker_count": int(reference["ticker"].nunique()),
        "ok_count": int((reference["reference_status"] == "ok").sum()),
        "common_stock_count": int(reference["is_common_stock"].sum()),
        "missing_market_cap_count": int(reference["market_cap"].isna().sum()),
        "missing_sector_count": int(reference["sector_missing"].sum()),
        "missing_industry_count": int(reference["industry_missing"].sum()),
        "missing_sub_industry_count": int(reference["sub_industry_missing"].sum()),
        "status_counts": reference["reference_status"].value_counts(dropna=False).to_dict(),
        "fetch_status_counts": fetch_audit["fetch_status"].value_counts(dropna=False).to_dict(),
        "as_of": args.as_of,
    }
    write_json(summary_path, summary)


if __name__ == "__main__":
    main()
