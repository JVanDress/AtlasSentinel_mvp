from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from reference.config import DEFAULT_PATHS
from reference.utils import (
    PipelineError,
    assert_no_future_dates,
    configure_logging,
    dataframe_memory_mb,
    deduplicate_prices,
    ensure_dir,
    list_raw_price_files,
    normalize_price_frame,
    read_price_file,
    utc_now_iso,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one canonical research prices parquet from per-ticker raw files.")
    parser.add_argument("--raw-input", default=str(DEFAULT_PATHS.raw_prices_dir))
    parser.add_argument("--output-dir", default=str(DEFAULT_PATHS.research_dir))
    parser.add_argument("--as-of", default=None, help="Optional maximum allowed date YYYY-MM-DD.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    raw_path = Path(args.raw_input)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    files = list_raw_price_files(raw_path)
    parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []

    for file_path in files:
        ticker = file_path.stem.upper().strip()
        status = "ok"
        error_message = None
        row_count = 0
        try:
            raw = read_price_file(file_path, minimal=False)
            normalized = normalize_price_frame(raw, fallback_ticker=ticker)
            row_count = len(normalized)
            if normalized.empty:
                status = "skip_empty"
            else:
                parts.append(normalized)
                ticker = str(normalized["ticker"].iloc[-1]).upper().strip()
        except Exception as exc:
            status = "error"
            error_message = str(exc)[:500]

        audit_rows.append(
            {
                "source_file": file_path.name,
                "ticker": ticker,
                "status": status,
                "error_message": error_message,
                "normalized_rows": row_count,
            }
        )

    if not parts:
        raise PipelineError("No valid price files could be normalized.")

    prices = pd.concat(parts, ignore_index=True)
    prices, duplicates_removed = deduplicate_prices(prices)
    assert_no_future_dates(prices, args.as_of)

    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices["dollar_volume"] = pd.to_numeric(prices["close"], errors="coerce") * pd.to_numeric(prices["volume"], errors="coerce")

    prices_path = output_dir / "research_prices_all.parquet"
    audit_path = output_dir / "research_prices_build_audit.csv"
    summary_path = output_dir / "research_prices_build_summary.json"

    prices.to_parquet(prices_path, index=False)
    pd.DataFrame(audit_rows).sort_values(["status", "ticker", "source_file"]).to_csv(audit_path, index=False)

    summary = {
        "run_timestamp_utc": utc_now_iso(),
        "raw_input": str(raw_path),
        "output_file": str(prices_path),
        "audit_file": str(audit_path),
        "files_scanned": len(files),
        "files_loaded": len(parts),
        "rows_written": int(len(prices)),
        "ticker_count": int(prices["ticker"].nunique()),
        "date_min": str(prices["date"].min().date()),
        "date_max": str(prices["date"].max().date()),
        "duplicates_removed": int(duplicates_removed),
        "memory_mb": round(dataframe_memory_mb(prices), 2),
        "as_of": args.as_of,
    }
    write_json(summary_path, summary)


if __name__ == "__main__":
    main()


