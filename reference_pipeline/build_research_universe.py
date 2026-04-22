from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from reference.config import DEFAULT_PATHS, DEFAULT_THRESHOLDS
from reference.utils import PipelineError, configure_logging, utc_now_iso, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stock-only and eligible research universes from prices + reference master.")
    parser.add_argument("--research-prices", default=str(DEFAULT_PATHS.research_dir / "research_prices_all.parquet"))
    parser.add_argument("--reference-master", default=str(DEFAULT_PATHS.reference_dir / "ticker_reference_master.parquet"))
    parser.add_argument("--output-dir", default=str(DEFAULT_PATHS.research_dir))
    parser.add_argument("--min-price", type=float, default=DEFAULT_THRESHOLDS.min_price)
    parser.add_argument("--min-avg-volume", type=float, default=DEFAULT_THRESHOLDS.min_avg_volume)
    parser.add_argument("--min-market-cap", type=float, default=DEFAULT_THRESHOLDS.min_market_cap)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_THRESHOLDS.prefilter_lookback_days)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
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

    prices = pd.read_parquet(research_prices_path)
    reference = pd.read_parquet(reference_master_path)

    reference["is_stock_reference_ok"] = (
        reference["is_common_stock"].fillna(False)
        & reference["market_cap"].notna()
        & (reference["market_cap"] >= args.min_market_cap)
        & (reference["reference_status"] == "ok")
    )

    work = prices.sort_values(["ticker", "date"]).copy()
    work["avg_close_last_window"] = work.groupby("ticker")["close"].transform(
        lambda s: s.rolling(args.lookback_days, min_periods=1).mean()
    )
    work["avg_volume_last_window"] = work.groupby("ticker")["volume"].transform(
        lambda s: s.rolling(args.lookback_days, min_periods=1).mean()
    )

    latest = work.groupby("ticker", as_index=False).tail(1)[["ticker", "avg_close_last_window", "avg_volume_last_window"]].copy()
    reference = reference.merge(latest, on="ticker", how="left", validate="1:1")
    reference["passes_price_threshold"] = reference["avg_close_last_window"] >= args.min_price
    reference["passes_volume_threshold"] = reference["avg_volume_last_window"] >= args.min_avg_volume
    reference["eligible"] = (
        reference["is_stock_reference_ok"]
        & reference["passes_price_threshold"].fillna(False)
        & reference["passes_volume_threshold"].fillna(False)
    )

    stock_only_tickers = set(reference.loc[reference["is_stock_reference_ok"], "ticker"].astype(str))
    eligible_tickers = set(reference.loc[reference["eligible"], "ticker"].astype(str))

    research_prices_stock_only = work.loc[work["ticker"].isin(stock_only_tickers)].copy()
    research_prices_eligible = work.loc[work["ticker"].isin(eligible_tickers)].copy()
    eligible_ticker_list = reference.loc[reference["eligible"]].sort_values("ticker").reset_index(drop=True)

    stock_only_path = output_dir / "research_prices_stock_only.parquet"
    eligible_prices_path = output_dir / "research_prices_eligible.parquet"
    eligible_tickers_path = output_dir / "eligible_ticker_list.parquet"
    universe_reference_path = output_dir / "research_universe_reference_snapshot.parquet"
    summary_path = output_dir / "research_universe_summary.json"

    research_prices_stock_only.to_parquet(stock_only_path, index=False)
    research_prices_eligible.to_parquet(eligible_prices_path, index=False)
    eligible_ticker_list.to_parquet(eligible_tickers_path, index=False)
    reference.to_parquet(universe_reference_path, index=False)

    summary = {
        "run_timestamp_utc": utc_now_iso(),
        "research_prices": str(research_prices_path),
        "reference_master": str(reference_master_path),
        "stock_only_rows": int(len(research_prices_stock_only)),
        "eligible_rows": int(len(research_prices_eligible)),
        "stock_only_ticker_count": int(research_prices_stock_only["ticker"].nunique()),
        "eligible_ticker_count": int(research_prices_eligible["ticker"].nunique()),
        "thresholds": {
            "min_price": args.min_price,
            "min_avg_volume": args.min_avg_volume,
            "min_market_cap": args.min_market_cap,
            "lookback_days": args.lookback_days,
        },
        "output_files": [
            str(stock_only_path),
            str(eligible_prices_path),
            str(eligible_tickers_path),
            str(universe_reference_path),
        ],
    }
    write_json(summary_path, summary)


if __name__ == "__main__":
    main()
