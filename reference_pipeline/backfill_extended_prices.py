#!/usr/bin/env python3
"""
Quintic Labs — Extended Stock Price Download

Pulls daily OHLCV bars from Polygon going back to 2015 for all tickers
in the current universe. Saves as a separate parquet that can be merged
with the existing panel to extend the training window.

Resumable: skips tickers already downloaded. Safe to re-run.

Usage:
    python backfill_extended_prices.py
    python backfill_extended_prices.py --start-date 2015-01-01 --end-date 2020-12-31
    python backfill_extended_prices.py --max-tickers 200

Requirements:
    pip install polygon-api-client pandas pyarrow numpy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from polygon import RESTClient

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from quintic_paths import data_dir, load_simple_dotenv, project_root


def parse_args():
    parser = argparse.ArgumentParser(description="Quintic Labs — Extended Stock Price Download")
    parser.add_argument("--start-date", default="2015-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", default="2020-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--max-tickers", type=int, default=0, help="Limit tickers (0 = all)")
    parser.add_argument("--output", default=None, help="Output parquet path")
    parser.add_argument("--universe", default=None, help="Universe parquet path")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Save checkpoint every N tickers")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_api_key():
    root = project_root()
    load_simple_dotenv(root, override=True)
    key = (os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY") or "").strip()
    if not key:
        sys.exit("Error: POLYGON_API_KEY or MASSIVE_API_KEY not found in .env")
    return key


def load_universe_tickers(path, max_tickers):
    df = pd.read_parquet(path, columns=["ticker"])
    tickers = (
        df["ticker"].astype(str).str.upper().str.strip()
        .replace("", pd.NA).dropna().drop_duplicates().sort_values().tolist()
    )
    if max_tickers > 0:
        tickers = tickers[:max_tickers]
    return tickers


def load_completed_tickers(output_path):
    """Load tickers already in the output file to skip."""
    if not output_path.exists():
        return set()
    try:
        df = pd.read_parquet(output_path, columns=["ticker"])
        return set(df["ticker"].astype(str).str.upper().str.strip().unique())
    except Exception:
        return set()


def fetch_ticker_history(client, ticker, start_date, end_date):
    """Fetch daily OHLCV bars for a single ticker."""
    try:
        aggs = client.get_aggs(ticker, 1, "day", start_date, end_date, limit=50000)
        rows = []
        for agg in aggs or []:
            ts = getattr(agg, "timestamp", None) or getattr(agg, "t", None)
            if ts is None:
                continue
            rows.append({
                "ticker": ticker,
                "date": pd.to_datetime(ts, unit="ms").normalize(),
                "open": float(getattr(agg, "open", 0) or getattr(agg, "o", 0) or 0),
                "high": float(getattr(agg, "high", 0) or getattr(agg, "h", 0) or 0),
                "low": float(getattr(agg, "low", 0) or getattr(agg, "l", 0) or 0),
                "close": float(getattr(agg, "close", 0) or getattr(agg, "c", 0) or 0),
                "volume": float(getattr(agg, "volume", 0) or getattr(agg, "v", 0) or 0),
                "vwap": float(getattr(agg, "vwap", 0) or getattr(agg, "vw", 0) or 0),
                "transactions": float(getattr(agg, "transactions", 0) or getattr(agg, "n", 0) or 0),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return pd.DataFrame()


def main():
    args = parse_args()
    api_key = require_api_key()
    client = RESTClient(api_key=api_key)

    root = project_root()
    dr = data_dir(root)

    universe_path = Path(args.universe) if args.universe else dr / "stocks_universe.parquet"
    output_path = Path(args.output) if args.output else dr / "extended_stock_prices.parquet"

    if not universe_path.exists():
        sys.exit(f"Error: universe not found: {universe_path}")

    tickers = load_universe_tickers(universe_path, args.max_tickers)
    completed = load_completed_tickers(output_path)
    remaining = [t for t in tickers if t not in completed]

    print(f"Quintic Labs — Extended Stock Price Download")
    print(f"Universe: {len(tickers)} tickers")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Output: {output_path}")

    if args.dry_run:
        est_minutes = len(remaining) * 0.5 / 60
        print(f"Estimated time: {est_minutes:.1f} minutes")
        print("Dry run complete.")
        return

    if not remaining:
        print("All tickers already downloaded. Nothing to do.")
        return

    # Load existing data
    if output_path.exists():
        existing = pd.read_parquet(output_path)
    else:
        existing = pd.DataFrame()

    total = len(remaining)
    t_start = time.time()
    new_frames = []
    failed = []

    for idx, ticker in enumerate(remaining, 1):
        df = fetch_ticker_history(client, ticker, args.start_date, args.end_date)

        if df.empty:
            failed.append(ticker)
        else:
            new_frames.append(df)

        if idx % 25 == 0 or idx == total:
            elapsed = time.time() - t_start
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (total - idx) / rate / 60 if rate > 0 else 0
            rows_so_far = sum(len(f) for f in new_frames)
            print(f"  [{idx}/{total}] downloaded={len(new_frames)} "
                  f"failed={len(failed)} rows={rows_so_far:,} "
                  f"rate={rate:.1f}/s ETA={eta:.1f}min")

        # Checkpoint
        if args.checkpoint_every > 0 and idx % args.checkpoint_every == 0 and new_frames:
            checkpoint = pd.concat([existing] + new_frames, ignore_index=True)
            checkpoint = checkpoint.sort_values(["ticker", "date"]).drop_duplicates(
                subset=["ticker", "date"], keep="last"
            ).reset_index(drop=True)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.to_parquet(output_path, index=False, engine="pyarrow")
            print(f"  Checkpoint saved: {len(checkpoint):,} rows")
            # Reset — fold new_frames into existing
            existing = checkpoint
            new_frames = []

        time.sleep(0.1)

    # Final save
    if new_frames:
        final = pd.concat([existing] + new_frames, ignore_index=True)
    else:
        final = existing

    final = final.sort_values(["ticker", "date"]).drop_duplicates(
        subset=["ticker", "date"], keep="last"
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(output_path, index=False, engine="pyarrow")

    total_time = time.time() - t_start

    status = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "tickers_requested": len(tickers),
        "tickers_downloaded": int(final["ticker"].nunique()) if not final.empty else 0,
        "tickers_failed": len(failed),
        "total_rows": int(len(final)),
        "output": str(output_path),
        "elapsed_seconds": round(total_time, 1),
        "failed_tickers_sample": failed[:20],
    }
    status_path = output_path.with_suffix(".status.json")
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    print(f"\nDone in {total_time/60:.1f} minutes")
    print(f"Output: {output_path}")
    print(f"Total rows: {len(final):,}")
    print(f"Tickers: {final['ticker'].nunique():,}" if not final.empty else "No data")
    if failed:
        print(f"Failed: {len(failed)} tickers")
    print(f"Status: {status_path}")


if __name__ == "__main__":
    main()
