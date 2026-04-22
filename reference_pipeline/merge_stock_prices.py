"""
Quintic Labs — Merge Extended + Universe Stock Prices
======================================================
Merges extended_stock_prices.parquet (2015-2020) with stocks_universe.parquet
(2021-2025) into a single combined file. Deduplicates on (ticker, date),
keeps the newer record on overlap, and sorts by (ticker, date).

Output: stocks_universe_merged.parquet  (does NOT overwrite originals)

Usage:
    python merge_stock_prices.py
"""

import sys
from pathlib import Path

import pandas as pd

from quintic_paths import data_dir, project_root


def main() -> int:
    root = project_root()
    stage = data_dir(root)

    extended_path = stage / "extended_stock_prices.parquet"
    universe_path = stage / "stocks_universe.parquet"
    output_path = stage / "stocks_universe_merged.parquet"

    # ── Load both files ──────────────────────────────────────────────────
    if not extended_path.exists():
        print(f"ERROR: {extended_path} not found")
        return 1
    if not universe_path.exists():
        print(f"ERROR: {universe_path} not found")
        return 1

    print("Loading extended_stock_prices.parquet ...")
    ext = pd.read_parquet(extended_path)
    print(f"  Extended: {len(ext):,} rows | {ext['ticker'].nunique():,} tickers | {ext['date'].min()} to {ext['date'].max()}")

    print("Loading stocks_universe.parquet ...")
    uni = pd.read_parquet(universe_path)
    print(f"  Universe: {len(uni):,} rows | {uni['ticker'].nunique():,} tickers | {uni['date'].min()} to {uni['date'].max()}")

    # ── Align columns ────────────────────────────────────────────────────
    ext_cols = set(ext.columns)
    uni_cols = set(uni.columns)
    shared_cols = sorted(ext_cols & uni_cols)

    if not shared_cols:
        print("ERROR: No shared columns between the two files")
        return 1

    extra_in_ext = ext_cols - uni_cols
    extra_in_uni = uni_cols - ext_cols

    if extra_in_ext:
        print(f"  Columns only in extended (will be dropped): {sorted(extra_in_ext)}")
    if extra_in_uni:
        print(f"  Columns only in universe (NaN for extended rows): {sorted(extra_in_uni)}")

    # ── Normalize date columns ───────────────────────────────────────────
    ext["date"] = pd.to_datetime(ext["date"], errors="coerce")
    uni["date"] = pd.to_datetime(uni["date"], errors="coerce")

    ext["ticker"] = ext["ticker"].astype(str).str.upper().str.strip()
    uni["ticker"] = uni["ticker"].astype(str).str.upper().str.strip()

    # ── Concatenate ──────────────────────────────────────────────────────
    # Use all columns from universe as the master schema
    # Extended rows will have NaN for any columns only in universe
    merged = pd.concat([ext, uni], ignore_index=True)
    print(f"\n  Concatenated: {len(merged):,} rows")

    # ── Deduplicate on (ticker, date) — keep universe (newer) on overlap ─
    # Since universe was concatenated second, keep='last' prefers it
    before = len(merged)
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
    merged = merged.drop_duplicates(subset=["ticker", "date"], keep="last").reset_index(drop=True)
    dupes_removed = before - len(merged)
    print(f"  Duplicates removed: {dupes_removed:,}")
    print(f"  Final merged: {len(merged):,} rows")

    # ── Verify ───────────────────────────────────────────────────────────
    print(f"\n  Date range: {merged['date'].min()} to {merged['date'].max()}")
    print(f"  Tickers: {merged['ticker'].nunique():,}")

    # Spot check: no nulls in core OHLCV
    for col in ["open", "high", "low", "close", "volume"]:
        if col in merged.columns:
            nulls = int(merged[col].isna().sum())
            if nulls > 0:
                print(f"  WARNING: {col} has {nulls:,} null values")

    # ── Save ─────────────────────────────────────────────────────────────
    merged.to_parquet(output_path, index=False)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n  Saved: {output_path}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print("  Originals NOT modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())