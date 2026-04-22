"""
Quintic Labs — Backfill Sector Labels
======================================
The 2015-2020 rows from extended_stock_prices.parquet have no sector/industry
labels. This script fills them using the ticker->sector mapping from 2021+ data.

A ticker's GICS sector does not change (barring rare reclassifications).

Input:  stocks_universe_merged.parquet (modified IN PLACE — backs up first)
Output: stocks_universe_merged.parquet (with sector/industry filled)
Backup: stocks_universe_merged.parquet.bak

Usage:
    python backfill_sector_labels.py
"""

import os, sys, shutil, time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
DATA_ROOT = os.getenv("DATA_ROOT", r"Z:\jvand\QUINTIC_V3\data\stage")


def main():
    path = os.path.join(DATA_ROOT, "stocks_universe_merged.parquet")
    backup = path + ".bak"

    if not os.path.exists(path):
        print(f"FATAL: {path} not found")
        sys.exit(1)

    print(f"Loading {path} ...")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    print(f"Rows: {len(df):,} | Tickers: {df['ticker'].nunique():,}")

    # ── Identify sector columns ──────────────────────────────────────────
    sector_cols = [c for c in df.columns if 'sector' in c.lower() or 'industry' in c.lower()]
    print(f"Sector/industry columns found: {sector_cols}")

    # ── Build ticker -> label mapping from 2021+ data ────────────────────
    post2021 = df[df["date"] >= "2021-01-01"].copy()
    pre2021_mask = df["date"] < "2021-01-01"
    pre2021_count = pre2021_mask.sum()

    print(f"Pre-2021 rows: {pre2021_count:,}")
    print(f"Post-2021 rows: {len(post2021):,}")

    for col in sector_cols:
        # Get the most common label per ticker from 2021+ data
        valid = post2021[post2021[col].notna()][["ticker", col]]
        if valid.empty:
            print(f"  {col}: no valid data in 2021+, skipping")
            continue

        # Mode per ticker (most frequent label)
        mapping = valid.groupby("ticker")[col].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)
        mapping = mapping.dropna().to_dict()

        # Count nulls before
        nulls_before = int(df[col].isna().sum())

        # Fill
        fill_mask = df[col].isna()
        df.loc[fill_mask, col] = df.loc[fill_mask, "ticker"].map(mapping)

        # Count nulls after
        nulls_after = int(df[col].isna().sum())
        filled = nulls_before - nulls_after

        print(f"  {col}: filled {filled:,} rows | remaining nulls: {nulls_after:,}")

    # ── Also create gics_sector if it doesn't exist ──────────────────────
    if "gics_sector" not in df.columns and "sector" in df.columns:
        print("  Creating gics_sector from sector column ...")
        df["gics_sector"] = df["sector"]
        print(f"  gics_sector nulls: {df['gics_sector'].isna().sum():,}")
    elif "gics_sector" in df.columns:
        # Fill gics_sector too
        valid = post2021[post2021["gics_sector"].notna()][["ticker", "gics_sector"]]
        if not valid.empty:
            mapping = valid.groupby("ticker")["gics_sector"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)
            mapping = mapping.dropna().to_dict()
            fill_mask = df["gics_sector"].isna()
            df.loc[fill_mask, "gics_sector"] = df.loc[fill_mask, "ticker"].map(mapping)
            print(f"  gics_sector nulls after fill: {df['gics_sector'].isna().sum():,}")

    # ── Backup original ──────────────────────────────────────────────────
    print(f"\nBacking up to {backup} ...")
    shutil.copy2(path, backup)

    # ── Save ─────────────────────────────────────────────────────────────
    df.to_parquet(path, index=False)
    file_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Saved: {path} ({file_mb:.1f} MB)")

    # ── Verify ───────────────────────────────────────────────────────────
    print("\nVerification:")
    for col in sector_cols + (["gics_sector"] if "gics_sector" not in sector_cols else []):
        if col in df.columns:
            nulls = df[col].isna().sum()
            unique = df[col].nunique()
            print(f"  {col}: {unique} unique values | {nulls:,} nulls remaining")

    pre = df[df["date"] < "2021-01-01"]
    print(f"\nPre-2021 sector coverage:")
    for col in sector_cols + (["gics_sector"] if "gics_sector" not in sector_cols else []):
        if col in pre.columns:
            coverage = pre[col].notna().sum()
            print(f"  {col}: {coverage:,} / {len(pre):,} ({coverage/len(pre)*100:.1f}%)")


if __name__ == "__main__":
    main()