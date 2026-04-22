"""
build_filtered_universe.py
--------------------------
Reads all_stocks_daily_panel_enriched.parquet, filters by:
  1. Close price >= $10 (most recent available date per ticker)
  2. 22-day ADV >= 300,000 shares
  3. Market cap >= $2B (from Polygon reference data)

Enriches with sector (GICS mapped from SIC) and asset type.
Splits output into stocks_universe.parquet and etf_universe.parquet.

Location: C:\QUINTIC_V3\scripts\reference_pipeline\
"""

import pandas as pd
import numpy as np
import requests
import time
import os
from dotenv import load_dotenv

# ── CONFIG ──────────────────────────────────────────────────────────────────
load_dotenv(r"C:\QUINTIC_V3\scripts\reference_pipeline\.env")

POLYGON_KEY = os.getenv("POLYGON_API_KEY")
if not POLYGON_KEY:
    raise ValueError("POLYGON_API_KEY not found in .env")

INPUT_FILE  = r"Z:\jvand\QUINTIC_V3\data\stage\all_stocks_daily_panel_enriched.parquet"
OUTPUT_DIR  = r"Z:\jvand\QUINTIC_V3\data\stage"

PRICE_FLOOR     = 10.0
ADV_FLOOR       = 300_000
ADV_WINDOW      = 22
MCAP_FLOOR      = 2_000_000_000  # $2B

POLYGON_BASE    = "https://api.polygon.io/v3/reference/tickers"
SLEEP_BETWEEN   = 0.05  # seconds between API calls (Unlimited plan)


# ── SIC-TO-GICS SECTOR MAPPING ─────────────────────────────────────────────
def sic_to_gics_sector(sic_code):
    """Map SIC code to approximate GICS sector."""
    if sic_code is None:
        return "Unknown"
    try:
        sic = int(sic_code)
    except (ValueError, TypeError):
        return "Unknown"

    if 1300 <= sic <= 1389 or 2900 <= sic <= 2999 or 4600 <= sic <= 4699:
        return "Energy"
    elif (1000 <= sic <= 1299 or 1400 <= sic <= 1499 or
          2400 <= sic <= 2499 or 2600 <= sic <= 2699 or
          2800 <= sic <= 2829 or 2840 <= sic <= 2899 or
          3000 <= sic <= 3099 or 3200 <= sic <= 3399):
        return "Materials"
    elif (1500 <= sic <= 1799 or 3400 <= sic <= 3599 or
          3700 <= sic <= 3719 or 3720 <= sic <= 3799 or
          3900 <= sic <= 3999 or 4000 <= sic <= 4299 or
          4400 <= sic <= 4599 or 4700 <= sic <= 4799 or
          8710 <= sic <= 8719 or 8740 <= sic <= 8749):
        return "Industrials"
    elif (2500 <= sic <= 2599 or 3100 <= sic <= 3199 or
          3710 <= sic <= 3716 or 5000 <= sic <= 5199 or
          5200 <= sic <= 5399 or 5500 <= sic <= 5599 or
          5600 <= sic <= 5699 or 5700 <= sic <= 5799 or
          5800 <= sic <= 5899 or 7000 <= sic <= 7299 or
          7800 <= sic <= 7829 or 8200 <= sic <= 8299):
        return "Consumer Discretionary"
    elif (2000 <= sic <= 2199 or 5400 <= sic <= 5499 or
          5900 <= sic <= 5999):
        return "Consumer Staples"
    elif (2830 <= sic <= 2836 or 3841 <= sic <= 3851 or
          5047 <= sic <= 5049 or 8000 <= sic <= 8099 or
          8730 <= sic <= 8734):
        return "Health Care"
    elif (6000 <= sic <= 6199 or 6200 <= sic <= 6299 or
          6300 <= sic <= 6499 or 6700 <= sic <= 6799):
        return "Financials"
    elif 6500 <= sic <= 6599:
        return "Real Estate"
    elif (3600 <= sic <= 3659 or 3670 <= sic <= 3699 or
          3800 <= sic <= 3840 or 7370 <= sic <= 7379 or
          7372 <= sic <= 7374 or 3570 <= sic <= 3579):
        return "Information Technology"
    elif (4800 <= sic <= 4899 or 2700 <= sic <= 2799 or
          3660 <= sic <= 3669 or 7830 <= sic <= 7899 or
          7900 <= sic <= 7999):
        return "Communication Services"
    elif 4900 <= sic <= 4999:
        return "Utilities"
    else:
        return "Unknown"


# ── STEP 1: LOAD & LOCAL FILTERS ───────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data and applying local filters...")
print("=" * 60)

df = pd.read_parquet(INPUT_FILE)
print(f"  Loaded: {df.shape[0]:,} rows, {df['ticker'].nunique():,} unique tickers")

# Sort for rolling calc
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

# Use only the most recent ADV_WINDOW trading days per ticker
latest = df.groupby("ticker").tail(ADV_WINDOW)

# Compute per-ticker stats on the recent window
ticker_stats = latest.groupby("ticker").agg(
    last_close=("close", "last"),
    avg_volume=("volume", "mean"),
    row_count=("volume", "count")
).reset_index()

# Filter: need at least ADV_WINDOW days of data, price >= 10, ADV >= 300K
survivors = ticker_stats[
    (ticker_stats["row_count"] >= ADV_WINDOW) &
    (ticker_stats["last_close"] >= PRICE_FLOOR) &
    (ticker_stats["avg_volume"] >= ADV_FLOOR)
].copy()

print(f"  After price >= ${PRICE_FLOOR} & 22-day ADV >= {ADV_FLOOR:,}:")
print(f"    {len(survivors):,} tickers survive (from {len(ticker_stats):,})")


# ── STEP 2: POLYGON REFERENCE DATA FOR SURVIVORS ───────────────────────────
print()
print("=" * 60)
print("STEP 2: Pulling reference data from Polygon...")
print("=" * 60)

tickers_to_check = survivors["ticker"].tolist()
total = len(tickers_to_check)

ref_data = []
errors = []

for i, ticker in enumerate(tickers_to_check, 1):
    if i % 100 == 0 or i == 1:
        print(f"  [{i}/{total}] Processing {ticker}...")

    try:
        url = f"{POLYGON_BASE}/{ticker}?apiKey={POLYGON_KEY}"
        resp = requests.get(url, timeout=10)

        if resp.status_code == 200:
            r = resp.json().get("results", {})
            ref_data.append({
                "ticker":       ticker,
                "asset_type":   r.get("type", "Unknown"),       # CS, ETF, etc.
                "sic_code":     r.get("sic_code"),
                "sic_desc":     r.get("sic_description", ""),
                "poly_mcap":    r.get("market_cap"),
                "name":         r.get("name", ""),
            })
        else:
            errors.append((ticker, resp.status_code))
            ref_data.append({
                "ticker": ticker, "asset_type": "Unknown",
                "sic_code": None, "sic_desc": "", "poly_mcap": None, "name": ""
            })
    except Exception as e:
        errors.append((ticker, str(e)))
        ref_data.append({
            "ticker": ticker, "asset_type": "Unknown",
            "sic_code": None, "sic_desc": "", "poly_mcap": None, "name": ""
        })

    time.sleep(SLEEP_BETWEEN)

ref_df = pd.DataFrame(ref_data)
print(f"  Done. {len(ref_df):,} tickers queried, {len(errors)} errors.")
if errors:
    print(f"  First 10 errors: {errors[:10]}")


# ── STEP 3: MAP SECTORS, MERGE, FILTER MARKET CAP ──────────────────────────
print()
print("=" * 60)
print("STEP 3: Mapping sectors and filtering by market cap...")
print("=" * 60)

# Map SIC → GICS sector
ref_df["gics_sector"] = ref_df["sic_code"].apply(sic_to_gics_sector)

# Use Polygon market cap if available; fall back to parquet market cap
# Get most recent parquet market cap per ticker
latest_mcap = df.dropna(subset=["market_cap"]).groupby("ticker")["market_cap"].last().reset_index()
latest_mcap.columns = ["ticker", "parquet_mcap"]

ref_df = ref_df.merge(latest_mcap, on="ticker", how="left")
ref_df["final_mcap"] = ref_df["poly_mcap"].fillna(ref_df["parquet_mcap"])

# Filter by market cap >= $2B
qualified = ref_df[ref_df["final_mcap"] >= MCAP_FLOOR].copy()
print(f"  After market cap >= ${MCAP_FLOOR/1e9:.0f}B: {len(qualified):,} tickers")


# ── STEP 4: SPLIT STOCKS vs ETFs ───────────────────────────────────────────
print()
print("=" * 60)
print("STEP 4: Splitting stocks vs ETFs...")
print("=" * 60)

stocks_ref = qualified[qualified["asset_type"] == "CS"]
etf_ref    = qualified[qualified["asset_type"] == "ETF"]
other_ref  = qualified[~qualified["asset_type"].isin(["CS", "ETF"])]

print(f"  Common stocks (CS):  {len(stocks_ref):,}")
print(f"  ETFs:                {len(etf_ref):,}")
print(f"  Other types:         {len(other_ref):,}")
if len(other_ref) > 0:
    print(f"    Types: {other_ref['asset_type'].value_counts().to_dict()}")


# ── STEP 5: BUILD ENRICHED PARQUET FILES ────────────────────────────────────
print()
print("=" * 60)
print("STEP 5: Building enriched parquet files...")
print("=" * 60)

# Columns to add from reference data
enrich_cols = ["ticker", "asset_type", "gics_sector", "sic_desc", "final_mcap", "name"]

def build_output(ticker_ref, label):
    """Filter original panel to qualified tickers & enrich."""
    tickers = ticker_ref["ticker"].tolist()
    panel = df[df["ticker"].isin(tickers)].copy()

    # Drop old (empty) sector/market_cap, merge enriched
    panel = panel.drop(columns=["sector", "market_cap"], errors="ignore")
    merge_df = ticker_ref[enrich_cols].rename(columns={
        "gics_sector": "sector",
        "sic_desc":    "industry_sic",
        "final_mcap":  "market_cap",
        "name":        "company_name",
    })
    panel = panel.merge(merge_df, on="ticker", how="left")

    print(f"  {label}: {len(tickers):,} tickers, {len(panel):,} rows")
    return panel

stocks_panel = build_output(stocks_ref, "Stocks")
etf_panel    = build_output(etf_ref,    "ETFs")


# ── STEP 6: SAVE ───────────────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 6: Saving output files...")
print("=" * 60)

stocks_path = os.path.join(OUTPUT_DIR, "stocks_universe.parquet")
etf_path    = os.path.join(OUTPUT_DIR, "etf_universe.parquet")

stocks_panel.to_parquet(stocks_path, index=False)
etf_panel.to_parquet(etf_path, index=False)

print(f"  Saved: {stocks_path}")
print(f"  Saved: {etf_path}")


# ── SUMMARY ─────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Input:           {df['ticker'].nunique():,} tickers")
print(f"  After price/ADV: {len(survivors):,} tickers")
print(f"  After mkt cap:   {len(qualified):,} tickers")
print(f"  Final stocks:    {len(stocks_ref):,} tickers  →  {stocks_path}")
print(f"  Final ETFs:      {len(etf_ref):,} tickers  →  {etf_path}")
print()
print("  Sector breakdown (stocks):")
if len(stocks_ref) > 0:
    for sec, cnt in stocks_ref["gics_sector"].value_counts().items():
        print(f"    {sec:30s} {cnt:,}")
print()
print("Done.")