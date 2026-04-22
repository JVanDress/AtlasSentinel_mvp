import os
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import requests

DATA_ROOT = Path(r"Z:\jvand\QUINTIC_V3\data")

INPUT_FILE = DATA_ROOT / "stage" / "all_stocks_daily_panel.parquet"
OUT_FILE   = DATA_ROOT / "stage" / "all_stocks_daily_panel_enriched.parquet"

load_dotenv(".env.txt")
API_KEY = os.getenv("POLYGON_API_KEY")
if not API_KEY:
    raise RuntimeError("POLYGON_API_KEY not found in .env.txt file")

print("Loading panel...")
df = pd.read_parquet(INPUT_FILE)
print(f"Loaded {len(df):,} rows | {df['ticker'].nunique():,} unique tickers")

# Debug: Check what Polygon actually returns for first 10 tickers
print("\n=== DEBUG: Checking Polygon response for first 10 tickers ===")
tickers_to_debug = sorted(df['ticker'].unique()[:10])

for ticker in tickers_to_debug:
    try:
        url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
        resp = requests.get(url, params={"apiKey": API_KEY}, timeout=20)
        if resp.status_code == 200:
            data = resp.json().get("results", {})
            print(f"\n{ticker}:")
            print("  sector          :", data.get("sector"))
            print("  gics_sector     :", data.get("gics_sector"))
            print("  industry        :", data.get("industry"))
            print("  gics_industry   :", data.get("gics_industry"))
            print("  sic_description :", data.get("sic_description"))
            print("  market_cap      :", data.get("market_cap"))
        else:
            print(f"\n{ticker}: HTTP {resp.status_code}")
    except Exception as e:
        print(f"\n{ticker}: Error {e}")
    time.sleep(0.3)

# Full enrichment
print("\nRunning full enrichment on all tickers...")
enriched = []

for i, ticker in enumerate(df['ticker'].unique(), 1):
    try:
        url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
        resp = requests.get(url, params={"apiKey": API_KEY}, timeout=20)
        
        if resp.status_code == 200:
            data = resp.json().get("results", {})
            enriched.append({
                "ticker": ticker,
                "sector": data.get("sector") or data.get("gics_sector"),
                "industry": data.get("industry") or data.get("gics_industry"),
                "sub_industry": data.get("sic_description"),
                "market_cap": data.get("market_cap"),
            })
        else:
            enriched.append({"ticker": ticker, "sector": None, "industry": None, "sub_industry": None, "market_cap": None})

        if i % 200 == 0:
            print(f"Processed {i:,} tickers...")

        time.sleep(0.12)
    except Exception as e:
        print(f"Error on {ticker}: {e}")
        enriched.append({"ticker": ticker, "sector": None, "industry": None, "sub_industry": None, "market_cap": None})

ref_df = pd.DataFrame(enriched)
df_enriched = df.merge(ref_df, on="ticker", how="left")

df_enriched.to_parquet(OUT_FILE, index=False)
print(f"\nSaved enriched file to: {OUT_FILE}")

summary = df_enriched.groupby("ticker").first()[["sector", "industry", "sub_industry", "market_cap"]].reset_index()
summary.to_csv(OUT_FILE.with_name("enriched_ticker_reference.csv"), index=False)
print("Saved ticker-level summary CSV as well.")