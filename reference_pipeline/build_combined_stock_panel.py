import pandas as pd
from pathlib import Path

# =========================================
# EXACT PATH FROM YOUR SCREENSHOT
# =========================================
PRICES_DIR = Path(r"Z:\jvand\QUINTIC_V3\data\raw\prices")

OUT_DIR = Path(r"Z:\jvand\QUINTIC_V3\data\stage")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "all_stocks_daily_panel.parquet"

print("Scanning the main prices folder for all .parquet files...")

files = list(PRICES_DIR.glob("*.parquet"))
print(f"Found {len(files):,} parquet files to combine...")

if len(files) == 0:
    print("ERROR: No .parquet files found in the folder.")
    print("Path being searched:", PRICES_DIR)
    print("Please make sure the files are in Z:\\jvand\\QUINTIC_V3\\data\\raw\\prices")
    exit()

# Combine all files
dfs = []
for i, file in enumerate(files, 1):
    try:
        df = pd.read_parquet(file)
        dfs.append(df)
        if i % 500 == 0:
            print(f"Processed {i:,}/{len(files):,} files...")
    except Exception as e:
        print(f"Error reading {file.name}: {e}")

print("\nMerging all data into ONE large panel...")
combined = pd.concat(dfs, ignore_index=True)

# Clean and sort
combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
combined = combined.sort_values(["ticker", "date"]).drop_duplicates(subset=["ticker", "date"], keep="last").reset_index(drop=True)

print(f"\nFinal combined panel created!")
print(f"Total rows: {len(combined):,}")
print(f"Total unique tickers: {combined['ticker'].nunique():,}")
print(f"Date range: {combined['date'].min().date()} to {combined['date'].max().date()}")

# Save the single large file
combined.to_parquet(OUT_FILE, index=False)
print(f"\nSAVED SUCCESSFULLY: {OUT_FILE}")

# Also save a CSV version for easy viewing
combined.to_csv(OUT_FILE.with_suffix(".csv"), index=False)
print("Also saved CSV version for easy viewing.")