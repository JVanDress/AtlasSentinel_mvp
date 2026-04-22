import pandas as pd
from pathlib import Path

# Exact path you gave me
file_path = Path(r"C:\The_Framework\QUINTIC\data\panel_with_sector_studies.parquet")

print("Loading file...")
df = pd.read_parquet(file_path)

print("\n=== FILE INFORMATION ===")
print(f"Total rows: {len(df):,}")
print(f"Unique tickers: {df['ticker'].nunique():,}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Columns: {list(df.columns)}")

print("\nFirst 3 rows:")
print(df.head(3))

print("\nLast 3 rows:")
print(df.tail(3))

print("\nDone.")