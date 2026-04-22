"""Check all parquet files in Z:\jvand\QUINTIC_V3\data\stage for corruption."""
import sys
from pathlib import Path

import pandas as pd

data_dir = Path(r"Z:\jvand\QUINTIC_V3\data\stage")
files = sorted(data_dir.glob("*.parquet"))

print(f"Found {len(files)} parquet files in {data_dir}\n")

if not files:
    print("No parquet files found.")
    sys.exit(1)

errors = 0
for f in files:
    try:
        df = pd.read_parquet(f)
        print(f"OK    {f.name}    rows={len(df):,}    cols={len(df.columns)}")
    except Exception as e:
        print(f"CORRUPT    {f.name}    {e}")
        errors += 1

print(f"\nTotal files: {len(files)}")
print(f"Errors: {errors}")