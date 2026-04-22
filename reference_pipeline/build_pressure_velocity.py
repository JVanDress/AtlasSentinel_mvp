"""
Quintic Labs — Daily Pressure Velocity Overlay
=================================================
Computes buying/selling pressure, momentum persistence, volume acceleration,
and directional velocity features from daily OHLCV data.

This is the daily-frequency version. When ARLO's streaming layer is live,
tick-level pressure velocity will replace this with intraday precision.

Input:  stocks_universe_merged.parquet (or stocks_universe.parquet)
Output: cleanroom_pressure_velocity_daily.parquet

Features produced (per ticker per date):
  - clv                        : Close Location Value (close-low)/(high-low)
  - buying_pressure             : Volume × CLV (Chaikin-style)
  - selling_pressure            : Volume × (1 - CLV)
  - pressure_ratio              : buying / (buying + selling)
  - pressure_ratio_5d           : 5-day rolling mean of pressure_ratio
  - pressure_ratio_20d          : 20-day rolling mean of pressure_ratio
  - volume_force                : Signed volume × log return
  - volume_force_5d             : 5-day rolling sum of volume_force
  - volume_force_20d            : 20-day rolling sum of volume_force
  - price_efficiency            : |close - open| / (high - low)
  - gap_pct                     : (open - prev_close) / prev_close
  - gap_direction               : sign of gap
  - consec_up_days              : Consecutive positive close-to-close days
  - consec_down_days            : Consecutive negative close-to-close days
  - run_length                  : abs(consec_up - consec_down) signed
  - velocity_5d                 : 5-day price velocity (regression slope)
  - velocity_20d                : 20-day price velocity (regression slope)
  - volume_accel                : volume / 20-day avg volume
  - volume_accel_5d_avg         : 5-day rolling mean of volume_accel
  - dollar_volume_accel         : dollar volume / 20-day avg dollar volume
  - ad_line                     : Cumulative Accumulation/Distribution
  - ad_line_slope_5d            : 5-day slope of AD line
  - ad_line_slope_20d           : 20-day slope of AD line
  - pressure_velocity_composite : Weighted combination of key signals

Usage:
    python build_pressure_velocity_daily.py
    python build_pressure_velocity_daily.py --input "Z:\\path\\to\\universe.parquet" --output "Z:\\path\\to\\output.parquet"
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from quintic_paths import data_dir, project_root
except ImportError:
    pass

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
DATA_ROOT = os.getenv("DATA_ROOT", r"Z:\jvand\QUINTIC_V3\data\stage")

EPS = 1e-9


def parse_args():
    p = argparse.ArgumentParser(description="Quintic Labs — Daily Pressure Velocity Overlay")
    p.add_argument("--input", default=None, help="Input parquet (default: DATA_ROOT/stocks_universe_merged.parquet)")
    p.add_argument("--output", default=None, help="Output parquet (default: DATA_ROOT/cleanroom_pressure_velocity_daily.parquet)")
    p.add_argument("--status-output", default=None)
    p.add_argument("--max-tickers", type=int, default=0, help="Limit tickers for testing (0=all)")
    return p.parse_args()


def compute_consecutive_runs(returns: pd.Series) -> tuple:
    """Compute consecutive up/down day counts."""
    signs = np.sign(returns).values
    n = len(signs)
    consec_up = np.zeros(n, dtype=np.float64)
    consec_down = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if np.isnan(signs[i]):
            consec_up[i] = 0
            consec_down[i] = 0
        elif signs[i] > 0:
            consec_up[i] = (consec_up[i - 1] + 1) if i > 0 else 1
            consec_down[i] = 0
        elif signs[i] < 0:
            consec_up[i] = 0
            consec_down[i] = (consec_down[i - 1] + 1) if i > 0 else 1
        else:
            consec_up[i] = 0
            consec_down[i] = 0

    return consec_up, consec_down


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling OLS slope (linear regression) over a window."""
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(y):
        if len(y) < window or np.isnan(y).any():
            return np.nan
        y_mean = y.mean()
        return ((x - x_mean) * (y - y_mean)).sum() / (x_var + EPS)

    return series.rolling(window, min_periods=window).apply(_slope, raw=True)


def compute_ticker_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all pressure velocity features for a single ticker."""
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)

    if n < 5:
        return pd.DataFrame()

    close = df["close"].astype(np.float64)
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    opn = df["open"].astype(np.float64)
    volume = df["volume"].astype(np.float64)
    vwap = df["vwap"].astype(np.float64) if "vwap" in df.columns else close.copy()

    # Replace zero/negative prices with NaN
    close = close.where(close > 0, np.nan)
    high = high.where(high > 0, np.nan)
    low = low.where(low > 0, np.nan)
    opn = opn.where(opn > 0, np.nan)

    hl_range = (high - low).clip(lower=EPS)
    log_return = np.log(close / close.shift(1))
    dollar_volume = close * volume

    # ── CLV and Pressure ─────────────────────────────────────────────────
    clv = ((close - low) / hl_range).clip(-1, 1)
    buying_pressure = volume * clv
    selling_pressure = volume * (1.0 - clv)
    bp_sp_sum = (buying_pressure + selling_pressure).clip(lower=EPS)
    pressure_ratio = buying_pressure / bp_sp_sum

    # ── Volume Force (directional volume) ────────────────────────────────
    volume_force = log_return * volume

    # ── Price Efficiency ─────────────────────────────────────────────────
    price_efficiency = (np.abs(close - opn) / hl_range).clip(0, 1)

    # ── Gap ───────────────────────────────────────────────────────────────
    prev_close = close.shift(1)
    gap_pct = ((opn - prev_close) / prev_close.clip(lower=EPS))
    gap_direction = np.sign(gap_pct)

    # ── Consecutive Runs ─────────────────────────────────────────────────
    consec_up, consec_down = compute_consecutive_runs(log_return)
    run_length = consec_up - consec_down

    # ── Volume Acceleration ──────────────────────────────────────────────
    vol_ma20 = volume.rolling(20, min_periods=10).mean()
    volume_accel = volume / vol_ma20.clip(lower=EPS)
    dvol_ma20 = dollar_volume.rolling(20, min_periods=10).mean()
    dollar_volume_accel = dollar_volume / dvol_ma20.clip(lower=EPS)

    # ── Accumulation/Distribution Line ───────────────────────────────────
    ad_flow = clv * volume
    ad_line = ad_flow.cumsum()

    # ── Assemble ─────────────────────────────────────────────────────────
    out = pd.DataFrame({
        "ticker": df["ticker"],
        "date": df["date"],
        "clv": clv,
        "buying_pressure": buying_pressure,
        "selling_pressure": selling_pressure,
        "pressure_ratio": pressure_ratio,
        "pressure_ratio_5d": pressure_ratio.rolling(5, min_periods=3).mean(),
        "pressure_ratio_20d": pressure_ratio.rolling(20, min_periods=10).mean(),
        "volume_force": volume_force,
        "volume_force_5d": volume_force.rolling(5, min_periods=3).sum(),
        "volume_force_20d": volume_force.rolling(20, min_periods=10).sum(),
        "price_efficiency": price_efficiency,
        "gap_pct": gap_pct,
        "gap_direction": gap_direction,
        "consec_up_days": consec_up,
        "consec_down_days": consec_down,
        "run_length": run_length,
        "velocity_5d": rolling_slope(close, 5),
        "velocity_20d": rolling_slope(close, 20),
        "volume_accel": volume_accel,
        "volume_accel_5d_avg": volume_accel.rolling(5, min_periods=3).mean(),
        "dollar_volume_accel": dollar_volume_accel,
        "ad_line": ad_line,
        "ad_line_slope_5d": rolling_slope(ad_line, 5),
        "ad_line_slope_20d": rolling_slope(ad_line, 20),
    })

    # ── Composite Score ──────────────────────────────────────────────────
    # Normalized blend of key signals
    pr_z = (out["pressure_ratio_20d"] - 0.5) * 2.0  # center at 0
    vf_z = np.tanh(out["volume_force_20d"] / (out["volume_force_20d"].rolling(60, min_periods=20).std().clip(lower=EPS)))
    rl_z = np.tanh(out["run_length"] / 5.0)
    va_z = np.tanh((out["volume_accel"] - 1.0) / 2.0)
    ad_z = np.tanh(out["ad_line_slope_20d"] / (out["ad_line_slope_20d"].rolling(60, min_periods=20).std().clip(lower=EPS)))

    out["pressure_velocity_composite"] = (
        0.30 * pr_z.fillna(0)
        + 0.25 * vf_z.fillna(0)
        + 0.20 * rl_z.fillna(0)
        + 0.10 * va_z.fillna(0)
        + 0.15 * ad_z.fillna(0)
    ).clip(-1, 1)

    return out


def main():
    args = parse_args()
    input_path = args.input or os.path.join(DATA_ROOT, "stocks_universe_merged.parquet")
    output_path = args.output or os.path.join(DATA_ROOT, "cleanroom_pressure_velocity_daily.parquet")
    status_path = args.status_output or output_path.replace(".parquet", "_status.json")

    if not os.path.exists(input_path):
        print(f"FATAL: Input not found: {input_path}")
        sys.exit(1)

    print(f"Loading {input_path} ...")
    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["ticker", "date", "close"]).copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    tickers = sorted(df["ticker"].unique())
    if args.max_tickers > 0:
        tickers = tickers[:args.max_tickers]
        df = df[df["ticker"].isin(tickers)].copy()

    print(f"Universe: {len(tickers):,} tickers | {len(df):,} rows")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # ── Process each ticker ──────────────────────────────────────────────
    results = []
    t0 = time.time()

    for tk in tqdm(tickers, desc="  Computing pressure velocity", unit="ticker"):
        tk_df = df[df["ticker"] == tk].copy()
        features = compute_ticker_features(tk_df)
        if len(features) > 0:
            results.append(features)

    if not results:
        print("FATAL: No features produced")
        sys.exit(1)

    out = pd.concat(results, ignore_index=True)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    # ── Ensure clean dtypes ──────────────────────────────────────────────
    for col in out.select_dtypes(include=["float64", "float32"]).columns:
        out[col] = out[col].astype("float64")

    # ── Verify no warnings ───────────────────────────────────────────────
    null_counts = out.drop(columns=["ticker", "date"]).isna().sum()
    feature_cols = [c for c in out.columns if c not in ("ticker", "date")]
    print(f"\nOutput: {len(out):,} rows | {len(feature_cols)} features")
    print(f"Date range: {out['date'].min()} to {out['date'].max()}")
    print(f"Tickers: {out['ticker'].nunique():,}")

    # ── Save ─────────────────────────────────────────────────────────────
    out.to_parquet(output_path, index=False)
    file_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved: {output_path} ({file_mb:.1f} MB)")

    # ── Status ───────────────────────────────────────────────────────────
    status = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "input": input_path,
        "output": output_path,
        "rows": int(len(out)),
        "tickers": int(out["ticker"].nunique()),
        "features": feature_cols,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    Path(status_path).write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"Status: {status_path}")
    print(f"Elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
