"""
Quintic Labs — Build Transformed Panel

Transforms raw OHLCV price data into the feature panel required by
OU features, sector overlay, and GMM regime studies.

Produces:
    - log_close, log_return_1d
    - garman_klass_var_1d (intraday volatility estimator)
    - volume_robust_z (volume z-score using MAD)
    - trade_size_robust_z (avg trade size z-score using MAD)
    - close_frac_diff (fractionally differenced close)
    - clv (close location value)
    - Sector ranks for all key features

Usage:
    python build_transformed_panel.py --input stocks_universe.parquet --output cleanroom_transformed_panel.parquet
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


MIN_STD = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(description="Quintic Labs — Build Transformed Panel")
    parser.add_argument("--input", "-i", required=True, help="Input OHLCV parquet")
    parser.add_argument("--output", "-o", default=None, help="Output transformed panel parquet")
    parser.add_argument("--max-tickers", type=int, default=0)
    return parser.parse_args()


def safe_divide(numerator, denominator):
    """Divide two series. Returns NaN where denominator is zero or near-zero."""
    if isinstance(denominator, pd.Series):
        denom = denominator.where(denominator.abs() > MIN_STD, other=np.nan)
    elif isinstance(denominator, np.ndarray):
        denom = np.where(np.abs(denominator) > MIN_STD, denominator, np.nan)
    else:
        if abs(denominator) <= MIN_STD:
            return np.nan
        denom = denominator
    return numerator / denom


def safe_log(series):
    """Natural log that returns NaN for zero or negative values without warnings."""
    if isinstance(series, pd.Series):
        clean = series.where(series > MIN_STD, other=np.nan)
        return np.log(clean)
    else:
        return np.log(series) if series > MIN_STD else np.nan


def robust_z(series, window=63, min_periods=40):
    """Z-score using median and MAD (robust to outliers)."""
    median = series.rolling(window, min_periods=min_periods).median()
    mad = series.rolling(window, min_periods=min_periods).apply(
        lambda x: float(np.median(np.abs(x - np.median(x)))), raw=True
    )
    # MAD to std approximation: std ~ 1.4826 * MAD
    mad_std = mad * 1.4826
    return safe_divide(series - median, mad_std)


def garman_klass_var(high, low, close, open_):
    """
    Garman-Klass volatility estimator — uses full OHLC bar information.
    GK = 0.5 * (log(H/L))^2 - (2*ln(2)-1) * (log(C/O))^2
    """
    log_hl = safe_log(safe_divide(high, low))
    log_co = safe_log(safe_divide(close, open_))
    return 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2


def close_location_value(high, low, close):
    """
    CLV: Where the close falls within the day's range [-1, 1].
    +1 = closed at high, -1 = closed at low, 0 = midpoint.
    """
    hl_range = high - low
    return safe_divide((close - low) - (high - close), hl_range)


def fractional_diff(series, d=0.4, threshold=1e-3):
    """
    Fractionally differenced series (Hosking 1981).
    d=0.4 balances stationarity with memory preservation.
    """
    weights = [1.0]
    for k in range(1, len(series)):
        w = weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
    weights = np.array(weights[::-1])

    result = np.full(len(series), np.nan)
    for i in range(len(weights) - 1, len(series)):
        window = series.iloc[i - len(weights) + 1:i + 1].values
        if np.isnan(window).any():
            continue
        result[i] = float(np.dot(weights, window))

    return pd.Series(result, index=series.index)


def transform_ticker(group):
    """Transform a single ticker's OHLCV into analysis features."""
    g = group.copy()
    close = g["close"].astype("float64")
    high = g["high"].astype("float64")
    low = g["low"].astype("float64")
    open_ = g["open"].astype("float64")
    volume = g["volume"].astype("float64")

    # Log close and returns — safe_log handles zero prices
    g["log_close"] = safe_log(close)
    g["log_return_1d"] = g["log_close"].diff()

    # Garman-Klass intraday variance — all divisions guarded
    g["garman_klass_var_1d"] = garman_klass_var(high, low, close, open_)

    # Close Location Value — guarded against zero range
    g["clv"] = close_location_value(high, low, close)

    # Volume robust z-score (63-day rolling, MAD-based)
    g["volume_robust_z"] = robust_z(volume, window=63, min_periods=40)

    # Average trade size and its z-score
    transactions = g.get("transactions")
    if transactions is not None:
        txn = pd.to_numeric(transactions, errors="coerce")
        g["avg_trade_size"] = safe_divide(volume, txn)
        g["trade_size_robust_z"] = robust_z(g["avg_trade_size"], window=63, min_periods=40)
    else:
        g["avg_trade_size"] = np.nan
        g["trade_size_robust_z"] = np.nan

    # Fractionally differenced close
    if len(g) >= 50:
        g["close_frac_diff"] = fractional_diff(close, d=0.4)
    else:
        g["close_frac_diff"] = np.nan

    return g


def add_sector_ranks(df):
    """Add cross-sectional sector ranks for key transformed features."""
    out = df.copy()

    sector_col = None
    for candidate in ("sector", "gics_sector"):
        if candidate in out.columns:
            sector_col = candidate
            break

    if sector_col is None:
        print("  No sector column — skipping sector ranks")
        return out

    if sector_col != "gics_sector":
        out["gics_sector"] = out[sector_col]

    rank_features = [
        "log_return_1d", "garman_klass_var_1d", "volume_robust_z",
        "trade_size_robust_z", "clv",
    ]

    for feature in rank_features:
        if feature not in out.columns:
            continue
        out[f"{feature}_sector_rank"] = out.groupby(["date", "gics_sector"])[feature].rank(
            method="average", pct=True
        )

    return out


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Error: input not found: {input_path}")

    output_path = (
        Path(args.output) if args.output
        else input_path.parent / "cleanroom_transformed_panel.parquet"
    )

    t_start = time.time()

    print(f"Loading {input_path} ...")
    df = pd.read_parquet(input_path)

    required = ["ticker", "date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"Missing required columns: {missing}")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).copy()
    df = df.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)

    tickers = df["ticker"].unique()
    if args.max_tickers > 0:
        tickers = tickers[:args.max_tickers]
        df = df[df["ticker"].isin(tickers)].copy()

    n_tickers = len(tickers)
    print(f"Transforming {n_tickers:,} tickers ...")

    parts = []
    for idx, (_, group) in enumerate(df.groupby("ticker", sort=False), start=1):
        parts.append(transform_ticker(group))
        if idx % 250 == 0 or idx == n_tickers:
            print(f"  [{idx:,}/{n_tickers:,}]")

    out = pd.concat(parts, axis=0).sort_values(
        ["ticker", "date"], kind="mergesort"
    ).reset_index(drop=True)

    print("Computing sector ranks ...")
    out = add_sector_ranks(out)

    # Final integrity check
    numeric = out.select_dtypes(include=[np.number]).columns
    inf_count = int(np.isinf(out[numeric].values).sum())
    if inf_count > 0:
        print(f"  Replacing {inf_count} inf values with NaN")
        out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False, engine="pyarrow")

    elapsed = time.time() - t_start
    transformed_cols = [c for c in out.columns if c not in required + [
        "sector", "industry", "gics_sector", "gics_industry",
        "vwap", "transactions", "timestamp", "otc",
        "sub_industry", "market_cap", "asset_type", "company_name",
    ]]

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output: {output_path}")
    print(f"Rows: {len(out):,}")
    print(f"Tickers: {out['ticker'].nunique():,}")
    print(f"Transformed features: {len(transformed_cols)}")


if __name__ == "__main__":
    main()
