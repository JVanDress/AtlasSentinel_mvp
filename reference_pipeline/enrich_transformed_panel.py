"""
Quintic Labs — Enrich Transformed Panel
=========================================
Adds all meaningful derived features to the transformed panel
before OU processing. Every feature here was selected because
it mean-reverts, and therefore benefits from OU stretch measurement.

Run AFTER build_transformed_panel.py, BEFORE build_ou_features.py.

Input:  cleanroom_transformed_panel_merged.parquet
Output: cleanroom_transformed_panel_enriched.parquet (new file, original untouched)

New features added:
  Price structure:
    - log_high_low_range        : ln(high/low), intraday vol proxy, mean-reverts
    - log_vwap_close            : ln(vwap/close), execution quality deviation
    - overnight_return          : ln(open/prev_close), gap signal
    - intraday_return           : ln(close/open), session direction
    - upper_shadow_pct          : (high - max(open,close)) / (high-low), rejection
    - lower_shadow_pct          : (min(open,close) - low) / (high-low), support
    - body_pct                  : abs(close-open) / (high-low), conviction/indecision
    - high_close_pct            : (high-close) / (high-low), where close sits in range
    - close_to_vwap_z20         : z-score of close/vwap ratio over 20d

  Volume structure:
    - log_dollar_volume         : ln(close * volume), liquidity scale
    - dollar_volume_z20         : z-score of dollar volume over 20d
    - relative_volume_20d       : volume / 20d SMA, mean-reverts to 1.0
    - volume_return_corr_20d    : rolling 20d correlation(volume, |return|)
    - txn_per_dollar_z20        : transactions per dollar volume, z-scored

  Momentum / mean-reversion:
    - ret_5d                    : 5-day cumulative return
    - ret_10d                   : 10-day cumulative return
    - ret_20d                   : 20-day cumulative return
    - ret_60d                   : 60-day cumulative return
    - rsi_14                    : RSI(14), classic mean-reverter (oscillates 0-100)
    - rsi_5                     : RSI(5), fast mean-reverter
    - price_vs_sma20            : close / SMA(20) - 1, mean-reverts
    - price_vs_sma50            : close / SMA(50) - 1, mean-reverts
    - price_vs_ema10            : close / EMA(10) - 1, mean-reverts
    - sma20_slope_5             : 5-day change in SMA(20), trend velocity
    - bollinger_pct             : (close - BB_lower) / (BB_upper - BB_lower)
    - atr_14_pct                : ATR(14) / close, normalized volatility

  Cross-sectional (sector ranks of new features):
    - rsi_14_sector_rank
    - relative_volume_20d_sector_rank
    - overnight_return_sector_rank
    - intraday_return_sector_rank
    - dollar_volume_z20_sector_rank
    - price_vs_sma20_sector_rank

Usage:
    python enrich_transformed_panel.py
    python enrich_transformed_panel.py --input "Z:\\...\\transformed.parquet" --output "Z:\\...\\enriched.parquet"
"""

import argparse, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
DATA_ROOT = os.getenv("DATA_ROOT", r"Z:\jvand\QUINTIC_V3\data\stage")
EPS = 1e-9


def parse_args():
    p = argparse.ArgumentParser(description="Quintic Labs — Enrich Transformed Panel")
    p.add_argument("--input", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--status-output", default=None)
    p.add_argument("--max-tickers", type=int, default=0)
    return p.parse_args()


def compute_rsi(close, period):
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.clip(lower=EPS)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_atr(high, low, close, period=14):
    """Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def enrich_ticker(df):
    """Add all derived features for a single ticker."""
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    if n < 20:
        return df

    close = df["close"].astype(np.float64).where(lambda x: x > 0, np.nan)
    high = df["high"].astype(np.float64).where(lambda x: x > 0, np.nan)
    low = df["low"].astype(np.float64).where(lambda x: x > 0, np.nan)
    opn = df["open"].astype(np.float64).where(lambda x: x > 0, np.nan)
    volume = df["volume"].astype(np.float64)
    vwap = df["vwap"].astype(np.float64) if "vwap" in df.columns else close.copy()
    transactions = df["transactions"].astype(np.float64) if "transactions" in df.columns else pd.Series(np.nan, index=df.index)

    hl_range = (high - low).clip(lower=EPS)
    prev_close = close.shift(1)
    dollar_volume = close * volume

    # ── Price Structure ──────────────────────────────────────────────────
    df["log_high_low_range"] = np.log(high / low.clip(lower=EPS))
    df["log_vwap_close"] = np.log(vwap / close.clip(lower=EPS))
    df["overnight_return"] = np.log(opn / prev_close.clip(lower=EPS))
    df["intraday_return"] = np.log(close / opn.clip(lower=EPS))

    max_oc = pd.concat([opn, close], axis=1).max(axis=1)
    min_oc = pd.concat([opn, close], axis=1).min(axis=1)
    df["upper_shadow_pct"] = ((high - max_oc) / hl_range).clip(0, 1)
    df["lower_shadow_pct"] = ((min_oc - low) / hl_range).clip(0, 1)
    df["body_pct"] = ((close - opn).abs() / hl_range).clip(0, 1)
    df["high_close_pct"] = ((high - close) / hl_range).clip(0, 1)

    vcr = close / vwap.clip(lower=EPS)
    vcr_mean = vcr.rolling(20, min_periods=10).mean()
    vcr_std = vcr.rolling(20, min_periods=10).std().clip(lower=EPS)
    df["close_to_vwap_z20"] = (vcr - vcr_mean) / vcr_std

    # ── Volume Structure ─────────────────────────────────────────────────
    df["log_dollar_volume"] = np.log(dollar_volume.clip(lower=1))
    dv_mean = dollar_volume.rolling(20, min_periods=10).mean()
    dv_std = dollar_volume.rolling(20, min_periods=10).std().clip(lower=EPS)
    df["dollar_volume_z20"] = (dollar_volume - dv_mean) / dv_std

    vol_sma20 = volume.rolling(20, min_periods=10).mean().clip(lower=EPS)
    df["relative_volume_20d"] = volume / vol_sma20

    log_ret_abs = np.log(close / prev_close.clip(lower=EPS)).abs()
    df["volume_return_corr_20d"] = volume.rolling(20, min_periods=15).corr(log_ret_abs)

    txn_per_dv = transactions / dollar_volume.clip(lower=EPS) * 1e6
    tpd_mean = txn_per_dv.rolling(20, min_periods=10).mean()
    tpd_std = txn_per_dv.rolling(20, min_periods=10).std().clip(lower=EPS)
    df["txn_per_dollar_z20"] = (txn_per_dv - tpd_mean) / tpd_std

    # ── Momentum / Mean-Reversion ────────────────────────────────────────
    log_return = np.log(close / prev_close.clip(lower=EPS))
    df["ret_5d"] = log_return.rolling(5, min_periods=3).sum()
    df["ret_10d"] = log_return.rolling(10, min_periods=7).sum()
    df["ret_20d"] = log_return.rolling(20, min_periods=15).sum()
    df["ret_60d"] = log_return.rolling(60, min_periods=40).sum()

    df["rsi_14"] = compute_rsi(close, 14)
    df["rsi_5"] = compute_rsi(close, 5)

    sma20 = close.rolling(20, min_periods=15).mean()
    sma50 = close.rolling(50, min_periods=35).mean()
    ema10 = close.ewm(span=10, min_periods=7, adjust=False).mean()
    df["price_vs_sma20"] = close / sma20.clip(lower=EPS) - 1.0
    df["price_vs_sma50"] = close / sma50.clip(lower=EPS) - 1.0
    df["price_vs_ema10"] = close / ema10.clip(lower=EPS) - 1.0
    df["sma20_slope_5"] = (sma20 - sma20.shift(5)) / sma20.shift(5).clip(lower=EPS)

    # Bollinger %B
    bb_std = close.rolling(20, min_periods=15).std()
    bb_upper = sma20 + 2 * bb_std
    bb_lower = sma20 - 2 * bb_std
    bb_width = (bb_upper - bb_lower).clip(lower=EPS)
    df["bollinger_pct"] = ((close - bb_lower) / bb_width).clip(-0.5, 1.5)

    # ATR normalized
    atr = compute_atr(high, low, close, 14)
    df["atr_14_pct"] = atr / close.clip(lower=EPS)

    return df


def add_sector_ranks(df, feature_cols, sector_col="gics_sector"):
    """Add sector-relative percentile ranks for specified columns."""
    if sector_col not in df.columns:
        return df

    for col in tqdm(feature_cols, desc="  Sector ranks", unit="feature"):
        rank_col = f"{col}_sector_rank"
        df[rank_col] = df.groupby(["date", sector_col])[col].rank(pct=True, method="average")

    return df


def main():
    args = parse_args()
    input_path = args.input or os.path.join(DATA_ROOT, "cleanroom_transformed_panel_merged.parquet")
    output_path = args.output or os.path.join(DATA_ROOT, "cleanroom_transformed_panel_enriched.parquet")
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
    print(f"Existing columns: {df.columns.size}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # ── Enrich each ticker ───────────────────────────────────────────────
    t0 = time.time()
    results = []
    for tk in tqdm(tickers, desc="  Enriching features", unit="ticker"):
        tk_df = df[df["ticker"] == tk].copy()
        enriched = enrich_ticker(tk_df)
        results.append(enriched)

    df = pd.concat(results, ignore_index=True)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # ── Sector ranks for new mean-reverting features ─────────────────────
    rank_features = [
        "rsi_14",
        "relative_volume_20d",
        "overnight_return",
        "intraday_return",
        "dollar_volume_z20",
        "price_vs_sma20",
        "bollinger_pct",
        "atr_14_pct",
        "log_high_low_range",
        "close_to_vwap_z20",
    ]
    existing_rank_features = [f for f in rank_features if f in df.columns]
    df = add_sector_ranks(df, existing_rank_features)

    # ── Clean infinities ─────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = int(np.isinf(df[numeric_cols].values).sum())
    if inf_count > 0:
        print(f"  Replacing {inf_count:,} inf values with NaN")
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # ── Ensure dtypes ────────────────────────────────────────────────────
    for col in df.select_dtypes(include=["float64", "float32"]).columns:
        df[col] = df[col].astype("float64")

    new_cols = [c for c in df.columns if c not in pd.read_parquet(input_path, columns=[]).columns]
    print(f"\nNew features added: {len(new_cols)}")
    print(f"Total columns: {df.columns.size}")
    print(f"Total rows: {len(df):,}")

    # ── Save ─────────────────────────────────────────────────────────────
    df.to_parquet(output_path, index=False)
    file_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved: {output_path} ({file_mb:.1f} MB)")
    print(f"Original file untouched: {input_path}")

    # ── Status ───────────────────────────────────────────────────────────
    status = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "input": input_path,
        "output": output_path,
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()),
        "original_columns": int(pd.read_parquet(input_path, columns=[]).columns.size),
        "total_columns": int(df.columns.size),
        "new_features": new_cols,
        "sector_rank_features": [f"{f}_sector_rank" for f in existing_rank_features],
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    Path(status_path).write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"Elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
