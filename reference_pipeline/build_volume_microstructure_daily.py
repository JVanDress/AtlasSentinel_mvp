"""
Quintic Labs — Daily Volume Microstructure Overlay (Dark Pool Proxy)
=====================================================================
Approximates institutional accumulation patterns and off-exchange behavior
from daily OHLCV data. When tick-level data with exchange IDs is available,
this will be replaced with true venue classification.

Input:  stocks_universe_merged.parquet
Output: cleanroom_volume_microstructure_daily.parquet

Features produced (per ticker per date):
  - vwap_deviation              : (close - vwap) / vwap — institutional execution quality
  - vwap_deviation_5d           : 5-day rolling mean
  - vwap_deviation_20d          : 20-day rolling mean
  - volume_vs_range             : volume / (high-low range) — volume intensity per unit move
  - volume_vs_range_z20         : z-score of volume_vs_range over 20d
  - stealth_accumulation        : High volume + small price change (institutional hiding)
  - stealth_score_5d            : 5-day rolling mean of stealth signal
  - stealth_score_20d           : 20-day rolling mean of stealth signal
  - txn_avg_size                : volume / transactions — average trade size proxy
  - txn_size_z20                : z-score of txn_avg_size over 20d
  - block_trade_proxy           : When txn_size >> normal, likely block/institutional
  - volume_concentration        : Fraction of 20d total volume in single day
  - smart_money_flow            : Directional volume weighted by close position in range
  - smart_money_flow_5d         : 5-day rolling sum
  - smart_money_flow_20d        : 20-day rolling sum
  - volume_divergence           : Volume trending up while |returns| trending down (accumulation)
  - absorption_ratio            : Volume absorbed per unit of price impact
  - absorption_ratio_z20        : z-score over 20d
  - microstructure_composite    : Weighted blend of key signals

Usage:
    python build_volume_microstructure_daily.py
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
    p = argparse.ArgumentParser(description="Quintic Labs — Daily Volume Microstructure Overlay")
    p.add_argument("--input", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--status-output", default=None)
    p.add_argument("--max-tickers", type=int, default=0)
    return p.parse_args()


def compute_ticker_features(df):
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    if n < 10:
        return pd.DataFrame()

    close = df["close"].astype(np.float64).where(lambda x: x > 0, np.nan)
    high = df["high"].astype(np.float64).where(lambda x: x > 0, np.nan)
    low = df["low"].astype(np.float64).where(lambda x: x > 0, np.nan)
    opn = df["open"].astype(np.float64).where(lambda x: x > 0, np.nan)
    volume = df["volume"].astype(np.float64)
    vwap = df["vwap"].astype(np.float64) if "vwap" in df.columns else close.copy()
    transactions = df["transactions"].astype(np.float64) if "transactions" in df.columns else pd.Series(np.nan, index=df.index)

    hl_range = (high - low).clip(lower=EPS)
    log_return = np.log(close / close.shift(1))
    abs_return = log_return.abs()
    dollar_volume = close * volume
    clv = ((close - low) / hl_range).clip(-1, 1)

    # ── VWAP Deviation (institutional execution quality) ─────────────────
    vwap_dev = (close - vwap) / vwap.clip(lower=EPS)

    # ── Volume vs Range (volume intensity) ───────────────────────────────
    vol_range = volume / hl_range
    vr_mean = vol_range.rolling(20, min_periods=10).mean()
    vr_std = vol_range.rolling(20, min_periods=10).std().clip(lower=EPS)
    vol_range_z = (vol_range - vr_mean) / vr_std

    # ── Stealth Accumulation (high volume, small move) ───────────────────
    vol_z = (volume - volume.rolling(20, min_periods=10).mean()) / volume.rolling(20, min_periods=10).std().clip(lower=EPS)
    ret_z = (abs_return - abs_return.rolling(20, min_periods=10).mean()) / abs_return.rolling(20, min_periods=10).std().clip(lower=EPS)
    stealth = (vol_z - ret_z).clip(-5, 5)  # High volume + low return = stealth

    # ── Transaction Size (block trade proxy) ─────────────────────────────
    txn_size = volume / transactions.clip(lower=1)
    txn_mean = txn_size.rolling(20, min_periods=10).mean()
    txn_std = txn_size.rolling(20, min_periods=10).std().clip(lower=EPS)
    txn_size_z = (txn_size - txn_mean) / txn_std
    block_proxy = (txn_size_z > 2.0).astype(np.float64)

    # ── Volume Concentration ─────────────────────────────────────────────
    vol_20sum = volume.rolling(20, min_periods=10).sum().clip(lower=EPS)
    vol_concentration = volume / vol_20sum

    # ── Smart Money Flow ─────────────────────────────────────────────────
    smf = clv * volume * np.sign(log_return.fillna(0))

    # ── Volume Divergence (accumulation signal) ──────────────────────────
    vol_trend = volume.rolling(10, min_periods=5).mean() / volume.rolling(20, min_periods=10).mean().clip(lower=EPS)
    ret_trend = abs_return.rolling(10, min_periods=5).mean() / abs_return.rolling(20, min_periods=10).mean().clip(lower=EPS)
    vol_divergence = (vol_trend - ret_trend).clip(-3, 3)

    # ── Absorption Ratio ─────────────────────────────────────────────────
    absorption = dollar_volume / (abs_return.clip(lower=EPS) * close * 1e6)
    abs_mean = absorption.rolling(20, min_periods=10).mean()
    abs_std = absorption.rolling(20, min_periods=10).std().clip(lower=EPS)
    absorption_z = (absorption - abs_mean) / abs_std

    out = pd.DataFrame({
        "ticker": df["ticker"],
        "date": df["date"],
        "vwap_deviation": vwap_dev,
        "vwap_deviation_5d": vwap_dev.rolling(5, min_periods=3).mean(),
        "vwap_deviation_20d": vwap_dev.rolling(20, min_periods=10).mean(),
        "volume_vs_range": vol_range,
        "volume_vs_range_z20": vol_range_z,
        "stealth_accumulation": stealth,
        "stealth_score_5d": stealth.rolling(5, min_periods=3).mean(),
        "stealth_score_20d": stealth.rolling(20, min_periods=10).mean(),
        "txn_avg_size": txn_size,
        "txn_size_z20": txn_size_z,
        "block_trade_proxy": block_proxy,
        "volume_concentration": vol_concentration,
        "smart_money_flow": smf,
        "smart_money_flow_5d": smf.rolling(5, min_periods=3).sum(),
        "smart_money_flow_20d": smf.rolling(20, min_periods=10).sum(),
        "volume_divergence": vol_divergence,
        "absorption_ratio": absorption,
        "absorption_ratio_z20": absorption_z,
    })

    # ── Composite ────────────────────────────────────────────────────────
    st_z = np.tanh(out["stealth_score_20d"].fillna(0) / 2.0)
    smf_z = np.tanh(out["smart_money_flow_20d"].fillna(0) / (out["smart_money_flow_20d"].rolling(60, min_periods=20).std().clip(lower=EPS)))
    vd_z = np.tanh(out["volume_divergence"].fillna(0))
    vw_z = np.tanh(out["vwap_deviation_20d"].fillna(0) * 20)
    ab_z = np.tanh(out["absorption_ratio_z20"].fillna(0) / 2.0)

    out["microstructure_composite"] = (
        0.25 * st_z + 0.25 * smf_z + 0.20 * vd_z + 0.15 * vw_z + 0.15 * ab_z
    ).clip(-1, 1)

    return out


def main():
    args = parse_args()
    input_path = args.input or os.path.join(DATA_ROOT, "stocks_universe_merged.parquet")
    output_path = args.output or os.path.join(DATA_ROOT, "cleanroom_volume_microstructure_daily.parquet")
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

    results = []
    t0 = time.time()
    for tk in tqdm(tickers, desc="  Computing microstructure", unit="ticker"):
        features = compute_ticker_features(df[df["ticker"] == tk].copy())
        if len(features) > 0:
            results.append(features)

    out = pd.concat(results, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    for col in out.select_dtypes(include=["float64", "float32"]).columns:
        out[col] = out[col].astype("float64")

    feature_cols = [c for c in out.columns if c not in ("ticker", "date")]
    print(f"\nOutput: {len(out):,} rows | {len(feature_cols)} features")

    out.to_parquet(output_path, index=False)
    file_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved: {output_path} ({file_mb:.1f} MB)")

    status = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "input": input_path, "output": output_path,
        "rows": int(len(out)), "tickers": int(out["ticker"].nunique()),
        "features": feature_cols, "elapsed_seconds": round(time.time() - t0, 1),
    }
    Path(status_path).write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"Elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
