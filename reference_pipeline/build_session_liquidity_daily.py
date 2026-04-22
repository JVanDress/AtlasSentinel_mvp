"""
Quintic Labs — Daily Session Liquidity Shadow
===============================================
Measures liquidity quality, depth, and regime from daily OHLCV data.
True session liquidity requires intraday venue data; this proxy uses
daily spread estimates, volume patterns, and transaction counts.

Input:  stocks_universe_merged.parquet
Output: cleanroom_session_liquidity_daily.parquet

Features produced (per ticker per date):
  - spread_proxy               : (high - low) / vwap — daily spread estimate
  - spread_proxy_5d            : 5-day rolling mean
  - spread_proxy_20d           : 20-day rolling mean
  - spread_narrowing           : Spread contracting vs 20d average (liquidity improving)
  - liquidity_score            : Composite daily liquidity (volume, spread, txn count)
  - liquidity_score_5d         : 5-day rolling mean
  - liquidity_score_20d        : 20-day rolling mean
  - liquidity_trend            : 5d vs 20d liquidity (improving/deteriorating)
  - dollar_depth               : Dollar volume / spread_proxy — how much $ moves price
  - dollar_depth_z20           : z-score over 20d
  - volume_stability           : Coefficient of variation of volume over 20d
  - txn_density                : transactions / (high - low) — trade density per price unit
  - txn_density_z20            : z-score over 20d
  - illiquidity_amihud         : Amihud illiquidity (|return| / dollar_volume)
  - illiquidity_amihud_5d      : 5-day rolling mean
  - illiquidity_amihud_20d     : 20-day rolling mean
  - liquidity_shock            : Sudden illiquidity spike (z-score > 2)
  - dryup_score                : Volume drying up relative to recent history
  - dryup_streak               : Consecutive days of below-average volume
  - session_liquidity_composite: Weighted blend of liquidity signals

Usage:
    python build_session_liquidity_daily.py
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
    p = argparse.ArgumentParser(description="Quintic Labs — Daily Session Liquidity Shadow")
    p.add_argument("--input", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--status-output", default=None)
    p.add_argument("--max-tickers", type=int, default=0)
    return p.parse_args()


def compute_dryup_streak(volume, avg_volume):
    """Count consecutive days where volume < average."""
    below = (volume < avg_volume).values
    n = len(below)
    streak = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if np.isnan(below[i]):
            streak[i] = 0
        elif below[i]:
            streak[i] = (streak[i - 1] + 1) if i > 0 else 1
        else:
            streak[i] = 0
    return streak


def compute_ticker_features(df):
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < 10:
        return pd.DataFrame()

    close = df["close"].astype(np.float64).where(lambda x: x > 0, np.nan)
    high = df["high"].astype(np.float64).where(lambda x: x > 0, np.nan)
    low = df["low"].astype(np.float64).where(lambda x: x > 0, np.nan)
    volume = df["volume"].astype(np.float64)
    vwap = df["vwap"].astype(np.float64) if "vwap" in df.columns else close.copy()
    transactions = df["transactions"].astype(np.float64) if "transactions" in df.columns else pd.Series(np.nan, index=df.index)

    hl_range = (high - low).clip(lower=EPS)
    log_return = np.log(close / close.shift(1))
    abs_return = log_return.abs()
    dollar_volume = close * volume

    # ── Spread Proxy ─────────────────────────────────────────────────────
    spread_proxy = hl_range / vwap.clip(lower=EPS)
    sp_20d = spread_proxy.rolling(20, min_periods=10).mean()
    spread_narrowing = (sp_20d - spread_proxy) / sp_20d.clip(lower=EPS)

    # ── Dollar Depth ─────────────────────────────────────────────────────
    dollar_depth = dollar_volume / spread_proxy.clip(lower=EPS)
    dd_mean = dollar_depth.rolling(20, min_periods=10).mean()
    dd_std = dollar_depth.rolling(20, min_periods=10).std().clip(lower=EPS)
    dollar_depth_z = (dollar_depth - dd_mean) / dd_std

    # ── Volume Stability ─────────────────────────────────────────────────
    vol_mean20 = volume.rolling(20, min_periods=10).mean().clip(lower=EPS)
    vol_std20 = volume.rolling(20, min_periods=10).std()
    vol_stability = vol_std20 / vol_mean20  # CV — lower = more stable

    # ── Transaction Density ──────────────────────────────────────────────
    txn_density = transactions / hl_range
    td_mean = txn_density.rolling(20, min_periods=10).mean()
    td_std = txn_density.rolling(20, min_periods=10).std().clip(lower=EPS)
    txn_density_z = (txn_density - td_mean) / td_std

    # ── Amihud Illiquidity ───────────────────────────────────────────────
    amihud = abs_return / dollar_volume.clip(lower=EPS) * 1e8
    amihud_5d = amihud.rolling(5, min_periods=3).mean()
    amihud_20d = amihud.rolling(20, min_periods=10).mean()

    # ── Liquidity Shock ──────────────────────────────────────────────────
    ami_std = amihud.rolling(20, min_periods=10).std().clip(lower=EPS)
    ami_z = (amihud - amihud_20d) / ami_std
    liq_shock = (ami_z > 2.0).astype(np.float64)

    # ── Dryup Score ──────────────────────────────────────────────────────
    dryup_score = 1.0 - (volume / vol_mean20).clip(0, 3)
    dryup_streak = compute_dryup_streak(volume, vol_mean20)

    # ── Liquidity Score ──────────────────────────────────────────────────
    # Higher = more liquid. Combine: high volume, low spread, high txn count
    vol_rank = (volume.rank(pct=True) * 2 - 1)  # -1 to 1
    spread_rank = (1 - spread_proxy.rank(pct=True)) * 2 - 1  # inverted
    txn_rank = transactions.rank(pct=True) * 2 - 1 if transactions.notna().any() else pd.Series(0, index=df.index)
    liquidity_score = (0.40 * vol_rank + 0.35 * spread_rank + 0.25 * txn_rank).clip(-1, 1)
    liq_5d = liquidity_score.rolling(5, min_periods=3).mean()
    liq_20d = liquidity_score.rolling(20, min_periods=10).mean()
    liq_trend = liq_5d - liq_20d

    out = pd.DataFrame({
        "ticker": df["ticker"],
        "date": df["date"],
        "spread_proxy": spread_proxy,
        "spread_proxy_5d": spread_proxy.rolling(5, min_periods=3).mean(),
        "spread_proxy_20d": sp_20d,
        "spread_narrowing": spread_narrowing,
        "liquidity_score": liquidity_score,
        "liquidity_score_5d": liq_5d,
        "liquidity_score_20d": liq_20d,
        "liquidity_trend": liq_trend,
        "dollar_depth": dollar_depth,
        "dollar_depth_z20": dollar_depth_z,
        "volume_stability": vol_stability,
        "txn_density": txn_density,
        "txn_density_z20": txn_density_z,
        "illiquidity_amihud": amihud,
        "illiquidity_amihud_5d": amihud_5d,
        "illiquidity_amihud_20d": amihud_20d,
        "liquidity_shock": liq_shock,
        "dryup_score": dryup_score,
        "dryup_streak": dryup_streak,
    })

    # ── Composite ────────────────────────────────────────────────────────
    ls_z = out["liquidity_score_20d"].fillna(0)
    lt_z = np.tanh(out["liquidity_trend"].fillna(0) * 5)
    am_z = -np.tanh(out["illiquidity_amihud_20d"].fillna(0) * 100)  # inverted: high illiq = bad
    dd_z_c = np.tanh(out["dollar_depth_z20"].fillna(0) / 2)
    du_z = -np.tanh(out["dryup_score"].fillna(0))  # inverted: dryup = bad

    out["session_liquidity_composite"] = (
        0.30 * ls_z + 0.25 * lt_z + 0.20 * am_z + 0.15 * dd_z_c + 0.10 * du_z
    ).clip(-1, 1)

    return out


def main():
    args = parse_args()
    input_path = args.input or os.path.join(DATA_ROOT, "stocks_universe_merged.parquet")
    output_path = args.output or os.path.join(DATA_ROOT, "cleanroom_session_liquidity_daily.parquet")
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
    for tk in tqdm(tickers, desc="  Computing session liquidity", unit="ticker"):
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