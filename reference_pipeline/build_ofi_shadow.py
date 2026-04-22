"""
Quintic Labs — Daily OFI Shadow Features
==========================================
Approximates Order Flow Imbalance signals from daily OHLCV data.
True OFI requires tick-level bid/ask changes; this proxy uses
price action and volume patterns to estimate directional flow.

Input:  stocks_universe_merged.parquet
Output: cleanroom_ofi_shadow_daily.parquet

Features produced (per ticker per date):
  - ofi_proxy                  : Signed volume × price impact direction
  - ofi_proxy_5d               : 5-day rolling sum
  - ofi_proxy_20d              : 20-day rolling sum
  - ofi_intensity              : OFI normalized by average volume
  - ofi_persistence_5d         : Fraction of last 5 days with same OFI sign
  - ofi_persistence_20d        : Fraction of last 20 days with same OFI sign
  - buy_volume_pct             : Estimated buy volume / total volume (using CLV)
  - sell_volume_pct             : Estimated sell volume / total volume
  - net_flow_ratio             : (buy - sell) / (buy + sell) volume ratio
  - net_flow_ratio_5d          : 5-day rolling mean
  - net_flow_ratio_20d         : 20-day rolling mean
  - flow_momentum              : Change in net_flow_ratio over 5 days
  - flow_acceleration          : Change in flow_momentum over 5 days
  - price_impact_per_volume    : |return| / volume — how much price moves per unit volume
  - price_impact_z20           : z-score of price impact over 20d
  - flow_volume_divergence     : Net flow trending up while price flat (hidden accumulation)
  - ofi_composite              : Weighted blend of key OFI signals

Usage:
    python build_ofi_shadow_daily.py
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
    p = argparse.ArgumentParser(description="Quintic Labs — Daily OFI Shadow Features")
    p.add_argument("--input", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--status-output", default=None)
    p.add_argument("--max-tickers", type=int, default=0)
    return p.parse_args()


def compute_ticker_features(df):
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < 10:
        return pd.DataFrame()

    close = df["close"].astype(np.float64).where(lambda x: x > 0, np.nan)
    high = df["high"].astype(np.float64).where(lambda x: x > 0, np.nan)
    low = df["low"].astype(np.float64).where(lambda x: x > 0, np.nan)
    opn = df["open"].astype(np.float64).where(lambda x: x > 0, np.nan)
    volume = df["volume"].astype(np.float64)

    hl_range = (high - low).clip(lower=EPS)
    log_return = np.log(close / close.shift(1))
    abs_return = log_return.abs()
    clv = ((close - low) / hl_range).clip(-1, 1)

    # ── OFI Proxy ────────────────────────────────────────────────────────
    # Approximate net order flow: signed volume weighted by price position
    ofi_proxy = clv * volume * np.sign(log_return.fillna(0))

    # ── Buy/Sell Volume Split ────────────────────────────────────────────
    # CLV > 0 means close nearer to high (buying), CLV < 0 means selling
    buy_frac = ((clv + 1) / 2).clip(0, 1)
    buy_volume = volume * buy_frac
    sell_volume = volume * (1 - buy_frac)
    total_vol = (buy_volume + sell_volume).clip(lower=EPS)
    buy_pct = buy_volume / total_vol
    sell_pct = sell_volume / total_vol
    net_flow_ratio = (buy_volume - sell_volume) / total_vol

    # ── OFI Intensity ────────────────────────────────────────────────────
    vol_avg20 = volume.rolling(20, min_periods=10).mean().clip(lower=EPS)
    ofi_intensity = ofi_proxy / vol_avg20

    # ── OFI Persistence (fraction of same-sign days) ─────────────────────
    ofi_sign = np.sign(ofi_proxy)

    def rolling_sign_frac(series, window):
        signs = series.values
        n = len(signs)
        result = np.full(n, np.nan)
        for i in range(window - 1, n):
            w = signs[i - window + 1:i + 1]
            valid = w[~np.isnan(w)]
            if len(valid) == 0:
                continue
            current = signs[i]
            if np.isnan(current):
                continue
            result[i] = (valid == current).sum() / len(valid)
        return pd.Series(result, index=series.index)

    ofi_persist_5 = rolling_sign_frac(ofi_sign, 5)
    ofi_persist_20 = rolling_sign_frac(ofi_sign, 20)

    # ── Flow Momentum / Acceleration ─────────────────────────────────────
    nfr_5d = net_flow_ratio.rolling(5, min_periods=3).mean()
    nfr_20d = net_flow_ratio.rolling(20, min_periods=10).mean()
    flow_momentum = nfr_5d - nfr_5d.shift(5)
    flow_acceleration = flow_momentum - flow_momentum.shift(5)

    # ── Price Impact per Volume ──────────────────────────────────────────
    price_impact = abs_return / volume.clip(lower=EPS) * 1e6
    pi_mean = price_impact.rolling(20, min_periods=10).mean()
    pi_std = price_impact.rolling(20, min_periods=10).std().clip(lower=EPS)
    price_impact_z = (price_impact - pi_mean) / pi_std

    # ── Flow-Volume Divergence ───────────────────────────────────────────
    nfr_trend = nfr_5d - nfr_20d
    ret_5d = log_return.rolling(5, min_periods=3).sum()
    flow_vol_div = (np.tanh(nfr_trend * 5) - np.tanh(ret_5d * 10)).clip(-2, 2)

    out = pd.DataFrame({
        "ticker": df["ticker"],
        "date": df["date"],
        "ofi_proxy": ofi_proxy,
        "ofi_proxy_5d": ofi_proxy.rolling(5, min_periods=3).sum(),
        "ofi_proxy_20d": ofi_proxy.rolling(20, min_periods=10).sum(),
        "ofi_intensity": ofi_intensity,
        "ofi_persistence_5d": ofi_persist_5,
        "ofi_persistence_20d": ofi_persist_20,
        "buy_volume_pct": buy_pct,
        "sell_volume_pct": sell_pct,
        "net_flow_ratio": net_flow_ratio,
        "net_flow_ratio_5d": nfr_5d,
        "net_flow_ratio_20d": nfr_20d,
        "flow_momentum": flow_momentum,
        "flow_acceleration": flow_acceleration,
        "price_impact_per_volume": price_impact,
        "price_impact_z20": price_impact_z,
        "flow_volume_divergence": flow_vol_div,
    })

    # ── Composite ────────────────────────────────────────────────────────
    oi_z = np.tanh(out["ofi_intensity"].fillna(0))
    nf_z = out["net_flow_ratio_20d"].fillna(0).clip(-1, 1)
    fp_z = (out["ofi_persistence_20d"].fillna(0.5) - 0.5) * 2
    fm_z = np.tanh(out["flow_momentum"].fillna(0) * 5)
    fd_z = np.tanh(out["flow_volume_divergence"].fillna(0))

    out["ofi_composite"] = (
        0.25 * oi_z + 0.25 * nf_z + 0.20 * fp_z + 0.15 * fm_z + 0.15 * fd_z
    ).clip(-1, 1)

    return out


def main():
    args = parse_args()
    input_path = args.input or os.path.join(DATA_ROOT, "stocks_universe_merged.parquet")
    output_path = args.output or os.path.join(DATA_ROOT, "cleanroom_ofi_shadow_daily.parquet")
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
    for tk in tqdm(tickers, desc="  Computing OFI features", unit="ticker"):
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
