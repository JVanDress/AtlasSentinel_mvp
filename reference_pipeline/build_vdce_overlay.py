"""
Quintic Labs — VDCE (Volatility-Driven Composite Exhaustion)

Measures per-stock stress before it shows in the broader market.
Tests whether individual stock stress events lead IWM stress events
by 5-10 trading days — a direct predictive signal for ARLO.

WHY ARLO NEEDS THIS:
    When individual stocks enter stress states (vol spike, drawdown
    acceleration, range expansion) BEFORE the index does, those
    stocks are leading indicators. ARLO can use the stress score
    to predict which stocks will underperform over 20-60 days,
    and the lead score to identify contagion risk.

Features produced per ticker per day:
    - vdce_stress: smoothed composite stress score
    - vdce_state: GREEN/YELLOW/ORANGE/RED
    - vdce_vol_ratio: short/long vol ratio (regime indicator)
    - vdce_vol_of_vol: volatility of volatility
    - vdce_dd_velocity: speed of drawdown deepening
    - vdce_range_z: range expansion z-score
    - vdce_vol_acceleration: volume acceleration

Per ticker (static, from lead-lag analysis):
    - vdce_leader_score: correlation of this stock's stress with
      future IWM stress events (higher = more predictive)

Usage:
    python build_vdce_overlay.py -i stocks_universe.parquet -o vdce_overlay.parquet
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quintic_paths import load_simple_dotenv, project_root

# Minimum standard deviation to consider a series non-constant.
# Below this, correlation is undefined and we skip rather than divide.
MIN_STD = 1e-12

LEAD_MIN = 5
LEAD_MAX = 10
MIN_DAYS_PER_TICKER = 200


def parse_args():
    parser = argparse.ArgumentParser(description="Quintic Labs — VDCE Stress Overlay")
    parser.add_argument("--input", "-i", required=True, help="Universe parquet with OHLCV")
    parser.add_argument("--output", "-o", default=None, help="Output parquet")
    parser.add_argument("--iwm-ticker", default="IWM", help="Index ticker for lead-lag (default IWM)")
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


# ===================================================================
# Safe math — every division is guarded at the source
# ===================================================================
def safe_divide(numerator, denominator):
    """Divide two series. Returns NaN where denominator is zero or near-zero."""
    if isinstance(denominator, pd.Series):
        denom = denominator.where(denominator.abs() > MIN_STD, other=np.nan)
    else:
        denom = denominator if abs(denominator) > MIN_STD else np.nan
    return numerator / denom


def zscore(s, win):
    """Rolling z-score. Returns NaN where std is zero (constant series)."""
    mu = s.rolling(win, min_periods=win).mean()
    sd = s.rolling(win, min_periods=win).std()
    sd = sd.where(sd.abs() > MIN_STD, other=np.nan)
    return (s - mu) / sd


def realized_vol(r, win):
    return r.rolling(win, min_periods=win).std()


def drawdown(close, win):
    peak = close.rolling(win, min_periods=win).max()
    peak = peak.where(peak.abs() > MIN_STD, other=np.nan)
    return (close / peak) - 1.0


def safe_corr(x, y):
    """
    Pearson correlation without division-by-zero.
    Returns NaN if either series has zero variance.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return np.nan

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_c = x - x_mean
    y_c = y - y_mean

    x_std = np.sqrt(np.mean(x_c ** 2))
    y_std = np.sqrt(np.mean(y_c ** 2))

    if x_std < MIN_STD or y_std < MIN_STD:
        return np.nan

    return float(np.mean(x_c * y_c) / (x_std * y_std))


# ===================================================================
# Core VDCE computation
# ===================================================================
def build_vdce_for_ticker(group):
    df = group.sort_values("date").copy()
    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    volume = df["volume"].astype("float64")

    ret_1 = close.pct_change()

    rv_5 = realized_vol(ret_1, 5)
    rv_20 = realized_vol(ret_1, 20)

    vol_ratio = safe_divide(rv_5, rv_20)
    vol_of_vol = rv_5.rolling(20, min_periods=20).std()

    dd_60 = drawdown(close, 60)
    dd_vel = dd_60.diff()

    range_pct = safe_divide(high - low, close)
    range_z = zscore(range_pct, 20)

    adv_20 = volume.rolling(20, min_periods=20).mean()
    vol_ratio2 = safe_divide(volume, adv_20)
    vol_z = zscore(vol_ratio2, 20)
    vol_vel = vol_z.diff()
    vol_acc = vol_vel.diff()

    if "vwap" in df.columns:
        vwap = df["vwap"].astype("float64")
        vwap_dev = safe_divide(close - vwap, close)
        vwap_dev_z = zscore(vwap_dev, 20)
    else:
        vwap_dev_z = pd.Series(0.0, index=df.index)

    stress_raw = (
        1.2 * zscore(vol_ratio, 20)
        + 1.0 * zscore(vol_of_vol, 20)
        + 1.0 * zscore(dd_vel, 20)
        + 1.0 * zscore(dd_60, 60)
        + 0.8 * range_z
        + 0.8 * zscore(vol_acc, 20)
        + 0.4 * vwap_dev_z
    )
    stress_raw = stress_raw.replace([np.inf, -np.inf], np.nan)
    stress = stress_raw.ewm(span=5, adjust=False).mean()

    state = pd.Series("GREEN", index=df.index)
    state = state.where(stress <= 1.0, "YELLOW")
    state = state.where(stress <= 2.0, "ORANGE")
    state = state.where(stress <= 3.0, "RED")

    state_rank = state.map({"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3})
    event_up = (state_rank.diff() > 0).astype(int)

    stress_vel_5d = stress - stress.shift(5)
    stress_accel = stress_vel_5d - stress_vel_5d.shift(5)

    df["vdce_stress"] = stress
    df["vdce_stress_raw"] = stress_raw
    df["vdce_state"] = state
    df["vdce_vol_ratio"] = vol_ratio
    df["vdce_vol_of_vol"] = vol_of_vol
    df["vdce_dd_60"] = dd_60
    df["vdce_dd_velocity"] = dd_vel
    df["vdce_range_z"] = range_z
    df["vdce_vol_acceleration"] = vol_acc
    df["vdce_event_up"] = event_up
    df["vdce_stress_vel_5d"] = stress_vel_5d
    df["vdce_stress_accel"] = stress_accel
    df["vdce_state_numeric"] = state_rank

    return df


# ===================================================================
# IWM lead-lag analysis
# ===================================================================
def lead_score(precursor, event_up, lag_lo, lag_hi):
    scores = []
    for lag in range(lag_lo, lag_hi + 1):
        x = precursor.shift(lag)
        y = event_up
        ok = x.notna() & y.notna()
        if ok.sum() < 80:
            continue
        c = safe_corr(x[ok].values, y[ok].values)
        if np.isfinite(c):
            scores.append(c)
    if not scores:
        return np.nan
    return float(np.mean(scores))


def compute_leader_scores(df, iwm_ticker):
    iwm = df[df["ticker"] == iwm_ticker].copy()
    if iwm.empty:
        print(f"  WARNING: {iwm_ticker} not found — skipping lead-lag")
        return pd.DataFrame(columns=["ticker", "vdce_leader_score"])

    iwm_v = build_vdce_for_ticker(iwm)
    iwm_events = iwm_v.set_index("date")["vdce_event_up"]

    # Self-test
    iwm_stress = iwm_v.set_index("date")["vdce_stress"]
    fut = pd.concat([iwm_events.shift(-k) for k in range(LEAD_MIN, LEAD_MAX + 1)],
                     axis=1).max(axis=1)
    aligned = pd.concat([iwm_stress, fut], axis=1).dropna()
    if len(aligned) >= 120:
        self_corr = safe_corr(aligned.iloc[:, 0].values, aligned.iloc[:, 1].values)
        if np.isfinite(self_corr):
            print(f"  IWM self-test: n={len(aligned)}, corr={self_corr:.4f}")

    results = []
    tickers = sorted(df["ticker"].unique())
    tickers = [t for t in tickers if t != iwm_ticker]

    for idx, ticker in enumerate(tickers, 1):
        group = df[df["ticker"] == ticker]
        if len(group) < MIN_DAYS_PER_TICKER:
            continue

        tv = build_vdce_for_ticker(group)
        precursor = tv.set_index("date")["vdce_stress"]
        combined = pd.concat([precursor, iwm_events], axis=1, sort=False)
        combined.columns = ["precursor_stress", "iwm_event_up"]
        combined = combined.dropna()

        if len(combined) < 200:
            continue

        score = lead_score(combined["precursor_stress"],
                          combined["iwm_event_up"], LEAD_MIN, LEAD_MAX)
        results.append({"ticker": ticker, "vdce_leader_score": score})

        if idx % 200 == 0:
            print(f"  Lead-lag: {idx}/{len(tickers)} tickers scanned")

    return pd.DataFrame(results)


# ===================================================================
# Main
# ===================================================================
def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Error: input not found: {input_path}")

    output_path = (
        Path(args.output) if args.output
        else input_path.parent / "cleanroom_vdce_overlay.parquet"
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
    df = df.dropna(subset=required).sort_values(["ticker", "date"]).reset_index(drop=True)

    tickers = df["ticker"].unique()
    if args.max_tickers > 0:
        tickers = tickers[:args.max_tickers]
        df = df[df["ticker"].isin(tickers)].copy()

    n_tickers = len(tickers)
    print(f"Processing {n_tickers:,} tickers ...")
    print(f"IWM ticker: {args.iwm_ticker}")

    # Auto-fetch IWM if not in universe
    if args.iwm_ticker not in df["ticker"].values:
        print(f"  {args.iwm_ticker} not in universe — fetching from Polygon ...")
        try:
            root = project_root()
            load_simple_dotenv(root, override=True)
            api_key = (os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY") or "").strip()
            if api_key:
                from polygon import RESTClient
                client = RESTClient(api_key=api_key)
                date_min = df["date"].min().strftime("%Y-%m-%d")
                date_max = df["date"].max().strftime("%Y-%m-%d")
                aggs = client.get_aggs(args.iwm_ticker, 1, "day", date_min, date_max)
                iwm_rows = []
                for agg in aggs or []:
                    ts = getattr(agg, "timestamp", None) or getattr(agg, "t", None)
                    iwm_rows.append({
                        "ticker": args.iwm_ticker,
                        "date": pd.to_datetime(ts, unit="ms").normalize(),
                        "open": float(getattr(agg, "open", 0) or getattr(agg, "o", 0)),
                        "high": float(getattr(agg, "high", 0) or getattr(agg, "h", 0)),
                        "low": float(getattr(agg, "low", 0) or getattr(agg, "l", 0)),
                        "close": float(getattr(agg, "close", 0) or getattr(agg, "c", 0)),
                        "volume": float(getattr(agg, "volume", 0) or getattr(agg, "v", 0)),
                    })
                if iwm_rows:
                    iwm_df = pd.DataFrame(iwm_rows)
                    df = pd.concat([df, iwm_df], ignore_index=True)
                    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
                    print(f"  Fetched {len(iwm_rows)} days of {args.iwm_ticker} data")
                else:
                    print(f"  WARNING: No {args.iwm_ticker} data returned from Polygon")
            else:
                print(f"  WARNING: No API key found — skipping {args.iwm_ticker} fetch")
        except Exception as e:
            print(f"  WARNING: Failed to fetch {args.iwm_ticker}: {e}")

    if args.dry_run:
        print("Dry run complete.")
        return

    # Build VDCE per ticker
    parts = []
    for idx, (_, group) in enumerate(df.groupby("ticker", sort=False), 1):
        parts.append(build_vdce_for_ticker(group))
        if idx % 250 == 0 or idx == n_tickers:
            print(f"  [{idx:,}/{n_tickers:,}] VDCE computed")

    out = pd.concat(parts, axis=0).sort_values(["ticker", "date"]).reset_index(drop=True)

    # Leader scores (IWM lead-lag)
    print("Computing IWM lead-lag scores ...")
    leaders = compute_leader_scores(df, args.iwm_ticker)
    if not leaders.empty:
        out = out.merge(leaders, on="ticker", how="left")
        top_leaders = leaders.dropna().sort_values("vdce_leader_score", ascending=False)
        print(f"  Top 10 leaders:")
        for _, row in top_leaders.head(10).iterrows():
            print(f"    {row['ticker']}: {row['vdce_leader_score']:.4f}")

    # Select output columns
    vdce_cols = [c for c in out.columns if c.startswith("vdce_")]
    keep_cols = ["ticker", "date"] + vdce_cols
    out = out[keep_cols]

    # Final safety check — should be unnecessary with safe_divide
    numeric = out.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(out[numeric].values).sum()
    if inf_count > 0:
        print(f"  WARNING: {inf_count} inf values found — replacing with NaN")
        out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False, engine="pyarrow")

    elapsed = time.time() - t_start

    status_path = output_path.with_suffix(".status.json")
    status = {
        "run_utc": pd.Timestamp.now("UTC").isoformat(),
        "input": str(input_path),
        "output": str(output_path),
        "tickers": n_tickers,
        "rows": len(out),
        "vdce_features": len(vdce_cols),
        "leaders_computed": len(leaders) if not leaders.empty else 0,
        "iwm_ticker": args.iwm_ticker,
        "elapsed_seconds": round(elapsed, 1),
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output: {output_path}")
    print(f"Rows: {len(out):,}")
    print(f"Tickers: {out['ticker'].nunique():,}")
    print(f"VDCE features: {len(vdce_cols)}")


if __name__ == "__main__":
    main()