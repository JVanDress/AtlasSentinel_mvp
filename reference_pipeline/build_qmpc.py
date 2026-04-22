"""
Quintic Labs — QMPC (Quintic Macro-Pressure Coefficient)

Measures how much of a stock's return is explained by macro factors
vs idiosyncratic behavior. High QMPC + negative return = the stock
is underperforming relative to what macro conditions would predict.
This is a forward-looking signal for 20/60/90 day mean reversion.

Definition:
    QMPC = sum_i [ Z_ticker - (beta_i * Z_macro_i) ]
    where:
    - Z_ticker = rolling z-score (window=20) of daily return
    - Z_macro_i = rolling z-score (window=20) of macro proxy return
    - beta_i = 60-day rolling correlation (ticker vs macro proxy)
    - i in {DXY proxy, Oil proxy, Yield proxy}

WHY ARLO NEEDS THIS:
    A stock dropping 10% while macro conditions are neutral has very
    different 60-day forward expectations than one dropping 10% because
    the entire market is repricing rates. QMPC separates the two.

Usage:
    python build_qmpc.py --universe stocks_universe.parquet --out qmpc_daily.parquet
    python build_qmpc.py --prices_csv prices.csv --dxy UUP --oil USO --yield_ticker IEF
    python build_qmpc.py --wide_csv macro_matrix.csv

Requirements:
    pip install pandas pyarrow numpy polygon-api-client python-dotenv
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quintic_paths import data_dir, load_simple_dotenv, project_root


DEFAULT_OUT = "qmpc_daily.parquet"
DEFAULT_MACRO_PROXIES = {"dxy": "UUP", "oil": "USO", "yield": "IEF"}
RATE_PAUSE = 0.2


def _log(msg: str) -> None:
    print(msg, flush=True)


def _die(msg: str) -> None:
    raise SystemExit(f"FATAL: {msg}")


def _ensure_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def _pct_ret(close: pd.Series) -> pd.Series:
    return close.pct_change(1)


def rolling_z(x: pd.Series, window: int) -> pd.Series:
    mu = x.rolling(window, min_periods=window).mean()
    sd = x.rolling(window, min_periods=window).std(ddof=0)
    return (x - mu) / sd.replace(0.0, np.nan)


def require_polygon_key() -> str:
    root = project_root()
    load_simple_dotenv(root, override=True)
    key = (os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY") or "").strip()
    if not key:
        _die("POLYGON_API_KEY or MASSIVE_API_KEY not found in environment or .env")
    return key


def fetch_macro_proxy(ticker: str, start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    """Fetch daily closes for a macro proxy ETF from Polygon."""
    from polygon import RESTClient

    client = RESTClient(api_key=api_key)
    aggs = client.get_aggs(ticker, 1, "day", start_date, end_date)
    rows = []
    for agg in aggs or []:
        ts = getattr(agg, "timestamp", None) or getattr(agg, "t", None)
        close = getattr(agg, "close", None) or getattr(agg, "c", None)
        if ts is not None and close is not None:
            rows.append({"date": pd.to_datetime(ts, unit="ms").normalize(), "close": float(close)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ticker"] = ticker
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return df


def compute_qmpc(
    equity_df: pd.DataFrame,
    macro_dfs: dict,
    z_window: int,
    beta_window: int,
    threshold: float,
) -> pd.DataFrame:
    """
    Core QMPC computation. No lookahead — all rolling windows are causal.

    Args:
        equity_df: DataFrame with ticker, date, close columns
        macro_dfs: dict of {label: DataFrame} for each macro proxy
        z_window: rolling z-score window
        beta_window: rolling correlation window
        threshold: QMPC threshold for positive_macro_debt flag
    """
    df = equity_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", "ticker", "close"]).sort_values(["ticker", "date"])
    df["ret_1"] = df.groupby("ticker")["close"].transform(_pct_ret)
    df["z_ticker"] = df.groupby("ticker")["ret_1"].transform(lambda s: rolling_z(s, z_window))

    # Build macro returns and z-scores per date
    macro_labels = []
    for label, macro_df in macro_dfs.items():
        m = macro_df[["date", "close"]].copy()
        m["date"] = pd.to_datetime(m["date"], errors="coerce").dt.normalize()
        m = m.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        m[f"{label}_ret_1"] = _pct_ret(m["close"])
        m[f"{label}_z"] = rolling_z(m[f"{label}_ret_1"], z_window)
        m = m[["date", f"{label}_ret_1", f"{label}_z"]]
        df = df.merge(m, on="date", how="left")
        macro_labels.append(label)

    # Rolling correlations (betas) per ticker — causal, no lookahead
    for label in macro_labels:
        def _roll_corr(g, lbl=label):
            return g["ret_1"].rolling(beta_window, min_periods=beta_window).corr(g[f"{lbl}_ret_1"])

        df[f"beta_{label}"] = df.groupby("ticker", group_keys=False).apply(_roll_corr)

    # QMPC = sum over i of (Z_ticker - beta_i * Z_macro_i)
    qmpc = pd.Series(0.0, index=df.index)
    for label in macro_labels:
        qmpc += df["z_ticker"] - df[f"beta_{label}"] * df[f"{label}_z"]
    df["qmpc"] = qmpc

    # Flag: positive macro debt = QMPC high but stock return negative
    # This means macro conditions should have supported the stock, but it dropped anyway
    df["positive_macro_debt"] = (df["qmpc"] > threshold) & (df["ret_1"] <= 0)

    # Select output columns
    out_cols = ["date", "ticker", "qmpc", "z_ticker"]
    out_cols += [f"beta_{label}" for label in macro_labels]
    out_cols += ["ret_1", "positive_macro_debt"]
    out = df[out_cols].dropna(subset=["qmpc"]).sort_values(["date", "ticker"])

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Quintic Labs — QMPC")
    ap.add_argument("--universe", type=str, default="",
                    help="Universe parquet (auto-fetches macro proxies from Polygon)")
    ap.add_argument("--wide_csv", type=str, default="",
                    help="Wide CSV with macro columns embedded")
    ap.add_argument("--prices_csv", type=str, default="",
                    help="Prices CSV with macro tickers as rows")
    ap.add_argument("--dxy", type=str, default="UUP")
    ap.add_argument("--oil", type=str, default="USO")
    ap.add_argument("--yield_ticker", type=str, default="IEF")
    ap.add_argument("--z_window", type=int, default=20)
    ap.add_argument("--beta_window", type=int, default=60)
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--start-date", type=str, default="2021-01-01")
    args = ap.parse_args()

    # Determine output path
    if args.out:
        out_path = Path(args.out)
    elif args.universe:
        out_path = Path(args.universe).parent / DEFAULT_OUT
    else:
        out_path = data_dir() / DEFAULT_OUT

    # Mode 1: Universe parquet + auto-fetch macro proxies
    if args.universe:
        universe_path = Path(args.universe)
        if not universe_path.exists():
            _die(f"Universe file not found: {universe_path}")

        _log(f"Loading universe: {universe_path}")
        df = pd.read_parquet(universe_path)

        required = ["ticker", "date", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            _die(f"Universe missing required columns: {missing}")

        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Date range
        start_date = args.start_date
        end_date = datetime.now(timezone.utc).date().isoformat()

        # Fetch macro proxies
        api_key = require_polygon_key()
        macro_tickers = {"dxy": args.dxy, "oil": args.oil, "yield": args.yield_ticker}
        macro_dfs = {}

        _log(f"Fetching macro proxies: {macro_tickers}")
        for label, ticker in macro_tickers.items():
            _log(f"  {label} -> {ticker} ...")
            macro_df = fetch_macro_proxy(ticker, start_date, end_date, api_key)
            if macro_df.empty:
                _die(f"No data for macro proxy {ticker}")
            macro_dfs[label] = macro_df
            _log(f"    {len(macro_df)} rows")
            time.sleep(RATE_PAUSE)

        # Compute QMPC
        _log("Computing QMPC ...")
        result = compute_qmpc(df, macro_dfs, args.z_window, args.beta_window, args.threshold)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(out_path, index=False, engine="pyarrow")
        _log(f"Saved -> {out_path} | rows={len(result):,} | tickers={result['ticker'].nunique():,}")
        return

    # Mode 2: Wide CSV
    if args.wide_csv:
        wide_path = Path(args.wide_csv)
        if not wide_path.exists():
            _die(f"wide_csv not found: {wide_path}")

        df = pd.read_csv(wide_path)
        for c in ("date", "ticker", "close", "dxy_close", "uso_close", "tnx_close"):
            if c not in df.columns:
                _die(f"wide_csv missing required column: {c}")

        df["date"] = _ensure_date(df["date"])
        df = df.dropna(subset=["date", "ticker"]).sort_values(["ticker", "date"])

        # Build macro dfs from wide columns
        mac = df[["date", "dxy_close", "uso_close", "tnx_close"]].drop_duplicates(subset=["date"]).sort_values("date")
        macro_dfs = {
            "dxy": mac.rename(columns={"dxy_close": "close"})[["date", "close"]],
            "oil": mac.rename(columns={"uso_close": "close"})[["date", "close"]],
            "yield": mac.rename(columns={"tnx_close": "close"})[["date", "close"]],
        }

        result = compute_qmpc(df, macro_dfs, args.z_window, args.beta_window, args.threshold)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        _log(f"Saved -> {out_path} | rows={len(result):,}")
        return

    # Mode 3: Prices CSV with macro tickers as rows
    if args.prices_csv:
        prices_path = Path(args.prices_csv)
        if not prices_path.exists():
            _die(f"prices_csv not found: {prices_path}")

        px = pd.read_csv(prices_path)
        cols = {c.lower(): c for c in px.columns}
        tcol = cols.get("ticker") or cols.get("symbol")
        dcol = cols.get("date") or cols.get("date_utc")
        ccol = cols.get("close")
        if not (tcol and dcol and ccol):
            _die(f"prices_csv must contain ticker/date/close. Found: {list(px.columns)[:25]}")

        px = px[[tcol, dcol, ccol]].rename(columns={tcol: "ticker", dcol: "date", ccol: "close"}).copy()
        px["date"] = _ensure_date(px["date"])
        px = px.dropna(subset=["date", "ticker"]).sort_values(["ticker", "date"])

        macro_tickers = {"dxy": args.dxy, "oil": args.oil, "yield": args.yield_ticker}
        macro_dfs = {}
        for label, ticker in macro_tickers.items():
            m = px[px["ticker"] == ticker][["date", "close"]].copy()
            if m.empty:
                _die(f"Macro tether ticker '{ticker}' not found in prices_csv")
            macro_dfs[label] = m

        result = compute_qmpc(px, macro_dfs, args.z_window, args.beta_window, args.threshold)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        _log(f"Saved -> {out_path} | rows={len(result):,}")
        return

    _die("Provide --universe, --wide_csv, or --prices_csv")


if __name__ == "__main__":
    main()