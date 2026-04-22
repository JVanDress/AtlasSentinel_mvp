#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from polygon import RESTClient

from quintic_paths import data_dir, load_simple_dotenv, project_root


DEFAULT_PROXIES = ("SPY", "TLT", "HYG", "LQD", "VIXY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cleanroom macro overlay from daily Polygon proxy ETFs and volatility proxies."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-parquet", type=Path, default=None)
    parser.add_argument("--status-output", type=Path, default=None)
    parser.add_argument("--proxies", default=",".join(DEFAULT_PROXIES))
    parser.add_argument("--lookback-days", type=int, default=500)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def runtime_project_root(cli_root: Path | None) -> Path:
    if cli_root is not None:
        return cli_root.expanduser()
    return project_root()


def resolve_path(path: Path | None, base_dir: Path, default_name: str, root_dir: Path | None = None) -> Path:
    if path is None:
        return base_dir / default_name

    candidate = path.expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate

    if root_dir is not None:
        rooted = root_dir / candidate
        if rooted.exists():
            return rooted
        if candidate.parts and candidate.parts[0] == base_dir.name:
            return root_dir / candidate

    if candidate.parts and candidate.parts[0] == base_dir.name:
        return base_dir.parent / candidate

    return base_dir / candidate


def require_polygon_key(root: Path) -> str:
    load_simple_dotenv(root, override=True)
    key = (os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("MASSIVE_API_KEY or POLYGON_API_KEY was not found in the environment.")
    return key


def parse_proxy_list(value: str) -> list[str]:
    proxies = [p.strip().upper() for p in str(value).split(",") if p.strip()]
    if not proxies:
        raise ValueError("At least one proxy ticker is required.")
    return proxies


def date_bounds(args: argparse.Namespace) -> tuple[str, str]:
    end_day = pd.Timestamp(args.end_date).date() if args.end_date else datetime.now(timezone.utc).date()
    if args.start_date:
        start_day = pd.Timestamp(args.start_date).date()
    else:
        start_day = end_day - timedelta(days=int(args.lookback_days))
    return start_day.isoformat(), end_day.isoformat()


def _agg_field(obj: Any, *names: str):
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
        return None

    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)

    try:
        as_dict = vars(obj)
    except Exception:
        as_dict = {}
    for name in names:
        if name in as_dict:
            return as_dict[name]
    return None


def _normalize_agg_date(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None:
        return pd.NaT
    if isinstance(value, (int, float, np.integer, np.floating)):
        ts = pd.to_datetime(value, unit="ms", utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_convert(None).normalize()


def fetch_daily_aggs(client: RESTClient, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    aggs = client.get_aggs(ticker, 1, "day", start_date, end_date)
    rows: list[dict[str, object]] = []
    for agg in aggs or []:
        row = {
            "ticker": ticker,
            "date": _normalize_agg_date(_agg_field(agg, "timestamp", "t", "date")),
            "open": pd.to_numeric(_agg_field(agg, "open", "o"), errors="coerce"),
            "high": pd.to_numeric(_agg_field(agg, "high", "h"), errors="coerce"),
            "low": pd.to_numeric(_agg_field(agg, "low", "l"), errors="coerce"),
            "close": pd.to_numeric(_agg_field(agg, "close", "c"), errors="coerce"),
            "volume": pd.to_numeric(_agg_field(agg, "volume", "v"), errors="coerce"),
            "vwap": pd.to_numeric(_agg_field(agg, "vwap", "vw"), errors="coerce"),
            "transactions": pd.to_numeric(_agg_field(agg, "transactions", "n"), errors="coerce"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return df


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def _expanding_pct_rank(s: pd.Series, min_periods: int = 60) -> pd.Series:
    """
    Expanding-window percentile rank — no lookahead.
    On date t, ranks the current value against all values up to t only.
    """
    def _rank_last(vals):
        if len(vals) < min_periods:
            return np.nan
        return float((vals < vals[-1]).sum()) / float(len(vals) - 1)
    return s.expanding(min_periods=min_periods).apply(_rank_last, raw=True)


def _z(s: pd.Series, w: int) -> pd.Series:
    mu = s.rolling(w, min_periods=w).mean()
    sd = s.rolling(w, min_periods=w).std().replace(0, np.nan)
    return (s - mu) / sd


def _fit_hmm_or_gmm(X: np.ndarray, n_states: int = 4) -> tuple:
    """Fit regime model on training data only. Returns (model, method_name)."""
    try:
        from hmmlearn.hmm import GaussianHMM

        hmm = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        hmm.fit(X)
        return hmm, "hmm"
    except Exception:
        pass

    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=n_states, random_state=42)
    gmm.fit(X)
    return gmm, "gmm"


def _expanding_window_regimes(
    df: pd.DataFrame,
    feat_cols: list,
    n_states: int = 4,
    min_train: int = 252,
    refit_every: int = 63,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Expanding-window regime detection — no lookahead.

    Fits HMM/GMM on data up to each refit point, then predicts forward
    until the next refit. The model never sees future data.

    Args:
        df: DataFrame with feat_cols populated
        feat_cols: columns to use as regime features
        n_states: number of regime states
        min_train: minimum training rows before first fit
        refit_every: refit the model every N trading days
    """
    xdf = df[feat_cols].dropna()
    n = len(xdf)

    states = np.full(len(df), np.nan)
    probs = np.full((len(df), n_states), np.nan)
    regime_model = "none"

    if n < min_train:
        return states, probs, regime_model

    # Map xdf index positions back to df positions
    xdf_positions = xdf.index

    # Determine refit points
    refit_points = list(range(min_train, n, refit_every))
    if refit_points[-1] != n:
        refit_points.append(n)

    for i, end_idx in enumerate(refit_points):
        # Training window: all data up to this point
        train_data = xdf.iloc[:end_idx][feat_cols].values.astype(float)

        # Prediction window: from this refit to the next
        if i + 1 < len(refit_points):
            predict_start = end_idx - refit_every if i > 0 else 0
            predict_end = refit_points[i + 1] if i + 1 < len(refit_points) else n
        else:
            predict_start = refit_points[i - 1] if i > 0 else 0
            predict_end = n

        # Only predict the NEW segment (avoid overwriting earlier predictions)
        if i == 0:
            pred_start = 0
        else:
            pred_start = refit_points[i - 1]
        pred_end = end_idx

        predict_data = xdf.iloc[pred_start:pred_end][feat_cols].values.astype(float)

        if len(predict_data) == 0:
            continue

        try:
            model, method = _fit_hmm_or_gmm(train_data, n_states)
            regime_model = method

            if hasattr(model, "predict"):
                pred_states = model.predict(predict_data)
            else:
                pred_states = model.predict(predict_data)

            if hasattr(model, "predict_proba"):
                pred_probs = model.predict_proba(predict_data)
            else:
                pred_probs = np.zeros((len(predict_data), n_states))

            # Map back to df index positions
            segment_indices = xdf_positions[pred_start:pred_end]
            states[segment_indices] = pred_states.astype(float)
            for k in range(min(n_states, pred_probs.shape[1])):
                probs[segment_indices, k] = pred_probs[:, k]

        except Exception:
            continue

    return states, probs, regime_model


def build_overlay(price_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    px = price_df.copy()
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px = px.dropna(subset=["date", "ticker", "close"]).sort_values(["ticker", "date"]).reset_index(drop=True)

    def pick_series(ticker: str, col: str = "close") -> pd.Series | None:
        d = px[px["ticker"] == ticker].copy()
        if d.empty:
            return None
        s = d.set_index("date")[col].astype(float)
        s.name = ticker
        return s

    spy = pick_series("SPY")
    if spy is None or spy.dropna().shape[0] < 120:
        raise RuntimeError("SPY not found or insufficient history in fetched macro proxies.")

    spy_ret = spy.pct_change(fill_method=None)
    vol5 = spy_ret.rolling(5, min_periods=5).std()
    vol20 = spy_ret.rolling(20, min_periods=20).std()
    mom20 = spy.pct_change(20, fill_method=None)
    mom60 = spy.pct_change(60, fill_method=None)

    idx = pd.to_datetime(spy.index)
    df = pd.DataFrame(index=idx)
    df.index.name = "date"
    df["macro_spy_ret_1d"] = spy_ret.values
    df["macro_spy_vol_5d"] = vol5.values
    df["macro_spy_vol_20d"] = vol20.values
    df["macro_spy_mom_20d"] = mom20.values
    df["macro_spy_mom_60d"] = mom60.values

    df["volatility_instability_pressure"] = _safe_div(df["macro_spy_vol_5d"], df["macro_spy_vol_20d"])
    df["market_liquidity_fragility_index"] = df["macro_spy_vol_20d"].diff().abs()
    df["financial_intermediation_stability_score"] = -df["volatility_instability_pressure"].rolling(10, min_periods=10).mean()
    df["systemic_instability_risk_score"] = (
        df["volatility_instability_pressure"].rolling(20, min_periods=20).mean()
        + df["market_liquidity_fragility_index"].rolling(20, min_periods=20).mean()
    )
    df["intermediation_stress_probability"] = _expanding_pct_rank(df["systemic_instability_risk_score"], min_periods=60)
    df["liquidity_impulse_score"] = df["macro_spy_mom_20d"]
    df["liquidity_impulse_5d"] = df["macro_spy_mom_20d"].diff(5)
    df["liquidity_impulse_20d"] = df["macro_spy_mom_60d"].diff(20)
    df["liquidity_regime_shift_probability"] = _expanding_pct_rank(df["liquidity_impulse_5d"].abs(), min_periods=60)

    z60 = _z(df["macro_spy_vol_20d"], 60).abs()
    df["structural_break_probability"] = _expanding_pct_rank(z60, min_periods=60)
    df["signal_stability_index"] = df["macro_spy_vol_20d"].rolling(60, min_periods=60).mean()
    df["regime_transition_risk"] = df["structural_break_probability"].diff().abs()

    for ticker in ("TLT", "HYG", "LQD", "VIXY"):
        s = pick_series(ticker)
        col = f"macro_{ticker.lower()}_ret_1d"
        if s is None:
            df[col] = np.nan
        else:
            r = s.pct_change(fill_method=None)
            df[col] = r.reindex(df.index).values

    df["cross_asset_pressure_index"] = (
        df["volatility_instability_pressure"].fillna(0.0)
        + (-df["macro_hyg_ret_1d"].fillna(0.0))
        + df["macro_vixy_ret_1d"].fillna(0.0)
    )
    df["equity_volatility_divergence_score"] = df["macro_spy_vol_5d"] - df["macro_spy_vol_20d"]
    df["sector_dispersion_pressure"] = spy_ret.rolling(10, min_periods=10).std().values

    feat_cols = [
        "macro_spy_vol_20d",
        "macro_spy_mom_20d",
        "volatility_instability_pressure",
        "liquidity_impulse_score",
        "cross_asset_pressure_index",
    ]
    for k in range(4):
        df[f"regime_prob_{k}"] = np.nan
    df["regime_state"] = np.nan
    regime_model = "none"

    # Expanding-window regime detection — no lookahead
    # Refits every 63 trading days using only past data
    states, probs, regime_model = _expanding_window_regimes(
        df, feat_cols, n_states=4, min_train=252, refit_every=63
    )
    df["regime_state"] = states
    for k in range(4):
        df[f"regime_prob_{k}"] = probs[:, k]

    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df, regime_model


def main() -> int:
    args = parse_args()
    root = runtime_project_root(args.project_root)
    data_root = data_dir(root)
    data_root.mkdir(parents=True, exist_ok=True)

    output_path = resolve_path(args.output_parquet, data_root, "cleanroom_macro_overlay.parquet", root_dir=root)
    status_path = resolve_path(args.status_output, data_root, "cleanroom_macro_overlay_status.json", root_dir=root)
    proxies = parse_proxy_list(args.proxies)
    start_date, end_date = date_bounds(args)

    print(f"Project root: {root}")
    print(f"Output parquet: {output_path}")
    print(f"Proxy tickers: {proxies}")
    print(f"Date window: {start_date} -> {end_date}")

    if args.dry_run:
        print("Dry run complete. No macro overlay written.")
        return 0

    api_key = require_polygon_key(root)
    client = RESTClient(api_key=api_key)

    frames = []
    missing: list[str] = []
    for ticker in proxies:
        try:
            df = fetch_daily_aggs(client, ticker, start_date, end_date)
        except Exception as exc:
            print(f"Macro fetch failed for {ticker}: {exc}")
            df = pd.DataFrame()
        if df.empty:
            missing.append(ticker)
            continue
        frames.append(df)

    if not frames:
        raise RuntimeError("No macro proxy history was fetched.")

    price_df = pd.concat(frames, ignore_index=True)
    overlay_df, regime_model = build_overlay(price_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_df.to_parquet(output_path, index=False)

    status = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "output_parquet": str(output_path),
        "proxy_tickers_requested": proxies,
        "proxy_tickers_missing": missing,
        "start_date": start_date,
        "end_date": end_date,
        "rows_written": int(len(overlay_df)),
        "regime_model": regime_model,
        "latest_date": str(pd.to_datetime(overlay_df["date"]).max().date()) if not overlay_df.empty else None,
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    print(f"Saved macro overlay: {output_path}")
    print(f"Saved status file: {status_path}")
    print(f"Rows: {len(overlay_df):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())