#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quintic_paths import data_dir, project_root
from build_volume_microstructure_overlay import (
    classify_venues,
    load_exchange_reference,
    load_frame,
    require_api_key,
    normalize_trades_with_venue,
    resolve_existing_input,
    resolve_path,
)


OUTPUT_NAME = "cleanroom_session_liquidity_shadow.parquet"
STATUS_NAME = "cleanroom_session_liquidity_shadow_status.json"
LATEST_NAME = "cleanroom_session_liquidity_shadow_latest.csv"
EXCHANGE_CACHE_NAME = "polygon_stock_exchange_reference.json"
LOOKBACK_SESSIONS = 20
MIN_LOOKBACK = 5
EASTERN_START = 570
EASTERN_END = 960
CONTEXT_CANDIDATES = [
    "tradeable_panel_final_fusion.parquet",
    "tradeable_panel_normalized.parquet",
    "tradeable_panel_sectorized.parquet",
]
CONTEXT_COLS = [
    "vwap_spread",
    "transactions",
    "trade_intensity",
    "intraday_return",
    "hl_range_pct",
    "ret_1d",
    "ret_20d_back",
    "volume_rvol_20",
    "dollar_volume_rvol_20",
    "log_volume_z_20",
    "log_dollar_volume_z_20",
    "sector",
    "industry",
    "gics_sector",
    "gics_industry",
    "rel_sector_strength_1d",
    "rel_industry_strength_1d",
    "rel_sector_log_volume",
    "rel_industry_log_volume",
    "rel_sector_log_dollar_volume",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a session-level liquidity shadow overlay from Polygon trades.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--trades-path", type=Path, default=None)
    parser.add_argument("--context-path", type=Path, default=None)
    parser.add_argument("--output-parquet", type=Path, default=None)
    parser.add_argument("--status-output", type=Path, default=None)
    parser.add_argument("--latest-csv", type=Path, default=None)
    parser.add_argument("--exchange-cache-output", type=Path, default=None)
    parser.add_argument("--lookback-sessions", type=int, default=LOOKBACK_SESSIONS)
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--session-date", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_context(base_dir: Path, context_path: Path | None) -> pd.DataFrame | None:
    if context_path is not None:
        paths = [context_path.expanduser()]
    else:
        paths = [base_dir / name for name in CONTEXT_CANDIDATES]
    for path in paths:
        if not path.is_absolute():
            path = base_dir / path
        if not path.exists():
            continue
        ctx = pd.read_parquet(path)
        ctx["ticker"] = ctx["ticker"].astype(str).str.upper().str.strip()
        ctx["date"] = pd.to_datetime(ctx["date"], errors="coerce").dt.normalize()
        keep = [c for c in ["ticker", "date"] + CONTEXT_COLS if c in ctx.columns]
        ctx = ctx[keep].copy()
        return ctx.rename(columns={c: f"ctx_{c}" for c in ctx.columns if c not in {"ticker", "date"}})
    return None


def build_shadow(trades: pd.DataFrame, lookback_sessions: int) -> pd.DataFrame:
    trades = trades.sort_values(["ticker", "date", "ts"], kind="mergesort").reset_index(drop=True).copy()
    trades["price_change_flag"] = trades.groupby(["ticker", "date"], sort=False)["trade_price"].diff().ne(0).astype(float).fillna(0.0)
    trades["off_notional"] = trades["trade_notional"] * trades["off_exchange_flag"]
    trades["lit_notional"] = trades["trade_notional"] * trades["lit_exchange_flag"]

    minute = (
        trades.groupby(["ticker", "date", "minute"], sort=False)
        .agg(
            minute_trade_open=("trade_price", "first"),
            minute_trade_high=("trade_price", "max"),
            minute_trade_low=("trade_price", "min"),
            minute_trade_close=("trade_price", "last"),
            minute_trade_notional=("trade_notional", "sum"),
            minute_trade_size=("trade_size", "sum"),
            minute_trade_count=("event_ns", "size"),
            minute_price_change_count=("price_change_flag", "sum"),
            minute_off_notional=("off_notional", "sum"),
            minute_lit_notional=("lit_notional", "sum"),
            minute_off_trade_count=("off_exchange_flag", "sum"),
            minute_lit_trade_count=("lit_exchange_flag", "sum"),
        )
        .reset_index()
    )

    minute["minute_trade_vwap"] = (minute["minute_trade_notional"] / minute["minute_trade_size"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    ref = minute["minute_trade_vwap"].fillna(minute["minute_trade_close"]).replace(0.0, np.nan)
    minute["minute_range_bps"] = (((minute["minute_trade_high"] - minute["minute_trade_low"]) / ref) * 10000.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    minute["minute_close_to_open_bps"] = (((minute["minute_trade_close"] - minute["minute_trade_open"]) / minute["minute_trade_open"].replace(0.0, np.nan)) * 10000.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    minute["update_proxy"] = (minute["minute_trade_count"] + minute["minute_price_change_count"]).astype(float)
    minute["minute_of_session"] = (minute["minute"].dt.hour * 60 + minute["minute"].dt.minute - EASTERN_START).clip(lower=0, upper=(EASTERN_END - EASTERN_START))

    grp = minute.groupby(["ticker", "date"], sort=False)
    minute["session_notional"] = grp["minute_trade_notional"].transform("sum")
    minute["session_trades"] = grp["minute_trade_count"].transform("sum")
    minute["session_off_notional"] = grp["minute_off_notional"].transform("sum")
    minute["session_off_trades"] = grp["minute_off_trade_count"].transform("sum")
    minute["minute_notional_share"] = (minute["minute_trade_notional"] / minute["session_notional"].replace(0.0, np.nan)).clip(0.0, 1.0).fillna(0.0)
    minute["minute_trade_share"] = (minute["minute_trade_count"] / minute["session_trades"].replace(0.0, np.nan)).clip(0.0, 1.0).fillna(0.0)
    minute["minute_off_notional_share"] = (minute["minute_off_notional"] / minute["session_notional"].replace(0.0, np.nan)).clip(0.0, 1.0).fillna(0.0)
    minute["minute_off_trade_share"] = (minute["minute_off_trade_count"] / minute["session_trades"].replace(0.0, np.nan)).clip(0.0, 1.0).fillna(0.0)
    minute["cum_notional_share"] = grp["minute_trade_notional"].cumsum() / minute["session_notional"].replace(0.0, np.nan)
    minute["cum_trade_share"] = grp["minute_trade_count"].cumsum() / minute["session_trades"].replace(0.0, np.nan)
    minute["cum_off_notional_share"] = grp["minute_off_notional"].cumsum() / minute["session_notional"].replace(0.0, np.nan)
    minute["cum_off_trade_share"] = grp["minute_off_trade_count"].cumsum() / minute["session_trades"].replace(0.0, np.nan)
    minute["cum_notional_share"] = minute["cum_notional_share"].clip(0.0, 1.0).fillna(0.0)
    minute["cum_trade_share"] = minute["cum_trade_share"].clip(0.0, 1.0).fillna(0.0)
    minute["cum_off_notional_share"] = minute["cum_off_notional_share"].clip(0.0, 1.0).fillna(0.0)
    minute["cum_off_trade_share"] = minute["cum_off_trade_share"].clip(0.0, 1.0).fillna(0.0)
    minute["idle_gap_minutes"] = (grp["minute_of_session"].diff().fillna(0.0) - 1.0).clip(lower=0.0)
    minute["minute_hhi_notional"] = minute["minute_notional_share"] ** 2
    minute["minute_hhi_trade"] = minute["minute_trade_share"] ** 2

    minute = minute.sort_values(["ticker", "minute_of_session", "date"], kind="mergesort").reset_index(drop=True)
    mgrp = minute.groupby(["ticker", "minute_of_session"], sort=False)
    min_periods = max(MIN_LOOKBACK, min(lookback_sessions, LOOKBACK_SESSIONS))
    for col in ["cum_notional_share", "cum_trade_share", "cum_off_notional_share", "cum_off_trade_share", "minute_range_bps", "update_proxy"]:
        minute[f"exp_{col}"] = mgrp[col].transform(lambda s: s.shift(1).rolling(lookback_sessions, min_periods=min_periods).mean())
        minute[f"{col}_dev"] = minute[col] - minute[f"exp_{col}"]
        minute[f"{col}_dev_abs"] = minute[f"{col}_dev"].abs()
        minute[f"{col}_dev_sq"] = minute[f"{col}_dev"] ** 2

    minute["curve_deficit"] = (-minute["cum_notional_share_dev"]).clip(lower=0.0)
    minute["curve_surplus"] = minute["cum_notional_share_dev"].clip(lower=0.0)
    minute["spread_widening_proxy"] = minute["minute_range_bps_dev"].clip(lower=0.0)
    minute["update_shortfall"] = (-minute["update_proxy_dev"]).clip(lower=0.0)
    minute["off_shift"] = minute["cum_off_notional_share_dev"].abs()

    minute["curve_deficit_open30"] = minute["curve_deficit"].where(minute["minute_of_session"] <= 30)
    minute["curve_deficit_midday"] = minute["curve_deficit"].where((minute["minute_of_session"] >= 120) & (minute["minute_of_session"] <= 240))
    minute["curve_deficit_late"] = minute["curve_deficit"].where(minute["minute_of_session"] >= 300)
    minute["spread_open30"] = minute["spread_widening_proxy"].where(minute["minute_of_session"] <= 30)
    minute["spread_midday"] = minute["spread_widening_proxy"].where((minute["minute_of_session"] >= 120) & (minute["minute_of_session"] <= 240))
    minute["spread_late"] = minute["spread_widening_proxy"].where(minute["minute_of_session"] >= 300)
    minute["update_open30"] = minute["update_shortfall"].where(minute["minute_of_session"] <= 30)
    minute["update_midday"] = minute["update_shortfall"].where((minute["minute_of_session"] >= 120) & (minute["minute_of_session"] <= 240))
    minute["update_late"] = minute["update_shortfall"].where(minute["minute_of_session"] >= 300)
    minute["off_shift_open30"] = minute["off_shift"].where(minute["minute_of_session"] <= 30)
    minute["off_shift_midday"] = minute["off_shift"].where((minute["minute_of_session"] >= 120) & (minute["minute_of_session"] <= 240))
    minute["off_shift_late"] = minute["off_shift"].where(minute["minute_of_session"] >= 300)

    summary = (
        minute.groupby(["ticker", "date"], sort=False)
        .agg(
            shadow_session_minutes=("minute", "size"),
            shadow_session_notional=("minute_trade_notional", "sum"),
            shadow_session_trades=("minute_trade_count", "sum"),
            shadow_session_off_notional=("minute_off_notional", "sum"),
            shadow_session_off_trades=("minute_off_trade_count", "sum"),
            shadow_session_idle_gap_mean=("idle_gap_minutes", "mean"),
            shadow_session_idle_gap_max=("idle_gap_minutes", "max"),
            shadow_session_hhi_notional=("minute_hhi_notional", "sum"),
            shadow_session_hhi_trade=("minute_hhi_trade", "sum"),
            shadow_session_range_bps_mean=("minute_range_bps", "mean"),
            shadow_session_range_bps_max=("minute_range_bps", "max"),
            shadow_update_proxy_mean=("update_proxy", "mean"),
            shadow_update_proxy_max=("update_proxy", "max"),
            shadow_curve_mae=("cum_notional_share_dev_abs", "mean"),
            shadow_curve_rmse=("cum_notional_share_dev_sq", "mean"),
            shadow_curve_peak_deficit=("curve_deficit", "max"),
            shadow_curve_peak_surplus=("curve_surplus", "max"),
            shadow_curve_open30=("curve_deficit_open30", "mean"),
            shadow_curve_midday=("curve_deficit_midday", "mean"),
            shadow_curve_late=("curve_deficit_late", "mean"),
            shadow_spread_mean=("spread_widening_proxy", "mean"),
            shadow_spread_max=("spread_widening_proxy", "max"),
            shadow_spread_open30=("spread_open30", "mean"),
            shadow_spread_midday=("spread_midday", "mean"),
            shadow_spread_late=("spread_late", "mean"),
            shadow_update_shortfall_mean=("update_shortfall", "mean"),
            shadow_update_shortfall_max=("update_shortfall", "max"),
            shadow_update_open30=("update_open30", "mean"),
            shadow_update_midday=("update_midday", "mean"),
            shadow_update_late=("update_late", "mean"),
            shadow_off_shift_mean=("off_shift", "mean"),
            shadow_off_shift_max=("off_shift", "max"),
            shadow_off_shift_open30=("off_shift_open30", "mean"),
            shadow_off_shift_midday=("off_shift_midday", "mean"),
            shadow_off_shift_late=("off_shift_late", "mean"),
            shadow_close_to_open_bps_mean=("minute_close_to_open_bps", "mean"),
        )
        .reset_index()
    )

    summary["shadow_session_off_share"] = (summary["shadow_session_off_notional"] / summary["shadow_session_notional"].replace(0.0, np.nan)).clip(0.0, 1.0).fillna(0.0)
    summary["shadow_session_off_trade_share"] = (summary["shadow_session_off_trades"] / summary["shadow_session_trades"].replace(0.0, np.nan)).clip(0.0, 1.0).fillna(0.0)
    summary["shadow_curve_rmse"] = np.sqrt(summary["shadow_curve_rmse"].clip(lower=0.0))
    return summary


def add_context(summary: pd.DataFrame, context: pd.DataFrame | None) -> pd.DataFrame:
    if context is None or context.empty:
        return summary.copy()
    return summary.merge(context, on=["ticker", "date"], how="left")


def add_scores(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True).copy()

    def col_or_zero(frame: pd.DataFrame, col: str) -> pd.Series:
        if col in frame.columns:
            return pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=frame.index, dtype=float)

    z_cols = [
        "shadow_curve_mae",
        "shadow_curve_rmse",
        "shadow_curve_peak_deficit",
        "shadow_spread_mean",
        "shadow_spread_max",
        "shadow_update_shortfall_mean",
        "shadow_update_shortfall_max",
        "shadow_off_shift_mean",
        "shadow_off_shift_max",
        "shadow_session_idle_gap_max",
        "shadow_session_hhi_notional",
    ]
    for col in z_cols:
        if col not in out.columns:
            continue
        mean = out.groupby("date", sort=False)[col].transform("mean")
        std = out.groupby("date", sort=False)[col].transform("std").replace(0.0, np.nan)
        out[f"{col}_cs_z"] = ((out[col] - mean) / std).replace([np.inf, -np.inf], np.nan).clip(-10.0, 10.0).fillna(0.0)

    pos = lambda s: s.clip(lower=0.0)
    drought_raw = (
        0.35 * pos(col_or_zero(out, "shadow_curve_peak_deficit_cs_z"))
        + 0.20 * pos(col_or_zero(out, "shadow_curve_mae_cs_z"))
        + 0.20 * pos(col_or_zero(out, "shadow_spread_mean_cs_z"))
        + 0.15 * pos(col_or_zero(out, "shadow_off_shift_max_cs_z"))
        + 0.10 * pos(col_or_zero(out, "shadow_session_idle_gap_max_cs_z"))
        + 0.05 * pos(-col_or_zero(out, "shadow_update_shortfall_mean_cs_z"))
    )
    spread_raw = 0.60 * pos(col_or_zero(out, "shadow_spread_mean_cs_z")) + 0.40 * pos(col_or_zero(out, "shadow_spread_max_cs_z"))
    update_raw = 0.60 * pos(col_or_zero(out, "shadow_update_shortfall_mean_cs_z")) + 0.40 * pos(col_or_zero(out, "shadow_update_shortfall_max_cs_z"))

    out["shadow_liquidity_drought_risk"] = 1.0 / (1.0 + np.exp(-drought_raw))
    out["shadow_spread_widening_risk"] = 1.0 / (1.0 + np.exp(-spread_raw))
    out["shadow_quote_update_intensity_risk"] = 1.0 / (1.0 + np.exp(-update_raw))
    out["shadow_expected_volume_curve_deviation"] = out["shadow_curve_mae"]
    out["shadow_off_exchange_share_shift"] = out["shadow_off_shift_mean"]
    out["shadow_quote_update_intensity"] = out["shadow_update_proxy_mean"]
    out["shadow_liquidity_shadow_score"] = (
        0.40 * out["shadow_liquidity_drought_risk"]
        + 0.25 * out["shadow_spread_widening_risk"]
        + 0.20 * out["shadow_quote_update_intensity_risk"]
        + 0.15 * pos(col_or_zero(out, "shadow_off_shift_max_cs_z"))
    ).clip(0.0, 1.0)
    return out


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    root = Path(args.project_root).expanduser().resolve() if args.project_root else project_root()
    base_dir = data_dir(root)
    output_path = resolve_path(args.output_parquet, base_dir, OUTPUT_NAME)
    status_path = resolve_path(args.status_output, base_dir, STATUS_NAME)
    latest_csv = resolve_path(args.latest_csv, base_dir, LATEST_NAME)
    exchange_cache = resolve_path(args.exchange_cache_output, base_dir, EXCHANGE_CACHE_NAME)

    trades_path = resolve_existing_input(
        base_dir,
        args.trades_path,
        ["polygon_trades.parquet", "polygon_trade_ticks.parquet", "intraday_trades.parquet", "trades.parquet", "trades.csv"],
        "trades",
    )
    context = load_context(base_dir, args.context_path)

    logging.info("Loading exchange reference")
    api_key = require_api_key(root)
    exchanges = load_exchange_reference(api_key, exchange_cache)
    logging.info("Loading trades from %s", trades_path)
    trades = load_frame(trades_path)
    trades = normalize_trades_with_venue(trades, max_tickers=args.max_tickers, session_date=None)
    trades = classify_venues(trades, exchanges)
    if trades.empty:
        raise RuntimeError("No usable trade rows were found after cleaning and regular-session filtering.")

    logging.info("Building session shadow features")
    shadow = build_shadow(trades, lookback_sessions=max(args.lookback_sessions, MIN_LOOKBACK))
    shadow = add_context(shadow, context)
    shadow = add_scores(shadow)
    shadow = shadow.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)

    if args.session_date:
        session_ts = pd.Timestamp(args.session_date).normalize()
        shadow = shadow[shadow["date"] == session_ts].copy()

    if args.dry_run:
        latest_date = shadow["date"].max()
        latest = shadow[shadow["date"] == latest_date].sort_values("shadow_liquidity_shadow_score", ascending=False).head(20)
        print(latest[["ticker", "shadow_liquidity_shadow_score", "shadow_liquidity_drought_risk", "shadow_spread_widening_risk", "shadow_quote_update_intensity_risk"]].to_string(index=False))
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shadow.to_parquet(output_path, index=False)
    latest_date = shadow["date"].max()
    latest = shadow[shadow["date"] == latest_date].sort_values("shadow_liquidity_shadow_score", ascending=False).head(50)
    latest.to_csv(latest_csv, index=False)

    status = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "session_liquidity_shadow",
        "trades_path": str(trades_path),
        "exchange_cache_path": str(exchange_cache),
        "context_path": str(args.context_path) if args.context_path else None,
        "rows": int(len(shadow)),
        "tickers": int(shadow["ticker"].nunique()) if not shadow.empty else 0,
        "date_min": str(shadow["date"].min().date()) if not shadow.empty else None,
        "date_max": str(latest_date.date()) if not shadow.empty and pd.notna(latest_date) else None,
        "context_merged": bool(context is not None and not context.empty),
        "avg_liquidity_drought_risk_latest": float(latest["shadow_liquidity_drought_risk"].mean()) if not latest.empty else None,
        "avg_spread_widening_risk_latest": float(latest["shadow_spread_widening_risk"].mean()) if not latest.empty else None,
        "avg_quote_update_intensity_risk_latest": float(latest["shadow_quote_update_intensity_risk"].mean()) if not latest.empty else None,
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    logging.info("Saved session liquidity shadow overlay to %s", output_path)
    logging.info("Saved latest session CSV to %s", latest_csv)
    logging.info("Saved status to %s", status_path)


if __name__ == "__main__":
    main()
