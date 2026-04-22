#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from quintic_paths import data_dir, project_root
from build_pressure_velocity_overlay import (
    EPS,
    load_frame,
    normalize_trades,
    merge_options_confirmation,
    add_rebalance_states,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Build a Quintic pressure-velocity overlay from intraday trades plus options confirmation. '
            'This is the Polygon-only fallback when stock quote REST is not entitled.'
        )
    )
    parser.add_argument('--project-root', type=Path, default=None)
    parser.add_argument('--trades-path', type=Path, default=None)
    parser.add_argument('--options-path', type=Path, default=None)
    parser.add_argument('--output-parquet', type=Path, default=None)
    parser.add_argument('--status-output', type=Path, default=None)
    parser.add_argument('--latest-long-csv', type=Path, default=None)
    parser.add_argument('--latest-short-csv', type=Path, default=None)
    parser.add_argument('--max-tickers', type=int, default=0)
    parser.add_argument('--session-date', default=None)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def runtime_project_root(cli_root: Path | None) -> Path:
    if cli_root is not None:
        return cli_root.expanduser()
    return project_root()


def resolve_path(path: Path | None, base_dir: Path, default_name: str) -> Path:
    if path is None:
        return base_dir / default_name
    candidate = path.expanduser()
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate


def resolve_existing_input(base_dir: Path, provided: Path | None, candidates: list[str], label: str) -> Path:
    if provided is not None:
        path = provided.expanduser()
        if not path.is_absolute():
            path = base_dir / path
        if not path.exists():
            raise FileNotFoundError(f'{label} path does not exist: {path}')
        return path
    for name in candidates:
        candidate = base_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'No default {label} file found under {base_dir}. Tried: {", ".join(candidates)}')


def _signed_tick(series: pd.Series) -> pd.Series:
    delta = series.diff()
    sign = np.sign(delta).replace(0, np.nan)
    sign = sign.ffill().fillna(0.0)
    return sign.clip(-1.0, 1.0)


def build_trades_minute_fallback(trades: pd.DataFrame) -> pd.DataFrame:
    work = trades.sort_values(['ticker', 'date', 'ts'], kind='mergesort').copy()
    group = work.groupby(['ticker', 'date'], sort=False)
    work['tick_sign'] = group['trade_price'].transform(_signed_tick)
    prev_price = group['trade_price'].shift(1)
    pct_change = ((work['trade_price'] - prev_price) / prev_price.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    work['aggressive_buy_flag'] = (work['tick_sign'] > 0.0).astype(float)
    work['aggressive_sell_flag'] = (work['tick_sign'] < 0.0).astype(float)
    work['through_ask_flag'] = ((pct_change > 0.0005) & (work['tick_sign'] > 0.0)).astype(float)
    work['through_bid_flag'] = ((pct_change < -0.0005) & (work['tick_sign'] < 0.0)).astype(float)

    work['aggressive_buy_notional'] = work['trade_notional'] * work['aggressive_buy_flag']
    work['aggressive_sell_notional'] = work['trade_notional'] * work['aggressive_sell_flag']
    work['through_ask_notional'] = work['trade_notional'] * work['through_ask_flag']
    work['through_bid_notional'] = work['trade_notional'] * work['through_bid_flag']

    minute = (
        work.groupby(['ticker', 'date', 'minute'], sort=False)
        .agg(
            trade_notional=('trade_notional', 'sum'),
            trade_size=('trade_size', 'sum'),
            trade_count=('event_ns', 'size'),
            aggressive_buy_notional=('aggressive_buy_notional', 'sum'),
            aggressive_sell_notional=('aggressive_sell_notional', 'sum'),
            through_ask_notional=('through_ask_notional', 'sum'),
            through_bid_notional=('through_bid_notional', 'sum'),
            aggressive_buy_trades=('aggressive_buy_flag', 'sum'),
            aggressive_sell_trades=('aggressive_sell_flag', 'sum'),
            through_ask_trades=('through_ask_flag', 'sum'),
            through_bid_trades=('through_bid_flag', 'sum'),
            minute_open=('trade_price', 'first'),
            minute_close=('trade_price', 'last'),
            minute_high=('trade_price', 'max'),
            minute_low=('trade_price', 'min'),
        )
        .reset_index()
    )

    denom = minute['trade_notional'].replace(0.0, np.nan)
    agg_denom = (minute['aggressive_buy_notional'] + minute['aggressive_sell_notional']).replace(0.0, np.nan)
    minute['aggressive_buy_share'] = (minute['aggressive_buy_notional'] / denom).clip(0.0, 1.0).fillna(0.0)
    minute['aggressive_sell_share'] = (minute['aggressive_sell_notional'] / denom).clip(0.0, 1.0).fillna(0.0)
    minute['through_ask_share'] = (minute['through_ask_notional'] / denom).clip(0.0, 1.0).fillna(0.0)
    minute['through_bid_share'] = (minute['through_bid_notional'] / denom).clip(0.0, 1.0).fillna(0.0)
    minute['net_aggression'] = ((minute['aggressive_buy_notional'] - minute['aggressive_sell_notional']) / (agg_denom + EPS)).clip(-1.0, 1.0).fillna(0.0)
    return minute.sort_values(['ticker', 'date', 'minute'], kind='mergesort').reset_index(drop=True)


def build_quote_proxy_minutes(trades_minute: pd.DataFrame) -> pd.DataFrame:
    minute = trades_minute[['ticker', 'date', 'minute', 'trade_count', 'minute_close', 'net_aggression']].copy()
    minute = minute.sort_values(['ticker', 'date', 'minute'], kind='mergesort').reset_index(drop=True)
    group = minute.groupby(['ticker', 'date'], sort=False)
    minute['price_step'] = group['minute_close'].diff().fillna(0.0)
    direction = np.sign(minute['price_step']).clip(-1.0, 1.0)
    minute['bid_price'] = minute['minute_close']
    minute['ask_price'] = minute['minute_close']
    minute['bid_size'] = 0.0
    minute['ask_size'] = 0.0
    minute['spread_mean'] = 0.0
    minute['spread_last'] = 0.0
    minute['mid_price'] = minute['minute_close']
    minute['quote_imbalance'] = minute['net_aggression'].clip(-1.0, 1.0)
    minute['quote_updates'] = minute['trade_count']
    minute['bid_up'] = (direction > 0.0).astype(float)
    minute['bid_down'] = (direction < 0.0).astype(float)
    minute['ask_up'] = (direction > 0.0).astype(float)
    minute['ask_down'] = (direction < 0.0).astype(float)
    minute['imbalance_step'] = group['quote_imbalance'].diff().fillna(0.0)
    minute['spread_change'] = 0.0
    return minute[['ticker', 'date', 'minute', 'bid_price', 'ask_price', 'bid_size', 'ask_size', 'spread_mean', 'spread_last', 'mid_price', 'quote_imbalance', 'quote_updates', 'bid_up', 'bid_down', 'ask_up', 'ask_down', 'imbalance_step', 'spread_change']]


def longest_streak(mask: pd.Series) -> int:
    best = 0
    current = 0
    for value in mask.astype(bool).tolist():
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def build_daily_overlay(quotes_minute: pd.DataFrame, trades_minute: pd.DataFrame) -> pd.DataFrame:
    minute = quotes_minute.merge(trades_minute, on=['ticker', 'date', 'minute'], how='outer')
    minute = minute.sort_values(['ticker', 'date', 'minute'], kind='mergesort').reset_index(drop=True)

    fill_zero = [
        'trade_notional', 'trade_size', 'trade_count',
        'aggressive_buy_notional', 'aggressive_sell_notional', 'through_ask_notional', 'through_bid_notional',
        'aggressive_buy_trades', 'aggressive_sell_trades', 'through_ask_trades', 'through_bid_trades',
        'aggressive_buy_share', 'aggressive_sell_share', 'through_ask_share', 'through_bid_share', 'net_aggression',
        'quote_updates', 'bid_up', 'bid_down', 'ask_up', 'ask_down', 'imbalance_step', 'spread_change',
    ]
    for col in fill_zero:
        if col in minute.columns:
            minute[col] = pd.to_numeric(minute[col], errors='coerce').fillna(0.0)

    for col in ['bid_price', 'ask_price', 'bid_size', 'ask_size', 'spread_mean', 'spread_last', 'mid_price', 'quote_imbalance']:
        if col in minute.columns:
            minute[col] = pd.to_numeric(minute[col], errors='coerce')

    minute['long_pulse'] = (
        0.55 * minute.get('aggressive_buy_share', 0.0)
        + 0.20 * minute.get('through_ask_share', 0.0)
        + 0.15 * minute.get('bid_up', 0.0)
        + 0.10 * minute.get('quote_imbalance', 0.0).clip(lower=0.0)
    ).clip(0.0, 1.0)
    minute['short_pulse'] = (
        0.55 * minute.get('aggressive_sell_share', 0.0)
        + 0.20 * minute.get('through_bid_share', 0.0)
        + 0.15 * minute.get('bid_down', 0.0)
        + 0.10 * (-minute.get('quote_imbalance', 0.0)).clip(lower=0.0)
    ).clip(0.0, 1.0)

    daily = (
        minute.groupby(['ticker', 'date'], sort=False)
        .agg(
            minute_rows=('minute', 'size'),
            quote_updates=('quote_updates', 'sum'),
            trade_count=('trade_count', 'sum'),
            trade_notional=('trade_notional', 'sum'),
            trade_size=('trade_size', 'sum'),
            aggressive_buy_notional=('aggressive_buy_notional', 'sum'),
            aggressive_sell_notional=('aggressive_sell_notional', 'sum'),
            through_ask_notional=('through_ask_notional', 'sum'),
            through_bid_notional=('through_bid_notional', 'sum'),
            aggressive_buy_share_mean=('aggressive_buy_share', 'mean'),
            aggressive_sell_share_mean=('aggressive_sell_share', 'mean'),
            through_ask_share_mean=('through_ask_share', 'mean'),
            through_bid_share_mean=('through_bid_share', 'mean'),
            net_aggression_mean=('net_aggression', 'mean'),
            bid_up_share=('bid_up', 'mean'),
            bid_down_share=('bid_down', 'mean'),
            ask_up_share=('ask_up', 'mean'),
            ask_down_share=('ask_down', 'mean'),
            spread_mean=('spread_mean', 'mean'),
            spread_last=('spread_last', 'last'),
            quote_imbalance_mean=('quote_imbalance', 'mean'),
            quote_imbalance_close=('quote_imbalance', 'last'),
            long_pulse_mean=('long_pulse', 'mean'),
            short_pulse_mean=('short_pulse', 'mean'),
        )
        .reset_index()
    )

    notional_denom = daily['trade_notional'].replace(0.0, np.nan)
    aggr_denom = (daily['aggressive_buy_notional'] + daily['aggressive_sell_notional']).replace(0.0, np.nan)
    daily['aggressive_buy_share'] = (daily['aggressive_buy_notional'] / notional_denom).clip(0.0, 1.0).fillna(0.0)
    daily['aggressive_sell_share'] = (daily['aggressive_sell_notional'] / notional_denom).clip(0.0, 1.0).fillna(0.0)
    daily['through_ask_share'] = (daily['through_ask_notional'] / notional_denom).clip(0.0, 1.0).fillna(0.0)
    daily['through_bid_share'] = (daily['through_bid_notional'] / notional_denom).clip(0.0, 1.0).fillna(0.0)
    daily['net_aggression'] = ((daily['aggressive_buy_notional'] - daily['aggressive_sell_notional']) / (aggr_denom + EPS)).clip(-1.0, 1.0).fillna(0.0)

    minute_groups = minute.groupby(['ticker', 'date'], sort=False)
    daily['long_pressure_minutes'] = minute_groups['long_pulse'].apply(lambda s: float((s >= 0.55).mean())).reset_index(level=[0,1], drop=True).values
    daily['short_pressure_minutes'] = minute_groups['short_pulse'].apply(lambda s: float((s >= 0.55).mean())).reset_index(level=[0,1], drop=True).values
    daily['long_max_streak'] = minute_groups['long_pulse'].apply(lambda s: longest_streak(s >= 0.55)).reset_index(level=[0,1], drop=True).values
    daily['short_max_streak'] = minute_groups['short_pulse'].apply(lambda s: longest_streak(s >= 0.55)).reset_index(level=[0,1], drop=True).values
    daily['long_streak_ratio'] = (daily['long_max_streak'] / daily['minute_rows'].replace(0, np.nan)).clip(0.0, 1.0).fillna(0.0)
    daily['short_streak_ratio'] = (daily['short_max_streak'] / daily['minute_rows'].replace(0, np.nan)).clip(0.0, 1.0).fillna(0.0)

    imbalance_long = ((daily['quote_imbalance_mean'] + 1.0) / 2.0).clip(0.0, 1.0)
    imbalance_short = ((-daily['quote_imbalance_mean'] + 1.0) / 2.0).clip(0.0, 1.0)
    book_lift_long = (daily['bid_up_share'] + daily['ask_up_share']) / 2.0
    book_drop_short = (daily['bid_down_share'] + daily['ask_down_share']) / 2.0

    daily['pressure_velocity_long_base'] = (
        0.40 * daily['aggressive_buy_share']
        + 0.20 * daily['through_ask_share']
        + 0.20 * daily['long_pressure_minutes']
        + 0.10 * daily['long_streak_ratio']
        + 0.10 * ((book_lift_long + imbalance_long) / 2.0).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    daily['pressure_velocity_short_base'] = (
        0.40 * daily['aggressive_sell_share']
        + 0.20 * daily['through_bid_share']
        + 0.20 * daily['short_pressure_minutes']
        + 0.10 * daily['short_streak_ratio']
        + 0.10 * ((book_drop_short + imbalance_short) / 2.0).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    return daily


def write_status(path: Path, *, trades_path: Path, options_path: Path | None, overlay: pd.DataFrame, options_merged: bool) -> None:
    latest_date = overlay['date'].max() if not overlay.empty else pd.NaT
    status = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'mode': 'trades_only_fallback',
        'trades_path': str(trades_path),
        'options_path': str(options_path) if options_path else None,
        'rows': int(len(overlay)),
        'tickers': int(overlay['ticker'].nunique()) if not overlay.empty else 0,
        'date_min': str(overlay['date'].min().date()) if not overlay.empty else None,
        'date_max': str(latest_date.date()) if not overlay.empty and pd.notna(latest_date) else None,
        'options_merged': bool(options_merged),
    }
    path.write_text(json.dumps(status, indent=2), encoding='utf-8')


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

    root = runtime_project_root(args.project_root)
    base_dir = data_dir(root)
    trades_path = resolve_existing_input(base_dir, args.trades_path, ['polygon_trades.parquet', 'intraday_trades.parquet', 'trades.parquet', 'trades.csv'], 'trades')
    output_path = resolve_path(args.output_parquet, base_dir, 'cleanroom_pressure_velocity_overlay.parquet')
    status_path = resolve_path(args.status_output, base_dir, 'cleanroom_pressure_velocity_overlay_status.json')
    latest_long_csv = resolve_path(args.latest_long_csv, base_dir, 'cleanroom_pressure_velocity_latest_long.csv')
    latest_short_csv = resolve_path(args.latest_short_csv, base_dir, 'cleanroom_pressure_velocity_latest_short.csv')

    options_path = None
    if args.options_path is not None:
        options_path = resolve_path(args.options_path, base_dir, args.options_path.name)
    else:
        candidate = base_dir / 'cleanroom_options_overlay_profiled.parquet'
        if candidate.exists():
            options_path = candidate

    logging.info('Loading trades from %s', trades_path)
    trades_raw = load_frame(trades_path)
    trades = normalize_trades(trades_raw, max_tickers=args.max_tickers, session_date=args.session_date)
    if trades.empty:
        raise RuntimeError('No usable trade rows were found after cleaning and regular-session filtering.')

    logging.info('Normalized trades: %s rows across %s tickers', f'{len(trades):,}', f'{trades["ticker"].nunique():,}')
    trades_minute = build_trades_minute_fallback(trades)
    quotes_minute = build_quote_proxy_minutes(trades_minute)
    overlay = build_daily_overlay(quotes_minute, trades_minute)
    overlay, options_merged = merge_options_confirmation(overlay, options_path)
    overlay = add_rebalance_states(overlay)
    overlay = overlay.sort_values(['ticker', 'date'], kind='mergesort').reset_index(drop=True)

    latest_date = overlay['date'].max()
    latest = overlay[overlay['date'] == latest_date].copy()
    latest_long = latest.sort_values(['hold_extension_long', 'pressure_velocity_long', 'exit_warning_long'], ascending=[False, False, True]).head(50)
    latest_short = latest.sort_values(['hold_extension_short', 'pressure_velocity_short', 'exit_warning_short'], ascending=[False, False, True]).head(50)

    if args.dry_run:
        logging.info('Dry run complete. Latest session date: %s', latest_date.date())
        logging.info('Top long hold-extension candidates:\n%s', latest_long[['ticker', 'hold_extension_long', 'rebalance_state_long']].to_string(index=False))
        logging.info('Top short hold-extension candidates:\n%s', latest_short[['ticker', 'hold_extension_short', 'rebalance_state_short']].to_string(index=False))
        return

    overlay.to_parquet(output_path, index=False)
    latest_long.to_csv(latest_long_csv, index=False)
    latest_short.to_csv(latest_short_csv, index=False)
    write_status(status_path, trades_path=trades_path, options_path=options_path, overlay=overlay, options_merged=options_merged)

    logging.info('Saved pressure-velocity overlay to %s', output_path)
    logging.info('Saved latest long view to %s', latest_long_csv)
    logging.info('Saved latest short view to %s', latest_short_csv)
    logging.info('Saved status to %s', status_path)


if __name__ == '__main__':
    main()
