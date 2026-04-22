"""
Quintic Labs — Technical Studies

Computes comprehensive technical indicators per ticker and cross-sectional
features within sector/industry groups. Designed to feed into the XGBoost
forecast model for 20/60/90 day probability prediction.

Features produced:
    - Momentum: cumulative returns, rate of change, MACD
    - Oscillators: RSI, Stochastic, CCI
    - Trend: SMAs, EMAs, slopes, distance from averages
    - Volume: OBV, ADL, volume ratio, volume trend
    - Volatility: ATR, Bollinger Bands, realized vol
    - Breakout: 20-day high/low flags
    - Cross-sectional: sector z-scores, percentile ranks

Usage:
    python technicals.py --input stocks_universe.parquet --output technicals.parquet
    python technicals.py -i Z:\\jvand\\QUINTIC_V3\\data\\stage\\stocks_universe.parquet

Requirements:
    pip install pandas pyarrow numpy
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ===================================================================
# CLI
# ===================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Quintic Labs — Technical Studies")
    parser.add_argument("--input", "-i", required=True, help="Input parquet with OHLCV data")
    parser.add_argument("--output", "-o", default=None,
                        help="Output parquet (default: <input_dir>/technicals.parquet)")
    parser.add_argument("--latest-csv", default=None,
                        help="Latest-day CSV snapshot (default: <input_dir>/technicals_latest.csv)")
    parser.add_argument("--max-tickers", type=int, default=0, help="Limit tickers for testing")
    return parser.parse_args()


# ===================================================================
# Indicator functions
# ===================================================================
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    signed_volume = np.where(direction > 0, volume, np.where(direction < 0, -volume, 0.0))
    return pd.Series(signed_volume, index=close.index, dtype="float64").cumsum()


def compute_adl(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    hl_range = (high - low).replace(0.0, np.nan)
    money_flow_mult = ((close - low) - (high - close)) / hl_range
    money_flow_mult = money_flow_mult.fillna(0.0)
    return (money_flow_mult * volume).cumsum()


def compute_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr_components = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1)
    return tr_components.max(axis=1)


def compute_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(period, min_periods=period).mean()
    mad = tp.rolling(period, min_periods=period).apply(
        lambda x: float(np.mean(np.abs(x - np.mean(x)))), raw=True
    )
    denom = (0.015 * mad).replace(0.0, np.nan)
    return (tp - sma_tp) / denom


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26,
                  signal: int = 9) -> tuple:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denom = float(((x - x_mean) ** 2).sum())

    def _slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        y = values.astype(float)
        return float(((x - x_mean) * (y - y.mean())).sum() / denom)

    return series.rolling(window, min_periods=window).apply(_slope, raw=True)


# ===================================================================
# Per-ticker technicals
# ===================================================================
def technicals_for_ticker(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    close = g["close"].astype("float64")
    high = g["high"].astype("float64")
    low = g["low"].astype("float64")
    volume = g["volume"].astype("float64")
    daily_ret = close.pct_change()
    log_volume = np.log(volume.replace(0.0, np.nan))

    # --- Momentum ---
    g["cum_ret_5d"] = close / close.shift(5) - 1.0
    g["cum_ret_10d"] = close / close.shift(10) - 1.0
    g["cum_ret_20d"] = close / close.shift(20) - 1.0
    g["cum_ret_60d"] = close / close.shift(60) - 1.0
    g["cum_ret_120d"] = close / close.shift(120) - 1.0

    # --- Oscillators ---
    g["rsi_14"] = compute_rsi(close, period=14)
    g["rsi_21"] = compute_rsi(close, period=21)

    roll_low_14 = low.rolling(14, min_periods=14).min()
    roll_high_14 = high.rolling(14, min_periods=14).max()
    stoch_range = (roll_high_14 - roll_low_14).replace(0.0, np.nan)
    g["stoch_k_14"] = 100.0 * (close - roll_low_14) / stoch_range
    g["stoch_d_3"] = g["stoch_k_14"].rolling(3, min_periods=3).mean()
    g["cci_20"] = compute_cci(high, low, close, period=20)

    # --- MACD ---
    macd_line, macd_signal, macd_hist = compute_macd(close, fast=12, slow=26, signal=9)
    g["macd_12_26"] = macd_line
    g["macd_signal_12_26_9"] = macd_signal
    g["macd_hist_12_26_9"] = macd_hist

    # --- Trend (moving averages) ---
    g["sma_10"] = close.rolling(10, min_periods=10).mean()
    g["sma_20"] = close.rolling(20, min_periods=20).mean()
    g["sma_50"] = close.rolling(50, min_periods=50).mean()
    g["sma_200"] = close.rolling(200, min_periods=200).mean()
    g["ema_10"] = close.ewm(span=10, adjust=False, min_periods=10).mean()
    g["ema_20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    g["ema_200"] = close.ewm(span=200, adjust=False, min_periods=200).mean()
    g["price_vs_sma10"] = close / g["sma_10"] - 1.0
    g["price_vs_sma20"] = close / g["sma_20"] - 1.0
    g["sma10_vs_sma20"] = g["sma_10"] / g["sma_20"] - 1.0
    g["sma20_slope_5"] = g["sma_20"] / g["sma_20"].shift(5) - 1.0
    g["ema10_slope_5"] = g["ema_10"] / g["ema_10"].shift(5) - 1.0
    g["dist_to_sma_50"] = (close - g["sma_50"]) / g["sma_50"]
    g["dist_to_ema_200"] = (close - g["ema_200"]) / g["ema_200"]
    g["price_above_200dma"] = (close > g["sma_200"]).astype("float64")
    g.loc[g["sma_200"].isna(), "price_above_200dma"] = np.nan

    # --- Volume ---
    g["vol_avg_20"] = volume.rolling(20, min_periods=20).mean()
    g["volume_ratio_20"] = volume / g["vol_avg_20"]
    g["volume_change_pct_5d"] = volume / volume.shift(5) - 1.0
    g["volume_trend_10d"] = rolling_slope(log_volume, 10)
    g["obv"] = compute_obv(close, volume)
    # OBV crosses zero — use 5-day difference normalized by avg volume
    g["obv_slope_5"] = (g["obv"] - g["obv"].shift(5)) / (g["vol_avg_20"] + 1.0)
    g["adl"] = compute_adl(high, low, close, volume)
    # ADL crosses zero — same normalization
    g["adl_slope_5"] = (g["adl"] - g["adl"].shift(5)) / (g["vol_avg_20"] + 1.0)

    # --- Volatility ---
    tr = compute_true_range(high, low, close)
    g["atr_14"] = tr.rolling(14, min_periods=14).mean()
    g["atr_pct_14"] = g["atr_14"] / close
    g["bb_mid_20"] = close.rolling(20, min_periods=20).mean()
    g["bb_std_20"] = close.rolling(20, min_periods=20).std()
    g["bb_upper_20"] = g["bb_mid_20"] + 2.0 * g["bb_std_20"]
    g["bb_lower_20"] = g["bb_mid_20"] - 2.0 * g["bb_std_20"]
    g["bb_width_20"] = (g["bb_upper_20"] - g["bb_lower_20"]) / g["bb_mid_20"]
    bb_denom = (g["bb_upper_20"] - g["bb_lower_20"]).replace(0.0, np.nan)
    g["bb_pos_20"] = (close - g["bb_lower_20"]) / bb_denom
    g["realized_vol_10"] = daily_ret.rolling(10, min_periods=10).std() * np.sqrt(252.0)
    g["realized_vol_20"] = daily_ret.rolling(20, min_periods=20).std() * np.sqrt(252.0)
    g["realized_vol_60"] = daily_ret.rolling(60, min_periods=60).std() * np.sqrt(252.0)

    # --- Breakout ---
    g["high_20"] = high.rolling(20, min_periods=20).max()
    g["low_20"] = low.rolling(20, min_periods=20).min()
    g["dist_from_20d_high"] = close / g["high_20"] - 1.0
    g["dist_from_20d_low"] = close / g["low_20"] - 1.0

    prior_high_20 = high.rolling(20, min_periods=20).max().shift(1)
    prior_low_20 = low.rolling(20, min_periods=20).min().shift(1)
    g["breakout_20d_high_flag"] = (close > prior_high_20).astype("float64")
    g["breakdown_20d_low_flag"] = (close < prior_low_20).astype("float64")
    g.loc[prior_high_20.isna(), "breakout_20d_high_flag"] = np.nan
    g.loc[prior_low_20.isna(), "breakdown_20d_low_flag"] = np.nan

    # --- ADX (trend strength) ---
    # WHY: ARLO needs to know if a trend will persist into 60/90 days.
    # RSI 70 + ADX 40 = strong trend likely to continue.
    # RSI 70 + ADX 15 = weak move likely to reverse.
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    plus_dm = (high - prev_high).clip(lower=0.0)
    minus_dm = (prev_low - low).clip(lower=0.0)
    # Zero out whichever is smaller
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)
    atr_14_smooth = tr.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean() / atr_14_smooth.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean() / atr_14_smooth.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    g["adx_14"] = dx.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
    g["plus_di_14"] = plus_di
    g["minus_di_14"] = minus_di
    # Directional pressure: +DI minus -DI, positive = bullish trend
    g["di_spread_14"] = plus_di - minus_di

    # --- Money Flow Index (volume-weighted RSI) ---
    # WHY: Separates price moves backed by volume conviction from thin moves.
    # High RSI + low MFI = price up without volume = likely to fade over 20-60 days.
    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume
    tp_diff = typical_price.diff()
    pos_flow = raw_money_flow.where(tp_diff > 0, 0.0)
    neg_flow = raw_money_flow.where(tp_diff < 0, 0.0)
    pos_flow_sum = pos_flow.rolling(14, min_periods=14).sum()
    neg_flow_sum = neg_flow.rolling(14, min_periods=14).sum()
    money_flow_ratio = pos_flow_sum / neg_flow_sum.replace(0.0, np.nan)
    g["mfi_14"] = 100.0 - (100.0 / (1.0 + money_flow_ratio))
    # RSI-MFI divergence: when these disagree, reversals are likely
    g["rsi_mfi_divergence"] = g["rsi_14"] - g["mfi_14"]

    # --- Volatility term structure ---
    # WHY: When short-term vol exceeds long-term, a regime change is happening.
    # vol_10/vol_60 > 1.5 = stress event, predicts higher future vol.
    # vol_10/vol_60 < 0.7 = unusual calm, often precedes explosive moves.
    g["vol_term_structure"] = g["realized_vol_10"] / g["realized_vol_60"].replace(0.0, np.nan)
    g["vol_term_structure_20_60"] = g["realized_vol_20"] / g["realized_vol_60"].replace(0.0, np.nan)

    # --- Momentum acceleration (second derivative) ---
    # WHY: A stock going up at a decreasing rate behaves very differently over
    # 60 days than one going up at an increasing rate. First derivative (momentum)
    # is already captured. This captures whether momentum is building or fading.
    g["mom_accel_5d"] = g["cum_ret_5d"] - g["cum_ret_5d"].shift(5)
    g["mom_accel_20d"] = g["cum_ret_20d"] - g["cum_ret_20d"].shift(20)

    # --- Volume-price divergence ---
    # WHY: When OBV trends down while price trends up, institutions are distributing.
    # This reliably precedes 20-60 day reversals.
    obv_slope_20 = rolling_slope(g["obv"], 20)
    price_slope_20 = rolling_slope(close, 20)
    # Normalize both to [-1, 1] range via sign * log(1+|slope|) for comparability
    g["vol_price_divergence_20d"] = np.sign(price_slope_20) * np.log1p(price_slope_20.abs()) - \
                                     np.sign(obv_slope_20) * np.log1p(obv_slope_20.abs())

    # --- Trend quality (R-squared of linear regression) ---
    # WHY: High R² = clean persistent trend ARLO can trust to continue.
    # Low R² = choppy action where mean-reversion signals dominate.
    def _rolling_r2(s, window):
        x = np.arange(window, dtype=float)
        x_mean = x.mean()
        ss_xx = float(((x - x_mean) ** 2).sum())
        def _r2(vals):
            if np.isnan(vals).any():
                return np.nan
            y = vals.astype(float)
            y_mean = y.mean()
            ss_yy = float(((y - y_mean) ** 2).sum())
            if ss_yy == 0:
                return 0.0
            ss_xy = float(((x - x_mean) * (y - y_mean)).sum())
            return (ss_xy ** 2) / (ss_xx * ss_yy)
        return s.rolling(window, min_periods=window).apply(_r2, raw=True)

    g["trend_r2_20d"] = _rolling_r2(close, 20)
    g["trend_r2_60d"] = _rolling_r2(close, 60)

    # --- 60-day channel position ---
    # WHY: Where the stock sits in its 60-day range directly predicts
    # mean reversion vs breakout behavior at 60/90 day horizons.
    high_60 = high.rolling(60, min_periods=60).max()
    low_60 = low.rolling(60, min_periods=60).min()
    channel_range_60 = (high_60 - low_60).replace(0.0, np.nan)
    g["channel_pos_60d"] = (close - low_60) / channel_range_60
    g["dist_from_60d_high"] = close / high_60 - 1.0
    g["dist_from_60d_low"] = close / low_60 - 1.0

    # --- VWAP distance (if available) ---
    # WHY: VWAP represents average institutional execution price.
    # Stocks persistently above VWAP have institutional accumulation.
    if "vwap" in g.columns:
        vwap = g["vwap"].astype("float64")
        g["vwap_distance"] = (close - vwap) / vwap
        vwap_dev = g["vwap_distance"]
        g["vwap_z_20d"] = (vwap_dev - vwap_dev.rolling(20, min_periods=20).mean()) / \
                           vwap_dev.rolling(20, min_periods=20).std().replace(0.0, np.nan)

    return g


# ===================================================================
# Cross-sectional features
# ===================================================================
def infer_group_cols(df: pd.DataFrame) -> tuple:
    sector_col = None
    industry_col = None
    for candidate in ("sector", "gics_sector"):
        if candidate in df.columns:
            sector_col = candidate
            break
    for candidate in ("industry", "gics_industry"):
        if candidate in df.columns:
            industry_col = candidate
            break
    return sector_col, industry_col


def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sector_col, industry_col = infer_group_cols(out)

    if sector_col is None:
        print("  No sector column found — skipping cross-sectional features")
        return out

    # Relative to sector mean
    rel_features = [
        "cum_ret_5d", "cum_ret_10d", "cum_ret_20d", "cum_ret_60d",
        "cum_ret_120d",
    ]
    for feature in rel_features:
        if feature not in out.columns:
            continue
        sector_mean = out.groupby(["date", sector_col])[feature].transform("mean")
        out[f"{feature}_rel_to_sector"] = out[feature] - sector_mean

    # Sector z-scores
    z_features = [
        "rsi_14", "rsi_21", "cci_20", "macd_hist_12_26_9",
        "realized_vol_20", "realized_vol_60",
        "dist_to_sma_50", "dist_to_ema_200",
        "volume_change_pct_5d", "volume_trend_10d",
        # New ARLO features
        "adx_14", "mfi_14", "rsi_mfi_divergence",
        "vol_term_structure", "mom_accel_5d", "mom_accel_20d",
        "trend_r2_20d", "trend_r2_60d", "channel_pos_60d",
        "vol_price_divergence_20d", "di_spread_14",
    ]
    for feature in z_features:
        if feature not in out.columns:
            continue
        sector_mean = out.groupby(["date", sector_col])[feature].transform("mean")
        sector_std = out.groupby(["date", sector_col])[feature].transform("std").replace(0.0, np.nan)
        out[f"{feature}_sector_z"] = (out[feature] - sector_mean) / sector_std

    # Cross-sectional and sector percentile ranks
    rank_map = {
        "cum_ret_20d": "ret_20d",
        "cum_ret_60d": "ret_60d",
        "cum_ret_120d": "ret_120d",
        # New: rank stocks by trend strength, momentum quality, volume conviction
        "adx_14": "adx",
        "mfi_14": "mfi",
        "trend_r2_60d": "trend_quality",
        "mom_accel_20d": "mom_accel",
    }
    for source, alias in rank_map.items():
        if source not in out.columns:
            continue
        out[f"cs_rank_{alias}"] = out.groupby("date")[source].rank(
            method="average", pct=True, ascending=False
        )
        out[f"sector_rank_{alias}"] = out.groupby(["date", sector_col])[source].rank(
            method="average", pct=True, ascending=False
        )
        if industry_col is not None:
            out[f"industry_rank_{alias}"] = out.groupby(["date", industry_col])[source].rank(
                method="average", pct=True, ascending=False
            )

    return out


# ===================================================================
# Main
# ===================================================================
def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Error: input file not found: {input_path}")

    output_path = (
        Path(args.output) if args.output
        else input_path.parent / "technicals.parquet"
    )
    latest_csv = (
        Path(args.latest_csv) if args.latest_csv
        else input_path.parent / "technicals_latest.csv"
    )

    t_start = time.time()

    # Load
    print(f"Loading {input_path} ...")
    df = pd.read_parquet(input_path)

    required = ["ticker", "date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"Error: missing required columns: {missing}")

    # Clean
    df = df.copy()
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
    print(f"Processing {n_tickers:,} tickers ...")

    # Compute per-ticker technicals
    parts = []
    for idx, (_, group) in enumerate(df.groupby("ticker", sort=False), start=1):
        parts.append(technicals_for_ticker(group))
        if idx % 250 == 0 or idx == n_tickers:
            print(f"  [{idx:,}/{n_tickers:,}] tickers")

    out_df = pd.concat(parts, axis=0).sort_values(
        ["ticker", "date"], kind="mergesort"
    ).reset_index(drop=True)

    # Cross-sectional features
    print("Computing cross-sectional features ...")
    out_df = add_cross_sectional_features(out_df)

    # Latest snapshot
    latest = out_df.groupby("ticker", sort=False).tail(1).copy()
    latest = latest.sort_values("ticker").reset_index(drop=True)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False, engine="pyarrow")
    latest.to_csv(latest_csv, index=False)

    elapsed = time.time() - t_start
    tech_cols = [c for c in out_df.columns if c not in required + ["sector", "industry",
                 "gics_sector", "gics_industry", "mkt_cap", "market_cap",
                 "vwap", "transactions", "timestamp", "otc", "asset_type",
                 "company_name", "in_universe"]]

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output: {output_path}")
    print(f"Latest: {latest_csv}")
    print(f"Rows: {len(out_df):,}")
    print(f"Tickers: {out_df['ticker'].nunique():,}")
    print(f"Technical features: {len(tech_cols)}")


if __name__ == "__main__":
    main()
