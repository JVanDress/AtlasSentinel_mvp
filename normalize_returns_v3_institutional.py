"""
normalize_returns_v3_institutional.py
-------------------------------------
Institutional-grade normalization pipeline for equity return features.

For each trailing horizon N in trailing_horizons, produces:
  ret_{N}d                  raw log return
  ret_{N}d_voladj           vol-adjusted return = ret / (vol_22d * sqrt(N))
  ret_{N}d_zscore           z-score of MAD-winsorized raw return by (date, sector)
  ret_{N}d_voladj_zscore    z-score of MAD-winsorized vol-adjusted return by (date, sector)
  ret_{N}d_pctrank          percentile rank of raw return by (date, sector)
  ret_{N}d_voladj_pctrank   percentile rank of vol-adjusted return by (date, sector)

Forward returns are preserved raw as targets:
  fwd_ret_{N}d

Quality gates (enabled by default) fail the run when core diagnostics degrade
below configured thresholds. Audit JSON is written beside the output parquet.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PipelineConfig:
    # Lag conventions
    filter_lag: int = 2
    conditioning_lag: int = 2

    # Universe filters
    min_price: float = 10.0
    min_adv_dollar: float = 1_000_000.0
    adv_window: int = 22

    # Eligibility history
    min_history_days: int = 60

    # Volatility inputs
    vol_window: int = 22
    vol_floor: float = 0.005

    # Return horizons
    trailing_horizons: List[int] = field(default_factory=lambda: [5, 21, 63])
    forward_horizons: List[int] = field(default_factory=lambda: [20, 60, 90])

    # Cross-sectional normalization
    norm_min_stocks: int = 10
    winsorize_mad_k: float = 5.0

    # Quality gates (percent values)
    min_universe_coverage_pct: float = 15.0
    min_sector_coverage_in_universe_pct: float = 80.0
    max_z_nan_rate_pct: float = 35.0


def fail(message: str) -> None:
    sys.exit(f"\nERROR: {message}\n")


def pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def safe_stat(value: float) -> Optional[float]:
    if pd.isna(value):
        return None
    return float(value)


def validate_schema(df: pd.DataFrame) -> None:
    required = {"ticker", "date", "close", "volume", "sector"}
    missing = sorted(required - set(df.columns))
    if missing:
        fail(f"Missing required columns: {missing}")


def validate_numeric_inputs(df: pd.DataFrame) -> None:
    # Convert to numeric first to make checks deterministic.
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    if df["close"].isna().any():
        fail(f"'close' contains non-numeric or missing values: {int(df['close'].isna().sum()):,} rows")

    non_positive_close = int((df["close"] <= 0).sum())
    if non_positive_close:
        fail(f"'close' must be > 0 for log returns; found {non_positive_close:,} rows")

    negative_volume = int((df["volume"] < 0).sum(skipna=True))
    if negative_volume:
        fail(f"'volume' must be >= 0; found {negative_volume:,} rows")


def winsorize_group(tmp: pd.DataFrame, value_col: str, k: float) -> pd.Series:
    med = tmp.groupby(["date", "sector"], sort=False)[value_col].transform("median")
    mad = (tmp[value_col] - med).abs().groupby([tmp["date"], tmp["sector"]]).transform("median")
    lo = med - k * mad
    hi = med + k * mad
    return tmp[value_col].clip(lower=lo, upper=hi)


def zscore_group(tmp: pd.DataFrame, value_col: str, min_count: int) -> pd.Series:
    mean = tmp.groupby(["date", "sector"], sort=False)[value_col].transform("mean")
    std = tmp.groupby(["date", "sector"], sort=False)[value_col].transform("std")
    count = tmp.groupby(["date", "sector"], sort=False)[value_col].transform("count")
    z = (tmp[value_col] - mean) / std.where(std > 0)
    return z.where(count >= min_count)


def pctrank_group(tmp: pd.DataFrame, value_col: str, min_count: int) -> pd.Series:
    rank = tmp.groupby(["date", "sector"], sort=False)[value_col].rank(pct=True)
    count = tmp.groupby(["date", "sector"], sort=False)[value_col].transform("count")
    return rank.where(count >= min_count)


def normalize_columns(
    df: pd.DataFrame,
    base_cols: List[str],
    cfg: PipelineConfig,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Optional[float]]], pd.Series, pd.Series]:
    required_runtime_cols = {"in_universe", "date", "sector"}
    missing_runtime = sorted(required_runtime_cols - set(df.columns))
    if missing_runtime:
        fail(f"normalize_columns missing required columns: {missing_runtime}")
    if not base_cols:
        fail("normalize_columns received empty base_cols; no trailing features were generated.")

    missing_base = sorted([c for c in base_cols if c not in df.columns])
    if missing_base:
        fail(f"normalize_columns missing base feature columns: {missing_base}")

    sector_key = df["sector"].astype("string").str.strip()
    sector_key = sector_key.where(sector_key.notna() & (sector_key != ""), pd.NA)

    universe_mask = df["in_universe"] & sector_key.notna()
    audit_cols: Dict[str, Dict[str, Optional[float]]] = {}

    for col in base_cols:
        z_col = f"{col}_zscore"
        p_col = f"{col}_pctrank"

        df[z_col] = np.nan
        df[p_col] = np.nan

        valid = universe_mask & df[col].notna()
        n_valid = int(valid.sum())
        if n_valid == 0:
            audit_cols[col] = {
                "n_valid": 0,
                "z_mean": None,
                "z_std": None,
                "z_nan_rate": None,
                "pct_min": None,
                "pct_max": None,
                "pct_nan_rate": None,
            }
            continue

        tmp = df.loc[valid, ["date", col]].copy()
        tmp["sector"] = sector_key.loc[valid].astype("string")
        tmp[col] = tmp[col].astype(np.float64)

        tmp[col] = winsorize_group(tmp, value_col=col, k=cfg.winsorize_mad_k)
        z = zscore_group(tmp, value_col=col, min_count=cfg.norm_min_stocks)

        rank_tmp = df.loc[valid, ["date", col]].copy()
        rank_tmp["sector"] = sector_key.loc[valid].astype("string")
        rank_tmp[col] = rank_tmp[col].astype(np.float64)
        p = pctrank_group(rank_tmp, value_col=col, min_count=cfg.norm_min_stocks)

        df.loc[tmp.index, z_col] = z.astype(np.float32)
        df.loc[rank_tmp.index, p_col] = p.astype(np.float32)

        audit_cols[col] = {
            "n_valid": n_valid,
            "z_mean": safe_stat(z.mean(skipna=True)),
            "z_std": safe_stat(z.std(skipna=True)),
            "z_nan_rate": float(z.isna().mean()),
            "pct_min": safe_stat(p.min(skipna=True)),
            "pct_max": safe_stat(p.max(skipna=True)),
            "pct_nan_rate": float(p.isna().mean()),
        }

    return df, audit_cols, universe_mask, sector_key


def run_quality_gates(
    cfg: PipelineConfig,
    audit: Dict[str, object],
    base_cols: List[str],
    enforce: bool,
) -> None:
    failures: List[str] = []

    universe_cov = float(audit.get("universe_coverage_pct", 0.0))
    sector_cov = float(audit.get("sector_coverage_in_universe_pct", 0.0))

    if universe_cov < cfg.min_universe_coverage_pct:
        failures.append(
            f"universe_coverage_pct={universe_cov:.2f}% < min_universe_coverage_pct={cfg.min_universe_coverage_pct:.2f}%"
        )
    if sector_cov < cfg.min_sector_coverage_in_universe_pct:
        failures.append(
            "sector_coverage_in_universe_pct="
            f"{sector_cov:.2f}% < min_sector_coverage_in_universe_pct={cfg.min_sector_coverage_in_universe_pct:.2f}%"
        )

    per_col = audit.get("per_column", {})
    for col in base_cols:
        stats = per_col.get(col)
        if not stats or stats["z_nan_rate"] is None:
            continue
        z_nan_pct = 100.0 * float(stats["z_nan_rate"])
        if z_nan_pct > cfg.max_z_nan_rate_pct:
            failures.append(
                f"{col}: z_nan_rate={z_nan_pct:.2f}% > max_z_nan_rate_pct={cfg.max_z_nan_rate_pct:.2f}%"
            )

    if failures and enforce:
        fail("Quality gates failed:\n  - " + "\n  - ".join(failures))

    if failures:
        print("\nWARNING: quality gates failed, but run continued (--disable-quality-gates).")
        for item in failures:
            print(f"  - {item}")


def main() -> None:
    default_input = r"Z:\jvand\QUINTIC_v3\data\stage\stocks_universe_merged.parquet"

    parser = argparse.ArgumentParser(description="Quintic V3 Institutional Normalization")
    parser.add_argument("-i", "--input", default=default_input, help="Input parquet file path")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output parquet file path (default: <input>_normalized_institutional.parquet)",
    )
    parser.add_argument(
        "--disable-quality-gates",
        action="store_true",
        help="Do not fail the run when quality gates are breached.",
    )
    parser.add_argument(
        "--min-adv-dollar",
        type=float,
        default=None,
        help="Override minimum lagged dollar ADV required for universe inclusion.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        fail(f"Input file not found: {input_path}")

    print(f"Reading: {input_path}")
    df = pd.read_parquet(input_path)
    cfg = PipelineConfig()
    if args.min_adv_dollar is not None:
        if args.min_adv_dollar <= 0:
            fail("--min-adv-dollar must be > 0")
        cfg.min_adv_dollar = float(args.min_adv_dollar)

    validate_schema(df)

    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        fail(f"Invalid dates found: {int(df['date'].isna().sum()):,} rows")

    validate_numeric_inputs(df)

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    if df.duplicated(subset=["ticker", "date"]).any():
        fail("Duplicate ticker/date rows found.")

    n_rows = len(df)
    print(f"Rows: {n_rows:,}  Tickers: {df['ticker'].nunique():,}  Dates: {df['date'].nunique():,}")
    if n_rows:
        print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")

    # Lagged filter/conditioning fields
    df["price_lag"] = df.groupby("ticker")["close"].shift(cfg.filter_lag)
    close_cond_lag = df.groupby("ticker")["close"].shift(cfg.conditioning_lag)
    vol_cond_lag = df.groupby("ticker")["volume"].shift(cfg.conditioning_lag)

    df["dollar_volume_lag"] = close_cond_lag * vol_cond_lag
    df["adv_22d"] = df.groupby("ticker")["dollar_volume_lag"].transform(
        lambda x: x.rolling(cfg.adv_window, min_periods=10).mean()
    )

    # Volatility from lagged daily log returns
    log_close_f64 = np.log(df["close"].astype(np.float64))
    daily_ret = log_close_f64.groupby(df["ticker"]).diff()
    ret_lag = daily_ret.groupby(df["ticker"]).shift(cfg.conditioning_lag)
    df["vol_22d"] = ret_lag.groupby(df["ticker"]).transform(
        lambda x: x.rolling(cfg.vol_window, min_periods=10).std()
    )
    df["vol_22d"] = df["vol_22d"].clip(lower=cfg.vol_floor)

    clean = (~df["price_lag"].isna()) & (~df["adv_22d"].isna()) & (~df["vol_22d"].isna())
    df["has_min_history"] = clean.groupby(df["ticker"]).cumsum() >= cfg.min_history_days

    df["in_universe"] = (
        (df["price_lag"] >= cfg.min_price)
        & (df["adv_22d"] >= cfg.min_adv_dollar)
        & df["has_min_history"]
    ).fillna(False)

    universe_rows = int(df["in_universe"].sum())
    print(f"In-universe rows: {universe_rows:,} ({pct(universe_rows, n_rows):.2f}%)")

    # Returns
    g = log_close_f64.groupby(df["ticker"])

    for h in cfg.trailing_horizons:
        if h <= 0:
            fail(f"All trailing horizons must be > 0; found {h}")
        df[f"ret_{h}d"] = (log_close_f64 - g.shift(h)).astype(np.float32)

    for h in cfg.forward_horizons:
        if h <= 0:
            fail(f"All forward horizons must be > 0; found {h}")
        df[f"fwd_ret_{h}d"] = (g.shift(-h) - log_close_f64).astype(np.float32)

    base_cols: List[str] = []
    for h in cfg.trailing_horizons:
        raw_col = f"ret_{h}d"
        voladj_col = f"ret_{h}d_voladj"
        if "vol_22d" not in df.columns:
            fail("vol_22d was not computed before vol-adjusted return generation.")
        scale = df["vol_22d"] * np.sqrt(h)
        df[voladj_col] = (df[raw_col] / scale.where(scale > 0)).astype(np.float32)
        base_cols.extend([raw_col, voladj_col])

    # Cross-sectional normalization
    df, audit_cols, universe_mask, sector_key = normalize_columns(df, base_cols, cfg)
    grp_sizes = df.loc[universe_mask].groupby([df.loc[universe_mask, "date"], sector_key.loc[universe_mask]]).size()
    audit_group_sizes = {
        "n_groups": int(len(grp_sizes)),
        "min": int(grp_sizes.min()) if len(grp_sizes) else None,
        "p10": int(grp_sizes.quantile(0.10)) if len(grp_sizes) else None,
        "p50": int(grp_sizes.quantile(0.50)) if len(grp_sizes) else None,
        "p90": int(grp_sizes.quantile(0.90)) if len(grp_sizes) else None,
        "max": int(grp_sizes.max()) if len(grp_sizes) else None,
        "n_below_min_stocks": int((grp_sizes < cfg.norm_min_stocks).sum()),
    }

    sector_covered_rows = int(universe_mask.sum())
    audit: Dict[str, object] = {
        "input_path": str(input_path),
        "config": asdict(cfg),
        "n_input_rows": n_rows,
        "n_output_rows": len(df),
        "n_tickers": int(df["ticker"].nunique()),
        "n_dates": int(df["date"].nunique()),
        "date_min": str(df["date"].min().date()) if n_rows else None,
        "date_max": str(df["date"].max().date()) if n_rows else None,
        "n_in_universe": universe_rows,
        "universe_coverage_pct": round(pct(universe_rows, n_rows), 4),
        "n_with_sector_in_universe": sector_covered_rows,
        "sector_coverage_in_universe_pct": round(pct(sector_covered_rows, universe_rows), 4),
        "group_sizes_date_sector": audit_group_sizes,
        "per_column": audit_cols,
    }

    run_quality_gates(
        cfg=cfg,
        audit=audit,
        base_cols=base_cols,
        enforce=not args.disable_quality_gates,
    )

    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(input_path.stem + "_normalized_institutional.parquet")
    )
    df.to_parquet(output_path, index=False, engine="pyarrow")

    audit["output_path"] = str(output_path)
    audit_path = output_path.with_name(output_path.stem + "__audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)

    print("\nNormalization complete.")
    print(f"Output: {output_path}")
    print(f"Audit:  {audit_path}")
    print(f"Rows:   {len(df):,}")
    print(f"Cols:   {len(df.columns):,}")
    print(f"In universe: {universe_rows:,} ({audit['universe_coverage_pct']:.2f}%)")
    print(
        "Group sizes (date, sector): "
        f"n={audit_group_sizes['n_groups']:,} "
        f"min={audit_group_sizes['min']} "
        f"p50={audit_group_sizes['p50']} "
        f"max={audit_group_sizes['max']} "
        f"below_{cfg.norm_min_stocks}={audit_group_sizes['n_below_min_stocks']}"
    )


if __name__ == "__main__":
    main()
