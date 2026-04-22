import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class PipelineConfig:
    filter_lag: int = 2
    conditioning_lag: int = 2

    min_price: float = 10.0
    min_adv_volume: float = 300_000.0
    adv_window: int = 22

    min_history_days: int = 60
    min_history_for_vol: int = 22

    trailing_horizons: List[int] = field(default_factory=lambda: [5, 21, 63])
    forward_horizons: List[int] = field(default_factory=lambda: [20, 60, 90])

    vol_window: int = 22
    vol_floor: float = 0.005

    norm_window: int = 63
    norm_min_periods: int = 20
    norm_min_stocks: int = 10
    winsorize_mad_k: float = 5.0

    sector_missing_warn_frac: float = 0.0025
    sector_missing_fail_frac: float = 0.02
    sector_conflict_warn_tickers: int = 5
    sector_conflict_fail_frac: float = 0.01

    has_unadjusted_price: bool = False


ALIASES = {
    "ticker": ["ticker", "symbol", "sym", "stock", "name", "permno", "id"],
    "date": ["date", "trade_date", "dt", "timestamp", "asofdate"],
    "close": ["close", "adj_close", "adjusted_close", "price", "px", "prc", "adjprc", "adj_price"],
    "close_unadj": ["close_unadj", "unadjusted_close", "raw_close", "rawprc", "unadj_close", "prcraw"],
    "volume": ["volume", "vol", "trading_volume", "shares_traded"],
    "sector": ["sector", "gics_sector", "sect", "sector_name"],
    "industry": ["industry", "gics_industry", "industry_name", "indgrp", "industry_group"],
    "mkt_cap": ["mkt_cap", "market_cap", "marketcap", "cap", "market_value"],
}

REQUIRED_COLUMNS = ["ticker", "date", "close"]


def fail(message: str) -> None:
    sys.exit(f"\nERROR: {message}\n")


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    lower_map = {c.lower().replace(" ", "_"): c for c in df.columns}
    mapping = {}
    for key, aliases in ALIASES.items():
        for alias in aliases:
            if alias in lower_map:
                mapping[key] = lower_map[alias]
                break
    return mapping


def standardize_columns(df: pd.DataFrame, col_map: Dict[str, str]) -> pd.DataFrame:
    reverse = {v: k for k, v in col_map.items()}
    out = df.rename(columns=reverse).copy()

    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        fail(
            f"Missing required columns: {missing}\n"
            f"Detected mapping: {col_map}\n"
            f"Available columns: {list(df.columns)}"
        )

    out["ticker"] = out["ticker"].astype(str).str.strip()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    bad_dates = int(out["date"].isna().sum())
    if bad_dates > 0:
        fail(f"{bad_dates:,} rows have invalid dates.")

    for col in ["close", "close_unadj", "volume", "mkt_cap"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["sector", "industry"]:
        if col in out.columns:
            out[col] = (
                out[col]
                .astype("string")
                .str.strip()
                .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
            )

    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    dupes = int(out.duplicated(subset=["ticker", "date"]).sum())
    if dupes > 0:
        fail(f"Found {dupes:,} duplicate (ticker, date) rows. Fix raw data first.")

    return out


def winsorize_by_group_mad(series: pd.Series, groups: pd.Series, k: float) -> pd.Series:
    med = series.groupby(groups).transform("median")
    mad = (series - med).abs().groupby(groups).transform("median")
    lo = med - k * mad
    hi = med + k * mad

    out = series.copy()
    mask = mad.notna() & (mad > 0) & series.notna()
    out.loc[mask] = np.minimum(np.maximum(series.loc[mask], lo.loc[mask]), hi.loc[mask])
    return out


def resolve_sector_history(df: pd.DataFrame, cfg: PipelineConfig) -> Tuple[pd.DataFrame, Dict]:
    out = df.copy()

    if "sector" not in out.columns:
        out["sector"] = pd.Series(pd.NA, index=out.index, dtype="string")

    original_missing = out["sector"].isna()

    non_null_sector = out.loc[out["sector"].notna(), ["ticker", "sector"]].copy()
    if non_null_sector.empty:
        summary = {
            "rows_total": int(len(out)),
            "original_missing_sector_rows": int(original_missing.sum()),
            "filled_from_ticker_history_rows": 0,
            "remaining_missing_sector_rows": int(original_missing.sum()),
            "conflict_ticker_count": 0,
            "conflict_tickers": [],
            "missing_sector_fraction_after_fill": 1.0 if len(out) else 0.0,
        }
        fail("Sector column exists but all sector values are missing.")

    sector_uniques = (
        non_null_sector.groupby("ticker")["sector"]
        .nunique(dropna=True)
        .rename("sector_unique_count")
    )

    consistent_tickers = sector_uniques[sector_uniques == 1].index
    conflict_tickers = sorted(sector_uniques[sector_uniques > 1].index.tolist())

    canonical_sector = (
        non_null_sector[non_null_sector["ticker"].isin(consistent_tickers)]
        .groupby("ticker")["sector"]
        .first()
    )

    out["sector_filled_from_ticker_history"] = False
    fillable = out["sector"].isna() & out["ticker"].isin(canonical_sector.index)
    if fillable.any():
        out.loc[fillable, "sector"] = out.loc[fillable, "ticker"].map(canonical_sector)
        out.loc[fillable, "sector_filled_from_ticker_history"] = True

    out["sector_conflict_flag"] = out["ticker"].isin(conflict_tickers)

    remaining_missing = int(out["sector"].isna().sum())
    missing_frac = remaining_missing / len(out) if len(out) else 0.0
    conflict_frac = len(conflict_tickers) / out["ticker"].nunique() if out["ticker"].nunique() else 0.0

    if len(conflict_tickers) > 0:
        print(f"WARNING: {len(conflict_tickers):,} ticker(s) have conflicting sector history.")
        print("         They will be flagged and carried forward without guessed sector backfill.")
        preview = conflict_tickers[:20]
        print(f"         Example conflicting tickers: {preview}")

    if missing_frac >= cfg.sector_missing_fail_frac:
        fail(
            f"Remaining missing sector rows after same-ticker fill are too high: "
            f"{remaining_missing:,} / {len(out):,} ({missing_frac:.2%})."
        )

    if conflict_frac >= cfg.sector_conflict_fail_frac:
        fail(
            f"Conflicting sector history is too widespread: "
            f"{len(conflict_tickers):,} ticker(s) / {out['ticker'].nunique():,} ({conflict_frac:.2%})."
        )

    if missing_frac >= cfg.sector_missing_warn_frac:
        print(
            f"WARNING: Remaining missing sector rows after fill: "
            f"{remaining_missing:,} / {len(out):,} ({missing_frac:.2%})."
        )

    if len(conflict_tickers) >= cfg.sector_conflict_warn_tickers:
        print(
            f"WARNING: Conflicting sector tickers reached warning threshold: "
            f"{len(conflict_tickers):,}."
        )

    summary = {
        "rows_total": int(len(out)),
        "original_missing_sector_rows": int(original_missing.sum()),
        "filled_from_ticker_history_rows": int(out["sector_filled_from_ticker_history"].sum()),
        "remaining_missing_sector_rows": remaining_missing,
        "conflict_ticker_count": int(len(conflict_tickers)),
        "conflict_tickers": conflict_tickers,
        "missing_sector_fraction_after_fill": float(missing_frac),
    }
    return out, summary


def add_basic_quality_flags(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    out = df.copy()

    out["_bad_price"] = out["close"].isna() | (out["close"] <= 0)

    if "volume" in out.columns:
        out["_bad_volume"] = out["volume"].isna() | (out["volume"] <= 0)
    else:
        out["_bad_volume"] = False

    out["_qc_fail"] = out["_bad_price"] | out["_bad_volume"]
    out["_qc_fail_lagged"] = (
        out.groupby("ticker")["_qc_fail"]
        .shift(cfg.filter_lag)
        .astype("boolean")
        .fillna(True)
        .astype(bool)
    )

    out.loc[out["close"] <= 0, "close"] = np.nan
    return out


def add_lagged_conditioning_variables(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    out = df.copy()

    price_col = "close_unadj" if "close_unadj" in out.columns else "close"
    out["price_filter_lagged"] = out.groupby("ticker")[price_col].shift(cfg.filter_lag)

    if "volume" in out.columns:
        shifted_volume = out.groupby("ticker")["volume"].shift(cfg.conditioning_lag)
        out["adv_22d_lagged"] = shifted_volume.groupby(out["ticker"]).transform(
            lambda s: s.rolling(cfg.adv_window, min_periods=max(10, cfg.adv_window // 2)).mean()
        )
    else:
        out["adv_22d_lagged"] = np.nan

    log_close = np.log(out["close"])
    log_ret_1d = log_close.groupby(out["ticker"]).diff()
    shifted_log_ret = log_ret_1d.groupby(out["ticker"]).shift(cfg.conditioning_lag)

    out["vol_22d_lagged"] = shifted_log_ret.groupby(out["ticker"]).transform(
        lambda s: s.rolling(cfg.vol_window, min_periods=cfg.min_history_for_vol).std()
    )
    out["vol_22d_lagged"] = out["vol_22d_lagged"].clip(lower=cfg.vol_floor)
    out["vol_22d_lagged_ann"] = out["vol_22d_lagged"] * np.sqrt(252.0)

    clean_lagged = ~out["_qc_fail_lagged"]
    out["_has_min_history"] = clean_lagged.groupby(out["ticker"]).cumsum() >= cfg.min_history_days

    return out


def add_universe_membership(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    out = df.copy()
    out["in_universe"] = (
        (out["price_filter_lagged"] >= cfg.min_price)
        & (out["adv_22d_lagged"] >= cfg.min_adv_volume)
        & out["_has_min_history"]
        & ~out["_qc_fail_lagged"]
    ).fillna(False)
    return out


def add_returns(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    out = df.copy()
    log_close = np.log(out["close"].astype(np.float64))
    g = log_close.groupby(out["ticker"])

    for h in cfg.trailing_horizons:
        out[f"ret_{h}d"] = (log_close - g.shift(h)).astype(np.float32)

    for h in cfg.forward_horizons:
        out[f"fwd_ret_{h}d"] = (g.shift(-h) - log_close).astype(np.float32)

    for h in cfg.trailing_horizons:
        out[f"ret_{h}d_vadj"] = (out[f"ret_{h}d"] / out["vol_22d_lagged"]).astype(np.float32)

    return out


def normalize_feature_by_sector(df: pd.DataFrame, col: str, cfg: PipelineConfig) -> pd.DataFrame:
    out = df.copy()

    valid = out["in_universe"] & out[col].notna() & out["sector"].notna()
    out[f"{col}_pctrank"] = np.nan
    out[f"{col}_zscore"] = np.nan

    if valid.sum() == 0:
        return out

    clip_groups = pd.Series(
        list(zip(out.loc[valid, "date"].values, out.loc[valid, "sector"].values)),
        index=out.index[valid],
    )

    clipped = winsorize_by_group_mad(
        pd.Series(out.loc[valid, col].values, index=out.index[valid]),
        clip_groups,
        cfg.winsorize_mad_k,
    )

    tmp = out.loc[valid, ["date", "sector", col]].copy()
    tmp[col] = clipped.values

    ranks = tmp.groupby(["date", "sector"])[col].rank(pct=True, method="average")
    counts = tmp.groupby(["date", "sector"])[col].transform("count")
    ranks = ranks.where(counts >= cfg.norm_min_stocks)
    out.loc[valid, f"{col}_pctrank"] = ranks.astype(np.float32).values

    daily = (
        tmp.groupby(["date", "sector"])[col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["sector", "date"])
    )
    daily.loc[daily["count"] < cfg.norm_min_stocks, ["mean", "std"]] = np.nan

    daily["lagged_mean"] = daily.groupby("sector")["mean"].transform(
        lambda s: s.shift(1).rolling(cfg.norm_window, min_periods=cfg.norm_min_periods).mean()
    )
    daily["lagged_std"] = daily.groupby("sector")["std"].transform(
        lambda s: s.shift(1).rolling(cfg.norm_window, min_periods=cfg.norm_min_periods).mean()
    )

    lookup = daily.set_index(["date", "sector"])[["lagged_mean", "lagged_std"]]
    reindexer = pd.MultiIndex.from_arrays(
        [out.loc[valid, "date"].values, out.loc[valid, "sector"].values],
        names=["date", "sector"],
    )
    params = lookup.reindex(reindexer)

    lm = params["lagged_mean"].values
    ls = params["lagged_std"].values
    vals = clipped.values.astype(np.float64)

    ok = ~np.isnan(lm) & ~np.isnan(ls) & (ls > 0)
    z = np.full(len(vals), np.nan, dtype=np.float32)
    z[ok] = ((vals[ok] - lm[ok]) / ls[ok]).astype(np.float32)

    out.loc[valid, f"{col}_zscore"] = z
    return out


def add_normalized_features(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    out = df.copy()
    feature_cols = [f"ret_{h}d" for h in cfg.trailing_horizons] + [f"ret_{h}d_vadj" for h in cfg.trailing_horizons]
    for col in feature_cols:
        out = normalize_feature_by_sector(out, col, cfg)
    return out


def add_sector_coverage_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sector_present"] = out["sector"].notna()
    out["sector_normalization_eligible"] = out["in_universe"] & out["sector"].notna()
    return out


def run_audit(df: pd.DataFrame, sector_summary: Dict, cfg: PipelineConfig) -> None:
    print("\nAUDIT")
    print("=" * 80)

    dupes = int(df.duplicated(subset=["ticker", "date"]).sum())
    print(f"Duplicate (ticker, date) rows:                    {dupes:,}")

    same_day_contam = int((df["in_universe"] & (df["close"].isna())).sum())
    print(f"In-universe rows with invalid same-day close:      {same_day_contam:,}")

    first_rows_blocked = int(df.groupby("ticker").head(cfg.filter_lag)["in_universe"].sum())
    print(f"In-universe rows in first {cfg.filter_lag} row(s):            {first_rows_blocked:,}")

    daily_universe = df.groupby("date")["in_universe"].sum()
    print(f"Daily universe median:                            {daily_universe.median():.0f}")
    print(f"Daily universe min:                               {daily_universe.min():.0f}")
    print(f"Daily universe max:                               {daily_universe.max():.0f}")

    for h in cfg.forward_horizons:
        col = f"fwd_ret_{h}d"
        pct_nan = df.groupby("ticker").tail(h)[col].isna().mean()
        print(f"Tail NaN rate for {col}:                          {pct_nan:.1%}")

    print("-" * 80)
    print(f"Original missing sector rows:                     {sector_summary['original_missing_sector_rows']:,}")
    print(f"Filled from same-ticker history:                  {sector_summary['filled_from_ticker_history_rows']:,}")
    print(f"Remaining missing sector rows:                    {sector_summary['remaining_missing_sector_rows']:,}")
    print(f"Conflicting sector-history tickers:               {sector_summary['conflict_ticker_count']:,}")
    print(f"Missing sector fraction after fill:               {sector_summary['missing_sector_fraction_after_fill']:.2%}")

    missing_in_universe = int((df["in_universe"] & df["sector"].isna()).sum())
    print(f"In-universe rows still missing sector:            {missing_in_universe:,}")

    elig = int(df["sector_normalization_eligible"].sum())
    in_uni = int(df["in_universe"].sum())
    elig_frac = (elig / in_uni) if in_uni else 0.0
    print(f"Sector normalization eligible in-universe rows:   {elig:,} / {in_uni:,} ({elig_frac:.2%})")

    print("=" * 80)


def build_metadata(df: pd.DataFrame, sector_summary: Dict, cfg: PipelineConfig) -> Dict:
    meta = {
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()),
        "date_min": str(df["date"].min().date()) if len(df) else None,
        "date_max": str(df["date"].max().date()) if len(df) else None,
        "trailing_horizons": cfg.trailing_horizons,
        "forward_horizons": cfg.forward_horizons,
        "adv_window": cfg.adv_window,
        "vol_window": cfg.vol_window,
        "min_price": cfg.min_price,
        "min_adv_volume": cfg.min_adv_volume,
        "sector_summary": sector_summary,
        "output_columns": list(df.columns),
    }
    return meta


def run_pipeline(df: pd.DataFrame, cfg: PipelineConfig) -> Tuple[pd.DataFrame, Dict]:
    out, sector_summary = resolve_sector_history(df, cfg)
    out = add_basic_quality_flags(out, cfg)
    out = add_lagged_conditioning_variables(out, cfg)
    out = add_universe_membership(out, cfg)
    out = add_returns(out, cfg)
    out = add_normalized_features(out, cfg)
    out = add_sector_coverage_flags(out)

    drop_cols = [c for c in out.columns if c.startswith("_")]
    out = out.drop(columns=drop_cols)
    return out, sector_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Quintic V3 normalization pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input parquet file")
    parser.add_argument("--output", "-o", default=None, help="Output parquet file")
    parser.add_argument("--audit", action="store_true", help="Print audit summary")
    parser.add_argument("--meta-out", default=None, help="Optional metadata JSON output path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        fail(f"Input file not found: {input_path}")

    print(f"Reading: {input_path}")
    df = pd.read_parquet(input_path)

    col_map = detect_columns(df)
    print(f"Detected columns: {col_map}")
    df = standardize_columns(df, col_map)

    cfg = PipelineConfig()
    out, sector_summary = run_pipeline(df, cfg)

    if args.audit:
        run_audit(out, sector_summary, cfg)

    output_path = Path(args.output) if args.output else Path(str(input_path).replace(".parquet", "_normalized.parquet"))
    out.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"\nSaved normalized file: {output_path}")
    print(f"Rows: {len(out):,} | Cols: {len(out.columns):,}")

    meta = build_metadata(out, sector_summary, cfg)
    meta_path = Path(args.meta_out) if args.meta_out else output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved metadata:        {meta_path}")

    print("\nCore output columns:")
    core_cols = [
        "ticker", "date", "close", "sector", "industry", "mkt_cap",
        "price_filter_lagged", "adv_22d_lagged", "vol_22d_lagged", "vol_22d_lagged_ann",
        "in_universe", "sector_present", "sector_filled_from_ticker_history",
        "sector_conflict_flag", "sector_normalization_eligible",
    ]
    core_cols += [f"ret_{h}d" for h in cfg.trailing_horizons]
    core_cols += [f"ret_{h}d_vadj" for h in cfg.trailing_horizons]
    core_cols += [f"ret_{h}d_pctrank" for h in cfg.trailing_horizons]
    core_cols += [f"ret_{h}d_zscore" for h in cfg.trailing_horizons]
    core_cols += [f"ret_{h}d_vadj_pctrank" for h in cfg.trailing_horizons]
    core_cols += [f"ret_{h}d_vadj_zscore" for h in cfg.trailing_horizons]
    core_cols += [f"fwd_ret_{h}d" for h in cfg.forward_horizons]

    for col in core_cols:
        if col in out.columns:
            print(f"  {col}")


if __name__ == "__main__":
    main()