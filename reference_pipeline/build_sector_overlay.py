#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quintic_paths import data_dir, project_root

MIN_STD = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cleanroom sector studies from the transformed and OU panels."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--transformed-parquet", type=Path, default=None)
    parser.add_argument("--ou-parquet", type=Path, default=None)
    parser.add_argument("--output-parquet", type=Path, default=None)
    parser.add_argument("--latest-csv", type=Path, default=None)
    parser.add_argument("--status-output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def runtime_project_root(cli_root):
    if cli_root is not None:
        return cli_root.expanduser()
    return project_root()


def resolve_path(path, base_dir, default_name, root_dir=None):
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


def load_panel(path, columns=None):
    if not path.exists():
        raise FileNotFoundError(f"Required parquet not found: {path}")
    df = pd.read_parquet(path, columns=columns)
    return df.copy()


def safe_corr_matrix(df_window):
    cols = list(df_window.columns)
    n = len(cols)
    mat = np.full((n, n), np.nan)

    for i in range(n):
        mat[i, i] = 1.0

    stds = np.zeros(n)
    centered_vals = [None] * n
    finite_masks = [None] * n

    for i, c in enumerate(cols):
        vals = df_window[c].values.astype(float)
        finite_mask = np.isfinite(vals)
        if finite_mask.sum() < 3:
            continue
        m = np.mean(vals[finite_mask])
        c_vals = vals - m
        c_vals[~finite_mask] = 0.0
        std = np.sqrt(np.mean(c_vals[finite_mask] ** 2))
        stds[i] = std
        centered_vals[i] = c_vals
        finite_masks[i] = finite_mask

    for i in range(n):
        if stds[i] < MIN_STD:
            continue
        for j in range(i + 1, n):
            if stds[j] < MIN_STD:
                continue
            both_ok = finite_masks[i] & finite_masks[j]
            if both_ok.sum() < 3:
                continue
            cov = np.mean(centered_vals[i][both_ok] * centered_vals[j][both_ok])
            denom = stds[i] * stds[j]
            if denom < MIN_STD:
                continue
            corr = cov / denom
            corr = max(-1.0, min(1.0, corr))
            mat[i, j] = corr
            mat[j, i] = corr

    return pd.DataFrame(mat, index=cols, columns=cols)


def compute_contagion(sector_daily):
    pivot = sector_daily.pivot(index="date", columns="gics_sector", values="sector_return_1d").sort_index()
    sectors = list(pivot.columns)
    out_rows = []
    if len(sectors) < 2:
        return pd.DataFrame(columns=["date", "gics_sector", "sector_contagion_score_20d"])

    for idx in range(19, len(pivot)):
        window = pivot.iloc[idx - 19:idx + 1]
        corr = safe_corr_matrix(window)
        asof_date = pivot.index[idx]
        for sector in sectors:
            if sector not in corr.columns:
                continue
            series = corr.loc[sector].drop(labels=[sector], errors="ignore").dropna()
            score = float(series.abs().mean()) if not series.empty else np.nan
            out_rows.append({
                "date": asof_date,
                "gics_sector": sector,
                "sector_contagion_score_20d": score,
            })
    return pd.DataFrame(out_rows)


def main() -> int:
    args = parse_args()
    root = runtime_project_root(args.project_root)
    data_root = data_dir(root)
    data_root.mkdir(parents=True, exist_ok=True)

    transformed_path = resolve_path(args.transformed_parquet, data_root, "cleanroom_transformed_panel.parquet", root_dir=root)
    ou_path = resolve_path(args.ou_parquet, data_root, "cleanroom_ou_panel.parquet", root_dir=root)
    output_path = resolve_path(args.output_parquet, data_root, "cleanroom_sector_overlay.parquet", root_dir=root)
    latest_csv = resolve_path(args.latest_csv, data_root, "cleanroom_sector_overlay_latest.csv", root_dir=root)
    status_path = resolve_path(args.status_output, data_root, "cleanroom_sector_overlay_status.json", root_dir=root)

    print(f"Project root: {root}")
    print(f"Transformed parquet: {transformed_path}")
    print(f"OU parquet: {ou_path}")
    print(f"Output parquet: {output_path}")

    if args.dry_run:
        print("Dry run complete.")
        return 0

    base_cols = ["ticker", "date", "gics_sector", "log_return_1d",
                 "garman_klass_var_1d", "volume_robust_z", "trade_size_robust_z"]
    transformed = load_panel(transformed_path, columns=base_cols)
    transformed["date"] = pd.to_datetime(transformed["date"], errors="coerce")
    transformed = transformed.dropna(subset=["ticker", "date", "gics_sector"]).copy()

    sector_daily = (
        transformed.groupby(["date", "gics_sector"], as_index=False)
        .agg(
            sector_member_count=("ticker", "nunique"),
            sector_return_1d=("log_return_1d", "mean"),
            sector_volatility_1d=("garman_klass_var_1d", "mean"),
            sector_volume_shock_1d=("volume_robust_z", "mean"),
            sector_trade_size_shock_1d=("trade_size_robust_z", "mean"),
        )
        .sort_values(["gics_sector", "date"])
        .reset_index(drop=True)
    )

    market_daily = (
        transformed.groupby("date", as_index=False)
        .agg(market_return_1d=("log_return_1d", "mean"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    market_daily["market_return_20d_mean"] = market_daily["market_return_1d"].rolling(20, min_periods=20).mean()
    sector_daily = sector_daily.merge(market_daily, on="date", how="left")

    sector_daily["sector_return_20d_mean"] = (
        sector_daily.groupby("gics_sector", sort=False)["sector_return_1d"]
        .transform(lambda s: s.rolling(20, min_periods=20).mean())
    )
    sector_daily["sector_relative_strength_20d"] = (
        sector_daily["sector_return_20d_mean"] - sector_daily["market_return_20d_mean"]
    )
    sector_daily["sector_realized_vol_20d"] = (
        sector_daily.groupby("gics_sector", sort=False)["sector_return_1d"]
        .transform(lambda s: s.rolling(20, min_periods=20).std())
    )
    sector_daily["sector_high_risk_flag"] = (sector_daily["sector_return_1d"] < -0.02).astype(int)

    ou_df = load_panel(ou_path)
    ou_df["date"] = pd.to_datetime(ou_df["date"], errors="coerce")
    ou_stretch_cols = [c for c in ou_df.columns if "_stretch_" in c and c.startswith("ou_")]
    if not ou_stretch_cols:
        print("  WARNING: No OU stretch columns found")

    sector_lookup = transformed[["ticker", "date", "gics_sector"]].drop_duplicates()
    ou_sector = ou_df[["ticker", "date"] + ou_stretch_cols].merge(
        sector_lookup, on=["ticker", "date"], how="left"
    )
    rename_dict = {col: f"sector_{col}" for col in ou_stretch_cols}
    ou_daily = (
        ou_sector.dropna(subset=["gics_sector"])
        .groupby(["date", "gics_sector"], as_index=False)
        [ou_stretch_cols].mean()
        .rename(columns=rename_dict)
    )

    sector_daily = sector_daily.merge(ou_daily, on=["date", "gics_sector"], how="left")

    print("Computing contagion scores ...")
    contagion = compute_contagion(sector_daily)
    sector_daily = sector_daily.merge(contagion, on=["date", "gics_sector"], how="left")
    sector_daily = sector_daily.sort_values(["date", "gics_sector"]).reset_index(drop=True)

    latest_date = sector_daily["date"].max()
    latest_df = sector_daily.loc[sector_daily["date"] == latest_date].copy()
    latest_df = latest_df.sort_values("sector_relative_strength_20d", ascending=False)

    numeric = sector_daily.select_dtypes(include=[np.number]).columns
    inf_count = int(np.isinf(sector_daily[numeric].values).sum())
    if inf_count > 0:
        print(f"  Replacing {inf_count} inf values with NaN")
        sector_daily[numeric] = sector_daily[numeric].replace([np.inf, -np.inf], np.nan)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sector_daily.to_parquet(output_path, index=False)
    latest_df.to_csv(latest_csv, index=False)

    status = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "transformed_parquet": str(transformed_path),
        "ou_parquet": str(ou_path),
        "output_parquet": str(output_path),
        "latest_csv": str(latest_csv),
        "rows_written": int(len(sector_daily)),
        "sectors": sorted(sector_daily["gics_sector"].dropna().astype(str).unique().tolist()),
        "latest_date": str(pd.to_datetime(latest_date).date()) if pd.notna(latest_date) else None,
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    print(f"Saved sector overlay: {output_path}")
    print(f"Saved latest sector CSV: {latest_csv}")
    print(f"Saved status file: {status_path}")
    print(f"Rows: {len(sector_daily):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
