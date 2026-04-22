#!/usr/bin/env python3
"""
Quintic Labs — OU Features (Parallel)
Build causal rolling OU features on the enriched transformed panel.
Uses multiprocessing across all CPU cores for speed.
Memory-safe: cleans infs per-ticker, no full-dataframe sort/copy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure quintic_paths is importable by worker processes on Windows
sys.path.insert(0, str(Path(__file__).resolve().parent))
from quintic_paths import data_dir, load_simple_dotenv, project_root

EPSILON = 1e-12


# ═══════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build causal rolling OU features (parallel)."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--input-parquet", type=Path, default=None)
    parser.add_argument("--output-parquet", type=Path, default=None)
    parser.add_argument("--status-output", type=Path, default=None)
    parser.add_argument("--group-column", default="ticker")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--features", default="log_close,garman_klass_var_1d,volume_robust_z,trade_size_robust_z")
    parser.add_argument("--windows", default="21,63,126")
    parser.add_argument("--min-window-obs", type=int, default=15)
    parser.add_argument("--min-source-history-days", type=int, default=1200)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0 = cpu_count - 1)")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════
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
    return base_dir / candidate


def parse_csv_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_windows(value):
    windows = [int(item) for item in parse_csv_list(value)]
    if not windows:
        raise ValueError("At least one OU window is required.")
    return sorted(dict.fromkeys(windows))


def validate_source_history(df, date_column, source_path, min_history_days):
    dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
    if dates.empty:
        raise RuntimeError(f"No valid dates found in {source_path}")
    min_date = dates.min().normalize()
    max_date = dates.max().normalize()
    span_days = int((max_date - min_date).days)
    if span_days < min_history_days:
        raise RuntimeError(
            f"{source_path}: {span_days} calendar days found, need {min_history_days}."
        )
    return min_date, max_date, span_days


def load_panel(path, group_column, date_column):
    if not path.exists():
        raise FileNotFoundError(f"Required parquet not found: {path}")
    df = pd.read_parquet(path)
    required = {group_column, date_column, "log_close", "garman_klass_var_1d", "volume_robust_z"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}")
    df = df.copy()
    df[group_column] = df[group_column].astype(str).str.upper().str.strip()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[group_column, date_column]).copy()
    df = df[df[group_column].ne("")].copy()
    return df.sort_values([group_column, date_column]).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# OU CORE MATH
# ═══════════════════════════════════════════════════════════════════════════
def feature_transform(series, feature_name):
    values = pd.to_numeric(series, errors="coerce")
    if feature_name == "garman_klass_var_1d":
        return np.log1p(values.clip(lower=0)), "log1p"
    return values.astype(float), "identity"


def _invalid_result():
    return {
        "valid": False, "b": np.nan, "mu": np.nan, "sigma_eq": np.nan,
        "half_life": np.nan, "stretch": np.nan, "r_squared": np.nan,
    }


def estimate_ou_from_history(history, current_value, min_window_obs):
    hist = np.asarray(history, dtype=float)
    hist = hist[np.isfinite(hist)]
    if hist.size < max(3, int(min_window_obs)):
        return _invalid_result()

    x = hist[:-1]
    y = hist[1:]
    if x.size < 2 or y.size < 2:
        return _invalid_result()

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_centered = x - x_mean
    y_centered = y - y_mean
    var_x = float(np.dot(x_centered, x_centered))

    if var_x <= EPSILON:
        return _invalid_result()

    cov_xy = float(np.dot(x_centered, y_centered))
    b = cov_xy / var_x

    if not np.isfinite(b) or b <= 0.0 or b >= 1.0:
        return _invalid_result()

    a = y_mean - (b * x_mean)
    mu = a / (1.0 - b)
    resid = y - (a + b * x)
    sigma_eps = float(np.std(resid, ddof=1)) if resid.size > 1 else np.nan
    if not np.isfinite(sigma_eps):
        sigma_eps = np.nan

    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(y_centered, y_centered))
    if ss_tot > EPSILON:
        r_squared = max(0.0, min(1.0, 1.0 - (ss_res / ss_tot)))
    else:
        r_squared = 0.0

    denom = 1.0 - (b ** 2)
    if denom <= EPSILON or not np.isfinite(sigma_eps):
        sigma_eq = np.nan
    else:
        sigma_eq = sigma_eps / np.sqrt(denom)

    half_life = np.log(2.0) / (-np.log(b))

    if np.isfinite(current_value) and np.isfinite(mu) and np.isfinite(sigma_eq) and sigma_eq > EPSILON:
        stretch = (current_value - mu) / sigma_eq
    else:
        stretch = np.nan

    return {
        "valid": bool(np.isfinite(stretch)),
        "b": float(b), "mu": float(mu),
        "sigma_eq": float(sigma_eq) if np.isfinite(sigma_eq) else np.nan,
        "half_life": float(half_life) if np.isfinite(half_life) else np.nan,
        "stretch": float(stretch) if np.isfinite(stretch) else np.nan,
        "r_squared": float(r_squared),
    }


def build_ou_columns(series, window, min_window_obs):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    length = len(values)

    out = {
        "b": np.full(length, np.nan),
        "mu": np.full(length, np.nan),
        "sigma_eq": np.full(length, np.nan),
        "half_life": np.full(length, np.nan),
        "stretch": np.full(length, np.nan),
        "r_squared": np.full(length, np.nan),
        "valid": np.zeros(length, dtype=bool),
    }

    for idx in range(window, length):
        current_value = values[idx]
        if not np.isfinite(current_value):
            continue
        history = values[idx - window:idx]
        result = estimate_ou_from_history(history, current_value, min_window_obs=min_window_obs)
        for key in ("b", "mu", "sigma_eq", "half_life", "stretch", "r_squared"):
            out[key][idx] = result[key]
        out["valid"][idx] = bool(result["valid"])

    stretch_s = pd.Series(out["stretch"])
    out["stretch_vel_5d"] = (stretch_s - stretch_s.shift(5)).values
    out["stretch_vel_10d"] = (stretch_s - stretch_s.shift(10)).values

    return pd.DataFrame(out, index=series.index)


def build_group_ou_features(group_df, feature_names, windows, min_window_obs):
    """Build OU features for one ticker."""
    new_cols = {}
    transform_map = {}

    for feature_name in feature_names:
        if feature_name not in group_df.columns:
            continue

        transformed_series, transform_name = feature_transform(group_df[feature_name], feature_name)
        label = feature_name if transform_name == "identity" else f"{feature_name}_{transform_name}"
        transform_map[label] = transform_name
        new_cols[f"ou_source_{label}"] = transformed_series.values

        for window in windows:
            ou = build_ou_columns(transformed_series, window=window, min_window_obs=min_window_obs)
            for col_name in ou.columns:
                new_cols[f"ou_{label}_{col_name}_{window}"] = ou[col_name].values

    result = pd.concat([group_df.reset_index(drop=True),
                        pd.DataFrame(new_cols, index=range(len(group_df)))], axis=1)
    return result, transform_map


def clean_infs_inplace(df):
    """Replace inf/-inf with NaN column by column — no full-dataframe copy."""
    for col in df.select_dtypes(include=[np.number]).columns:
        arr = df[col].values
        mask = np.isinf(arr)
        if mask.any():
            arr[mask] = np.nan


# ═══════════════════════════════════════════════════════════════════════════
# PARALLEL WORKER (must be at module level for Windows multiprocessing)
# ═══════════════════════════════════════════════════════════════════════════
def _process_one_ticker(args):
    """Process a single ticker group. Called by Pool workers."""
    group_df, feature_names, windows, min_window_obs = args
    try:
        result, transform_map = build_group_ou_features(
            group_df=group_df,
            feature_names=feature_names,
            windows=windows,
            min_window_obs=min_window_obs,
        )
        clean_infs_inplace(result)
        return result, transform_map, None
    except Exception as e:
        ticker = group_df.iloc[0].get("ticker", "unknown") if len(group_df) > 0 else "unknown"
        return None, {}, f"{ticker}: {e}"


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    root = runtime_project_root(args.project_root)
    load_simple_dotenv(root, override=False)

    env_data_root = os.getenv("DATA_ROOT", "").strip()
    if env_data_root:
        data_root = Path(env_data_root) / "data" / "stage"
    else:
        data_root = data_dir(root)
    data_root.mkdir(parents=True, exist_ok=True)

    input_path = resolve_path(args.input_parquet, data_root, "cleanroom_transformed_panel.parquet", root_dir=root)
    output_path = resolve_path(args.output_parquet, data_root, "cleanroom_ou_panel.parquet", root_dir=root)
    status_path = resolve_path(args.status_output, data_root, "cleanroom_ou_panel_status.json", root_dir=root)

    feature_names = parse_csv_list(args.features)
    windows = parse_windows(args.windows)

    panel = load_panel(input_path, group_column=args.group_column, date_column=args.date_column)
    source_start, source_end, source_span_days = validate_source_history(
        panel, date_column=args.date_column, source_path=input_path,
        min_history_days=int(args.min_source_history_days),
    )

    group_values = sorted(panel[args.group_column].dropna().unique().tolist())
    if int(args.max_groups) > 0:
        keep_values = set(group_values[:int(args.max_groups)])
        panel = panel[panel[args.group_column].isin(keep_values)].copy()

    groups = list(panel.groupby(args.group_column, sort=True))

    n_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 1)

    print(f"Project root: {root}")
    print(f"Input parquet: {input_path}")
    print(f"Output parquet: {output_path}")
    print(f"Source history: {source_start.date()} -> {source_end.date()} ({source_span_days} calendar days)")
    print(f"Groups to transform: {len(groups):,}")
    print(f"Features requested: {feature_names}")
    print(f"OU windows: {windows}")
    print(f"Parallel workers: {n_workers}")

    if args.dry_run:
        print("Dry run complete.")
        return 0

    # Build work items — each is (group_df, feature_names, windows, min_window_obs)
    work_items = [
        (group_df, feature_names, windows, int(args.min_window_obs))
        for _, group_df in groups
    ]

    # Free the panel — workers have their own copies of group DataFrames
    del panel

    # Process in parallel
    frames = []
    transform_catalog = {}
    errors = []

    with Pool(processes=n_workers) as pool:
        results_iter = pool.imap(_process_one_ticker, work_items)
        for result_df, transform_map, error in tqdm(results_iter, total=len(work_items),
                                                     desc="  OU features", unit="ticker"):
            if error:
                errors.append(error)
                continue
            if result_df is not None:
                transform_catalog.update(transform_map)
                frames.append(result_df)

    if errors:
        print(f"\n  WARNING: {len(errors)} tickers had errors:")
        for e in errors[:10]:
            print(f"    {e}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")

    if not frames:
        print("FATAL: No frames produced")
        return 1

    # Concat — no sort needed, data is already grouped by ticker
    print(f"Concatenating {len(frames):,} ticker frames ...")
    output_df = pd.concat(frames, ignore_index=True)

    # Free frames immediately
    del frames

    # Save
    print("Saving to parquet ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)

    # Collect stretch counts for status
    valid_counts = {}
    for column in output_df.columns:
        if "_stretch_" in column and column.startswith("ou_") and not any(
            column.endswith(s) for s in ("_vel_5d", "_vel_10d")
        ):
            for w in windows:
                if column.endswith(f"_{w}"):
                    valid_counts[column] = int(output_df[column].notna().sum())

    status = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "input_parquet": str(input_path),
        "output_parquet": str(output_path),
        "group_column": args.group_column,
        "date_column": args.date_column,
        "features_requested": feature_names,
        "windows": windows,
        "feature_transforms": transform_catalog,
        "groups_transformed": int(output_df[args.group_column].nunique()),
        "rows_written": int(len(output_df)),
        "columns_written": int(output_df.columns.size),
        "source_history_start": source_start.date().isoformat(),
        "source_history_end": source_end.date().isoformat(),
        "source_history_days": int(source_span_days),
        "parallel_workers": n_workers,
        "errors": errors,
        "ou_stretch_notna_counts": valid_counts,
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    print(f"\nSaved OU panel: {output_path}")
    print(f"Rows: {len(output_df):,}")
    print(f"Columns: {output_df.columns.size}")
    print(f"Tickers: {output_df[args.group_column].nunique():,}")
    print(f"File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())