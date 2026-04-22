#!/usr/bin/env python3
"""
Quintic Labs — GMM Regime Detection (Memory-Safe)
Fits a train-only cleanroom GMM regime layer on OU stretch features.
Only loads the columns it needs — works on 15GB+ OU panels without crashing.
Output is a LEAN parquet: ticker, date, regime columns only.
The master panel merge joins this with the full OU panel.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from quintic_paths import data_dir, project_root

DEFAULT_MIN_HISTORY_DAYS = 1820
DEFAULT_FEATURES = (
    "close_frac_diff",
    "log_return_1d",
    "volume_robust_z_sector_rank",
    "trade_size_robust_z_sector_rank",
    "clv_sector_rank",
    "garman_klass_var_1d_sector_rank",
    "log_return_1d_sector_rank",
    "ou_log_close_stretch_63",
    "ou_garman_klass_var_1d_log1p_stretch_63",
    "ou_volume_robust_z_stretch_63",
    "ou_trade_size_robust_z_stretch_63",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a train-only cleanroom GMM regime layer (memory-safe)."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--input-parquet", type=Path, default=None)
    parser.add_argument("--output-parquet", type=Path, default=None)
    parser.add_argument("--status-output", type=Path, default=None)
    parser.add_argument("--model-output", type=Path, default=None)
    parser.add_argument("--scaler-output", type=Path, default=None)
    parser.add_argument("--group-column", default="ticker")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--feature-columns", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--train-start-date", default="2015-01-01")
    parser.add_argument("--train-end-date", default="2024-12-31")
    parser.add_argument("--component-grid", default="3,4,5,6")
    parser.add_argument("--covariance-type", default="diag", choices=["full", "tied", "diag", "spherical"])
    parser.add_argument("--reg-covar", type=float, default=1e-6)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--max-train-rows", type=int, default=250000)
    parser.add_argument("--clip-lower-quantile", type=float, default=0.01)
    parser.add_argument("--clip-upper-quantile", type=float, default=0.99)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-source-history-days", type=int, default=DEFAULT_MIN_HISTORY_DAYS)
    parser.add_argument("--max-groups", type=int, default=0)
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
    return base_dir / candidate


def parse_csv_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_int_grid(value):
    return sorted(set(int(x) for x in parse_csv_list(value)))


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


def load_panel_lean(path, group_column, date_column, feature_columns):
    """Load ONLY the columns GMM needs — not the full 906-column file."""
    if not path.exists():
        raise FileNotFoundError(f"Required parquet not found: {path}")

    # Read just the columns we need
    import pyarrow.parquet as pq
    all_parquet_cols = pq.read_schema(path).names
    cols_to_load = [group_column, date_column]
    present_features = []
    for f in feature_columns:
        if f in all_parquet_cols:
            cols_to_load.append(f)
            present_features.append(f)

    cols_to_load = list(dict.fromkeys(cols_to_load))  # deduplicate, preserve order

    print(f"  Loading {len(cols_to_load)} columns (of {len(all_parquet_cols)} available) ...")
    df = pd.read_parquet(path, columns=cols_to_load)

    df[group_column] = df[group_column].astype(str).str.upper().str.strip()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[group_column, date_column])
    df = df[df[group_column].ne("")]
    df = df.sort_values([group_column, date_column]).reset_index(drop=True)

    return df, present_features


def choose_present_features(df, requested):
    present = [f for f in requested if f in df.columns]
    if not present:
        raise RuntimeError(f"None of the requested feature columns found. Requested: {requested}")
    return present


def select_train_frame(df, date_column, feature_columns, train_start, train_end):
    mask = (df[date_column] >= train_start) & (df[date_column] <= train_end)
    return df.loc[mask].copy()


def compute_clip_bounds(df, feature_columns, lower_quantile, upper_quantile):
    bounds = {}
    for col in feature_columns:
        vals = df[col].dropna()
        if vals.empty:
            bounds[col] = (float("-inf"), float("inf"))
            continue
        bounds[col] = (
            float(vals.quantile(lower_quantile)),
            float(vals.quantile(upper_quantile)),
        )
    return bounds


def clip_feature_frame(df, clip_bounds):
    out = df.copy()
    for col, (lo, hi) in clip_bounds.items():
        if col in out.columns:
            out[col] = out[col].clip(lower=lo, upper=hi)
    return out


def sample_training_frame(df, max_rows, random_state):
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=random_state)


def fit_best_gmm(train_matrix, component_grid, covariance_type, reg_covar, n_init, max_iter, random_state):
    best_model = None
    best_bic = float("inf")
    candidate_scores = {}

    for n_comp in tqdm(component_grid, desc="  GMM grid search", unit="comp"):
        gmm = GaussianMixture(
            n_components=n_comp,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        gmm.fit(train_matrix)
        bic = gmm.bic(train_matrix)
        aic = gmm.aic(train_matrix)
        candidate_scores[str(n_comp)] = {"bic": float(bic), "aic": float(aic)}
        if bic < best_bic:
            best_bic = bic
            best_model = gmm

    return best_model, candidate_scores


def main():
    args = parse_args()
    root = runtime_project_root(args.project_root)
    data_root = data_dir(root)
    data_root.mkdir(parents=True, exist_ok=True)

    input_path = resolve_path(args.input_parquet, data_root, "cleanroom_ou_panel.parquet", root_dir=root)
    output_path = resolve_path(args.output_parquet, data_root, "cleanroom_gmm_panel.parquet", root_dir=root)
    status_path = resolve_path(args.status_output, data_root, "cleanroom_gmm_status.json", root_dir=root)
    model_path = resolve_path(args.model_output, data_root, "cleanroom_gmm_model.joblib", root_dir=root)
    scaler_path = resolve_path(args.scaler_output, data_root, "cleanroom_gmm_scaler.joblib", root_dir=root)

    feature_columns_requested = parse_csv_list(args.feature_columns)
    component_grid = parse_int_grid(args.component_grid)
    train_start = pd.Timestamp(args.train_start_date)
    train_end = pd.Timestamp(args.train_end_date)
    if train_end < train_start:
        raise ValueError("train-end-date must be >= train-start-date")

    # MEMORY-SAFE: only load the columns GMM needs
    panel, feature_columns = load_panel_lean(
        input_path,
        group_column=args.group_column,
        date_column=args.date_column,
        feature_columns=feature_columns_requested,
    )

    source_start, source_end, source_span_days = validate_source_history(
        panel, date_column=args.date_column, source_path=input_path,
        min_history_days=int(args.min_source_history_days),
    )

    if int(args.max_groups) > 0:
        keep_values = set(sorted(panel[args.group_column].dropna().unique().tolist())[:int(args.max_groups)])
        panel = panel[panel[args.group_column].isin(keep_values)]

    panel["gmm_ready"] = panel[feature_columns].notna().all(axis=1)
    panel["gmm_sample_period"] = np.where(panel[args.date_column] <= train_end, "train_or_earlier", "post_train")

    train_df = select_train_frame(panel, args.date_column, feature_columns, train_start, train_end)
    train_ready = train_df[train_df["gmm_ready"]].copy()
    if train_ready.empty:
        raise RuntimeError("No training rows are fully populated for the requested GMM features.")

    clip_bounds = compute_clip_bounds(
        train_ready, feature_columns,
        lower_quantile=float(args.clip_lower_quantile),
        upper_quantile=float(args.clip_upper_quantile),
    )
    train_ready_clipped = clip_feature_frame(train_ready, clip_bounds)
    train_used = sample_training_frame(train_ready_clipped, max_rows=int(args.max_train_rows), random_state=int(args.random_state))

    print(f"Project root: {root}")
    print(f"Input parquet: {input_path}")
    print(f"Output parquet: {output_path}")
    print(f"Source history: {source_start.date()} -> {source_end.date()} ({source_span_days} calendar days)")
    print(f"Feature columns used: {feature_columns}")
    print(f"Training window: {train_start.date()} -> {train_end.date()}")
    print(f"Panel rows: {len(panel):,}")
    print(f"Panel tickers: {panel[args.group_column].nunique():,}")
    print(f"Train rows available: {len(train_ready):,}")
    print(f"Train rows used for GMM fit: {len(train_used):,}")
    print(f"Component grid: {component_grid}")

    if args.dry_run:
        print("Dry run complete.")
        return 0

    # ── Fit GMM ──────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_used[feature_columns].to_numpy(dtype=float))
    gmm_model, candidate_scores = fit_best_gmm(
        train_matrix=X_train,
        component_grid=component_grid,
        covariance_type=str(args.covariance_type),
        reg_covar=float(args.reg_covar),
        n_init=int(args.n_init),
        max_iter=int(args.max_iter),
        random_state=int(args.random_state),
    )

    # ── Apply to full panel ──────────────────────────────────────────────
    ready_mask = panel["gmm_ready"].to_numpy(dtype=bool)
    n_rows = len(panel)
    regime_label = np.full(n_rows, np.nan, dtype=float)
    regime_confidence = np.full(n_rows, np.nan, dtype=float)
    regime_log_likelihood = np.full(n_rows, np.nan, dtype=float)
    proba_matrix = np.full((n_rows, gmm_model.n_components), np.nan, dtype=float)

    if ready_mask.any():
        panel_ready_clipped = clip_feature_frame(panel.loc[ready_mask, feature_columns], clip_bounds)
        X_full = scaler.transform(panel_ready_clipped[feature_columns].to_numpy(dtype=float))
        labels = gmm_model.predict(X_full)
        probs = gmm_model.predict_proba(X_full)
        scores = gmm_model.score_samples(X_full)
        regime_label[ready_mask] = labels.astype(float)
        regime_confidence[ready_mask] = probs.max(axis=1)
        regime_log_likelihood[ready_mask] = scores
        proba_matrix[ready_mask, :] = probs

    # ── Build LEAN output: ticker, date, regime columns only ─────────────
    output_df = pd.DataFrame({
        args.group_column: panel[args.group_column].values,
        args.date_column: panel[args.date_column].values,
        "gmm_regime_label": regime_label,
        "gmm_regime_confidence": regime_confidence,
        "gmm_log_likelihood": regime_log_likelihood,
    })
    for idx in range(gmm_model.n_components):
        output_df[f"gmm_regime_prob_{idx}"] = proba_matrix[:, idx]

    # No sort needed — data came in sorted from load_panel_lean
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)

    # ── Save model and scaler ────────────────────────────────────────────
    joblib.dump(gmm_model, model_path)
    joblib.dump({
        "scaler": scaler,
        "feature_columns": feature_columns,
        "clip_bounds": clip_bounds,
        "clip_lower_quantile": float(args.clip_lower_quantile),
        "clip_upper_quantile": float(args.clip_upper_quantile),
    }, scaler_path)

    # ── Status ───────────────────────────────────────────────────────────
    regime_counts = pd.Series(regime_label).dropna().value_counts().sort_index()
    status = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "input_parquet": str(input_path),
        "output_parquet": str(output_path),
        "model_output": str(model_path),
        "scaler_output": str(scaler_path),
        "group_column": args.group_column,
        "date_column": args.date_column,
        "feature_columns_requested": feature_columns_requested,
        "feature_columns_used": feature_columns,
        "train_window_start": train_start.date().isoformat(),
        "train_window_end": train_end.date().isoformat(),
        "component_grid": component_grid,
        "selected_n_components": int(gmm_model.n_components),
        "covariance_type": args.covariance_type,
        "train_rows_available": int(len(train_ready)),
        "train_rows_used": int(len(train_used)),
        "panel_rows": int(len(output_df)),
        "panel_tickers": int(output_df[args.group_column].nunique()),
        "rows_with_regime": int(pd.Series(regime_label).notna().sum()),
        "regime_label_counts": {str(int(k)): int(v) for k, v in regime_counts.items()},
        "candidate_scores": candidate_scores,
        "source_history_start": source_start.date().isoformat(),
        "source_history_end": source_end.date().isoformat(),
        "source_history_days": int(source_span_days),
        "output_type": "lean_regime_only",
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    print(f"\nSaved GMM regime panel: {output_path}")
    print(f"Saved model: {model_path}")
    print(f"Saved scaler: {scaler_path}")
    print(f"Selected n_components: {gmm_model.n_components}")
    print(f"Rows: {len(output_df):,}")
    print(f"Tickers: {output_df[args.group_column].nunique():,}")
    print(f"Rows with regime: {int(pd.Series(regime_label).notna().sum()):,}")
    print(f"Avg confidence: {float(np.nanmean(regime_confidence)):.4f}")
    print(f"File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())