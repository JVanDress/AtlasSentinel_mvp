# baseline_opus.py
"""
Baseline Study — Opus
---------------------
Purpose: Establish benchmark binary classification AUCs for QUINTIC_V3,
BEFORE any overlays are added, using only normalized price-derived features
from normalize_opus.py output.

Targets: probability that fwd_ret_{h}d > 0 for h in {20, 60, 90}.

Features: every column ending in "_zscore" or "_pctrank" (produced by
normalize_opus.py). Forward-looking columns (starting with "fwd_") are
explicitly excluded.

SEALED HOLDOUT ENFORCEMENT:
  Build window:  2015-01-02  through  2024-04-18   (inclusive)
  Sealed holdout: 2024-04-19 through  2026-04-07   (never used)
  Any row dated >= 2024-04-19 is dropped before any fit/evaluate call.

Validation: expanding-window walk-forward CV with a purge of `horizon`
trading days between training end and test start. Purge prevents the
target of late training samples from overlapping with test-period prices.

Hyperparameters: fixed, no tuning. Baseline is baseline.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Iterator

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss


# =============================================================================
# SEALED HOLDOUT CONSTANTS — DO NOT MODIFY
# =============================================================================
BUILD_START = pd.Timestamp("2015-01-02")
BUILD_END   = pd.Timestamp("2024-04-18")
HOLDOUT_START = pd.Timestamp("2024-04-19")
HOLDOUT_END   = pd.Timestamp("2026-04-07")


@dataclass
class ModelConfig:
    horizons: List[int] = field(default_factory=lambda: [20, 60, 90])
    n_splits: int = 5

    # XGBoost
    n_estimators: int = 150
    max_depth: int = 4
    learning_rate: float = 0.05
    min_child_weight: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    tree_method: str = "hist"
    device: str = "cpu"

    # Quality filters
    min_feature_completeness: float = 0.8
    min_rows_per_fold: int = 10_000


def fail(msg: str) -> None:
    sys.exit(f"\nERROR: {msg}\n")


# =============================================================================
# Feature / target builders
# =============================================================================

def build_feature_set(df: pd.DataFrame) -> List[str]:
    """Pattern match: columns ending in _zscore or _pctrank, excluding fwd_ prefix."""
    features = []
    for c in df.columns:
        if c.startswith("fwd_"):
            continue
        if c.endswith("_zscore") or c.endswith("_pctrank"):
            features.append(c)
    return sorted(features)


def compute_targets(df: pd.DataFrame, cfg: ModelConfig) -> Tuple[pd.DataFrame, List[Tuple[int, str]]]:
    """
    Binary directional target per horizon: y_up_{h}d = 1 iff fwd_ret_{h}d > 0.
    NaN target for rows where fwd_ret is NaN (end-of-sample rows).
    """
    targets = []
    for h in cfg.horizons:
        src = f"fwd_ret_{h}d"
        if src not in df.columns:
            fail(f"Missing forward return column: {src}")
        tgt = f"y_up_{h}d"
        fr = df[src]
        df[tgt] = np.where(fr.isna(), np.nan, (fr > 0).astype(np.float32))
        targets.append((h, tgt))
    return df, targets


# =============================================================================
# Purged walk-forward CV
# =============================================================================

def purged_walk_forward_splits(
    sorted_dates: np.ndarray, n_splits: int, horizon: int
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window walk-forward splitter operating on the unique sorted
    trading dates of the build window.

    Splits total dates into (n_splits + 1) equal chunks. For fold k:
      train = dates[0 : (k+1)*chunk - horizon]    (purged by `horizon` days)
      test  = dates[(k+1)*chunk : (k+2)*chunk]

    Yields (train_dates, test_dates) numpy arrays.
    """
    total = len(sorted_dates)
    if total < (n_splits + 1) * (horizon + 10):
        return  # not enough data for meaningful splits

    chunk = total // (n_splits + 1)

    for k in range(n_splits):
        test_start = (k + 1) * chunk
        test_end = (k + 2) * chunk if k < n_splits - 1 else total
        train_end = test_start - horizon

        if train_end <= 0:
            continue

        train_dates = sorted_dates[:train_end]
        test_dates = sorted_dates[test_start:test_end]

        if len(train_dates) == 0 or len(test_dates) == 0:
            continue

        yield train_dates, test_dates


# =============================================================================
# Train + evaluate one horizon
# =============================================================================

def train_and_eval(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    horizon: int,
    cfg: ModelConfig,
) -> Optional[Dict]:
    """
    Walk-forward CV. Returns dict of per-fold metrics + feature importance
    from the final fold. None if insufficient data.
    """
    work = df[df[target_col].notna()].copy()
    if len(work) < cfg.min_rows_per_fold:
        return None

    sorted_dates = np.sort(work["date"].unique())
    fold_results = []
    final_model = None

    for fold_idx, (train_dates, test_dates) in enumerate(
        purged_walk_forward_splits(sorted_dates, cfg.n_splits, horizon)
    ):
        tr_mask = work["date"].isin(train_dates)
        te_mask = work["date"].isin(test_dates)

        X_tr = work.loc[tr_mask, features]
        y_tr = work.loc[tr_mask, target_col].astype(int)
        X_te = work.loc[te_mask, features]
        y_te = work.loc[te_mask, target_col].astype(int)

        if len(X_tr) < cfg.min_rows_per_fold or len(X_te) < cfg.min_rows_per_fold:
            continue
        if y_tr.nunique() < 2 or y_te.nunique() < 2:
            continue

        model = xgb.XGBClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            min_child_weight=cfg.min_child_weight,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            tree_method=cfg.tree_method,
            device=cfg.device,
            objective="binary:logistic",
            eval_metric="auc",
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]

        fold_results.append({
            "fold": fold_idx,
            "train_rows": int(len(X_tr)),
            "test_rows": int(len(X_te)),
            "train_start": str(pd.Timestamp(train_dates[0]).date()),
            "train_end":   str(pd.Timestamp(train_dates[-1]).date()),
            "test_start":  str(pd.Timestamp(test_dates[0]).date()),
            "test_end":    str(pd.Timestamp(test_dates[-1]).date()),
            "gap_days_between_train_end_and_test_start": int(
                np.busday_count(
                    pd.Timestamp(train_dates[-1]).date(),
                    pd.Timestamp(test_dates[0]).date(),
                )
            ),
            "pos_rate_train": float(y_tr.mean()),
            "pos_rate_test":  float(y_te.mean()),
            "auc":     float(roc_auc_score(y_te, p)),
            "logloss": float(log_loss(y_te, p)),
            "brier":   float(brier_score_loss(y_te, p)),
        })
        final_model = model

    if not fold_results or final_model is None:
        return None

    importance = dict(sorted(
        zip(features, final_model.feature_importances_.tolist()),
        key=lambda kv: -kv[1],
    ))

    return {
        "fold_results": fold_results,
        "importance": importance,
        "n_samples": int(len(work)),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Baseline Study — Opus")
    parser.add_argument("-i", "--input", required=True, help="Normalized parquet (output of normalize_opus.py)")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory (default: <input_dir>/baseline_opus)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU (CUDA) for XGBoost")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        fail(f"Input not found: {input_path}")

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "baseline_opus"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = ModelConfig()
    if args.gpu:
        cfg.device = "cuda"

    start_time = time.time()

    print(f"Loading: {input_path}")
    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        fail("Invalid dates in input.")
    print(f"Loaded {len(df):,} rows across {df['date'].nunique():,} dates "
          f"({df['date'].min().date()} -> {df['date'].max().date()})")

    # ---- Sealed holdout enforcement ----
    pre = len(df)
    df = df[df["date"] <= BUILD_END].copy()
    excluded = pre - len(df)
    print(f"\nSEALED HOLDOUT ENFORCEMENT:")
    print(f"  Build end: {BUILD_END.date()}")
    print(f"  Rows removed (holdout): {excluded:,}")
    print(f"  Rows remaining (build): {len(df):,}")
    if len(df) == 0:
        fail("No build-window rows remain after holdout enforcement.")

    # ---- Universe filter ----
    if "in_universe" not in df.columns:
        fail("Input must have 'in_universe' column. Run normalize_opus.py first.")
    df = df[df["in_universe"]].copy()
    print(f"In-universe rows in build window: {len(df):,}")
    if len(df) == 0:
        fail("No in-universe rows in the build window.")

    # ---- Feature selection ----
    features = build_feature_set(df)
    if not features:
        fail("No features found. Expected columns ending in _zscore or _pctrank.")
    print(f"\nFeatures ({len(features)}):")
    for f in features:
        print(f"  - {f}")

    # ---- Feature completeness filter ----
    completeness = df[features].notna().mean(axis=1)
    keep = completeness >= cfg.min_feature_completeness
    removed = (~keep).sum()
    df = df[keep].copy()
    print(f"\nFeature completeness filter (>= {cfg.min_feature_completeness*100:.0f}%): "
          f"removed {removed:,}, kept {len(df):,}")
    if len(df) == 0:
        fail("No rows pass feature completeness filter.")

    # ---- Targets ----
    df, targets = compute_targets(df, cfg)

    # ---- Run CV per horizon ----
    print(f"\n{'='*70}")
    print(f"Running walk-forward CV for {len(targets)} horizons...")
    print(f"{'='*70}")
    all_results = {}
    for h, target_col in tqdm(targets, desc="Horizons"):
        print(f"\n--- Horizon {h}d  ({target_col}) ---")
        result = train_and_eval(df, features, target_col, h, cfg)
        if result is None:
            print("  SKIPPED — insufficient data")
            all_results[target_col] = None
            continue
        aucs = [f["auc"] for f in result["fold_results"]]
        print(f"  Folds: {len(aucs)}")
        for fr in result["fold_results"]:
            print(f"    Fold {fr['fold']}: "
                  f"train {fr['train_start']}->{fr['train_end']} ({fr['train_rows']:,}) | "
                  f"test {fr['test_start']}->{fr['test_end']} ({fr['test_rows']:,}) | "
                  f"gap={fr['gap_days_between_train_end_and_test_start']}d | "
                  f"AUC={fr['auc']:.4f}")
        print(f"  AUC median: {np.median(aucs):.4f}  mean: {np.mean(aucs):.4f}  std: {np.std(aucs):.4f}")
        all_results[target_col] = result

    # ---- Write metadata ----
    summary = {
        "input": str(input_path),
        "build_window": [str(BUILD_START.date()), str(BUILD_END.date())],
        "sealed_holdout": [str(HOLDOUT_START.date()), str(HOLDOUT_END.date())],
        "n_rows_used": int(len(df)),
        "n_features": len(features),
        "features": features,
        "config": asdict(cfg),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "per_horizon": {},
    }
    for k, v in all_results.items():
        if v is None:
            summary["per_horizon"][k] = None
            continue
        aucs = [f["auc"] for f in v["fold_results"]]
        summary["per_horizon"][k] = {
            "fold_results": v["fold_results"],
            "auc_median": float(np.median(aucs)),
            "auc_mean":   float(np.mean(aucs)),
            "auc_std":    float(np.std(aucs)),
            "top_10_features": dict(list(v["importance"].items())[:10]),
            "n_samples": v["n_samples"],
        }

    meta_path = output_dir / "baseline_opus_metadata.json"
    meta_path.write_text(json.dumps(summary, indent=2, default=str))

    # ---- Final summary table ----
    print(f"\n{'='*70}")
    print(f"BASELINE OPUS — FINAL CV SUMMARY")
    print(f"{'='*70}")
    print(f"{'Target':<20}  {'AUC median':>12}  {'AUC mean':>10}  {'AUC std':>10}  {'folds':>6}")
    print(f"{'-'*70}")
    for k, r in summary["per_horizon"].items():
        if r is None:
            print(f"{k:<20}  {'SKIPPED':>12}")
        else:
            print(f"{k:<20}  {r['auc_median']:>12.4f}  {r['auc_mean']:>10.4f}  "
                  f"{r['auc_std']:>10.4f}  {len(r['fold_results']):>6d}")
    print(f"{'='*70}")
    print(f"Metadata: {meta_path}")
    print(f"Elapsed:  {summary['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()