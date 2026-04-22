"""
baseline_forecast_institutional.py
----------------------------------
Walk-forward forecasting baseline for 20/60/90 day horizons using
normalized study features (unitless only; no dollar sizing assumptions).

Outputs:
  - metrics_summary.json
  - fold_metrics.csv
  - daily_ic.csv
  - predictions.parquet
  - feature_importance.csv

This baseline focuses on forecast quality + tradability diagnostics:
  - RMSE / hit rate
  - Daily cross-sectional Spearman/Pearson IC
  - Decile spread (Q10-Q1), gross and cost-adjusted
  - Unitless turnover and max drawdown from normalized signal weights
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "\nERROR: scikit-learn is required. Install with: pip install scikit-learn\n"
    ) from exc


def fail(message: str) -> None:
    raise SystemExit(f"\nERROR: {message}\n")


def parse_horizons(raw: str) -> List[int]:
    vals: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            val = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid horizon value: {token}") from exc
        if val <= 0:
            raise argparse.ArgumentTypeError(f"Horizon must be > 0: {token}")
        vals.append(val)
    if not vals:
        raise argparse.ArgumentTypeError("No valid horizons were provided.")
    return sorted(set(vals))


@dataclass
class BaselineConfig:
    horizons: List[int] = field(default_factory=lambda: [20, 60, 90])
    n_splits: int = 5
    min_rows_train: int = 15_000
    min_rows_test: int = 5_000
    min_days_per_fold: int = 40
    min_feature_completeness: float = 0.70
    min_names_per_day: int = 30
    n_deciles: int = 10
    ridge_alpha: float = 1.0
    cost_bps: float = 10.0
    feature_mode: str = "all_numeric"
    purge_days_override: int = -1
    max_abs_weight_z: float = 3.0


def build_feature_set(
    df: pd.DataFrame,
    feature_mode: str,
    include_prefixes: Sequence[str],
    exclude_prefixes: Sequence[str],
) -> List[str]:
    protected = ("ticker", "date", "in_universe", "close", "volume")

    normalized = [
        c
        for c in df.columns
        if (c.endswith("_zscore") or c.endswith("_pctrank"))
        and not c.startswith("fwd_")
        and c not in protected
    ]

    core_candidates = [
        "ret_5d_zscore",
        "ret_21d_zscore",
        "ret_63d_zscore",
        "ret_5d_pctrank",
        "ret_21d_pctrank",
        "ret_63d_pctrank",
        "ret_5d_voladj_zscore",
        "ret_21d_voladj_zscore",
        "ret_63d_voladj_zscore",
        "ret_5d_voladj_pctrank",
        "ret_21d_voladj_pctrank",
        "ret_63d_voladj_pctrank",
    ]

    if feature_mode == "core":
        selected = [c for c in core_candidates if c in df.columns]
    elif feature_mode in {"normalized", "all"}:
        selected = list(normalized)
    else:
        # Full-system mode: include all numeric lagged features, excluding obvious
        # identifiers and target/leakage columns.
        numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
        selected = []
        for col in sorted(numeric_cols):
            if col in protected:
                continue
            if col.startswith("fwd_") or col.startswith("y_"):
                continue
            if col in {
                "pred",
                "target",
                "fold",
                "horizon",
            }:
                continue
            if "target" in col.lower() or "label" in col.lower() or "future" in col.lower():
                continue
            selected.append(col)

    # Optional family inclusion by prefix (useful for overlays that are numeric
    # but may not follow _zscore/_pctrank naming conventions).
    if include_prefixes:
        numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
        for col in sorted(numeric_cols):
            if col.startswith("fwd_") or col.startswith("y_"):
                continue
            if col in protected:
                continue
            if any(col.startswith(pref) for pref in include_prefixes):
                selected.append(col)

    # Optional family exclusion by prefix.
    if exclude_prefixes:
        selected = [
            c for c in selected if not any(c.startswith(pref) for pref in exclude_prefixes)
        ]

    # Keep deterministic, unique ordering.
    selected = sorted(set(selected))
    return selected


def generate_walk_forward_splits(
    sorted_dates: np.ndarray,
    n_splits: int,
    purge_days: int,
    min_days_per_fold: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    total = len(sorted_dates)
    if total < (n_splits + 1) * min_days_per_fold:
        return

    chunk = total // (n_splits + 1)
    if chunk < min_days_per_fold:
        return

    for k in range(n_splits):
        test_start = (k + 1) * chunk
        test_end = (k + 2) * chunk if k < n_splits - 1 else total
        train_end = test_start - purge_days
        if train_end <= 0:
            continue

        train_dates = sorted_dates[:train_end]
        test_dates = sorted_dates[test_start:test_end]
        if len(train_dates) < min_days_per_fold or len(test_dates) < min_days_per_fold:
            continue
        yield train_dates, test_dates


def daily_ic(pred_df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for dt, g in pred_df.groupby("date", sort=True):
        g = g.dropna(subset=["pred", "target"])
        n = len(g)
        if n < 5:
            continue
        if g["pred"].std(ddof=0) == 0 or g["target"].std(ddof=0) == 0:
            continue
        out_rows.append(
            {
                "date": dt,
                "n": n,
                "spearman_ic": g["pred"].corr(g["target"], method="spearman"),
                "pearson_ic": g["pred"].corr(g["target"], method="pearson"),
            }
        )
    return pd.DataFrame(out_rows)


def decile_spread_by_day(
    pred_df: pd.DataFrame,
    n_deciles: int,
    min_names_per_day: int,
) -> pd.DataFrame:
    rows = []
    for dt, g in pred_df.groupby("date", sort=True):
        g = g.dropna(subset=["pred", "target"])
        if len(g) < min_names_per_day:
            continue
        try:
            q = pd.qcut(g["pred"], q=n_deciles, labels=False, duplicates="drop")
        except ValueError:
            continue
        if q.nunique(dropna=True) < 2:
            continue
        q = q.astype(float)
        top_bucket = int(np.nanmax(q.values))
        bot_bucket = int(np.nanmin(q.values))
        top = g.loc[q == top_bucket, "target"].mean()
        bot = g.loc[q == bot_bucket, "target"].mean()
        rows.append(
            {
                "date": dt,
                "n": len(g),
                "q_top": top_bucket,
                "q_bot": bot_bucket,
                "top_mean_target": float(top),
                "bot_mean_target": float(bot),
                "decile_spread": float(top - bot),
            }
        )
    return pd.DataFrame(rows)


def build_unitless_weights(
    pred_df: pd.DataFrame,
    min_names_per_day: int,
    max_abs_weight_z: float,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for dt, g in pred_df.groupby("date", sort=True):
        g = g.dropna(subset=["pred", "target"]).copy()
        if len(g) < min_names_per_day:
            continue
        std = g["pred"].std(ddof=0)
        if std == 0 or np.isnan(std):
            continue
        z = (g["pred"] - g["pred"].mean()) / std
        z = z.clip(lower=-max_abs_weight_z, upper=max_abs_weight_z)
        gross = np.abs(z).sum()
        if gross <= 0 or np.isnan(gross):
            continue
        g["weight"] = z / gross
        rows.append(g[["date", "ticker", "target", "pred", "weight"]])

    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "target", "pred", "weight"])
    return pd.concat(rows, axis=0, ignore_index=True)


def compute_turnover(weights: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    if weights.empty:
        return np.nan, pd.DataFrame(columns=["date", "turnover"])

    daily_turn = []
    prev = pd.Series(dtype=float)
    for dt, g in weights.groupby("date", sort=True):
        cur = g.set_index("ticker")["weight"].astype(float)
        union_idx = cur.index.union(prev.index)
        cur_u = cur.reindex(union_idx, fill_value=0.0)
        prev_u = prev.reindex(union_idx, fill_value=0.0)
        turnover = float(np.abs(cur_u - prev_u).sum() / 2.0)
        daily_turn.append({"date": dt, "turnover": turnover})
        prev = cur

    turn_df = pd.DataFrame(daily_turn)
    return float(turn_df["turnover"].mean()), turn_df


def compute_drawdown_from_log_returns(log_returns: pd.Series) -> float:
    if log_returns.empty:
        return np.nan
    equity = np.exp(log_returns.cumsum())
    rolling_peak = np.maximum.accumulate(equity.values)
    dd = equity.values / rolling_peak - 1.0
    return float(dd.min())


def horizon_non_overlapping_backtest(
    weights_df: pd.DataFrame,
    horizon: int,
    cost_bps: float,
) -> Dict[str, float]:
    """
    Horizon-aligned return diagnostics using non-overlapping cohorts.

    We evaluate portfolio event returns at rebalance dates using fwd_ret_{h}d
    targets and sample every `horizon`th rebalance to avoid overlap leakage in
    the reported return path.
    """
    if weights_df.empty:
        return {
            "cohorts_used": 0,
            "turnover_mean": np.nan,
            "cost_drag": np.nan,
            "event_return_gross_mean": np.nan,
            "event_return_net_mean": np.nan,
            "return_gross_annualized": np.nan,
            "return_net_annualized": np.nan,
            "max_drawdown_worst": np.nan,
        }

    per_date = (
        weights_df.groupby("date", sort=True)
        .apply(lambda g: float(np.sum(g["weight"].values * g["target"].values)))
        .rename("event_return")
    )
    if per_date.empty:
        return {
            "cohorts_used": 0,
            "turnover_mean": np.nan,
            "cost_drag": np.nan,
            "event_return_gross_mean": np.nan,
            "event_return_net_mean": np.nan,
            "return_gross_annualized": np.nan,
            "return_net_annualized": np.nan,
            "max_drawdown_worst": np.nan,
        }

    ann_factor = 252.0 / float(horizon)
    max_offsets = min(horizon, len(per_date))

    gross_events: List[float] = []
    net_events: List[float] = []
    gross_annualized: List[float] = []
    net_annualized: List[float] = []
    mdds: List[float] = []
    turnovers: List[float] = []

    for offset in range(max_offsets):
        cohort = per_date.iloc[offset::horizon]
        if len(cohort) == 0:
            continue

        cohort_dates = set(cohort.index.tolist())
        cohort_weights = weights_df[weights_df["date"].isin(cohort_dates)].copy()
        cohort_turnover_mean, _ = compute_turnover(cohort_weights)
        if np.isnan(cohort_turnover_mean):
            continue

        cost_drag = (cost_bps / 10_000.0) * cohort_turnover_mean
        gross_mean = float(cohort.mean())
        net_mean = gross_mean - cost_drag

        gross_events.append(gross_mean)
        net_events.append(net_mean)
        gross_annualized.append(gross_mean * ann_factor)
        net_annualized.append(net_mean * ann_factor)
        turnovers.append(float(cohort_turnover_mean))
        mdds.append(compute_drawdown_from_log_returns(cohort))

    if not gross_events:
        return {
            "cohorts_used": 0,
            "turnover_mean": np.nan,
            "cost_drag": np.nan,
            "event_return_gross_mean": np.nan,
            "event_return_net_mean": np.nan,
            "return_gross_annualized": np.nan,
            "return_net_annualized": np.nan,
            "max_drawdown_worst": np.nan,
        }

    out_turnover = float(np.mean(turnovers))
    out_cost_drag = (cost_bps / 10_000.0) * out_turnover
    return {
        "cohorts_used": int(len(gross_events)),
        "turnover_mean": out_turnover,
        "cost_drag": float(out_cost_drag),
        "event_return_gross_mean": float(np.mean(gross_events)),
        "event_return_net_mean": float(np.mean(net_events)),
        "return_gross_annualized": float(np.mean(gross_annualized)),
        "return_net_annualized": float(np.mean(net_annualized)),
        "max_drawdown_worst": float(np.min(mdds)),
    }


def feature_completeness_filter(
    df: pd.DataFrame,
    features: Sequence[str],
    min_completeness: float,
) -> pd.Series:
    if not features:
        return pd.Series(False, index=df.index)
    completeness = df[list(features)].notna().mean(axis=1)
    return completeness >= min_completeness


def train_and_evaluate_horizon(
    df: pd.DataFrame,
    features: List[str],
    horizon: int,
    cfg: BaselineConfig,
) -> Tuple[List[Dict[str, object]], pd.DataFrame, pd.DataFrame]:
    target_col = f"fwd_ret_{horizon}d"
    if target_col not in df.columns:
        return [], pd.DataFrame(), pd.DataFrame()

    work = df[df[target_col].notna()].copy()
    if work.empty:
        return [], pd.DataFrame(), pd.DataFrame()

    sorted_dates = np.sort(work["date"].unique())
    purge_days = horizon if cfg.purge_days_override < 0 else cfg.purge_days_override

    fold_summaries: List[Dict[str, object]] = []
    fold_predictions: List[pd.DataFrame] = []
    importance_rows: List[Dict[str, object]] = []

    for fold_idx, (train_dates, test_dates) in enumerate(
        generate_walk_forward_splits(
            sorted_dates=sorted_dates,
            n_splits=cfg.n_splits,
            purge_days=purge_days,
            min_days_per_fold=cfg.min_days_per_fold,
        )
    ):
        tr_mask = work["date"].isin(train_dates)
        te_mask = work["date"].isin(test_dates)

        train_df = work.loc[tr_mask, ["ticker", "date", target_col] + features].copy()
        test_df = work.loc[te_mask, ["ticker", "date", target_col] + features].copy()

        if len(train_df) < cfg.min_rows_train or len(test_df) < cfg.min_rows_test:
            continue

        x_train = train_df[features].fillna(0.0)
        y_train = train_df[target_col].astype(np.float64)
        x_test = test_df[features].fillna(0.0)
        y_test = test_df[target_col].astype(np.float64)

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", Ridge(alpha=cfg.ridge_alpha)),
            ]
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        pred_df = test_df[["ticker", "date"]].copy()
        pred_df["target"] = y_test.values
        pred_df["pred"] = pred.astype(np.float64)
        pred_df["horizon"] = horizon
        pred_df["fold"] = fold_idx

        # Core error / direction metrics.
        rmse = float(np.sqrt(np.mean((pred_df["pred"] - pred_df["target"]) ** 2)))
        hit_rate = float(
            np.mean(np.sign(pred_df["pred"].values) == np.sign(pred_df["target"].values))
        )

        # Daily cross-sectional IC.
        ic_df = daily_ic(pred_df)
        ic_mean = float(ic_df["spearman_ic"].mean()) if not ic_df.empty else np.nan
        ic_std = float(ic_df["spearman_ic"].std(ddof=0)) if not ic_df.empty else np.nan
        ic_ir = float(ic_mean / ic_std * math.sqrt(len(ic_df))) if ic_std and not np.isnan(ic_std) else np.nan

        # Decile spread diagnostics.
        decile_df = decile_spread_by_day(
            pred_df=pred_df, n_deciles=cfg.n_deciles, min_names_per_day=cfg.min_names_per_day
        )
        spread_gross = float(decile_df["decile_spread"].mean()) if not decile_df.empty else np.nan

        # Unitless portfolio path for turnover/cost/drawdown diagnostics.
        weights_df = build_unitless_weights(
            pred_df=pred_df,
            min_names_per_day=cfg.min_names_per_day,
            max_abs_weight_z=cfg.max_abs_weight_z,
        )
        turnover_mean_daily, turnover_df = compute_turnover(weights_df)
        cost_drag = (
            (cfg.cost_bps / 10_000.0) * turnover_mean_daily
            if not np.isnan(turnover_mean_daily)
            else np.nan
        )
        spread_net = spread_gross - cost_drag if not np.isnan(spread_gross) and not np.isnan(cost_drag) else np.nan

        # Proper return diagnostics for horizon forecasts: non-overlapping cohorts.
        bt = horizon_non_overlapping_backtest(
            weights_df=weights_df,
            horizon=horizon,
            cost_bps=cfg.cost_bps,
        )
        turnover_mean = bt["turnover_mean"]
        max_dd = bt["max_drawdown_worst"]

        fold_summaries.append(
            {
                "horizon": horizon,
                "fold": fold_idx,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "train_start": str(pd.Timestamp(train_dates[0]).date()),
                "train_end": str(pd.Timestamp(train_dates[-1]).date()),
                "test_start": str(pd.Timestamp(test_dates[0]).date()),
                "test_end": str(pd.Timestamp(test_dates[-1]).date()),
                "purge_days": int(purge_days),
                "rmse": rmse,
                "hit_rate": hit_rate,
                "spearman_ic_mean": ic_mean,
                "spearman_ic_ir": ic_ir,
                "pearson_ic_mean": float(ic_df["pearson_ic"].mean()) if not ic_df.empty else np.nan,
                "decile_spread_gross": spread_gross,
                "turnover_mean": turnover_mean,
                "cost_drag_from_turnover": bt["cost_drag"],
                "decile_spread_net": spread_net,
                "max_drawdown_unitless": max_dd,
                "event_return_gross_mean": bt["event_return_gross_mean"],
                "event_return_net_mean": bt["event_return_net_mean"],
                "return_gross_annualized": bt["return_gross_annualized"],
                "return_net_annualized": bt["return_net_annualized"],
                "cohorts_used": bt["cohorts_used"],
            }
        )

        pred_df = pred_df.merge(ic_df[["date", "spearman_ic", "pearson_ic"]], on="date", how="left")
        pred_df["turnover"] = np.nan
        if not turnover_df.empty:
            pred_df = pred_df.drop(columns=["turnover"]).merge(turnover_df, on="date", how="left")
        fold_predictions.append(pred_df)

        ridge = model.named_steps["ridge"]
        for name, coef in zip(features, ridge.coef_):
            importance_rows.append(
                {"horizon": horizon, "fold": fold_idx, "feature": name, "coef": float(coef)}
            )

    pred_all = pd.concat(fold_predictions, ignore_index=True) if fold_predictions else pd.DataFrame()
    importance_df = pd.DataFrame(importance_rows)
    return fold_summaries, pred_all, importance_df


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if fold_metrics.empty:
        return out
    for horizon, g in fold_metrics.groupby("horizon"):
        out[str(horizon)] = {
            "folds": int(len(g)),
            "rmse_mean": float(g["rmse"].mean()),
            "hit_rate_mean": float(g["hit_rate"].mean()),
            "spearman_ic_mean": float(g["spearman_ic_mean"].mean()),
            "decile_spread_gross_mean": float(g["decile_spread_gross"].mean()),
            "decile_spread_net_mean": float(g["decile_spread_net"].mean()),
            "turnover_mean": float(g["turnover_mean"].mean()),
            "return_gross_annualized_mean": float(g["return_gross_annualized"].mean()),
            "return_net_annualized_mean": float(g["return_net_annualized"].mean()),
            "max_drawdown_unitless_worst": float(g["max_drawdown_unitless"].min()),
        }
    return out


def append_experiment_registry(
    output_dir: Path,
    summary: Dict[str, object],
    args: argparse.Namespace,
) -> Path:
    registry_path = output_dir / "experiment_registry.csv"
    horizon_summary: Dict[str, Dict[str, float]] = summary.get("horizon_summary", {})

    row: Dict[str, object] = {
        "run_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "input_path": summary.get("input_path"),
        "output_dir": summary.get("output_dir"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "n_rows_used": summary.get("n_rows_used"),
        "n_features": summary.get("n_features"),
        "universe_source": summary.get("universe_source"),
        "horizons": ",".join(str(h) for h in summary.get("config", {}).get("horizons", [])),
        "feature_mode": summary.get("config", {}).get("feature_mode"),
        "include_prefixes": ",".join(args.include_prefix),
        "exclude_prefixes": ",".join(args.exclude_prefix),
        "cost_bps": summary.get("config", {}).get("cost_bps"),
        "ridge_alpha": summary.get("config", {}).get("ridge_alpha"),
        "n_splits": summary.get("config", {}).get("n_splits"),
        "min_feature_completeness": summary.get("config", {}).get("min_feature_completeness"),
        "purge_days_override": summary.get("config", {}).get("purge_days_override"),
    }

    for horizon_key, metrics in horizon_summary.items():
        row[f"h{horizon_key}_folds"] = metrics.get("folds")
        row[f"h{horizon_key}_ic"] = metrics.get("spearman_ic_mean")
        row[f"h{horizon_key}_spread_gross"] = metrics.get("decile_spread_gross_mean")
        row[f"h{horizon_key}_spread_net"] = metrics.get("decile_spread_net_mean")
        row[f"h{horizon_key}_turnover"] = metrics.get("turnover_mean")
        row[f"h{horizon_key}_ret_gross_ann"] = metrics.get("return_gross_annualized_mean")
        row[f"h{horizon_key}_ret_net_ann"] = metrics.get("return_net_annualized_mean")
        row[f"h{horizon_key}_mdd"] = metrics.get("max_drawdown_unitless_worst")
        row[f"h{horizon_key}_hit_rate"] = metrics.get("hit_rate_mean")
        row[f"h{horizon_key}_rmse"] = metrics.get("rmse_mean")

    row_df = pd.DataFrame([row])
    if registry_path.exists():
        existing = pd.read_csv(registry_path)
        out_df = pd.concat([existing, row_df], ignore_index=True)
    else:
        out_df = row_df
    out_df.to_csv(registry_path, index=False)
    return registry_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast baseline (institutional)")
    parser.add_argument("-i", "--input", required=True, help="Normalized parquet input file")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory (default: <input_dir>/baseline_forecast_institutional)",
    )
    parser.add_argument(
        "--horizons",
        type=parse_horizons,
        default=parse_horizons("20,60,90"),
        help="Comma-separated forward horizons (default: 20,60,90)",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Walk-forward fold count")
    parser.add_argument(
        "--feature-mode",
        choices=["all_numeric", "normalized", "all", "core"],
        default="all_numeric",
        help=(
            "'all_numeric' = all numeric lagged features (best full-system mode), "
            "'normalized'/'all' = normalized-only features, "
            "'core' = return-family core set"
        ),
    )
    parser.add_argument(
        "--include-prefix",
        action="append",
        default=[],
        help="Optional feature family prefix to include (repeatable).",
    )
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="Optional feature family prefix to exclude (repeatable).",
    )
    parser.add_argument("--min-feature-completeness", type=float, default=0.70)
    parser.add_argument("--min-rows-train", type=int, default=15_000)
    parser.add_argument("--min-rows-test", type=int, default=5_000)
    parser.add_argument("--min-days-per-fold", type=int, default=40)
    parser.add_argument("--min-names-per-day", type=int, default=30)
    parser.add_argument("--cost-bps", type=float, default=10.0, help="Turnover cost in bps")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--purge-days",
        type=int,
        default=-1,
        help="Purge gap between train/test in trading days; -1 means use horizon value.",
    )
    parser.add_argument(
        "--ignore-in-universe",
        action="store_true",
        help="Ignore input in_universe column even if present.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        fail(f"Input not found: {input_path}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else input_path.parent / "baseline_forecast_institutional"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = BaselineConfig(
        horizons=args.horizons,
        n_splits=args.n_splits,
        min_rows_train=args.min_rows_train,
        min_rows_test=args.min_rows_test,
        min_days_per_fold=args.min_days_per_fold,
        min_feature_completeness=args.min_feature_completeness,
        min_names_per_day=args.min_names_per_day,
        ridge_alpha=args.ridge_alpha,
        cost_bps=args.cost_bps,
        feature_mode=args.feature_mode,
        purge_days_override=args.purge_days,
    )

    t0 = time.time()
    print(f"Loading: {input_path}")
    df = pd.read_parquet(input_path)
    if "ticker" not in df.columns or "date" not in df.columns:
        fail("Input must include ticker and date columns.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = int(df["date"].isna().sum())
    if bad_dates:
        print(f"WARNING: dropping {bad_dates:,} rows with invalid dates.")
        df = df[df["date"].notna()].copy()

    df["ticker"] = df["ticker"].astype(str).str.strip()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    dup = int(df.duplicated(["ticker", "date"]).sum())
    if dup:
        print(f"WARNING: dropping {dup:,} duplicate ticker/date rows (keeping last).")
        df = df.drop_duplicates(["ticker", "date"], keep="last")

    # Universe handling.
    if (not args.ignore_in_universe) and ("in_universe" in df.columns):
        univ = df["in_universe"]
        if not pd.api.types.is_bool_dtype(univ):
            univ = univ.astype("string").str.strip().str.lower().isin(["1", "true", "t", "yes", "y"])
        df = df[univ.fillna(False)].copy()
        universe_source = "input_in_universe"
    else:
        universe_source = "all_rows"

    features = build_feature_set(
        df=df,
        feature_mode=cfg.feature_mode,
        include_prefixes=args.include_prefix,
        exclude_prefixes=args.exclude_prefix,
    )
    if not features:
        fail("No usable features found after feature selection.")

    keep_mask = feature_completeness_filter(df, features, cfg.min_feature_completeness)
    removed = int((~keep_mask).sum())
    df = df.loc[keep_mask].copy()
    if df.empty:
        fail("No rows remain after feature completeness filter.")

    print(
        f"Rows: {len(df):,} | Features: {len(features)} | "
        f"Dropped by completeness: {removed:,} | Universe source: {universe_source}"
    )

    all_fold_rows: List[Dict[str, object]] = []
    all_preds: List[pd.DataFrame] = []
    all_importance: List[pd.DataFrame] = []

    for horizon in cfg.horizons:
        print(f"\n=== Horizon {horizon}d ===")
        fold_rows, pred_df, imp_df = train_and_evaluate_horizon(df, features, horizon, cfg)
        if not fold_rows:
            print("Skipped: insufficient data for configured folds.")
            continue
        fold_frame = pd.DataFrame(fold_rows)
        all_fold_rows.extend(fold_rows)
        all_preds.append(pred_df)
        all_importance.append(imp_df)
        print(
            f"Folds={len(fold_frame)} | "
            f"IC={fold_frame['spearman_ic_mean'].mean():+.4f} | "
            f"RetNetAnn={fold_frame['return_net_annualized'].mean():+.2%} | "
            f"MDD={fold_frame['max_drawdown_unitless'].min():+.2%} | "
            f"SpreadNet={fold_frame['decile_spread_net'].mean():+.5f}"
        )

    fold_metrics = pd.DataFrame(all_fold_rows)
    predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    importance = pd.concat(all_importance, ignore_index=True) if all_importance else pd.DataFrame()

    if fold_metrics.empty:
        fail("No successful folds across requested horizons. Relax filters or fold settings.")

    summary = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "elapsed_seconds": round(time.time() - t0, 2),
        "n_rows_used": int(len(df)),
        "n_features": int(len(features)),
        "features": features,
        "config": asdict(cfg),
        "universe_source": universe_source,
        "horizon_summary": summarize_fold_metrics(fold_metrics),
    }

    # Persist outputs.
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    if not predictions.empty:
        predictions.to_parquet(output_dir / "predictions.parquet", index=False)
        daily_ic_out = (
            predictions.groupby(["horizon", "fold", "date"], as_index=False)[["spearman_ic", "pearson_ic", "turnover"]]
            .mean()
        )
        daily_ic_out.to_csv(output_dir / "daily_ic.csv", index=False)
    else:
        pd.DataFrame(columns=["horizon", "fold", "date", "spearman_ic", "pearson_ic", "turnover"]).to_csv(
            output_dir / "daily_ic.csv", index=False
        )

    if not importance.empty:
        # Average absolute coefficients across folds.
        imp = (
            importance.assign(abs_coef=lambda x: x["coef"].abs())
            .groupby(["horizon", "feature"], as_index=False)["abs_coef"]
            .mean()
            .sort_values(["horizon", "abs_coef"], ascending=[True, False])
        )
        imp.to_csv(output_dir / "feature_importance.csv", index=False)
    else:
        pd.DataFrame(columns=["horizon", "feature", "abs_coef"]).to_csv(
            output_dir / "feature_importance.csv", index=False
        )

    registry_path = append_experiment_registry(output_dir=output_dir, summary=summary, args=args)

    print("\nBaseline forecasting complete.")
    print(f"Output directory: {output_dir}")
    print(
        "Wrote: metrics_summary.json, fold_metrics.csv, daily_ic.csv, "
        "predictions.parquet, feature_importance.csv, experiment_registry.csv"
    )
    print(f"Registry: {registry_path}")


if __name__ == "__main__":
    main()
