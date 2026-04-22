#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quintic_paths import data_dir, project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile GMM regime durations and transitions on a cleanroom regime panel."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--input-parquet", type=Path, default=None)
    parser.add_argument("--runs-parquet", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--transitions-csv", type=Path, default=None)
    parser.add_argument("--status-output", type=Path, default=None)
    parser.add_argument("--group-column", default="ticker")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--regime-column", default="gmm_regime_label")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def runtime_project_root(cli_root: Path | None) -> Path:
    if cli_root is not None:
        return cli_root.expanduser()
    return project_root()


def resolve_path(path: Path | None, base_dir: Path, default_name: str, root_dir: Path | None = None) -> Path:
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


def load_panel(path: Path, group_column: str, date_column: str, regime_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required parquet not found: {path}")

    df = pd.read_parquet(path)
    required = {group_column, date_column, regime_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}")

    df = df.copy()
    df[group_column] = df[group_column].astype(str).str.upper().str.strip()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df[regime_column] = pd.to_numeric(df[regime_column], errors="coerce")
    df = df.dropna(subset=[group_column, date_column]).copy()
    df = df[df[group_column].ne("")].copy()
    return df.sort_values([group_column, date_column]).reset_index(drop=True)


def build_runs(df: pd.DataFrame, group_column: str, date_column: str, regime_column: str) -> pd.DataFrame:
    assigned = df.dropna(subset=[regime_column]).copy()
    if assigned.empty:
        raise RuntimeError("No rows with non-null regime labels were found.")

    assigned["_regime_change"] = (
        assigned[group_column].ne(assigned[group_column].shift(1))
        | assigned[regime_column].ne(assigned[regime_column].shift(1))
    )
    assigned["_run_id"] = assigned["_regime_change"].cumsum()

    runs = (
        assigned.groupby([group_column, "_run_id", regime_column], as_index=False)
        .agg(
            start_date=(date_column, "min"),
            end_date=(date_column, "max"),
            duration_rows=(date_column, "size"),
        )
        .sort_values([group_column, "start_date"])
        .reset_index(drop=True)
    )
    runs["duration_calendar_days"] = (runs["end_date"] - runs["start_date"]).dt.days + 1
    return runs


def summarize_runs(runs: pd.DataFrame, regime_column: str) -> pd.DataFrame:
    summary = (
        runs.groupby(regime_column)
        .agg(
            n_runs=("duration_rows", "size"),
            mean_duration_rows=("duration_rows", "mean"),
            median_duration_rows=("duration_rows", "median"),
            p90_duration_rows=("duration_rows", lambda s: float(np.quantile(s, 0.90))),
            max_duration_rows=("duration_rows", "max"),
            mean_calendar_days=("duration_calendar_days", "mean"),
            median_calendar_days=("duration_calendar_days", "median"),
            max_calendar_days=("duration_calendar_days", "max"),
        )
        .reset_index()
        .sort_values(regime_column)
        .reset_index(drop=True)
    )
    return summary


def summarize_transitions(df: pd.DataFrame, group_column: str, date_column: str, regime_column: str) -> pd.DataFrame:
    assigned = df.dropna(subset=[regime_column]).copy()
    assigned = assigned.sort_values([group_column, date_column]).reset_index(drop=True)
    assigned["next_group"] = assigned[group_column].shift(-1)
    assigned["next_regime"] = assigned[regime_column].shift(-1)
    transitions = assigned[assigned[group_column].eq(assigned["next_group"])].copy()
    transitions = transitions[transitions[regime_column].ne(transitions["next_regime"])].copy()
    if transitions.empty:
        return pd.DataFrame(columns=["from_regime", "to_regime", "transition_count", "transition_prob"])

    out = (
        transitions.groupby([regime_column, "next_regime"])
        .size()
        .rename("transition_count")
        .reset_index()
        .rename(columns={regime_column: "from_regime", "next_regime": "to_regime"})
        .sort_values(["from_regime", "transition_count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    out["transition_prob"] = out["transition_count"] / out.groupby("from_regime")["transition_count"].transform("sum")
    return out


def main() -> int:
    args = parse_args()
    root = runtime_project_root(args.project_root)
    data_root = data_dir(root)
    data_root.mkdir(parents=True, exist_ok=True)

    input_path = resolve_path(args.input_parquet, data_root, "cleanroom_gmm_panel.parquet", root_dir=root)
    runs_path = resolve_path(args.runs_parquet, data_root, "cleanroom_gmm_regime_runs.parquet", root_dir=root)
    summary_path = resolve_path(args.summary_csv, data_root, "cleanroom_gmm_regime_duration_summary.csv", root_dir=root)
    transitions_path = resolve_path(args.transitions_csv, data_root, "cleanroom_gmm_regime_transitions.csv", root_dir=root)
    status_path = resolve_path(args.status_output, data_root, "cleanroom_gmm_regime_diagnostics.json", root_dir=root)

    panel = load_panel(
        input_path,
        group_column=args.group_column,
        date_column=args.date_column,
        regime_column=args.regime_column,
    )
    runs = build_runs(panel, group_column=args.group_column, date_column=args.date_column, regime_column=args.regime_column)
    summary = summarize_runs(runs, regime_column=args.regime_column)
    transitions = summarize_transitions(panel, group_column=args.group_column, date_column=args.date_column, regime_column=args.regime_column)

    print(f"Project root: {root}")
    print(f"Input parquet: {input_path}")
    print(f"Assigned rows: {int(panel[args.regime_column].notna().sum()):,}")
    print(f"Tickers with regimes: {int(panel.loc[panel[args.regime_column].notna(), args.group_column].nunique()):,}")
    print(f"Runs found: {len(runs):,}")

    if args.dry_run:
        print("Dry run complete. No diagnostics written.")
        return 0

    runs_path.parent.mkdir(parents=True, exist_ok=True)
    runs.to_parquet(runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    transitions.to_csv(transitions_path, index=False)

    status = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "input_parquet": str(input_path),
        "runs_parquet": str(runs_path),
        "summary_csv": str(summary_path),
        "transitions_csv": str(transitions_path),
        "group_column": args.group_column,
        "date_column": args.date_column,
        "regime_column": args.regime_column,
        "assigned_rows": int(panel[args.regime_column].notna().sum()),
        "tickers_with_regimes": int(panel.loc[panel[args.regime_column].notna(), args.group_column].nunique()),
        "runs_found": int(len(runs)),
        "regimes_profiled": summary[args.regime_column].dropna().astype(int).tolist() if not summary.empty else [],
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    print(f"Saved runs parquet: {runs_path}")
    print(f"Saved duration summary: {summary_path}")
    print(f"Saved transitions summary: {transitions_path}")
    print(f"Saved status file: {status_path}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
