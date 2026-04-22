#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from polygon import RESTClient

from quintic_paths import data_dir, load_simple_dotenv, project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cleanroom options pressure/sentiment overlay and append daily snapshots."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--universe-parquet", type=Path, default=None)
    parser.add_argument("--prices-cache", type=Path, default=None)
    parser.add_argument("--output-parquet", type=Path, default=None)
    parser.add_argument("--status-output", type=Path, default=None)
    parser.add_argument("--asof-date", default=None, help="YYYY-MM-DD; defaults to today in UTC")
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--max-contracts", type=int, default=400)
    parser.add_argument("--max-expiry-trading-days", type=int, default=180)
    parser.add_argument(
        "--replace-date",
        action="store_true",
        help="Replace rows for the as-of date instead of appending duplicates",
    )
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
        if candidate.parts and candidate.parts[0] == base_dir.name:
            return root_dir / candidate

    if candidate.parts and candidate.parts[0] == base_dir.name:
        return base_dir.parent / candidate

    return base_dir / candidate


def require_polygon_key(root: Path) -> str:
    load_simple_dotenv(root, override=True)
    key = (os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("MASSIVE_API_KEY or POLYGON_API_KEY was not found in the environment.")
    return key


def parse_asof_date(value: str | None) -> date:
    if value:
        return pd.Timestamp(value).date()
    return datetime.now(timezone.utc).date()


def load_universe(universe_path: Path, max_tickers: int) -> list[str]:
    if not universe_path.exists():
        raise FileNotFoundError(f"Required universe parquet not found: {universe_path}")
    df = pd.read_parquet(universe_path)
    if "ticker" not in df.columns:
        raise KeyError(f"Missing required column 'ticker' in {universe_path}")
    tickers = (
        df["ticker"].astype(str).str.upper().str.strip().replace("", pd.NA).dropna().drop_duplicates().tolist()
    )
    tickers = sorted(tickers)
    if max_tickers and max_tickers > 0:
        tickers = tickers[: int(max_tickers)]
    if not tickers:
        raise RuntimeError("Universe is empty after cleaning ticker symbols.")
    return tickers


def load_existing_output(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def load_module_from_candidates(root: Path, module_name: str):
    candidates = [
        root / f"{module_name}.py",
        root / "scripts" / f"{module_name}.py",
        root / "scripts" / "reference_pipeline" / f"{module_name}.py",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        spec = importlib.util.spec_from_file_location(module_name, candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module
    raise ModuleNotFoundError(f"Could not locate {module_name}.py under {root}, {root / 'scripts'}, or {root / 'scripts' / 'reference_pipeline'}")


def load_first_available_module(root: Path, module_names: tuple[str, ...]):
    last_error: Exception | None = None
    for module_name in module_names:
        try:
            return load_module_from_candidates(root, module_name)
        except ModuleNotFoundError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise ModuleNotFoundError("No module names were provided.")


def main() -> int:
    args = parse_args()
    root = runtime_project_root(args.project_root)
    data_root = data_dir(root)
    data_root.mkdir(parents=True, exist_ok=True)

    pressure_mod = load_first_available_module(root, ("options_pressure_daily_fixed",))
    ratios_mod = load_first_available_module(root, ("options_ratios_fixed", "options_ratios"))
    utils_mod = load_first_available_module(root, ("options_utils_fixed", "options_utils"))

    compute_options_pressure = pressure_mod.compute_options_pressure
    compute_options_ratios = ratios_mod.compute_options_ratios
    fetch_chain_dataframe = utils_mod.fetch_chain_dataframe
    get_spot_price = utils_mod.get_spot_price

    universe_path = resolve_path(args.universe_parquet, data_root, "stocks_universe.parquet", root_dir=root)
    prices_cache = resolve_path(args.prices_cache, data_root, "cleanroom_transformed_panel.parquet", root_dir=root)
    output_path = resolve_path(args.output_parquet, data_root, "cleanroom_options_overlay.parquet", root_dir=root)
    status_path = resolve_path(args.status_output, data_root, "cleanroom_options_overlay_status.json", root_dir=root)
    asof_day = parse_asof_date(args.asof_date)

    tickers = load_universe(universe_path, max_tickers=int(args.max_tickers))
    api_key = require_polygon_key(root)
    client = RESTClient(api_key=api_key)

    print(f"Project root: {root}")
    print(f"Universe parquet: {universe_path}")
    print(f"Prices cache: {prices_cache}")
    print(f"Output parquet: {output_path}")
    print(f"As-of date: {asof_day.isoformat()}")
    print(f"Tickers requested: {len(tickers):,}")

    if args.dry_run:
        print("Dry run complete. No options overlay written.")
        return 0

    rows: list[dict[str, object]] = []
    failed: list[dict[str, str]] = []
    total = len(tickers)
    for idx, ticker in enumerate(tickers, start=1):
        spot, spot_source = get_spot_price(client, prices_cache, ticker)
        if spot is None or spot <= 0:
            failed.append({"ticker": ticker, "reason": "spot_unavailable"})
            continue

        chain_df = fetch_chain_dataframe(
            client,
            ticker=ticker,
            asof_date=asof_day.isoformat(),
            max_contracts=int(args.max_contracts) if args.max_contracts else None,
            max_expiry_trading_days=int(args.max_expiry_trading_days) if args.max_expiry_trading_days else None,
        )
        if chain_df.empty:
            failed.append({"ticker": ticker, "reason": "empty_chain"})
            continue

        try:
            pressure = compute_options_pressure(chain_df, spot=float(spot), asof_date=asof_day)
            ratios = compute_options_ratios(chain_df, spot=float(spot), asof_utc=asof_day)
        except Exception as exc:
            failed.append({"ticker": ticker, "reason": f"metrics_error:{type(exc).__name__}"})
            continue

        row: dict[str, object] = {
            "ticker": ticker,
            "date": pd.Timestamp(asof_day),
            "spot_price": float(spot),
            "spot_source": spot_source,
            "contracts_fetched": int(len(chain_df)),
        }
        row.update(pressure)
        row.update(ratios)
        rows.append(row)

        if idx % 25 == 0 or idx == total:
            print(f"Processed {idx:,}/{total:,} tickers | succeeded={len(rows):,} failed={len(failed):,}")

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        raise RuntimeError("No options overlay rows were produced.")

    existing = load_existing_output(output_path)
    if not existing.empty:
        combined = pd.concat([existing, result_df], ignore_index=True)
    else:
        combined = result_df.copy()

    combined["ticker"] = combined["ticker"].astype(str).str.upper().str.strip()
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["ticker", "date"]).copy()
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

    if args.replace_date:
        combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)

    status = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "universe_parquet": str(universe_path),
        "prices_cache": str(prices_cache),
        "output_parquet": str(output_path),
        "asof_date": asof_day.isoformat(),
        "tickers_requested": int(len(tickers)),
        "tickers_succeeded": int(result_df["ticker"].nunique()),
        "rows_written_today": int(len(result_df)),
        "rows_total": int(len(combined)),
        "failed": failed[:100],
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    print(f"Saved options overlay: {output_path}")
    print(f"Saved status file: {status_path}")
    print(f"Rows written today: {len(result_df):,}")
    print(f"Total rows in overlay history: {len(combined):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())