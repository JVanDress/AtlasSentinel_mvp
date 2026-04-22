from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger("quintic.v3")

PRICE_REQUIRED_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume"]
PRICE_MINIMAL_COLUMNS = ["ticker", "date", "close", "volume"]


class PipelineError(Exception):
    pass


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def list_raw_price_files(path: Path) -> list[Path]:
    if not path.exists() or not path.is_dir():
        raise PipelineError(f"Raw prices directory not found: {path}")
    files = sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in {".parquet", ".pq", ".csv"}
    )
    if not files:
        raise PipelineError(f"No parquet/csv files found in {path}")
    return files


def read_price_file(file_path: Path, minimal: bool = False) -> pd.DataFrame:
    desired_columns = PRICE_MINIMAL_COLUMNS if minimal else PRICE_REQUIRED_COLUMNS
    if file_path.suffix.lower() in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(file_path, columns=desired_columns)
        except Exception:
            return pd.read_parquet(file_path)
    try:
        return pd.read_csv(file_path, usecols=desired_columns)
    except Exception:
        return pd.read_csv(file_path)


def normalize_price_frame(df: pd.DataFrame, fallback_ticker: str) -> pd.DataFrame:
    work = df.copy()
    work.columns = [str(c).strip().lower() for c in work.columns]

    if "ticker" not in work.columns:
        work["ticker"] = fallback_ticker
    if "date" not in work.columns:
        raise PipelineError("Price file missing required column: date")
    if "close" not in work.columns:
        raise PipelineError("Price file missing required column: close")
    if "volume" not in work.columns:
        raise PipelineError("Price file missing required column: volume")

    work["ticker"] = work["ticker"].astype(str).str.upper().str.strip()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")

    numeric_candidates = [c for c in ["open", "high", "low", "close", "volume", "vwap", "transactions"] if c in work.columns]
    for column in numeric_candidates:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    for required in ["open", "high", "low"]:
        if required not in work.columns:
            work[required] = pd.NA

    keep = [c for c in ["ticker", "date", "open", "high", "low", "close", "volume", "vwap", "transactions"] if c in work.columns]
    work = work[keep]
    work = work.dropna(subset=["ticker", "date", "close", "volume"])
    work = work.sort_values(["ticker", "date"]).reset_index(drop=True)
    return work


def deduplicate_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(prices)
    deduped = prices.drop_duplicates(subset=["ticker", "date"], keep="last").sort_values(["ticker", "date"]).reset_index(drop=True)
    removed = before - len(deduped)
    return deduped, removed


def assert_no_future_dates(prices: pd.DataFrame, as_of: str | None) -> None:
    if prices.empty:
        return
    max_allowed = pd.Timestamp(as_of) if as_of else pd.Timestamp.utcnow().normalize()
    future_rows = prices.loc[prices["date"] > max_allowed]
    if not future_rows.empty:
        raise PipelineError(
            f"Found {len(future_rows)} price rows after allowed max date {max_allowed.date()}"
        )


def stable_reference_industry(sector: object, industry: object, sic_description: object, type_description: object) -> str:
    for value in [industry, sic_description, sector]:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    type_text = "" if type_description is None else str(type_description).strip()
    return type_text or "UNCLASSIFIED"


def dataframe_memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum()) / (1024.0 * 1024.0)


def env_or_default(env_name: str, default: str | None = None) -> str | None:
    value = os.getenv(env_name)
    return value if value not in {None, ""} else default


def rows_to_dicts(df: pd.DataFrame, max_rows: int = 50) -> list[dict]:
    if df.empty:
        return []
    return df.head(max_rows).to_dict(orient="records")
