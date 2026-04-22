from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import RuntimeConfig
from .utils import PipelineError

DEFAULT_TYPES_URL = "https://api.polygon.io/v3/reference/tickers/types"
DEFAULT_OVERVIEW_URL_TEMPLATE = "https://api.polygon.io/v3/reference/tickers/{ticker}"
COMMON_STOCK_TYPE_HINTS = {
    "CS",
    "COMMONSTOCK",
    "COMMON STOCK",
    "CLASS A COMMON STOCK",
    "CLASS B COMMON STOCK",
    "ORDINARYSHARES",
    "ORDINARY SHARES",
}


@dataclass(frozen=True)
class PolygonTickerOverview:
    row: dict[str, object] | None
    fetch_status: str


def build_session(config: RuntimeConfig) -> requests.Session:
    retry = Retry(
        total=5,
        read=5,
        connect=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": config.user_agent})
    return session


def polygon_get(session: requests.Session, url: str, params: dict[str, object], timeout: float) -> dict:
    response = session.get(url, params=params, timeout=timeout)
    if response.status_code == 404:
        return {"status": "NOT_FOUND", "results": None, "message": "Ticker not found"}
    if response.status_code >= 400:
        raise PipelineError(
            f"Polygon request failed: status={response.status_code} url={response.url} body={response.text[:500]}"
        )
    return response.json()


def fetch_ticker_types(session: requests.Session, config: RuntimeConfig) -> pd.DataFrame:
    payload = polygon_get(
        session,
        DEFAULT_TYPES_URL,
        {"asset_class": "stocks", "apiKey": config.polygon_api_key},
        timeout=config.request_timeout_seconds,
    )
    rows: list[dict[str, object]] = []
    for item in payload.get("results", []) or []:
        rows.append(
            {
                "type_code": item.get("code") or item.get("ticker_type") or item.get("type"),
                "type_description": item.get("description") or item.get("name"),
            }
        )
    return pd.DataFrame(rows)


def fetch_ticker_overview(session: requests.Session, ticker: str, config: RuntimeConfig) -> PolygonTickerOverview:
    params: dict[str, object] = {"apiKey": config.polygon_api_key}
    if config.as_of:
        params["date"] = config.as_of

    payload = polygon_get(
        session,
        DEFAULT_OVERVIEW_URL_TEMPLATE.format(ticker=ticker),
        params=params,
        timeout=config.request_timeout_seconds,
    )

    if payload.get("status") == "NOT_FOUND":
        return PolygonTickerOverview(row=None, fetch_status="ticker_not_found")

    result = payload.get("results") or {}
    if not result:
        return PolygonTickerOverview(row=None, fetch_status="empty_result")

    row = {
        "ticker": ticker,
        "active": result.get("active"),
        "name": result.get("name"),
        "market": result.get("market"),
        "locale": result.get("locale"),
        "primary_exchange": result.get("primary_exchange"),
        "currency_name": result.get("currency_name"),
        "list_date": result.get("list_date"),
        "delisted_utc": result.get("delisted_utc"),
        "cik": result.get("cik"),
        "sic_code": result.get("sic_code"),
        "sic_description": result.get("sic_description"),
        "ticker_root": result.get("ticker_root"),
        "ticker_suffix": result.get("ticker_suffix"),
        "type_code": result.get("type"),
        "market_cap": result.get("market_cap"),
        "weighted_shares_outstanding": result.get("weighted_shares_outstanding"),
        "share_class_shares_outstanding": result.get("share_class_shares_outstanding"),
        "sector": result.get("sector") or result.get("gics_sector") or result.get("sector_name"),
        "industry": result.get("industry") or result.get("gics_industry") or result.get("industry_name"),
        "sub_industry": result.get("sub_industry") or result.get("gics_sub_industry") or result.get("sub_industry_name"),
        "source": "polygon_ticker_overview",
        "reference_as_of": config.as_of,
    }
    return PolygonTickerOverview(row=row, fetch_status="ok")


def normalize_type_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).upper().replace("-", " ").replace("_", " ").strip()


def is_common_stock_row(row: pd.Series) -> bool:
    code = normalize_type_text(row.get("type_code"))
    description = normalize_type_text(row.get("type_description"))
    name = normalize_type_text(row.get("name"))

    if code in COMMON_STOCK_TYPE_HINTS or description in COMMON_STOCK_TYPE_HINTS:
        return True
    if "COMMON STOCK" in description or "COMMON STOCK" in name:
        return True
    if description.startswith("CLASS ") and "COMMON STOCK" in description:
        return True
    if any(
        bad in description
        for bad in ["ETF", "FUND", "TRUST", "ETN", "ADR", "WARRANT", "UNIT", "PREFERRED", "RIGHT"]
    ):
        return False
    return False


def reference_status_row(row: pd.Series) -> str:
    if pd.isna(row.get("name")):
        return "missing_overview"
    if row.get("active") is False:
        return "inactive"
    if not row.get("is_common_stock"):
        return "non_common_stock"
    if pd.isna(row.get("market_cap")):
        return "missing_market_cap"
    return "ok"


def fetch_reference_rows(tickers: list[str], config: RuntimeConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not config.polygon_api_key:
        raise PipelineError("Missing Polygon API key. Pass --api-key or set POLYGON_API_KEY.")

    session = build_session(config)
    type_lookup = fetch_ticker_types(session, config)

    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    total = len(tickers)

    for index, ticker in enumerate(tickers, start=1):
        fetch_status = "ok"
        error_message = None
        try:
            result = fetch_ticker_overview(session, ticker, config)
            fetch_status = result.fetch_status
            if result.row is not None:
                rows.append(result.row)
        except Exception as exc:
            fetch_status = "error"
            error_message = str(exc)[:500]

        audits.append(
            {
                "ticker": ticker,
                "fetch_status": fetch_status,
                "error_message": error_message,
                "sequence_number": index,
                "total_tickers": total,
            }
        )
        if config.sleep_seconds_between_calls > 0:
            time.sleep(config.sleep_seconds_between_calls)

    reference = pd.DataFrame(rows)
    audit = pd.DataFrame(audits).sort_values(["fetch_status", "ticker"]).reset_index(drop=True)
    if reference.empty:
        return reference, audit

    if not type_lookup.empty:
        reference = reference.merge(
            type_lookup[["type_code", "type_description"]].drop_duplicates(),
            how="left",
            on="type_code",
            validate="m:1",
        )
    else:
        reference["type_description"] = None

    numeric_columns = ["market_cap", "weighted_shares_outstanding", "share_class_shares_outstanding"]
    for column in numeric_columns:
        reference[column] = pd.to_numeric(reference[column], errors="coerce")

    reference["is_common_stock"] = reference.apply(is_common_stock_row, axis=1)
    reference["reference_status"] = reference.apply(reference_status_row, axis=1)
    reference = reference.sort_values(["reference_status", "ticker"]).reset_index(drop=True)
    return reference, audit
