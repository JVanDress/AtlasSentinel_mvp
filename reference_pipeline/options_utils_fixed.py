from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# Helpers
# ---------------------------


def _safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _sleep_backoff(attempt: int) -> None:
    base = min(2 ** max(attempt - 1, 0), 30)
    jitter = (attempt % 3) * 0.25
    time.sleep(base + jitter)


def _classify_exception(e: Exception) -> str:
    msg = str(e).lower()
    if "429" in msg or "rate limit" in msg or "too many requests" in msg:
        return "rate_limited"
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if "temporarily" in msg or "transient" in msg or "connection reset" in msg:
        return "transient"
    return "exception"


# ---------------------------
# Spot price helpers
# ---------------------------


def _spot_from_cache_close(prices_cache: Path, ticker: str) -> Optional[float]:
    try:
        if not prices_cache.exists():
            return None

        df = pd.read_parquet(prices_cache)

        if df.empty or "ticker" not in df.columns or "close" not in df.columns:
            return None

        t = str(ticker).upper().strip()
        df = df[df["ticker"].astype(str).str.upper().str.strip() == t]

        if df.empty:
            return None

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")

        px = _safe_float(df["close"].iloc[-1], default=np.nan)

        if np.isfinite(px) and px > 0:
            return float(px)

        return None

    except Exception:
        return None


def iter_options_chain(client: Any, underlying: str, params: Optional[Dict[str, Any]] = None) -> Iterable[Any]:
    params = params or {}

    try:
        return client.list_snapshot_options_chain(underlying_asset=underlying, params=params)
    except TypeError:
        return client.list_snapshot_options_chain(underlying, params=params)


def _spot_from_chain_first(client: Any, ticker: str, retries: int = 4) -> Optional[float]:
    if not hasattr(client, "list_snapshot_options_chain"):
        return None

    for attempt in range(1, retries + 1):
        try:
            it = iter_options_chain(client, ticker, params={"limit": 1})
            first = next(iter(it), None)

            if first is None:
                return None

            spot = getattr(getattr(first, "underlying_asset", None), "price", None)
            px = _safe_float(spot, default=np.nan)

            if np.isfinite(px) and px > 0:
                return float(px)

            return None

        except Exception as e:
            if _classify_exception(e) in {"rate_limited", "timeout", "transient"} and attempt < retries:
                _sleep_backoff(attempt)
                continue

            return None

    return None


def _spot_from_last_trade(client: Any, ticker: str, retries: int = 4) -> Optional[float]:
    if not hasattr(client, "get_last_trade"):
        return None

    for attempt in range(1, retries + 1):
        try:
            t = client.get_last_trade(ticker)
            px = _safe_float(getattr(t, "price", None), default=np.nan)

            if np.isfinite(px) and px > 0:
                return float(px)

            return None

        except Exception as e:
            if _classify_exception(e) in {"rate_limited", "timeout", "transient"} and attempt < retries:
                _sleep_backoff(attempt)
                continue

            return None

    return None


def get_spot_price(client: Any, prices_cache: Path, ticker: str) -> Tuple[Optional[float], str]:
    t = str(ticker).upper().strip()

    spot = _spot_from_cache_close(prices_cache, t)
    if spot is not None:
        return spot, "cache_close"

    spot = _spot_from_chain_first(client, t)
    if spot is not None:
        return spot, "options_chain_spot"

    spot = _spot_from_last_trade(client, t)
    if spot is not None:
        return spot, "last_trade"

    return None, "none"


# ---------------------------------------------------
# Chain fetch -> normalized dataframe
# ---------------------------------------------------


def fetch_chain_dataframe(
    client: Any,
    ticker: str,
    asof_date: str,
    max_contracts: Optional[int] = None,
    max_expiry_trading_days: Optional[int] = None,
) -> pd.DataFrame:
    rows = []
    asof_ts = pd.to_datetime(asof_date, errors="coerce")
    asof_day = asof_ts.date() if pd.notna(asof_ts) else None

    try:
        chain = iter_options_chain(client, ticker)

        for i, c in enumerate(chain):
            if max_contracts and i >= max_contracts:
                break

            details = getattr(c, "details", None)
            day = getattr(c, "day", None)
            greeks = getattr(c, "greeks", None)

            strike = _safe_float(getattr(details, "strike_price", None))
            expiration = getattr(details, "expiration_date", None)
            contract_type = getattr(details, "contract_type", None)
            volume = _safe_float(getattr(day, "volume", None))
            open_interest = _safe_float(getattr(c, "open_interest", None))
            iv = _safe_float(getattr(c, "implied_volatility", None))
            gamma = _safe_float(getattr(greeks, "gamma", None))
            theta = _safe_float(getattr(greeks, "theta", None))

            if max_expiry_trading_days is not None and asof_day is not None:
                exp_ts = pd.to_datetime(expiration, errors="coerce")
                if pd.isna(exp_ts):
                    continue
                exp_day = exp_ts.date()
                dte_trading = int(np.busday_count(asof_day.isoformat(), exp_day.isoformat()))
                if dte_trading < 0 or dte_trading > int(max_expiry_trading_days):
                    continue

            rows.append(
                {
                    "ticker": ticker,
                    "contract": getattr(c, "ticker", None),
                    # Canonical names expected by downstream studies.
                    "strike": strike,
                    "strike_price": strike,
                    "expiry": expiration,
                    "expiration": expiration,
                    "expiration_date": expiration,
                    "type": contract_type,
                    "contract_type": contract_type,
                    "volume": volume,
                    "open_interest": open_interest,
                    "iv": iv,
                    "implied_volatility": iv,
                    "gamma": gamma,
                    "theta": theta,
                }
            )

        return pd.DataFrame(rows)

    except Exception as e:
        print(f"Chain fetch failed for {ticker}: {e}")
        return pd.DataFrame()
