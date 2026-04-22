from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, Iterable, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RatioConfig:
    # moneyness bands around spot: (name, low_pct, high_pct)
    # where pct is relative distance from spot: (K/spot - 1)
    moneyness_bands: Tuple[Tuple[str, float, float], ...] = (
        ("near_2pct", -0.02, 0.02),
        ("near_5pct", -0.05, 0.05),
        ("below_5pct", -0.05, 0.0),
        ("above_5pct", 0.0, 0.05),
        ("below_10pct", -0.10, 0.0),
        ("above_10pct", 0.0, 0.10),
    )

    # DTE buckets: (name, min_dte, max_dte_inclusive)
    dte_buckets: Tuple[Tuple[str, int, int], ...] = (
        ("dte_0_7", 0, 7),
        ("dte_8_30", 8, 30),
        ("dte_31_90", 31, 90),
        ("dte_91_180", 91, 180),
        ("dte_181_plus", 181, 10_000),
    )

    eps: float = 1e-12


def _to_date(x) -> Optional[date]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    # string parse
    try:
        return pd.to_datetime(x).date()
    except Exception:
        return None


def _safe_ratio(numer: float, denom: float, eps: float) -> float:
    denom = float(denom)
    numer = float(numer)
    if abs(denom) < eps:
        return 0.0
    return numer / denom


def compute_options_ratios(
    chain: pd.DataFrame,
    spot: float,
    asof_utc: date,
    cfg: RatioConfig = RatioConfig(),
) -> Dict[str, float]:
    """
    Returns a stable, minimal, *correct* set of ratios.

    All ratios are expressed as shares in [0,1] where appropriate:
      - put_share_oi = put_oi / (put_oi + call_oi)
      - call_share_oi = call_oi / (put_oi + call_oi)
    plus:
      - pcr_oi = put_oi / call_oi  (classic PCR; unbounded)
      - pcr_vol = put_vol / call_vol (if volume exists)
    """

    out: Dict[str, float] = {}

    if chain is None or chain.empty or spot is None or spot <= 0:
        # return zeros with predictable keys
        for b, _, _ in cfg.moneyness_bands:
            out[f"put_share_oi_{b}"] = 0.0
            out[f"call_share_oi_{b}"] = 0.0
            out[f"pcr_oi_{b}"] = 0.0
        for d, _, _ in cfg.dte_buckets:
            out[f"put_share_oi_{d}"] = 0.0
            out[f"call_share_oi_{d}"] = 0.0
            out[f"pcr_oi_{d}"] = 0.0
        out["put_share_oi_all"] = 0.0
        out["call_share_oi_all"] = 0.0
        out["pcr_oi_all"] = 0.0
        return out

    df = chain.copy()

    # Normalize columns
    # contract_type
    if "contract_type" not in df.columns:
        raise ValueError("chain missing contract_type")
    df["contract_type"] = df["contract_type"].astype(str).str.lower().str.strip()

    # strike
    strike_col = "strike" if "strike" in df.columns else ("strike_price" if "strike_price" in df.columns else None)
    if strike_col is None:
        raise ValueError("chain missing strike/strike_price")
    df["strike"] = pd.to_numeric(df[strike_col], errors="coerce")

    # expiration
    exp_col = "expiration" if "expiration" in df.columns else ("expiration_date" if "expiration_date" in df.columns else None)
    if exp_col is None:
        raise ValueError("chain missing expiration/expiration_date")
    df["_exp_date"] = df[exp_col].apply(_to_date)

    # OI and volume
    oi_col = "open_interest" if "open_interest" in df.columns else ("oi" if "oi" in df.columns else None)
    if oi_col is None:
        raise ValueError("chain missing open_interest/oi")
    df["open_interest"] = pd.to_numeric(df[oi_col], errors="coerce").fillna(0.0)

    vol_col = "volume" if "volume" in df.columns else None
    if vol_col is not None:
        df["volume"] = pd.to_numeric(df[vol_col], errors="coerce").fillna(0.0)
    else:
        df["volume"] = 0.0  # keep stable

    # Derived columns
    df = df.dropna(subset=["strike"])
    df["_mny"] = (df["strike"] / float(spot)) - 1.0  # relative moneyness
    df["_dte"] = df["_exp_date"].apply(lambda d: (d - asof_utc).days if d is not None else np.nan)
    df["_dte"] = pd.to_numeric(df["_dte"], errors="coerce")

    # Helper to compute shares/ratios on a slice
    def summarize(slice_df: pd.DataFrame, suffix: str) -> None:
        call_oi = float(slice_df.loc[slice_df["contract_type"] == "call", "open_interest"].sum())
        put_oi = float(slice_df.loc[slice_df["contract_type"] == "put", "open_interest"].sum())
        tot_oi = call_oi + put_oi

        out[f"put_share_oi_{suffix}"] = _safe_ratio(put_oi, tot_oi, cfg.eps)
        out[f"call_share_oi_{suffix}"] = _safe_ratio(call_oi, tot_oi, cfg.eps)
        out[f"pcr_oi_{suffix}"] = _safe_ratio(put_oi, call_oi, cfg.eps)  # classic PCR

        call_vol = float(slice_df.loc[slice_df["contract_type"] == "call", "volume"].sum())
        put_vol = float(slice_df.loc[slice_df["contract_type"] == "put", "volume"].sum())
        out[f"pcr_vol_{suffix}"] = _safe_ratio(put_vol, call_vol, cfg.eps)

    # All-chain summary
    summarize(df, "all")

    # Moneyness-band ratios (near spot is the money)
    for name, lo, hi in cfg.moneyness_bands:
        sl = df[(df["_mny"] >= lo) & (df["_mny"] <= hi)]
        summarize(sl, name)

    # DTE-bucket ratios
    # only include non-negative DTE (ignore expired)
    dfx = df[df["_dte"].notna() & (df["_dte"] >= 0)].copy()
    for name, lo, hi in cfg.dte_buckets:
        sl = dfx[(dfx["_dte"] >= lo) & (dfx["_dte"] <= hi)]
        summarize(sl, name)

    return out