
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PressureConfig:
    eps: float = 1e-12
    near_spot_width: float = 0.05
    near_expiry_days: int = 7
    min_gamma_contract_coverage: float = 0.60
    min_gamma_oi_coverage: float = 0.60


def _to_date(x) -> Optional[date]:
    if x is None:
        return None
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    try:
        return pd.to_datetime(x).date()
    except Exception:
        return None


def compute_options_pressure(
    chain: pd.DataFrame,
    spot: float,
    asof_date: date,
    cfg: PressureConfig = PressureConfig(),
) -> Dict[str, float]:
    """
    Real pressure metrics from actual chain data.

    NOTE:
    - gamma_proxy is NOT true dealer GEX.
    - theta_pressure is a decay-weighted OI proxy.
    """
    zero = {
        "opt_call_vol": 0.0,
        "opt_put_vol": 0.0,
        "opt_put_call_vol_ratio": 0.0,
        "opt_call_oi": 0.0,
        "opt_put_oi": 0.0,
        "opt_put_call_oi_ratio": 0.0,
        "opt_near_expiry_call_pressure": 0.0,
        "opt_near_expiry_put_pressure": 0.0,
        "opt_strike_pressure_near_spot": 0.0,
        "gamma_proxy": 0.0,
        "theta_pressure": 0.0,
        "opt_iv_mean": 0.0,
        "contracts_processed": 0.0,
    }
    if chain is None or chain.empty or spot is None or spot <= 0:
        return zero

    df = chain.copy()
    df["strike_price"] = pd.to_numeric(df.get("strike_price", df.get("strike")), errors="coerce")
    df["open_interest"] = pd.to_numeric(df.get("open_interest", df.get("oi")), errors="coerce").fillna(0.0)
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0.0)
    df["implied_volatility"] = pd.to_numeric(df.get("implied_volatility"), errors="coerce")
    df["gamma"] = pd.to_numeric(df.get("gamma"), errors="coerce")
    df["theta"] = pd.to_numeric(df.get("theta"), errors="coerce")
    df["contract_type"] = df["contract_type"].astype(str).str.lower().str.strip()
    df["_exp_date"] = df.get("expiration_date", df.get("expiration")).apply(_to_date)

    df = df.dropna(subset=["strike_price"]).copy()
    if df.empty:
        return zero

    df["_mny"] = (df["strike_price"] / float(spot)) - 1.0
    df["_abs_mny"] = df["_mny"].abs()
    df["_dte"] = df["_exp_date"].apply(lambda d: (d - asof_date).days if d is not None else np.nan)
    df["_dte"] = pd.to_numeric(df["_dte"], errors="coerce")
    df = df[df["_dte"].isna() | (df["_dte"] >= 0)].copy()

    calls = df[df["contract_type"] == "call"].copy()
    puts = df[df["contract_type"] == "put"].copy()

    call_oi = float(calls["open_interest"].sum())
    put_oi = float(puts["open_interest"].sum())
    call_vol = float(calls["volume"].sum())
    put_vol = float(puts["volume"].sum())

    near_spot = df[df["_abs_mny"] <= cfg.near_spot_width].copy()
    if not near_spot.empty:
        call_near_oi = float(near_spot.loc[near_spot["contract_type"] == "call", "open_interest"].sum())
        put_near_oi = float(near_spot.loc[near_spot["contract_type"] == "put", "open_interest"].sum())
        strike_pressure = (call_near_oi - put_near_oi) / (call_near_oi + put_near_oi + cfg.eps)
    else:
        strike_pressure = 0.0

    near_expiry = df[df["_dte"].notna() & (df["_dte"] <= cfg.near_expiry_days)].copy()
    if not near_expiry.empty:
        near_expiry["_vel_weight"] = 1.0 / np.sqrt(near_expiry["_dte"].replace(0, 0.5))
        near_call_pressure = float(
            (near_expiry.loc[near_expiry["contract_type"] == "call", "open_interest"] *
             near_expiry.loc[near_expiry["contract_type"] == "call", "_vel_weight"]).sum()
        )
        near_put_pressure = float(
            (near_expiry.loc[near_expiry["contract_type"] == "put", "open_interest"] *
             near_expiry.loc[near_expiry["contract_type"] == "put", "_vel_weight"]).sum()
        )
    else:
        near_call_pressure = 0.0
        near_put_pressure = 0.0

    # Gamma proxy: use greeks when available; otherwise use near-spot OI weighted by 1/sqrt(DTE)
    gdf = near_spot.copy()
    if not gdf.empty:
        gdf["_vel_weight"] = 1.0 / np.sqrt(gdf["_dte"].replace(0, 0.5).fillna(cfg.near_expiry_days))
        gamma_rows = gdf[gdf["gamma"].notna()].copy()
        gamma_contract_coverage = float(len(gamma_rows)) / float(len(gdf))
        gamma_oi_coverage = float(gamma_rows["open_interest"].sum()) / (float(gdf["open_interest"].sum()) + cfg.eps)
        if (
            not gamma_rows.empty
            and gamma_contract_coverage >= cfg.min_gamma_contract_coverage
            and gamma_oi_coverage >= cfg.min_gamma_oi_coverage
        ):
            gamma_rows["_gamma_weighted"] = gamma_rows["gamma"] * gamma_rows["open_interest"] * gamma_rows["_vel_weight"]
            call_gamma = float(gamma_rows.loc[gamma_rows["contract_type"] == "call", "_gamma_weighted"].sum())
            put_gamma = float(gamma_rows.loc[gamma_rows["contract_type"] == "put", "_gamma_weighted"].sum())
        else:
            call_gamma = float(
                (gdf.loc[gdf["contract_type"] == "call", "open_interest"] *
                 gdf.loc[gdf["contract_type"] == "call", "_vel_weight"]).sum()
            )
            put_gamma = float(
                (gdf.loc[gdf["contract_type"] == "put", "open_interest"] *
                 gdf.loc[gdf["contract_type"] == "put", "_vel_weight"]).sum()
            )
        gamma_proxy = (call_gamma - put_gamma) / (abs(call_gamma) + abs(put_gamma) + cfg.eps)
    else:
        gamma_proxy = 0.0

    # Theta pressure proxy: larger absolute theta near expiry implies more decay pressure
    tdf = near_expiry.copy()
    if not tdf.empty:
        if tdf["theta"].notna().any():
            tdf["_theta_weighted"] = tdf["theta"].abs().fillna(0.0) * tdf["open_interest"]
            theta_pressure = float(tdf["_theta_weighted"].sum()) / (float(tdf["open_interest"].sum()) + cfg.eps)
        else:
            tdf["_theta_proxy"] = tdf["open_interest"] / np.sqrt(tdf["_dte"].replace(0, 0.5))
            theta_pressure = float(tdf["_theta_proxy"].sum()) / (float(tdf["open_interest"].sum()) + cfg.eps)
    else:
        theta_pressure = 0.0

    iv_mean = float(df["implied_volatility"].dropna().mean()) if df["implied_volatility"].notna().any() else 0.0

    return {
        "opt_call_vol": call_vol,
        "opt_put_vol": put_vol,
        "opt_put_call_vol_ratio": put_vol / (call_vol + cfg.eps),
        "opt_call_oi": call_oi,
        "opt_put_oi": put_oi,
        "opt_put_call_oi_ratio": put_oi / (call_oi + cfg.eps),
        "opt_near_expiry_call_pressure": near_call_pressure,
        "opt_near_expiry_put_pressure": near_put_pressure,
        "opt_strike_pressure_near_spot": strike_pressure,
        "gamma_proxy": gamma_proxy,
        "theta_pressure": theta_pressure,
        "opt_iv_mean": iv_mean,
        "contracts_processed": float(len(df)),
    }
