"""
Quintic Labs — Fundamentals Overlay

Pulls financial statements and daily ratios from Polygon/Massive
Financials API (requires Stocks Advanced plan) and engineers features
for 20/60/90 day probability forecasting.

Features produced:
    - Daily valuation ratios (P/E, P/S, P/B, EV/EBITDA) + sector ranks
    - Profitability metrics (ROE, ROA, margins) + sector ranks
    - Leverage & liquidity (D/E, current ratio, quick ratio)
    - Growth metrics (revenue, EPS, EBITDA QoQ and YoY)
    - Quality signals (FCF yield, earnings quality, margin trends)
    - Composite scores (value, quality, momentum-adjusted value)

Usage:
    python build_fundamentals_overlay.py --universe Z:\\jvand\\QUINTIC_V3\\data\\stage\\stocks_universe.parquet
    python build_fundamentals_overlay.py --universe universe.parquet --output Z:\\jvand\\QUINTIC_V3\\data\\stage\\fundamentals_overlay.parquet
    python build_fundamentals_overlay.py --universe universe.parquet --ratios-only

Requirements:
    pip install polygon-api-client pandas pyarrow numpy requests
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


# ===================================================================
# Configuration
# ===================================================================
FINANCIALS_BASE = "https://api.polygon.io/stocks/financials/v1"
RATE_LIMIT_PAUSE = 0.15   # seconds between API calls
RETRY_PAUSE = 2.0
MAX_RETRIES = 3
EPS = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(description="Quintic Labs — Fundamentals Overlay")
    parser.add_argument("--universe", "-u", required=True, help="Universe parquet path")
    parser.add_argument("--output", "-o", default=None, help="Output parquet path")
    parser.add_argument("--status-output", default=None, help="Status JSON path")
    parser.add_argument("--api-key", default=None, help="Override API key (prefer .env file)")
    parser.add_argument("--ratios-only", action="store_true", help="Only pull daily ratios (faster)")
    parser.add_argument("--max-tickers", type=int, default=0, help="Limit tickers for testing")
    parser.add_argument("--max-quarters", type=int, default=8, help="Quarters of history for growth metrics")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _locate_and_load_dotenv():
    """
    Search for .env file in standard locations and load it.
    Search order: current dir, script dir, project root candidates.
    """
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parents[1] / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    # Also check common project root env vars
    for env_var in ("QUINTIC_PROJECT_ROOT",):
        root = os.environ.get(env_var)
        if root:
            candidates.append(Path(root).expanduser() / ".env")

    for path in candidates:
        if path.exists():
            load_dotenv(path, override=True)
            return str(path)

    # Fallback: let dotenv search normally
    load_dotenv(override=True)
    return None


def get_api_key(cli_key=None):
    """Load API key from .env file, environment, or CLI override."""
    env_path = _locate_and_load_dotenv()

    key = (
        cli_key
        or os.environ.get("POLYGON_API_KEY")
        or os.environ.get("MASSIVE_API_KEY")
        or ""
    ).strip()

    if not key:
        locations_checked = ".env file" + (f" ({env_path})" if env_path else " (not found)")
        sys.exit(
            f"Error: API key not found.\n"
            f"  Checked: {locations_checked}, POLYGON_API_KEY env, MASSIVE_API_KEY env\n"
            f"  Fix: Add POLYGON_API_KEY=your_key to your .env file"
        )
    return key


def load_universe(path, max_tickers):
    p = Path(path)
    if not p.exists():
        sys.exit(f"Error: universe file not found: {p}")
    df = pd.read_parquet(p)
    tickers = (
        df["ticker"].astype(str).str.upper().str.strip()
        .replace("", pd.NA).dropna().drop_duplicates().sort_values().tolist()
    )
    if max_tickers > 0:
        tickers = tickers[:max_tickers]
    return tickers


# ===================================================================
# API helpers
# ===================================================================
def _api_get(url, params, api_key):
    """GET with retries and rate limit handling."""
    params["apiKey"] = api_key
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = RETRY_PAUSE * attempt * 2
                print(f"    Rate limited, waiting {wait:.0f}s ...")
                time.sleep(wait)
                continue
            if resp.status_code == 403:
                return None, "forbidden"
            if resp.status_code == 404:
                return None, "not_found"
            resp.raise_for_status()
            data = resp.json()
            return data, "ok"
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_PAUSE * attempt)
                continue
            return None, "timeout"
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_PAUSE * attempt)
                continue
            return None, f"error:{str(e)[:100]}"
    return None, "max_retries"


def fetch_ratios(ticker, api_key):
    """Fetch daily ratios for a ticker."""
    url = f"{FINANCIALS_BASE}/ratios"
    params = {"ticker": ticker, "limit": 1}
    data, status = _api_get(url, params, api_key)
    if data is None or "results" not in data or not data["results"]:
        return None, status
    return data["results"][0], "ok"


def fetch_income_statements(ticker, api_key, max_quarters=8):
    """Fetch quarterly income statements."""
    url = f"{FINANCIALS_BASE}/income-statements"
    params = {
        "tickers": ticker,
        "timeframe": "quarterly",
        "limit": max_quarters,
        "order": "desc",
    }
    data, status = _api_get(url, params, api_key)
    if data is None or "results" not in data or not data["results"]:
        return [], status
    return data["results"], "ok"


def fetch_balance_sheets(ticker, api_key, max_quarters=8):
    """Fetch quarterly balance sheets."""
    url = f"{FINANCIALS_BASE}/balance-sheets"
    params = {
        "tickers": ticker,
        "timeframe": "quarterly",
        "limit": max_quarters,
        "order": "desc",
    }
    data, status = _api_get(url, params, api_key)
    if data is None or "results" not in data or not data["results"]:
        return [], status
    return data["results"], "ok"


def fetch_cash_flow(ticker, api_key, max_quarters=8):
    """Fetch quarterly cash flow statements."""
    url = f"{FINANCIALS_BASE}/cash-flow-statements"
    params = {
        "tickers": ticker,
        "timeframe": "quarterly",
        "limit": max_quarters,
        "order": "desc",
    }
    data, status = _api_get(url, params, api_key)
    if data is None or "results" not in data or not data["results"]:
        return [], status
    return data["results"], "ok"


# ===================================================================
# Feature engineering from ratios
# ===================================================================
def _sf(val, default=np.nan):
    """Safe float conversion."""
    if val is None:
        return default
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (ValueError, TypeError):
        return default


def engineer_ratio_features(ticker, ratios):
    """Extract and engineer features from the daily ratios snapshot."""
    if ratios is None:
        return None

    row = {"ticker": ticker}

    # Raw ratios
    row["fund_pe"] = _sf(ratios.get("price_to_earnings"))
    row["fund_ps"] = _sf(ratios.get("price_to_sales"))
    row["fund_pb"] = _sf(ratios.get("price_to_book"))
    row["fund_ev_ebitda"] = _sf(ratios.get("ev_to_ebitda"))
    row["fund_ev_sales"] = _sf(ratios.get("ev_to_sales"))
    row["fund_pcf"] = _sf(ratios.get("price_to_cash_flow"))
    row["fund_pfcf"] = _sf(ratios.get("price_to_free_cash_flow"))

    # Profitability
    row["fund_roe"] = _sf(ratios.get("return_on_equity"))
    row["fund_roa"] = _sf(ratios.get("return_on_assets"))
    row["fund_eps"] = _sf(ratios.get("earnings_per_share"))

    # Leverage & liquidity
    row["fund_debt_equity"] = _sf(ratios.get("debt_to_equity"))
    row["fund_current_ratio"] = _sf(ratios.get("current"))
    row["fund_quick_ratio"] = _sf(ratios.get("quick"))
    row["fund_cash_ratio"] = _sf(ratios.get("cash"))

    # Yield & size
    row["fund_div_yield"] = _sf(ratios.get("dividend_yield"))
    row["fund_fcf"] = _sf(ratios.get("free_cash_flow"))
    row["fund_ev"] = _sf(ratios.get("enterprise_value"))
    row["fund_mktcap_ratio"] = _sf(ratios.get("market_cap"))
    row["fund_avg_volume"] = _sf(ratios.get("average_volume"))
    row["fund_price"] = _sf(ratios.get("price"))

    # Derived: FCF yield (FCF / market cap)
    mktcap = _sf(ratios.get("market_cap"))
    fcf = _sf(ratios.get("free_cash_flow"))
    if np.isfinite(mktcap) and mktcap > 0 and np.isfinite(fcf):
        row["fund_fcf_yield"] = fcf / mktcap
    else:
        row["fund_fcf_yield"] = np.nan

    # Derived: earnings yield (1 / PE)
    pe = row["fund_pe"]
    if np.isfinite(pe) and pe > 0:
        row["fund_earnings_yield"] = 1.0 / pe
    else:
        row["fund_earnings_yield"] = np.nan

    # Date from ratios
    row["fund_ratios_date"] = ratios.get("date")

    return row


# ===================================================================
# Feature engineering from quarterly statements
# ===================================================================
def _safe_pct_change(current, previous):
    """Compute percentage change, handling zeros and NaN."""
    if current is None or previous is None:
        return np.nan
    c, p = _sf(current), _sf(previous)
    if not np.isfinite(c) or not np.isfinite(p) or abs(p) < EPS:
        return np.nan
    return (c - p) / abs(p)


def engineer_growth_features(ticker, income_stmts, balance_sheets, cash_flows):
    """
    Compute growth and trend metrics from quarterly financial statements.
    All metrics are backward-looking (no lookahead).
    """
    row = {"ticker": ticker}

    if not income_stmts or len(income_stmts) < 2:
        return row

    # Sort by period_end descending (most recent first)
    stmts = sorted(income_stmts, key=lambda x: x.get("period_end", ""), reverse=True)

    latest = stmts[0]
    prev_q = stmts[1] if len(stmts) >= 2 else None
    prev_y = stmts[4] if len(stmts) >= 5 else None  # same quarter last year

    # Revenue
    rev_now = _sf(latest.get("revenue"))
    rev_prev_q = _sf(prev_q.get("revenue")) if prev_q else np.nan
    rev_prev_y = _sf(prev_y.get("revenue")) if prev_y else np.nan

    row["fund_revenue_qoq"] = _safe_pct_change(rev_now, rev_prev_q)
    row["fund_revenue_yoy"] = _safe_pct_change(rev_now, rev_prev_y)

    # Gross margin
    gp_now = _sf(latest.get("gross_profit"))
    if np.isfinite(rev_now) and rev_now > 0 and np.isfinite(gp_now):
        row["fund_gross_margin"] = gp_now / rev_now
    else:
        row["fund_gross_margin"] = np.nan

    # Operating margin
    oi_now = _sf(latest.get("operating_income"))
    if np.isfinite(rev_now) and rev_now > 0 and np.isfinite(oi_now):
        row["fund_operating_margin"] = oi_now / rev_now
    else:
        row["fund_operating_margin"] = np.nan

    # Net margin
    ni_now = _sf(latest.get("consolidated_net_income_loss",
                            latest.get("net_income_loss_attributable_common_shareholders")))
    if np.isfinite(rev_now) and rev_now > 0 and np.isfinite(ni_now):
        row["fund_net_margin"] = ni_now / rev_now
    else:
        row["fund_net_margin"] = np.nan

    # Margin trend (current - previous quarter)
    if prev_q:
        rev_pq = _sf(prev_q.get("revenue"))
        gp_pq = _sf(prev_q.get("gross_profit"))
        oi_pq = _sf(prev_q.get("operating_income"))
        if np.isfinite(rev_pq) and rev_pq > 0:
            prev_gm = gp_pq / rev_pq if np.isfinite(gp_pq) else np.nan
            prev_om = oi_pq / rev_pq if np.isfinite(oi_pq) else np.nan
            row["fund_gross_margin_chg"] = row["fund_gross_margin"] - prev_gm if np.isfinite(prev_gm) else np.nan
            row["fund_operating_margin_chg"] = row["fund_operating_margin"] - prev_om if np.isfinite(prev_om) else np.nan
        else:
            row["fund_gross_margin_chg"] = np.nan
            row["fund_operating_margin_chg"] = np.nan
    else:
        row["fund_gross_margin_chg"] = np.nan
        row["fund_operating_margin_chg"] = np.nan

    # EPS growth
    eps_now = _sf(latest.get("diluted_earnings_per_share", latest.get("basic_earnings_per_share")))
    eps_prev_q = _sf(prev_q.get("diluted_earnings_per_share", prev_q.get("basic_earnings_per_share"))) if prev_q else np.nan
    eps_prev_y = _sf(prev_y.get("diluted_earnings_per_share", prev_y.get("basic_earnings_per_share"))) if prev_y else np.nan
    row["fund_eps_qoq"] = _safe_pct_change(eps_now, eps_prev_q)
    row["fund_eps_yoy"] = _safe_pct_change(eps_now, eps_prev_y)

    # EBITDA growth
    ebitda_now = _sf(latest.get("ebitda"))
    ebitda_prev_q = _sf(prev_q.get("ebitda")) if prev_q else np.nan
    ebitda_prev_y = _sf(prev_y.get("ebitda")) if prev_y else np.nan
    row["fund_ebitda_qoq"] = _safe_pct_change(ebitda_now, ebitda_prev_q)
    row["fund_ebitda_yoy"] = _safe_pct_change(ebitda_now, ebitda_prev_y)

    # Revenue acceleration (is growth accelerating or decelerating?)
    if len(stmts) >= 3:
        rev_2q_ago = _sf(stmts[2].get("revenue"))
        growth_prev = _safe_pct_change(rev_prev_q, rev_2q_ago)
        growth_now = row["fund_revenue_qoq"]
        if np.isfinite(growth_now) and np.isfinite(growth_prev):
            row["fund_revenue_acceleration"] = growth_now - growth_prev
        else:
            row["fund_revenue_acceleration"] = np.nan
    else:
        row["fund_revenue_acceleration"] = np.nan

    # Latest filing date (for point-in-time validation)
    row["fund_latest_filing_date"] = latest.get("filing_date")
    row["fund_latest_period_end"] = latest.get("period_end")

    # --- Cash flow features ---
    if cash_flows and len(cash_flows) >= 1:
        cf_sorted = sorted(cash_flows, key=lambda x: x.get("period_end", ""), reverse=True)
        cf_latest = cf_sorted[0]

        ocf = _sf(cf_latest.get("net_cash_from_operating_activities"))
        capex = _sf(cf_latest.get("purchase_of_property_plant_and_equipment"))

        row["fund_operating_cf"] = ocf
        row["fund_capex"] = capex

        # FCF from statements (ocf + capex, capex is negative)
        if np.isfinite(ocf) and np.isfinite(capex):
            row["fund_fcf_quarterly"] = ocf + capex
        else:
            row["fund_fcf_quarterly"] = np.nan

        # Earnings quality: OCF / Net Income (>1 = high quality)
        if np.isfinite(ocf) and np.isfinite(ni_now) and abs(ni_now) > EPS:
            row["fund_earnings_quality"] = ocf / abs(ni_now)
        else:
            row["fund_earnings_quality"] = np.nan

        # Capex intensity: capex / revenue
        if np.isfinite(capex) and np.isfinite(rev_now) and rev_now > 0:
            row["fund_capex_intensity"] = abs(capex) / rev_now
        else:
            row["fund_capex_intensity"] = np.nan
    else:
        row["fund_operating_cf"] = np.nan
        row["fund_capex"] = np.nan
        row["fund_fcf_quarterly"] = np.nan
        row["fund_earnings_quality"] = np.nan
        row["fund_capex_intensity"] = np.nan

    # --- Balance sheet features ---
    if balance_sheets and len(balance_sheets) >= 1:
        bs_sorted = sorted(balance_sheets, key=lambda x: x.get("period_end", ""), reverse=True)
        bs_latest = bs_sorted[0]

        total_assets = _sf(bs_latest.get("total_assets"))
        total_equity = _sf(bs_latest.get("total_equity"))
        total_liab = _sf(bs_latest.get("total_liabilities"))
        cash = _sf(bs_latest.get("cash_and_equivalents"))
        debt_current = _sf(bs_latest.get("debt_current"))
        debt_lt = _sf(bs_latest.get("long_term_debt_and_capital_lease_obligations"))

        # Net debt
        total_debt = 0.0
        if np.isfinite(debt_current):
            total_debt += debt_current
        if np.isfinite(debt_lt):
            total_debt += debt_lt
        if np.isfinite(cash):
            row["fund_net_debt"] = total_debt - cash
        else:
            row["fund_net_debt"] = np.nan

        # Asset turnover (revenue / total assets)
        if np.isfinite(rev_now) and np.isfinite(total_assets) and total_assets > 0:
            row["fund_asset_turnover"] = (rev_now * 4) / total_assets  # annualize quarterly rev
        else:
            row["fund_asset_turnover"] = np.nan

        # Equity ratio
        if np.isfinite(total_equity) and np.isfinite(total_assets) and total_assets > 0:
            row["fund_equity_ratio"] = total_equity / total_assets
        else:
            row["fund_equity_ratio"] = np.nan
    else:
        row["fund_net_debt"] = np.nan
        row["fund_asset_turnover"] = np.nan
        row["fund_equity_ratio"] = np.nan

    return row


# ===================================================================
# Cross-sectional features
# ===================================================================
def add_sector_ranks(df, sector_col="sector"):
    """
    Add cross-sectional percentile ranks within sector for key ratios.
    Lower rank = cheaper/better for valuation; higher = better for profitability.
    """
    if sector_col not in df.columns:
        print(f"  WARNING: No '{sector_col}' column found, skipping sector ranks")
        return df

    # Valuation: lower is cheaper (rank ascending)
    val_cols = ["fund_pe", "fund_ps", "fund_pb", "fund_ev_ebitda", "fund_pfcf"]
    for col in val_cols:
        if col in df.columns:
            df[f"{col}_sector_rank"] = df.groupby(sector_col)[col].rank(
                pct=True, method="average", ascending=True
            )

    # Profitability: higher is better (rank descending)
    prof_cols = ["fund_roe", "fund_roa", "fund_gross_margin", "fund_operating_margin",
                 "fund_earnings_quality", "fund_fcf_yield"]
    for col in prof_cols:
        if col in df.columns:
            df[f"{col}_sector_rank"] = df.groupby(sector_col)[col].rank(
                pct=True, method="average", ascending=False
            )

    # Growth: higher is better (rank descending)
    growth_cols = ["fund_revenue_yoy", "fund_eps_yoy", "fund_ebitda_yoy",
                   "fund_revenue_acceleration"]
    for col in growth_cols:
        if col in df.columns:
            df[f"{col}_sector_rank"] = df.groupby(sector_col)[col].rank(
                pct=True, method="average", ascending=False
            )

    # Leverage: lower D/E is safer (rank ascending)
    if "fund_debt_equity" in df.columns:
        df["fund_debt_equity_sector_rank"] = df.groupby(sector_col)["fund_debt_equity"].rank(
            pct=True, method="average", ascending=True
        )

    return df


def add_composite_scores(df):
    """
    Build composite scores from individual fundamentals.
    These are the features that will drive 60/90-day predictions.
    """
    # Value composite: average of valuation sector ranks (lower = cheaper)
    val_rank_cols = [c for c in df.columns if c.endswith("_sector_rank")
                     and any(v in c for v in ["pe", "ps", "pb", "ev_ebitda", "pfcf"])]
    if val_rank_cols:
        df["fund_value_composite"] = df[val_rank_cols].mean(axis=1)

    # Quality composite: average of profitability + earnings quality ranks
    qual_rank_cols = [c for c in df.columns if c.endswith("_sector_rank")
                      and any(v in c for v in ["roe", "roa", "gross_margin",
                                                "operating_margin", "earnings_quality"])]
    if qual_rank_cols:
        df["fund_quality_composite"] = df[qual_rank_cols].mean(axis=1)

    # Growth composite
    growth_rank_cols = [c for c in df.columns if c.endswith("_sector_rank")
                        and any(v in c for v in ["revenue_yoy", "eps_yoy",
                                                  "ebitda_yoy", "acceleration"])]
    if growth_rank_cols:
        df["fund_growth_composite"] = df[growth_rank_cols].mean(axis=1)

    # GARP score (Growth at Reasonable Price): growth rank / value rank
    # High GARP = good growth + reasonable valuation
    if "fund_growth_composite" in df.columns and "fund_value_composite" in df.columns:
        # Invert value composite so higher = cheaper
        df["fund_garp_score"] = (1 - df["fund_value_composite"]) * df["fund_growth_composite"]

    return df


# ===================================================================
# Main pipeline
# ===================================================================
def main():
    args = parse_args()
    api_key = get_api_key(args.api_key)
    tickers = load_universe(args.universe, args.max_tickers)

    output_path = Path(args.output) if args.output else Path(args.universe).parent / "fundamentals_overlay.parquet"
    status_path = Path(args.status_output) if args.status_output else output_path.with_suffix(".status.json")

    print(f"Quintic Labs — Fundamentals Overlay")
    print(f"Universe: {len(tickers)} tickers")
    print(f"Output: {output_path}")
    print(f"Mode: {'ratios only' if args.ratios_only else 'full (ratios + statements)'}")
    print(f"Quarters of history: {args.max_quarters}")

    if args.dry_run:
        print("Dry run complete.")
        return

    t_start = time.time()
    rows = []
    failed = []
    total = len(tickers)

    for idx, ticker in enumerate(tickers, 1):
        # --- Ratios ---
        ratios, r_status = fetch_ratios(ticker, api_key)
        time.sleep(RATE_LIMIT_PAUSE)

        ratio_features = engineer_ratio_features(ticker, ratios)
        if ratio_features is None:
            ratio_features = {"ticker": ticker}
            if r_status != "ok":
                failed.append({"ticker": ticker, "reason": f"ratios:{r_status}"})

        if not args.ratios_only:
            # --- Income statements ---
            income, i_status = fetch_income_statements(ticker, api_key, args.max_quarters)
            time.sleep(RATE_LIMIT_PAUSE)

            # --- Balance sheets ---
            balance, b_status = fetch_balance_sheets(ticker, api_key, args.max_quarters)
            time.sleep(RATE_LIMIT_PAUSE)

            # --- Cash flow ---
            cashflow, c_status = fetch_cash_flow(ticker, api_key, args.max_quarters)
            time.sleep(RATE_LIMIT_PAUSE)

            growth_features = engineer_growth_features(ticker, income, balance, cashflow)
            ratio_features.update(growth_features)
        else:
            # Minimal growth features from ratios alone
            pass

        rows.append(ratio_features)

        if idx % 25 == 0 or idx == total:
            elapsed = time.time() - t_start
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (total - idx) / rate if rate > 0 else 0
            print(f"  [{idx}/{total}] succeeded={len(rows)} failed={len(failed)} "
                  f"rate={rate:.1f}/s ETA={eta/60:.1f}min")

    # Build dataframe
    df = pd.DataFrame(rows)
    print(f"\nRaw features: {len(df)} rows x {len(df.columns)} cols")

    # Load sector info from universe for cross-sectional ranking
    universe_df = pd.read_parquet(args.universe)
    sector_cols = ["ticker", "sector"]
    if "sector" not in universe_df.columns and "gics_sector" in universe_df.columns:
        universe_df = universe_df.rename(columns={"gics_sector": "sector"})
    if "sector" in universe_df.columns:
        sector_lookup = universe_df[["ticker", "sector"]].drop_duplicates(subset=["ticker"], keep="last")
        sector_lookup["ticker"] = sector_lookup["ticker"].astype(str).str.upper().str.strip()
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df = df.merge(sector_lookup, on="ticker", how="left")

    # Add sector ranks and composites
    df = add_sector_ranks(df, sector_col="sector")
    df = add_composite_scores(df)

    # Add timestamp
    df["fund_snapshot_date"] = datetime.now(timezone.utc).date().isoformat()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"Saved: {output_path}")
    print(f"  {len(df)} rows x {len(df.columns)} columns")

    # Feature coverage report
    fund_cols = [c for c in df.columns if c.startswith("fund_")]
    coverage = df[fund_cols].notna().mean().sort_values(ascending=False)
    print(f"\nFeature coverage (top 20):")
    for name, pct in coverage.head(20).items():
        print(f"  {name}: {pct:.1%}")

    # Status
    status = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(output_path),
        "tickers_requested": total,
        "tickers_succeeded": len(df),
        "tickers_failed": len(failed),
        "mode": "ratios_only" if args.ratios_only else "full",
        "quarters_pulled": args.max_quarters,
        "total_features": len(fund_cols),
        "failed_tickers": failed[:50],
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"Status: {status_path}")

    total_time = time.time() - t_start
    print(f"\nDone in {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()