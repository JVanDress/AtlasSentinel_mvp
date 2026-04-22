"""
Quintic Labs - Historical Options Backfill (Parallel — Opus)
Massive.com Options Chain Snapshot
Endpoint: GET /v3/snapshot/options/{ticker}
Auth: Authorization: Bearer <key>
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ── ENV ──────────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")
API_KEY = os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")
DATA_ROOT = os.getenv("DATA_ROOT", r"Z:\jvand\QUINTIC_V3\data\stage")

if not API_KEY:
    print("FATAL: Neither MASSIVE_API_KEY nor POLYGON_API_KEY found in .env")
    sys.exit(1)

BASE = "https://api.massive.com/v3/snapshot/options"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def test_api():
    r = requests.get(
        f"{BASE}/AAPL",
        params={"as_of": "2025-04-01", "limit": 1},
        headers=HEADERS,
        timeout=15,
    )
    if r.status_code != 200:
        print(f"FATAL: API test failed. Status={r.status_code} Body={r.text[:300]}")
        sys.exit(1)
    count = len(r.json().get("results", []))
    print(f"API test passed: AAPL returned {count} contract(s) via api.massive.com")


def fetch_chain(ticker, as_of, max_contracts=2000, max_retries=3):
    """Fetch full options chain snapshot. Retries on network errors."""
    all_results = []
    params = {"as_of": as_of, "limit": 250}
    url = f"{BASE}/{ticker}"
    retries = 0

    while url and len(all_results) < max_contracts:
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 404:
                return []
            if r.status_code == 429:
                time.sleep(min(2 ** retries, 30))
                retries += 1
                if retries > max_retries:
                    return all_results
                continue
            if r.status_code != 200:
                return all_results
            data = r.json()
            all_results.extend(data.get("results", []))
            nxt = data.get("next_url")
            if nxt and len(all_results) < max_contracts:
                url = nxt
                params = {}
            else:
                break
            retries = 0
        except (requests.Timeout, requests.ConnectionError):
            retries += 1
            if retries > max_retries:
                return all_results
            time.sleep(min(2 ** retries, 30))
        except Exception:
            return all_results

    return all_results[:max_contracts]


def aggregate(ticker, date_str, contracts, spot):
    """Aggregate contract-level data into per-ticker per-date features."""
    if not contracts:
        return None
    calls, puts = [], []
    for c in contracts:
        det = c.get("details", {})
        gr = c.get("greeks", {})
        dy = c.get("day", {})
        ct = det.get("contract_type", "").lower()
        row = {
            "strike": det.get("strike_price"),
            "oi": c.get("open_interest", 0) or 0,
            "volume": dy.get("volume", 0) or 0,
            "iv": c.get("implied_volatility"),
            "delta": gr.get("delta"),
            "gamma": gr.get("gamma"),
            "theta": gr.get("theta"),
            "vega": gr.get("vega"),
        }
        if ct == "call":
            calls.append(row)
        elif ct == "put":
            puts.append(row)

    call_oi = sum(r["oi"] for r in calls)
    put_oi = sum(r["oi"] for r in puts)
    call_vol = sum(r["volume"] for r in calls)
    put_vol = sum(r["volume"] for r in puts)

    def wavg_iv(rows):
        v = [(r["iv"], r["oi"]) for r in rows if r["iv"] is not None and r["oi"] > 0]
        if not v:
            ivs = [r["iv"] for r in rows if r["iv"] is not None]
            return float(np.mean(ivs)) if ivs else np.nan
        i, w = zip(*v)
        return float(np.average(i, weights=w))

    iv_c, iv_p = wavg_iv(calls), wavg_iv(puts)

    def net_greek(field, cr, pr):
        t = 0.0
        for r in cr:
            v = r.get(field)
            if v is not None:
                t += v * r["oi"] * 100
        for r in pr:
            v = r.get(field)
            if v is not None:
                t += v * r["oi"] * 100
        return t

    gex = 0.0
    for r in calls:
        g = r.get("gamma")
        if g is not None:
            gex += g * r["oi"] * 100 * spot
    for r in puts:
        g = r.get("gamma")
        if g is not None:
            gex -= g * r["oi"] * 100 * spot

    dex = 0.0
    for r in calls:
        d = r.get("delta")
        if d is not None:
            dex += d * r["oi"] * 100 * spot
    for r in puts:
        d = r.get("delta")
        if d is not None:
            dex += d * r["oi"] * 100 * spot

    strike_oi = {}
    for r in calls + puts:
        s = r["strike"]
        if s is not None:
            strike_oi[s] = strike_oi.get(s, 0) + r["oi"]
    mp = max(strike_oi, key=strike_oi.get) if strike_oi else np.nan

    return {
        "ticker": ticker, "date": date_str,
        "num_call_contracts": len(calls), "num_put_contracts": len(puts),
        "total_call_oi": call_oi, "total_put_oi": put_oi,
        "total_call_volume": call_vol, "total_put_volume": put_vol,
        "pc_ratio_oi": (put_oi / call_oi) if call_oi > 0 else np.nan,
        "pc_ratio_volume": (put_vol / call_vol) if call_vol > 0 else np.nan,
        "avg_iv_calls": iv_c, "avg_iv_puts": iv_p,
        "iv_skew": (iv_p - iv_c) if not (np.isnan(iv_p) or np.isnan(iv_c)) else np.nan,
        "net_delta": net_greek("delta", calls, puts),
        "net_gamma": net_greek("gamma", calls, puts),
        "net_theta": net_greek("theta", calls, puts),
        "net_vega": net_greek("vega", calls, puts),
        "gex": gex, "dex": dex,
        "max_pain_strike": mp, "spot_price": spot,
    }


def process_one_ticker(tk, ds, spot, max_contracts):
    """Worker function — called concurrently. Returns aggregated row or None."""
    if spot is None or spot <= 0:
        return None
    contracts = fetch_chain(tk, ds, max_contracts)
    if not contracts:
        return None
    return aggregate(tk, ds, contracts, spot)


def main():
    p = argparse.ArgumentParser(description="Quintic Labs Options Backfill (Parallel - Opus)")
    default_start = (datetime.now() - timedelta(days=365*5)).strftime("%Y-%m-%d")
    p.add_argument("--start-date", default=default_start,
                   help=f"Start date YYYY-MM-DD (default: 5 years ago = {default_start})")
    p.add_argument("--end-date", default=datetime.now().strftime("%Y-%m-%d"),
                   help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--max-tickers", type=int, default=0,
                   help="Limit number of tickers (0 = all)")
    p.add_argument("--max-contracts", type=int, default=2000,
                   help="Max contracts per ticker per date (default: 2000)")
    p.add_argument("--workers", type=int, default=60,
                   help="Concurrent workers (default: 60)")
    p.add_argument("--universe", default=None)
    p.add_argument("--output", default=None)
    a = p.parse_args()

    uni_path = a.universe or os.path.join(DATA_ROOT, "stocks_universe.parquet")
    out_path = a.output or os.path.join(DATA_ROOT, "cleanroom_options_overlay_opus.parquet")
    stat_path = out_path.replace(".parquet", ".backfill_status.json")

    # ── Startup API check ────────────────────────────────────────────────
    test_api()

    # ── Load universe ────────────────────────────────────────────────────
    print("Loading universe ...")
    uni = pd.read_parquet(uni_path)
    tickers = sorted(uni["ticker"].unique().tolist())
    if a.max_tickers > 0:
        tickers = tickers[:a.max_tickers]

    print("Building spot price lookup ...")
    uni["ds"] = pd.to_datetime(uni["date"]).dt.strftime("%Y-%m-%d")
    # Vectorized — faster than row-iter
    spot_lk = dict(zip(
        zip(uni["ticker"].values, uni["ds"].values),
        uni["close"].values,
    ))
    print(f"  Loaded {len(spot_lk):,} (ticker, date) spot prices")

    # ── Date range ───────────────────────────────────────────────────────
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(a.start_date, a.end_date)]

    # ── Resume support ───────────────────────────────────────────────────
    done = set()
    if os.path.exists(stat_path):
        with open(stat_path) as f:
            done = set(json.load(f).get("completed_dates", []))
    remaining = [d for d in dates if d not in done]

    print(f"\nQuintic Labs Options Backfill (PARALLEL — Opus)")
    print(f"Tickers: {len(tickers):,} | Dates: {len(dates):,} | Remaining: {len(remaining):,}")
    print(f"Workers: {a.workers} | Max contracts: {a.max_contracts}")
    print(f"Date range: {a.start_date} -> {a.end_date}")
    print(f"Output: {out_path}")

    if not remaining:
        print("All dates done.")
        return

    # ── Resume existing rows ─────────────────────────────────────────────
    rows = []
    if os.path.exists(out_path):
        try:
            rows = pd.read_parquet(out_path).to_dict("records")
            print(f"Resumed {len(rows):,} existing rows")
        except Exception:
            pass

    # ── Main backfill loop: outer = dates, inner = parallel tickers ──────
    t0 = time.time()
    for ds in tqdm(remaining, desc="Dates", unit="day"):
        t_date = time.time()
        rows_this_date = 0

        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures = {
                pool.submit(process_one_ticker, tk, ds, spot_lk.get((tk, ds)), a.max_contracts): tk
                for tk in tickers
            }
            with tqdm(total=len(futures), desc=f"  {ds}", unit="tk", leave=False) as pbar:
                for fut in as_completed(futures):
                    result = fut.result()
                    if result is not None:
                        rows.append(result)
                        rows_this_date += 1
                    pbar.update(1)

        done.add(ds)

        # Save incrementally after each date
        if rows:
            df = pd.DataFrame(rows)
            for c in df.select_dtypes(include=["float64", "float32"]).columns:
                df[c] = df[c].astype("float64")
            df.to_parquet(out_path, index=False)

        with open(stat_path, "w") as f:
            json.dump({
                "completed_dates": sorted(done),
                "last_updated": datetime.now().isoformat(),
                "total_rows": len(rows),
            }, f, indent=2)

        dt = time.time() - t_date
        rate = rows_this_date / dt if dt > 0 else 0
        print(f"    {ds}: {rows_this_date:,} rows in {dt:.1f}s ({rate:.1f} tickers/sec)")

    elapsed = (time.time() - t0) / 3600
    print(f"\nDone in {elapsed:.2f}h | Total rows: {len(rows):,}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()