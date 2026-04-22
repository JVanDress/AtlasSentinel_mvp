"""
Quintic Labs — Social Sentiment Overlay (Grok/X)

Uses Grok's real-time X/Twitter access to measure social sentiment
per ticker. No other AI model has direct access to the X firehose.

WHY ARLO NEEDS THIS:
    Social sentiment on Twitter moves stocks 1-5 days before news
    outlets report. Retail trader mood, influencer calls, and viral
    threads create momentum that persists into 20-day windows.
    This is alpha no fundamental or technical study can capture.

Features produced:
    - social_sentiment: overall bullish/bearish score [-1, 1]
    - social_volume: relative discussion volume (low/normal/high/viral)
    - social_bullish_pct / social_bearish_pct / social_neutral_pct
    - social_confidence: how confident Grok is in the reading [0, 1]
    - social_momentum: is sentiment shifting (improving/stable/deteriorating)
    - social_notable_themes: key topics driving discussion

Cost efficiency:
    - Uses grok-4-1-fast ($0.20/M input, $0.50/M output)
    - Batches 10 tickers per API call (~16x fewer calls)
    - Caches daily results — won't re-fetch same day
    - ~1,600 tickers ≈ 160 API calls ≈ ~$0.50-1.00 per run

Usage:
    python build_social_sentiment_overlay.py --universe stocks_universe.parquet \\
        --output cleanroom_social_sentiment_overlay.parquet

Requirements:
    pip install pandas pyarrow numpy requests python-dotenv
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from quintic_paths import load_simple_dotenv, project_root


GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_MODEL = "grok-4-1-fast"
BATCH_SIZE = 10  # tickers per API call
RATE_PAUSE = 1.0  # seconds between API calls
MAX_RETRIES = 3


def parse_args():
    parser = argparse.ArgumentParser(description="Quintic Labs — Social Sentiment (Grok/X)")
    parser.add_argument("--universe", "-u", required=True, help="Universe parquet")
    parser.add_argument("--output", "-o", default=None, help="Output parquet")
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_grok_key() -> str:
    root = project_root()
    load_simple_dotenv(root, override=True)
    key = os.getenv("XAI_API_KEY", "").strip()
    if not key:
        raise SystemExit("XAI_API_KEY not found in .env")
    return key


def load_universe(path: Path, max_tickers: int) -> list[str]:
    df = pd.read_parquet(path, columns=["ticker"])
    tickers = (
        df["ticker"].astype(str).str.upper().str.strip()
        .replace("", pd.NA).dropna().drop_duplicates().sort_values().tolist()
    )
    if max_tickers > 0:
        tickers = tickers[:max_tickers]
    return tickers


def build_batch_prompt(tickers: list[str]) -> str:
    ticker_list = ", ".join(tickers)
    return f"""Analyze the current Twitter/X sentiment for these stock tickers: {ticker_list}

For EACH ticker, assess the recent social media discussion and provide:
1. sentiment: overall score from -1.0 (extremely bearish) to 1.0 (extremely bullish)
2. volume: discussion level as "none", "low", "normal", "high", or "viral"
3. bullish_pct: percentage of bullish posts (0.0 to 1.0)
4. bearish_pct: percentage of bearish posts (0.0 to 1.0)
5. neutral_pct: percentage of neutral posts (0.0 to 1.0)
6. confidence: your confidence in this reading (0.0 to 1.0)
7. momentum: sentiment direction as "improving", "stable", or "deteriorating"
8. themes: 1-3 word summary of key discussion topics (max 3 themes)

Respond ONLY with valid JSON array. No markdown, no explanation. Example format:
[{{"ticker":"AAPL","sentiment":0.3,"volume":"high","bullish_pct":0.5,"bearish_pct":0.2,"neutral_pct":0.3,"confidence":0.8,"momentum":"stable","themes":["earnings beat","services growth"]}}]"""


def call_grok_api(prompt: str, api_key: str) -> dict | None:
    """Call Grok API with retries and rate limiting."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROK_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a financial social media analyst. Analyze Twitter/X sentiment for stocks. Return only valid JSON."
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 2000,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(GROK_API_URL, headers=headers, json=payload, timeout=60)

            if resp.status_code == 429:
                wait = RATE_PAUSE * attempt * 3
                print(f"    Rate limited, waiting {wait:.0f}s ...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            # Extract token usage for cost tracking
            usage = data.get("usage", {})

            # Extract response text
            content = data["choices"][0]["message"]["content"].strip()

            # Clean markdown fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            if content.startswith("json"):
                content = content[4:].strip()

            return {
                "content": content,
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            }

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                time.sleep(RATE_PAUSE * attempt)
                continue
            return None
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RATE_PAUSE * attempt)
                continue
            print(f"    API error: {e}")
            return None

    return None


def parse_grok_response(content: str, expected_tickers: list[str]) -> list[dict]:
    """Parse JSON response, handle failures gracefully."""
    try:
        results = json.loads(content)
        if isinstance(results, dict):
            results = [results]
        if not isinstance(results, list):
            return []
        return results
    except json.JSONDecodeError:
        # Try to extract JSON from messy response
        try:
            start = content.index("[")
            end = content.rindex("]") + 1
            return json.loads(content[start:end])
        except (ValueError, json.JSONDecodeError):
            return []


def empty_row(ticker: str, date_str: str) -> dict:
    return {
        "ticker": ticker,
        "date": pd.Timestamp(date_str),
        "social_sentiment": np.nan,
        "social_volume": "none",
        "social_bullish_pct": np.nan,
        "social_bearish_pct": np.nan,
        "social_neutral_pct": np.nan,
        "social_confidence": 0.0,
        "social_momentum": "stable",
        "social_themes": "",
        "social_status": "no_data",
    }


def clean_ticker(raw: str) -> str:
    """Normalize ticker from Grok response — strip $, spaces, dots."""
    return raw.upper().strip().lstrip("$").replace(".", "").replace("-", "").strip()


def result_to_row(result: dict, date_str: str) -> dict:
    ticker = clean_ticker(str(result.get("ticker", "")))
    return {
        "ticker": ticker,
        "date": pd.Timestamp(date_str),
        "social_sentiment": float(result.get("sentiment", 0.0)),
        "social_volume": str(result.get("volume", "none")),
        "social_bullish_pct": float(result.get("bullish_pct", 0.0)),
        "social_bearish_pct": float(result.get("bearish_pct", 0.0)),
        "social_neutral_pct": float(result.get("neutral_pct", 0.0)),
        "social_confidence": float(result.get("confidence", 0.0)),
        "social_momentum": str(result.get("momentum", "stable")),
        "social_themes": " | ".join(result.get("themes", [])) if isinstance(result.get("themes"), list) else str(result.get("themes", "")),
        "social_status": "ok",
    }


def main():
    args = parse_args()
    api_key = require_grok_key()
    universe_path = Path(args.universe)
    if not universe_path.exists():
        raise SystemExit(f"Universe not found: {universe_path}")

    output_path = (
        Path(args.output) if args.output
        else universe_path.parent / "cleanroom_social_sentiment_overlay.parquet"
    )

    tickers = load_universe(universe_path, args.max_tickers)
    today = datetime.now(timezone.utc).date().isoformat()
    batch_size = max(1, args.batch_size)

    print(f"Quintic Labs — Social Sentiment (Grok/X)")
    print(f"Universe: {len(tickers)} tickers")
    print(f"Output: {output_path}")
    print(f"Batch size: {batch_size}")
    print(f"Model: {GROK_MODEL}")
    print(f"Date: {today}")

    if args.dry_run:
        print(f"Estimated API calls: {len(tickers) // batch_size + 1}")
        print("Dry run complete.")
        return

    # Batch tickers
    batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]

    all_rows = []
    total_input_tokens = 0
    total_output_tokens = 0
    failed_batches = 0

    for batch_idx, batch in enumerate(batches, 1):
        prompt = build_batch_prompt(batch)
        response = call_grok_api(prompt, api_key)

        if response is None:
            failed_batches += 1
            for ticker in batch:
                all_rows.append(empty_row(ticker, today))
            print(f"  [{batch_idx}/{len(batches)}] FAILED batch: {batch}")
        else:
            total_input_tokens += response.get("input_tokens", 0)
            total_output_tokens += response.get("output_tokens", 0)

            results = parse_grok_response(response["content"], batch)
            result_tickers = {clean_ticker(str(r.get("ticker", ""))) for r in results}

            for result in results:
                all_rows.append(result_to_row(result, today))

            # Fill in any tickers that Grok didn't return
            for ticker in batch:
                if ticker not in result_tickers:
                    all_rows.append(empty_row(ticker, today))

            succeeded = len(result_tickers & set(batch))
            print(f"  [{batch_idx}/{len(batches)}] {succeeded}/{len(batch)} tickers scored")

        time.sleep(RATE_PAUSE)

    # Build dataframe
    df = pd.DataFrame(all_rows)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Encode volume as numeric for ARLO
    volume_map = {"none": 0, "low": 1, "normal": 2, "high": 3, "viral": 4}
    df["social_volume_numeric"] = df["social_volume"].map(volume_map).fillna(0).astype(int)

    # Encode momentum as numeric
    momentum_map = {"deteriorating": -1, "stable": 0, "improving": 1}
    df["social_momentum_numeric"] = df["social_momentum"].map(momentum_map).fillna(0).astype(int)

    # Load and append to existing history
    if output_path.exists():
        try:
            existing = pd.read_parquet(output_path)
            existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
            # Remove today's data if re-running
            existing = existing[existing["date"].dt.date.astype(str) != today]
            df = pd.concat([existing, df], ignore_index=True)
        except Exception:
            pass

    df = df.sort_values(["ticker", "date"]).drop_duplicates(
        subset=["ticker", "date"], keep="last"
    ).reset_index(drop=True)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")

    # Cost estimate
    input_cost = (total_input_tokens / 1_000_000) * 0.20
    output_cost = (total_output_tokens / 1_000_000) * 0.50
    total_cost = input_cost + output_cost

    # Status
    status_path = output_path.with_suffix(".status.json")
    status = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "date": today,
        "model": GROK_MODEL,
        "tickers_requested": len(tickers),
        "tickers_scored": int((df["social_status"] == "ok").sum()),
        "batches_total": len(batches),
        "batches_failed": failed_batches,
        "total_rows": len(df),
        "unique_dates": int(df["date"].nunique()),
        "tokens_input": total_input_tokens,
        "tokens_output": total_output_tokens,
        "estimated_cost_usd": round(total_cost, 4),
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    print(f"\nSaved: {output_path}")
    print(f"Rows today: {len(all_rows):,}")
    print(f"Total rows: {len(df):,}")
    print(f"Tokens: {total_input_tokens:,} in / {total_output_tokens:,} out")
    print(f"Estimated cost: ${total_cost:.4f}")
    print(f"Status: {status_path}")


if __name__ == "__main__":
    main()
