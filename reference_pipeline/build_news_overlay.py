#!/usr/bin/env python3
"""
Quintic Labs — News Sentiment Overlay

Institutional-grade news sentiment for 20/60/90 day probability forecasting.

WHY ARLO NEEDS THIS:
    Raw sentiment (positive/negative) is table stakes. What predicts 60-day
    returns is the PATTERN — sentiment momentum (improving vs deteriorating),
    consensus strength (do all articles agree or conflict), news volume
    anomalies (sudden spike in coverage = event), and sentiment relative
    to recent history (is today's sentiment unusual for this stock).

Features produced:
    Daily snapshot (per ticker):
    - sent_score: aggregate FinBERT sentiment [-1, 1]
    - sent_dispersion: std of per-article scores (disagreement)
    - sent_max / sent_min: most extreme article scores
    - sent_consensus: 1 - dispersion (agreement strength)
    - article_count_raw / article_count_unique
    - pos_pct / neg_pct / neu_pct

    Rolling features (computed from accumulated daily history):
    - sent_momentum_5d / 10d / 20d: rolling mean of daily sentiment
    - sent_reversal_5d: sentiment change over 5 days
    - sent_vol_20d: volatility of sentiment (unstable = uncertain)
    - news_volume_z_20d: article count z-score vs recent history
    - sent_trend_slope_10d: is sentiment trending up or down

Usage:
    python build_news_overlay.py --project-root C:\\QUINTIC_V3 \\
        --universe-parquet Z:\\jvand\\...\\stocks_universe.parquet \\
        --output-parquet Z:\\jvand\\...\\cleanroom_news_overlay.parquet \\
        --use-finbert --replace-date

Requirements:
    pip install pandas pyarrow polygon-api-client python-dotenv transformers torch
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from polygon import RESTClient

from quintic_paths import data_dir, load_simple_dotenv, project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quintic Labs — News Sentiment Overlay"
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--universe-parquet", type=Path, default=None)
    parser.add_argument("--output-parquet", type=Path, default=None)
    parser.add_argument("--status-output", type=Path, default=None)
    parser.add_argument("--asof-date", default=None)
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--max-articles", type=int, default=10,
                        help="Articles per ticker (default 10 for institutional use)")
    parser.add_argument("--use-finbert", action="store_true")
    parser.add_argument("--replace-date", action="store_true")
    parser.add_argument("--skip-rolling", action="store_true",
                        help="Skip rolling feature computation")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def runtime_project_root(cli_root: Path | None) -> Path:
    if cli_root is not None:
        return cli_root.expanduser()
    return project_root()


def resolve_path(path: Path | None, base_dir: Path, default_name: str,
                  root_dir: Path | None = None) -> Path:
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
    key = (os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("POLYGON_API_KEY or MASSIVE_API_KEY not found.")
    return key


def parse_asof_date(value: str | None) -> str:
    if value:
        return pd.Timestamp(value).date().isoformat()
    return datetime.now(timezone.utc).date().isoformat()


def load_universe(path: Path, max_tickers: int) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Universe not found: {path}")
    df = pd.read_parquet(path, columns=["ticker"])
    tickers = (
        df["ticker"].astype(str).str.upper().str.strip()
        .replace("", pd.NA).dropna().drop_duplicates().sort_values().tolist()
    )
    if max_tickers and max_tickers > 0:
        tickers = tickers[:int(max_tickers)]
    if not tickers:
        raise RuntimeError("Universe is empty.")
    return tickers


def load_existing_output(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


# ===================================================================
# Title extraction and normalization
# ===================================================================
def extract_title(item: Any) -> str:
    try:
        return str(item.title)
    except Exception:
        pass
    try:
        return str(item["title"])
    except Exception:
        return ""


def extract_published(item: Any) -> str | None:
    for attr in ("published_utc", "published_at", "published_at_utc"):
        try:
            value = getattr(item, attr)
            if value:
                return str(value)
        except Exception:
            pass
    for key in ("published_utc", "published_at", "published_at_utc"):
        try:
            value = item[key]
            if value:
                return str(value)
        except Exception:
            pass
    return None


def normalize_title(title: str) -> str:
    return " ".join((title or "").strip().lower().split())


# ===================================================================
# Sentiment scoring
# ===================================================================
def build_finbert_pipeline(enabled: bool):
    if not enabled:
        return None, "disabled"
    try:
        from transformers import pipeline
        return pipeline("sentiment-analysis", model="ProsusAI/finbert",
                        truncation=True, device=-1), "finbert"
    except Exception:
        return None, "unavailable"


POSITIVE_HINTS = {
    "beats", "beat", "upgrade", "upgrades", "outperform", "buy", "bullish",
    "surge", "surges", "jump", "jumps", "gain", "gains", "strong", "record",
    "growth", "expands", "expansion", "profit", "profits", "raises", "raised",
    "partnership", "contract", "approval", "approved",
}

NEGATIVE_HINTS = {
    "miss", "misses", "missed", "downgrade", "downgrades", "underperform",
    "sell", "bearish", "drop", "drops", "fall", "falls", "slump", "weak",
    "lawsuit", "probe", "investigation", "cut", "cuts", "warns", "warning",
    "decline", "declines", "loss", "losses", "bankruptcy",
}


def heuristic_score_single(title: str) -> float:
    """Score a single title using keyword heuristics. Returns [-1, 1]."""
    words = set(normalize_title(title).replace("-", " ").split())
    pos = len(words & POSITIVE_HINTS)
    neg = len(words & NEGATIVE_HINTS)
    if pos == 0 and neg == 0:
        return 0.0
    raw = float(pos - neg) / max(pos + neg, 1)
    return max(min(raw, 1.0), -1.0)


def score_titles_detailed(nlp, titles: list[str]) -> dict:
    """
    Score titles and return per-article scores for dispersion analysis.
    Returns dict with aggregate and per-article metrics.
    """
    result = {
        "sent_score": 0.0,
        "pos_pct": 0.0,
        "neg_pct": 0.0,
        "neu_pct": 1.0,
        "sent_dispersion": 0.0,
        "sent_max": 0.0,
        "sent_min": 0.0,
        "sent_consensus": 1.0,
    }

    if not titles:
        return result

    per_article_scores = []

    if nlp is not None:
        # FinBERT scoring
        preds = nlp(titles)
        pos = neg = neu = score_sum = prob_sum = 0.0
        for pred in preds:
            label = str(pred["label"]).lower()
            prob = float(pred["score"])
            prob_sum += prob
            if "pos" in label:
                pos += prob
                score_sum += prob
                per_article_scores.append(prob)
            elif "neg" in label:
                neg += prob
                score_sum -= prob
                per_article_scores.append(-prob)
            else:
                neu += prob
                per_article_scores.append(0.0)
        denom = prob_sum if prob_sum > 0 else 1.0
        result["sent_score"] = float(score_sum / denom)
        result["pos_pct"] = float(pos / denom)
        result["neg_pct"] = float(neg / denom)
        result["neu_pct"] = float(neu / denom)
    else:
        # Heuristic scoring
        for title in titles:
            s = heuristic_score_single(title)
            per_article_scores.append(s)

        scores_arr = np.array(per_article_scores)
        result["sent_score"] = float(scores_arr.mean())
        pos_count = (scores_arr > 0).sum()
        neg_count = (scores_arr < 0).sum()
        neu_count = (scores_arr == 0).sum()
        total = len(scores_arr)
        result["pos_pct"] = float(pos_count / total)
        result["neg_pct"] = float(neg_count / total)
        result["neu_pct"] = float(neu_count / total)

    # Per-article dispersion metrics
    if len(per_article_scores) >= 2:
        scores_arr = np.array(per_article_scores)
        result["sent_dispersion"] = float(np.std(scores_arr))
        result["sent_max"] = float(np.max(scores_arr))
        result["sent_min"] = float(np.min(scores_arr))
        result["sent_consensus"] = float(max(0.0, 1.0 - 2.0 * result["sent_dispersion"]))
    elif len(per_article_scores) == 1:
        result["sent_max"] = per_article_scores[0]
        result["sent_min"] = per_article_scores[0]
        result["sent_consensus"] = 1.0

    return result


# ===================================================================
# News fetching
# ===================================================================
def fetch_news_rows(client: RESTClient, ticker: str,
                     max_articles: int) -> tuple[list[str], list[str], str]:
    items: list[Any] = []
    try:
        if hasattr(client, "list_ticker_news"):
            for idx, item in enumerate(client.list_ticker_news(
                ticker=ticker, limit=max_articles
            )):
                items.append(item)
                if idx + 1 >= max_articles:
                    break
        elif hasattr(client, "get_ticker_news"):
            items = list(
                client.get_ticker_news(ticker, limit=max_articles) or []
            )[:max_articles]
        else:
            return [], [], "client_missing_news_method"
    except Exception as exc:
        return [], [], f"fetch_fail:{type(exc).__name__}"

    titles_raw = [t for t in (extract_title(item) for item in items) if t]
    published = [v for v in (extract_published(item) for item in items) if v]

    seen: set[str] = set()
    titles = []
    for title in titles_raw:
        key = normalize_title(title)
        if not key or key in seen:
            continue
        seen.add(key)
        titles.append(title)
    if not titles:
        return [], published, "thin"
    return titles_raw, published, "ok"


# ===================================================================
# Rolling features from accumulated history
# ===================================================================
def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling sentiment features from accumulated daily history.
    Only uses past data — no lookahead.

    WHY: A single day's sentiment is noise. The TREND of sentiment
    over 5/10/20 days is signal. ARLO needs to know if sentiment
    is improving or deteriorating, and how volatile it is.
    """
    out = df.copy()
    out = out.sort_values(["ticker", "date"])

    for col in ["sent_score", "article_count_unique"]:
        if col not in out.columns:
            continue

        grouped = out.groupby("ticker", sort=False)[col]

        if col == "sent_score":
            out["sent_momentum_5d"] = grouped.transform(
                lambda s: s.rolling(5, min_periods=3).mean()
            )
            out["sent_momentum_10d"] = grouped.transform(
                lambda s: s.rolling(10, min_periods=5).mean()
            )
            out["sent_momentum_20d"] = grouped.transform(
                lambda s: s.rolling(20, min_periods=10).mean()
            )
            out["sent_reversal_5d"] = grouped.transform(
                lambda s: s - s.shift(5)
            )
            out["sent_vol_20d"] = grouped.transform(
                lambda s: s.rolling(20, min_periods=10).std()
            )

            def _slope_10(s):
                x = np.arange(10, dtype=float)
                x_mean = x.mean()
                denom = float(((x - x_mean) ** 2).sum())
                def _slope(vals):
                    if np.isnan(vals).any():
                        return np.nan
                    y = vals.astype(float)
                    return float(((x - x_mean) * (y - y.mean())).sum() / denom)
                return s.rolling(10, min_periods=10).apply(_slope, raw=True)

            out["sent_trend_slope_10d"] = grouped.transform(_slope_10)

        if col == "article_count_unique":
            roll_mean = grouped.transform(
                lambda s: s.rolling(20, min_periods=10).mean()
            )
            roll_std = grouped.transform(
                lambda s: s.rolling(20, min_periods=10).std().replace(0.0, np.nan)
            )
            out["news_volume_z_20d"] = (out[col] - roll_mean) / roll_std

    return out


# ===================================================================
# Main
# ===================================================================
def main() -> int:
    args = parse_args()
    root = runtime_project_root(args.project_root)
    data_root = data_dir(root)
    data_root.mkdir(parents=True, exist_ok=True)

    universe_path = resolve_path(args.universe_parquet, data_root,
                                  "polygon_equity_universe.parquet", root_dir=root)
    output_path = resolve_path(args.output_parquet, data_root,
                                "cleanroom_news_overlay.parquet", root_dir=root)
    status_path = resolve_path(args.status_output, data_root,
                                "cleanroom_news_overlay_status.json", root_dir=root)
    asof_date = parse_asof_date(args.asof_date)

    print(f"Project root: {root}")
    print(f"Universe parquet: {universe_path}")
    print(f"Output parquet: {output_path}")
    print(f"As-of date: {asof_date}")
    print(f"Articles per ticker: {args.max_articles}")

    tickers = load_universe(universe_path, max_tickers=int(args.max_tickers))
    print(f"Tickers: {len(tickers):,}")

    if args.dry_run:
        print("Dry run complete.")
        return 0

    api_key = require_polygon_key(root)
    client = RESTClient(api_key=api_key)
    nlp, sentiment_mode = build_finbert_pipeline(bool(args.use_finbert))
    if nlp is None and sentiment_mode != "disabled":
        sentiment_mode = "heuristic_fallback"

    print(f"Sentiment mode: {sentiment_mode}")

    rows: list[dict[str, object]] = []
    failed: list[dict[str, str]] = []
    total = len(tickers)

    for idx, ticker in enumerate(tickers, start=1):
        titles_raw, published_list, status = fetch_news_rows(
            client, ticker, int(args.max_articles)
        )

        seen: set[str] = set()
        unique_titles = []
        for title in titles_raw:
            key = normalize_title(title)
            if key and key not in seen:
                seen.add(key)
                unique_titles.append(title)

        scores = score_titles_detailed(nlp, unique_titles[:int(args.max_articles)])

        row = {
            "ticker": ticker,
            "date": pd.Timestamp(asof_date),
            "article_count_raw": int(len(titles_raw)),
            "article_count_unique": int(len(unique_titles)),
            "titles": " | ".join(unique_titles),
            "top_headline": unique_titles[0][:240] if unique_titles else "",
            "latest_published_utc": published_list[0] if published_list else None,
            "sent_score": scores["sent_score"],
            "pos_pct": scores["pos_pct"],
            "neg_pct": scores["neg_pct"],
            "neu_pct": scores["neu_pct"],
            "sent_dispersion": scores["sent_dispersion"],
            "sent_max": scores["sent_max"],
            "sent_min": scores["sent_min"],
            "sent_consensus": scores["sent_consensus"],
            "sentiment_mode": sentiment_mode,
            "status": status,
        }
        rows.append(row)

        if status.startswith("fetch_fail"):
            failed.append({"ticker": ticker, "reason": status})

        if idx % 25 == 0 or idx == total:
            print(f"  [{idx:,}/{total:,}] rows={len(rows):,} "
                  f"failures={len(failed):,}")

    result_df = pd.DataFrame(rows)

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

    if not args.skip_rolling and len(combined["date"].unique()) >= 3:
        print("Computing rolling sentiment features ...")
        combined = compute_rolling_features(combined)
    else:
        print("Skipping rolling features (need 3+ days of accumulated history)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)

    status_out = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "universe_parquet": str(universe_path),
        "output_parquet": str(output_path),
        "asof_date": asof_date,
        "tickers_requested": int(len(tickers)),
        "rows_written_today": int(len(result_df)),
        "rows_total": int(len(combined)),
        "unique_dates": int(combined["date"].nunique()),
        "sentiment_mode": sentiment_mode,
        "articles_per_ticker": args.max_articles,
        "failed": failed[:100],
    }
    status_path.write_text(json.dumps(status_out, indent=2), encoding="utf-8")

    print(f"Saved: {output_path}")
    print(f"Rows today: {len(result_df):,}")
    print(f"Total rows: {len(combined):,}")
    print(f"Unique dates: {combined['date'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())