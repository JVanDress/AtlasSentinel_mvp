#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from quintic_paths import data_dir, load_simple_dotenv, project_root


QUERY_ENDPOINT = "https://api.sec-api.io"
EXTRACTOR_ENDPOINT = "https://api.sec-api.io/extractor"

HEDGE_TERMS = (
    "actually",
    "to be fair",
    "frankly",
    "look",
    "i mean",
    "somewhat",
    "relatively",
    "approximately",
    "roughly",
    "cautious",
)

UNCERTAINTY_TERMS = (
    "uncertain",
    "uncertainty",
    "volatile",
    "volatility",
    "challenging",
    "headwind",
    "headwinds",
    "pressure",
    "softness",
    "risk",
    "risks",
    "may",
    "could",
    "potentially",
)

POSITIVE_GUIDANCE_TERMS = (
    "raise guidance",
    "raised guidance",
    "increasing guidance",
    "strong demand",
    "record revenue",
    "record earnings",
    "improved margin",
    "margin expansion",
    "accelerating",
    "outperform",
    "ahead of expectations",
    "healthy pipeline",
    "returning capital",
    "buyback",
)

NEGATIVE_GUIDANCE_TERMS = (
    "lower guidance",
    "lowered guidance",
    "reducing guidance",
    "withdrawing guidance",
    "macro pressure",
    "margin pressure",
    "inventory correction",
    "weakened demand",
    "challenging environment",
    "substantial doubt",
    "restructuring",
    "impairment",
    "litigation",
)

POSITIVE_TONE_TERMS = (
    "strong",
    "improved",
    "growth",
    "resilient",
    "confidence",
    "momentum",
    "expanding",
    "stabilized",
    "better than expected",
    "disciplined",
)

NEGATIVE_TONE_TERMS = (
    "weaker",
    "decline",
    "pressure",
    "softness",
    "deteriorated",
    "challenging",
    "constrained",
    "headwind",
    "slower",
    "below expectations",
)

# Forward-looking language: management discussing future plans/expectations
# WHY: Filings heavy on forward language signal management confidence.
# Research shows forward-looking ratio predicts 60-90 day returns.
FORWARD_LOOKING_TERMS = (
    "will", "expect", "expects", "expected", "anticipate", "anticipates",
    "plan", "plans", "planned", "planning", "intend", "intends",
    "believe", "believes", "forecast", "project", "projects", "projected",
    "outlook", "guidance", "target", "targets", "goal", "goals",
    "initiative", "initiatives", "strategy", "pipeline", "roadmap",
    "upcoming", "going forward", "looking ahead", "next quarter",
    "next year", "fiscal year", "remain confident", "well positioned",
)

# Backward-looking language: management reviewing past performance
BACKWARD_LOOKING_TERMS = (
    "was", "were", "achieved", "realized", "recorded", "reported",
    "delivered", "generated", "completed", "resulted", "reflected",
    "experienced", "declined", "decreased", "increased", "grew",
    "compared to", "year over year", "quarter over quarter",
    "prior year", "last quarter", "previous quarter", "prior quarter",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cleanroom SEC management tone overlay from recent 8-K, 10-Q, and 10-K filings."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--universe-parquet", type=Path, default=None)
    parser.add_argument("--output-parquet", type=Path, default=None)
    parser.add_argument("--latest-csv", type=Path, default=None)
    parser.add_argument("--status-output", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--asof-date", default=None)
    parser.add_argument("--lookback-days", type=int, default=120)
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--size-per-ticker", type=int, default=12)
    parser.add_argument("--pause-ms", type=int, default=0)
    parser.add_argument("--use-finbert", action="store_true")
    parser.add_argument("--replace-date", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
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


def require_sec_api_key(root: Path) -> str:
    load_simple_dotenv(root)
    key = (os.getenv("SEC_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("SEC_API_KEY was not found in the environment.")
    return key


def parse_asof_date(value: str | None) -> pd.Timestamp:
    if value:
        return pd.Timestamp(value).tz_localize(None).normalize()
    return pd.Timestamp(datetime.now(timezone.utc).date())


def load_universe(path: Path, max_tickers: int) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Required universe parquet not found: {path}")
    df = pd.read_parquet(path, columns=["ticker"])
    tickers = (
        df["ticker"].astype(str).str.upper().str.strip().replace("", pd.NA).dropna().drop_duplicates().sort_values().tolist()
    )
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


def build_finbert_pipeline(enabled: bool):
    if not enabled:
        return None, "disabled"
    try:
        from transformers import pipeline
        model = pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True, device=-1)
        return model, "finbert"
    except Exception:
        return None, "heuristic_fallback"


def load_anthropic_key() -> str | None:
    """Load Anthropic API key from environment."""
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    return key if key else None


CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
_claude_total_input_tokens = 0
_claude_total_output_tokens = 0


def claude_tone_score(text: str, section_type: str, api_key: str) -> dict | None:
    """
    Use Claude Sonnet to analyze filing section tone.
    Returns structured scores that are far richer than FinBERT's pos/neg/neutral.

    Cost: ~500 input tokens + ~150 output tokens per call
    At Sonnet pricing (~$3/M input, $15/M output): ~$0.004 per call
    """
    global _claude_total_input_tokens, _claude_total_output_tokens

    if not text or not text.strip() or len(text.strip()) < 50:
        return None

    # Truncate to ~3000 chars to control costs while getting enough context
    cleaned = " ".join(text.split())[:3000]

    prompt = f"""Analyze this {section_type} section from an SEC filing. Return ONLY valid JSON with these exact fields:
- tone_score: overall sentiment from -1.0 (very negative) to 1.0 (very positive)
- guidance_signal: forward guidance strength from -1.0 (lowering/withdrawing) to 1.0 (raising/strong)
- confidence_level: management's expressed confidence 0.0 to 1.0
- hedging_level: degree of hedging language 0.0 (direct) to 1.0 (heavily hedged)

Text:
{cleaned}"""

    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 200,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }

    for attempt in range(2):
        try:
            resp = requests.post(CLAUDE_API_URL, headers=headers, json=payload, timeout=30)

            if resp.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            if resp.status_code == 529:  # overloaded
                time.sleep(5.0)
                continue

            resp.raise_for_status()
            data = resp.json()

            # Track tokens
            usage = data.get("usage", {})
            _claude_total_input_tokens += usage.get("input_tokens", 0)
            _claude_total_output_tokens += usage.get("output_tokens", 0)

            # Parse response
            content = data["content"][0]["text"].strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            if content.startswith("json"):
                content = content[4:].strip()

            result = json.loads(content)
            return {
                "tone_score": float(np.clip(result.get("tone_score", 0.0), -1.0, 1.0)),
                "guidance_signal": float(np.clip(result.get("guidance_signal", 0.0), -1.0, 1.0)),
                "confidence_level": float(np.clip(result.get("confidence_level", 0.5), 0.0, 1.0)),
                "hedging_level": float(np.clip(result.get("hedging_level", 0.0), 0.0, 1.0)),
            }

        except (json.JSONDecodeError, KeyError, IndexError):
            return None
        except Exception:
            if attempt == 0:
                time.sleep(1.0)
                continue
            return None

    return None


def chunk_text(text: str, max_chars: int = 1200, max_chunks: int = 5) -> list[str]:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    words = cleaned.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        add_len = len(word) + (1 if current else 0)
        if current and current_len + add_len > max_chars:
            chunks.append(" ".join(current))
            if len(chunks) >= max_chunks:
                break
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += add_len
    if current and len(chunks) < max_chunks:
        chunks.append(" ".join(current))
    return chunks[:max_chunks]


def count_phrase_hits(text_lower: str, phrases: tuple[str, ...]) -> int:
    return sum(text_lower.count(phrase) for phrase in phrases)


def heuristic_tone_metrics(text: str) -> dict[str, float]:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return {
            "tone_score": 0.0,
            "guidance_signal": 0.0,
            "hedge_density": 0.0,
            "uncertainty_density": 0.0,
            "forward_looking_ratio": 0.0,
            "avg_sentence_length": 0.0,
            "word_count": 0.0,
        }

    text_lower = cleaned.lower()
    words = re.findall(r"\b[\w'-]+\b", text_lower)
    word_count = max(len(words), 1)

    pos_hits = count_phrase_hits(text_lower, POSITIVE_TONE_TERMS) + count_phrase_hits(text_lower, POSITIVE_GUIDANCE_TERMS)
    neg_hits = count_phrase_hits(text_lower, NEGATIVE_TONE_TERMS) + count_phrase_hits(text_lower, NEGATIVE_GUIDANCE_TERMS)
    hedge_hits = count_phrase_hits(text_lower, HEDGE_TERMS)
    uncertainty_hits = count_phrase_hits(text_lower, UNCERTAINTY_TERMS)
    guidance_pos = count_phrase_hits(text_lower, POSITIVE_GUIDANCE_TERMS)
    guidance_neg = count_phrase_hits(text_lower, NEGATIVE_GUIDANCE_TERMS)

    guidance_signal = float(np.clip((guidance_pos - guidance_neg) / max(guidance_pos + guidance_neg, 1), -1.0, 1.0))
    raw_tone = (pos_hits - neg_hits) / max(pos_hits + neg_hits, 1)

    hedge_density = (hedge_hits / word_count) * 1000.0
    uncertainty_density = (uncertainty_hits / word_count) * 1000.0
    penalty = min(0.35, (hedge_density / 100.0) + (uncertainty_density / 80.0))
    tone_score = float(np.clip(raw_tone - penalty + (0.20 * guidance_signal), -1.0, 1.0))

    # Forward-looking language ratio
    # WHY: Management confident about the future uses more forward terms.
    # A ratio > 1.0 means more forward than backward language = bullish signal.
    forward_hits = count_phrase_hits(text_lower, FORWARD_LOOKING_TERMS)
    backward_hits = count_phrase_hits(text_lower, BACKWARD_LOOKING_TERMS)
    total_directional = forward_hits + backward_hits
    if total_directional > 0:
        forward_looking_ratio = float(forward_hits / total_directional)
    else:
        forward_looking_ratio = 0.5  # neutral when no directional language found

    # Readability: average words per sentence
    # WHY: Complex language (long sentences) correlates with obfuscation of
    # bad news. Documented in Li (2008) "Annual Report Readability".
    # Higher = more complex = potentially hiding problems.
    sentences = re.split(r'[.!?]+', cleaned)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip().split()) >= 3]
    if sentences:
        avg_sentence_length = float(word_count / max(len(sentences), 1))
    else:
        avg_sentence_length = 0.0

    return {
        "tone_score": tone_score,
        "guidance_signal": guidance_signal,
        "hedge_density": float(hedge_density),
        "uncertainty_density": float(uncertainty_density),
        "forward_looking_ratio": forward_looking_ratio,
        "avg_sentence_length": avg_sentence_length,
        "word_count": float(word_count),
    }


def finbert_tone_score(nlp, text: str) -> float | None:
    if nlp is None:
        return None
    chunks = chunk_text(text)
    if not chunks:
        return None
    try:
        preds = nlp(chunks)
    except Exception:
        return None
    score_sum = 0.0
    weight_sum = 0.0
    for chunk, pred in zip(chunks, preds):
        label = str(pred.get("label", "")).lower()
        prob = float(pred.get("score", 0.0))
        weight = max(len(chunk.split()), 1)
        signed = 0.0
        if "pos" in label:
            signed = prob
        elif "neg" in label:
            signed = -prob
        score_sum += signed * weight
        weight_sum += weight
    if weight_sum <= 0:
        return None
    return float(np.clip(score_sum / weight_sum, -1.0, 1.0))


def score_text(nlp, text: str, section_type: str = "filing",
               anthropic_key: str | None = None) -> dict[str, float]:
    """
    Score filing text using best available engine:
    1. Claude Sonnet (if ANTHROPIC_API_KEY available) — best quality
    2. FinBERT (if --use-finbert) — decent quality
    3. Heuristic keywords — always available fallback
    """
    metrics = heuristic_tone_metrics(text)

    # Try Claude first (best quality)
    if anthropic_key:
        claude_result = claude_tone_score(text, section_type, anthropic_key)
        if claude_result is not None:
            # Claude provides richer scores — blend with heuristic for robustness
            metrics["tone_score"] = float(np.clip(
                0.75 * claude_result["tone_score"] + 0.25 * metrics["tone_score"],
                -1.0, 1.0
            ))
            metrics["guidance_signal"] = float(np.clip(
                0.75 * claude_result["guidance_signal"] + 0.25 * metrics["guidance_signal"],
                -1.0, 1.0
            ))
            metrics["confidence_level"] = claude_result["confidence_level"]
            metrics["hedging_level"] = claude_result["hedging_level"]
            return metrics

    # Fall back to FinBERT
    finbert_score = finbert_tone_score(nlp, text)
    if finbert_score is not None:
        metrics["tone_score"] = float(np.clip(
            0.65 * finbert_score + 0.35 * metrics["tone_score"],
            -1.0, 1.0
        ))

    # Heuristic-only metrics are already computed
    return metrics


def sec_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": api_key, "Content-Type": "application/json"}


def run_query(api_key: str, query: str, size: int) -> list[dict[str, Any]]:
    payload = {
        "query": {"query_string": {"query": query}},
        "from": "0",
        "size": str(size),
        "sort": [{"filedAt": {"order": "desc"}}],
    }
    response = requests.post(QUERY_ENDPOINT, headers=sec_headers(api_key), json=payload, timeout=45)
    response.raise_for_status()
    return list(response.json().get("filings", []))


def filing_date_value(filing: dict[str, Any] | None) -> pd.Timestamp:
    if not filing:
        return pd.NaT
    return pd.to_datetime(filing.get("filedAt"), utc=True, errors="coerce")


def normalize_items(raw_items: Any) -> set[str]:
    items: set[str] = set()
    if raw_items is None:
        return items
    if isinstance(raw_items, str):
        candidates = re.split(r"[,;|]+", raw_items)
    elif isinstance(raw_items, (list, tuple, set)):
        candidates = list(raw_items)
    else:
        candidates = [str(raw_items)]
    for value in candidates:
        token = str(value).strip().lower()
        if not token:
            continue
        items.add(token)
        items.add(token.replace(".", "-"))
        items.add(token.replace("-", "."))
    return items


def filing_url(filing: dict[str, Any] | None) -> str:
    if not filing:
        return ""
    for key in ("linkToFilingDetails", "linkToHtml", "linkToTxt"):
        value = str(filing.get(key, "") or "").strip()
        if value:
            return value
    return ""


def latest_by_form(filings: list[dict[str, Any]], form_prefixes: tuple[str, ...]) -> dict[str, Any] | None:
    prefixes = tuple(prefix.upper() for prefix in form_prefixes)
    for filing in filings:
        form_type = str(filing.get("formType", "")).upper()
        if any(form_type.startswith(prefix) for prefix in prefixes):
            return filing
    return None


def latest_earnings_8k(filings: list[dict[str, Any]]) -> dict[str, Any] | None:
    for filing in filings:
        form_type = str(filing.get("formType", "")).upper()
        if not form_type.startswith("8-K"):
            continue
        items = normalize_items(filing.get("items"))
        if {"2.02", "2-02", "2-2"} & items:
            return filing
        if {"7.01", "7-01", "7-1"} & items:
            return filing
    return None


def extract_section(api_key: str, url: str, item: str, retries: int = 3) -> str:
    if not url:
        return ""
    params = {"url": url, "item": item, "type": "text", "token": api_key}
    for attempt in range(retries):
        response = requests.get(EXTRACTOR_ENDPOINT, params=params, timeout=60)
        response.raise_for_status()
        text = response.text.strip()
        if text.lower() == "processing":
            time.sleep(0.75 * (attempt + 1))
            continue
        return text
    return ""


def choose_mdna_item(form_type: str) -> str | None:
    ft = str(form_type).upper()
    if ft.startswith("10-Q"):
        return "part1item2"
    if ft.startswith("10-K"):
        return "7"
    return None


def choose_risk_item(form_type: str) -> str | None:
    ft = str(form_type).upper()
    if ft.startswith("10-Q"):
        return "part2item1a"
    if ft.startswith("10-K"):
        return "1A"
    return None


def build_ticker_cache_path(cache_dir: Path, ticker: str, asof_date: pd.Timestamp) -> Path:
    safe_ticker = re.sub(r"[^A-Z0-9._-]+", "_", ticker.upper())
    return cache_dir / f"{safe_ticker}_{asof_date.date().isoformat()}.json"


def fetch_ticker_payload(
    ticker: str,
    api_key: str,
    asof_date: pd.Timestamp,
    lookback_days: int,
    size_per_ticker: int,
    cache_dir: Path,
    force_refresh: bool,
) -> dict[str, Any]:
    cache_path = build_ticker_cache_path(cache_dir, ticker, asof_date)
    if cache_path.exists() and not force_refresh:
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    start_date = (asof_date - pd.Timedelta(days=int(lookback_days))).date().isoformat()
    end_date = asof_date.date().isoformat()
    query = (
        f'ticker:{ticker} AND filedAt:[{start_date} TO {end_date}] AND '
        '(formType:"8-K" OR formType:"8-K/A" OR formType:"10-Q" OR formType:"10-Q/A" OR formType:"10-K" OR formType:"10-K/A")'
    )
    filings = run_query(api_key, query=query, size=size_per_ticker)
    payload = {"ticker": ticker, "filings": filings}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_row(
    ticker: str,
    asof_date: pd.Timestamp,
    payload: dict[str, Any],
    api_key: str,
    nlp,
    anthropic_key: str | None = None,
) -> dict[str, Any]:
    filings = list(payload.get("filings", []))
    filings = sorted(filings, key=lambda x: filing_date_value(x), reverse=True)

    thirty_days_ago = asof_date.tz_localize("UTC") - pd.Timedelta(days=30)
    filing_count_30d = 0
    for filing in filings:
        filed_at = filing_date_value(filing)
        if pd.notna(filed_at) and filed_at >= thirty_days_ago:
            filing_count_30d += 1

    latest_8k = latest_earnings_8k(filings)
    latest_qk = latest_by_form(filings, ("10-Q", "10-K"))

    earnings_text = ""
    if latest_8k is not None:
        url = filing_url(latest_8k)
        items = normalize_items(latest_8k.get("items"))
        if {"2.02", "2-02", "2-2"} & items:
            earnings_text = extract_section(api_key, url, "2-2")
        if not earnings_text and {"7.01", "7-01", "7-1"} & items:
            earnings_text = extract_section(api_key, url, "7-1")

    mdna_text = ""
    risk_text = ""
    if latest_qk is not None:
        url = filing_url(latest_qk)
        mdna_item = choose_mdna_item(str(latest_qk.get("formType", "")))
        risk_item = choose_risk_item(str(latest_qk.get("formType", "")))
        if mdna_item:
            mdna_text = extract_section(api_key, url, mdna_item)
        if risk_item:
            risk_text = extract_section(api_key, url, risk_item)

    earnings_metrics = score_text(nlp, earnings_text, section_type="earnings release",
                                   anthropic_key=anthropic_key)
    mdna_metrics = score_text(nlp, mdna_text, section_type="management discussion and analysis (MD&A)",
                               anthropic_key=anthropic_key)
    risk_metrics = score_text(nlp, risk_text, section_type="risk factors",
                               anthropic_key=anthropic_key)

    latest_event = latest_8k or latest_qk
    latest_event_dt = filing_date_value(latest_event)
    filing_recency_days = np.nan
    if pd.notna(latest_event_dt):
        filing_recency_days = float((asof_date.tz_localize("UTC") - latest_event_dt).days)

    uncertainty_penalty = min(1.0, (earnings_metrics["uncertainty_density"] + mdna_metrics["uncertainty_density"]) / 40.0)
    hedge_penalty = min(1.0, (earnings_metrics["hedge_density"] + mdna_metrics["hedge_density"]) / 40.0)
    management_tone_score = float(
        np.clip(
            (0.45 * mdna_metrics["tone_score"])
            + (0.35 * earnings_metrics["tone_score"])
            + (0.20 * earnings_metrics["guidance_signal"])
            - (0.10 * uncertainty_penalty)
            - (0.10 * hedge_penalty),
            -1.0,
            1.0,
        )
    )

    # Forward-looking ratio: weighted average across sections
    # MD&A weighted higher because it's where management discusses outlook
    mdna_fwd = mdna_metrics.get("forward_looking_ratio", 0.5)
    earn_fwd = earnings_metrics.get("forward_looking_ratio", 0.5)
    forward_looking_ratio = float(0.60 * mdna_fwd + 0.40 * earn_fwd)

    # Readability: average sentence length across MD&A and earnings
    # Higher = more complex language = potential obfuscation
    mdna_asl = mdna_metrics.get("avg_sentence_length", 0.0)
    earn_asl = earnings_metrics.get("avg_sentence_length", 0.0)
    if mdna_asl > 0 and earn_asl > 0:
        avg_sentence_length = float(0.60 * mdna_asl + 0.40 * earn_asl)
    else:
        avg_sentence_length = float(max(mdna_asl, earn_asl))

    # Risk section word count: raw length for cross-filing comparison
    # WHY: Growth in risk section length across filings predicts
    # future negative events independent of risk tone
    risk_section_word_count = float(risk_metrics.get("word_count", 0.0))

    return {
        "date": asof_date.date().isoformat(),
        "ticker": ticker,
        "management_tone_score": management_tone_score,
        "earnings_release_tone": float(earnings_metrics["tone_score"]),
        "guidance_tone": float(earnings_metrics["guidance_signal"]),
        "mdna_tone": float(mdna_metrics["tone_score"]),
        "risk_factor_tone": float(risk_metrics["tone_score"]),
        "hedge_density": float(max(earnings_metrics["hedge_density"], mdna_metrics["hedge_density"])),
        "uncertainty_density": float(max(earnings_metrics["uncertainty_density"], mdna_metrics["uncertainty_density"])),
        "forward_looking_ratio": forward_looking_ratio,
        "avg_sentence_length": avg_sentence_length,
        "risk_section_word_count": risk_section_word_count,
        "sec_filing_count_30d": int(filing_count_30d),
        "earnings_event_flag": int(latest_8k is not None),
        "filing_recency_days": filing_recency_days,
        "source_8k_filed_at": latest_8k.get("filedAt") if latest_8k else None,
        "source_8k_form_type": latest_8k.get("formType") if latest_8k else None,
        "source_qk_filed_at": latest_qk.get("filedAt") if latest_qk else None,
        "source_qk_form_type": latest_qk.get("formType") if latest_qk else None,
        "extract_status": "ok" if (earnings_text or mdna_text or risk_text) else "no_extractable_sections",
        "latest_event_at": latest_event_dt.isoformat() if pd.notna(latest_event_dt) else None,
        "source_url_8k": filing_url(latest_8k) if latest_8k else None,
        "source_url_qk": filing_url(latest_qk) if latest_qk else None,
    }


def main() -> int:
    args = parse_args()
    root = runtime_project_root(args.project_root)
    data_root = data_dir(root)
    data_root.mkdir(parents=True, exist_ok=True)

    universe_path = resolve_path(args.universe_parquet, data_root, "polygon_equity_universe.parquet", root_dir=root)
    output_path = resolve_path(args.output_parquet, data_root, "cleanroom_management_tone_overlay.parquet", root_dir=root)
    latest_csv = resolve_path(args.latest_csv, data_root, "cleanroom_management_tone_overlay_latest.csv", root_dir=root)
    status_path = resolve_path(args.status_output, data_root, "cleanroom_management_tone_overlay_status.json", root_dir=root)
    cache_dir = resolve_path(args.cache_dir, data_root, "sec_api_cache/management_tone", root_dir=root)
    asof_date = parse_asof_date(args.asof_date)

    print(f"Project root: {root}")
    print(f"Universe parquet: {universe_path}")
    print(f"Output parquet: {output_path}")
    print(f"As-of date: {asof_date.date().isoformat()}")

    tickers = load_universe(universe_path, max_tickers=int(args.max_tickers))
    print(f"Tickers requested: {len(tickers):,}")

    if args.dry_run:
        print("Dry run complete. No management tone overlay written.")
        return 0

    api_key = require_sec_api_key(root)
    nlp, scorer_mode = build_finbert_pipeline(args.use_finbert)

    # Load Anthropic key for Claude-powered scoring
    anthropic_key = load_anthropic_key()
    if anthropic_key:
        scorer_mode = "claude_sonnet"
        print(f"Scoring engine: Claude Sonnet (with heuristic blend)")
    elif scorer_mode == "finbert":
        print(f"Scoring engine: FinBERT (with heuristic blend)")
    else:
        print(f"Scoring engine: heuristic only")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    cache_dir.mkdir(parents=True, exist_ok=True)

    for idx, ticker in enumerate(tickers, start=1):
        try:
            payload = fetch_ticker_payload(
                ticker=ticker,
                api_key=api_key,
                asof_date=asof_date,
                lookback_days=int(args.lookback_days),
                size_per_ticker=int(args.size_per_ticker),
                cache_dir=cache_dir,
                force_refresh=bool(args.force_refresh),
            )
            row = build_row(ticker=ticker, asof_date=asof_date, payload=payload,
                           api_key=api_key, nlp=nlp, anthropic_key=anthropic_key)
            rows.append(row)
        except Exception as exc:
            failures.append({"ticker": ticker, "error": f"{type(exc).__name__}: {exc}"})
            rows.append(
                {
                    "date": asof_date.date().isoformat(),
                    "ticker": ticker,
                    "management_tone_score": np.nan,
                    "earnings_release_tone": np.nan,
                    "guidance_tone": np.nan,
                    "mdna_tone": np.nan,
                    "risk_factor_tone": np.nan,
                    "hedge_density": np.nan,
                    "uncertainty_density": np.nan,
                    "forward_looking_ratio": np.nan,
                    "avg_sentence_length": np.nan,
                    "risk_section_word_count": np.nan,
                    "sec_filing_count_30d": np.nan,
                    "earnings_event_flag": 0,
                    "filing_recency_days": np.nan,
                    "source_8k_filed_at": None,
                    "source_8k_form_type": None,
                    "source_qk_filed_at": None,
                    "source_qk_form_type": None,
                    "extract_status": "error",
                    "latest_event_at": None,
                    "source_url_8k": None,
                    "source_url_qk": None,
                }
            )
        if args.pause_ms and idx < len(tickers):
            time.sleep(max(int(args.pause_ms), 0) / 1000.0)
        if idx % 50 == 0 or idx == len(tickers):
            print(f"Processed {idx:,}/{len(tickers):,} tickers")

    overlay = pd.DataFrame(rows)
    overlay["date"] = pd.to_datetime(overlay["date"], errors="coerce")
    overlay = overlay.sort_values(["date", "ticker"]).reset_index(drop=True)

    existing = load_existing_output(output_path)
    if not existing.empty:
        if "date" in existing.columns:
            existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
        if args.replace_date:
            existing = existing.loc[existing["date"] != asof_date].copy()
        merged = pd.concat([existing, overlay], ignore_index=True)
        merged = merged.sort_values(["date", "ticker"]).drop_duplicates(subset=["date", "ticker"], keep="last")
    else:
        merged = overlay.copy()

    # Compute tone CHANGE features from accumulated filing history
    # WHY: The change in management tone across consecutive filings is
    # more predictive than the level. A tone drop from +0.6 to +0.2
    # predicts 60-90 day underperformance even though +0.2 is still positive.
    if merged["date"].nunique() >= 2:
        merged = merged.sort_values(["ticker", "date"])

        for col in ["management_tone_score", "mdna_tone", "guidance_tone",
                     "forward_looking_ratio", "risk_section_word_count"]:
            if col not in merged.columns:
                continue
            delta_col = f"{col}_delta"
            merged[delta_col] = merged.groupby("ticker", sort=False)[col].transform(
                lambda s: s - s.shift(1)
            )

        # Risk section growth rate: pct change in risk word count
        # WHY: A 50% increase in risk factor length across filings is a
        # strong predictor of future negative events (new risks added)
        if "risk_section_word_count" in merged.columns:
            merged["risk_section_growth"] = merged.groupby("ticker", sort=False)[
                "risk_section_word_count"
            ].transform(
                lambda s: s.pct_change().clip(-5.0, 5.0)
            )

        merged = merged.sort_values(["date", "ticker"]).reset_index(drop=True)
    else:
        print("  Only 1 filing date — tone change features require accumulated history")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)

    latest_df = overlay.copy().sort_values("management_tone_score", ascending=False)
    latest_df.to_csv(latest_csv, index=False)

    latest_date = overlay["date"].max()
    status = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "universe_parquet": str(universe_path),
        "output_parquet": str(output_path),
        "latest_csv": str(latest_csv),
        "status_output": str(status_path),
        "cache_dir": str(cache_dir),
        "rows_written_latest": int(len(overlay)),
        "rows_total": int(len(merged)),
        "tickers_requested": int(len(tickers)),
        "tickers_failed": int(len(failures)),
        "failed_examples": failures[:10],
        "latest_date": str(pd.to_datetime(latest_date).date()) if pd.notna(latest_date) else None,
        "scorer_mode": scorer_mode,
        "lookback_days": int(args.lookback_days),
        "size_per_ticker": int(args.size_per_ticker),
        "claude_input_tokens": _claude_total_input_tokens,
        "claude_output_tokens": _claude_total_output_tokens,
        "claude_estimated_cost_usd": round(
            (_claude_total_input_tokens / 1_000_000 * 3.0) +
            (_claude_total_output_tokens / 1_000_000 * 15.0), 4
        ),
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    print(f"Saved management tone overlay: {output_path}")
    print(f"Saved latest management tone CSV: {latest_csv}")
    print(f"Saved status file: {status_path}")
    print(f"Latest rows: {len(overlay):,}")
    if _claude_total_input_tokens > 0:
        cost = (_claude_total_input_tokens / 1_000_000 * 3.0) + (_claude_total_output_tokens / 1_000_000 * 15.0)
        print(f"Claude tokens: {_claude_total_input_tokens:,} in / {_claude_total_output_tokens:,} out")
        print(f"Claude cost: ${cost:.4f}")
    if failures:
        print(f"Failures: {len(failures):,} (see status file for examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
