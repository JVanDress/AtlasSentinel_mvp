#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quintic_paths import data_dir, project_root


EPSILON = 1e-12


@dataclass
class MacroContext:
    score: float
    confidence: float
    label: str
    source: str
    asof_date: pd.Timestamp | pd.NaT


@dataclass
class VdceContext:
    stress_score: float
    state: str
    asof_date: pd.Timestamp | pd.NaT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a combined 20d/60d/90d regime summary with commentary."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--gmm-panel", type=Path, default=None)
    parser.add_argument("--macro-overlay", type=Path, default=None)
    parser.add_argument("--macro-fallback", type=Path, default=None)
    parser.add_argument("--sector-panel", type=Path, default=None)
    parser.add_argument("--technicals-parquet", type=Path, default=None)
    parser.add_argument("--vdce-csv", type=Path, default=None)
    parser.add_argument("--vdce-leaders-csv", type=Path, default=None)
    parser.add_argument("--calibration-csv", type=Path, default=None)
    parser.add_argument("--output-parquet", type=Path, default=None)
    parser.add_argument("--latest-csv", type=Path, default=None)
    parser.add_argument("--status-output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def runtime_project_root(cli_root: Path | None) -> Path:
    if cli_root is not None:
        return cli_root.expanduser()
    return project_root()


def resolve_path(
    path: Path | None,
    base_dir: Path,
    default_name: str,
    root_dir: Path | None = None,
) -> Path:
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


def clip(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    if value is None or not np.isfinite(value):
        return float(np.nan)
    return float(np.clip(value, lower, upper))


def safe_tanh(value: float, scale: float) -> float:
    if value is None or not np.isfinite(value):
        return float(np.nan)
    denom = float(scale) if float(scale) > EPSILON else 1.0
    return float(np.tanh(value / denom))


def pick_latest_per_ticker(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=["ticker", date_col]).copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out = out.sort_values(["ticker", date_col]).drop_duplicates(["ticker"], keep="last")
    return out.reset_index(drop=True)


def score_to_label(score: float) -> str:
    if not np.isfinite(score):
        return "unknown"
    if score >= 0.45:
        return "risk_on"
    if score >= 0.15:
        return "constructive"
    if score > -0.15:
        return "neutral"
    if score > -0.45:
        return "caution"
    return "risk_off"


def compute_calibration_quality(calibration_path: Path) -> dict[int, float]:
    if not calibration_path.exists():
        return {20: 0.70, 60: 0.70, 90: 0.70}

    cal = pd.read_csv(calibration_path)
    qualities: dict[int, float] = {}
    for _, row in cal.iterrows():
        try:
            horizon = int(row["horizon_days"])
            brier = float(row["weighted_brier"])
        except Exception:
            continue
        quality = np.clip(1.0 - (brier / 0.10), 0.50, 0.98)
        qualities[horizon] = float(quality)
    for horizon in (20, 60, 90):
        qualities.setdefault(horizon, 0.70)
    return qualities


def compute_stock_regime_map(gmm_df: pd.DataFrame) -> dict[float, float]:
    valid = gmm_df[gmm_df["gmm_regime_label"].notna()].copy()
    if valid.empty:
        return {}

    grouped = (
        valid.groupby("gmm_regime_label", as_index=False)
        .agg(
            median_return=("log_return_1d", "median"),
            median_vol=("realized_vol_20", "median"),
            median_drawdown=("drawdown_60", "median"),
            median_stretch=("ou_log_close_stretch_63", "median"),
        )
        .fillna(0.0)
    )
    grouped["return_rank"] = grouped["median_return"].rank(pct=True)
    grouped["vol_rank"] = 1.0 - grouped["median_vol"].rank(pct=True)
    grouped["drawdown_rank"] = grouped["median_drawdown"].rank(pct=True)
    grouped["stretch_rank"] = grouped["median_stretch"].rank(pct=True)
    grouped["composite"] = (
        0.35 * grouped["return_rank"]
        + 0.25 * grouped["vol_rank"]
        + 0.25 * grouped["drawdown_rank"]
        + 0.15 * grouped["stretch_rank"]
    )
    grouped["score"] = (grouped["composite"] - 0.5) * 2.0
    return {
        float(label): float(score)
        for label, score in zip(grouped["gmm_regime_label"], grouped["score"])
    }


def build_macro_context(macro_overlay_path: Path, macro_fallback_path: Path) -> MacroContext:
    if macro_overlay_path.exists():
        overlay = pd.read_parquet(macro_overlay_path).copy()
        overlay["date"] = pd.to_datetime(overlay["date"], errors="coerce")
        overlay = overlay.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        if overlay.empty:
            raise RuntimeError(f"Macro overlay exists but is empty: {macro_overlay_path}")

        state_map: dict[float, float] = {}
        if "regime_state" in overlay.columns:
            prof = (
                overlay.dropna(subset=["regime_state"])
                .groupby("regime_state", as_index=False)
                .agg(
                    median_mom=("macro_spy_mom_20d", "median"),
                    median_pressure=("volatility_instability_pressure", "median"),
                    median_cross_asset=("cross_asset_pressure_index", "median"),
                    median_systemic=("systemic_instability_risk_score", "median"),
                )
            )
            if not prof.empty:
                prof["mom_rank"] = prof["median_mom"].rank(pct=True)
                prof["pressure_rank"] = 1.0 - prof["median_pressure"].rank(pct=True)
                prof["cross_asset_rank"] = 1.0 - prof["median_cross_asset"].rank(pct=True)
                prof["systemic_rank"] = 1.0 - prof["median_systemic"].rank(pct=True)
                prof["composite"] = (
                    0.35 * prof["mom_rank"]
                    + 0.25 * prof["pressure_rank"]
                    + 0.20 * prof["cross_asset_rank"]
                    + 0.20 * prof["systemic_rank"]
                )
                prof["score"] = (prof["composite"] - 0.5) * 2.0
                state_map = {
                    float(label): float(score)
                    for label, score in zip(prof["regime_state"], prof["score"])
                }

        latest = overlay.iloc[-1]
        regime_state = latest.get("regime_state")
        regime_score = state_map.get(float(regime_state), 0.0) if pd.notna(regime_state) else 0.0
        prob_cols = [col for col in overlay.columns if col.startswith("regime_prob_")]
        confidence = float(latest[prob_cols].max()) if prob_cols else 0.60
        score = clip(
            0.55 * regime_score
            + 0.20 * safe_tanh(float(latest.get("macro_spy_mom_20d", 0.0)), 0.10)
            + 0.15 * (-safe_tanh(float(latest.get("volatility_instability_pressure", 0.0)) - 1.0, 0.40))
            + 0.10 * (-safe_tanh(float(latest.get("cross_asset_pressure_index", 0.0)), 0.75))
        )
        label = score_to_label(score)
        return MacroContext(
            score=float(score if np.isfinite(score) else 0.0),
            confidence=float(np.clip(confidence, 0.35, 0.99)),
            label=label,
            source="cleanroom_macro_overlay",
            asof_date=pd.to_datetime(latest["date"], errors="coerce"),
        )

    if not macro_fallback_path.exists():
        return MacroContext(score=0.0, confidence=0.40, label="unknown", source="missing", asof_date=pd.NaT)

    macro = pd.read_parquet(macro_fallback_path).copy()
    macro["date"] = pd.to_datetime(macro["date"], errors="coerce")
    macro = macro.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if macro.empty:
        return MacroContext(score=0.0, confidence=0.40, label="unknown", source="empty_macro_20d", asof_date=pd.NaT)

    window = macro.tail(min(252, len(macro))).copy()
    for col in ("vix", "high_yield_spread", "us_2s10s_spread", "vix_5d_change", "dxy_5d_change"):
        if col in window.columns:
            window[col] = pd.to_numeric(window[col], errors="coerce")

    latest = window.iloc[-1]

    def latest_z(col: str) -> float:
        if col not in window.columns:
            return 0.0
        series = pd.to_numeric(window[col], errors="coerce").dropna()
        if len(series) < 20:
            return 0.0
        std = float(series.std())
        if not np.isfinite(std) or std <= EPSILON:
            return 0.0
        return float((series.iloc[-1] - series.mean()) / std)

    score = clip(
        -0.30 * latest_z("vix")
        - 0.30 * latest_z("high_yield_spread")
        + 0.20 * latest_z("us_2s10s_spread")
        - 0.10 * max(0.0, latest_z("vix_5d_change"))
        - 0.10 * abs(latest_z("dxy_5d_change"))
    )
    label = score_to_label(score)
    return MacroContext(
        score=float(score if np.isfinite(score) else 0.0),
        confidence=0.55,
        label=label,
        source="macro_20d_fallback",
        asof_date=pd.to_datetime(latest["date"], errors="coerce"),
    )


def build_vdce_context(vdce_csv_path: Path) -> VdceContext:
    if not vdce_csv_path.exists():
        return VdceContext(stress_score=0.0, state="unknown", asof_date=pd.NaT)
    vdce = pd.read_csv(vdce_csv_path)
    if vdce.empty:
        return VdceContext(stress_score=0.0, state="unknown", asof_date=pd.NaT)
    vdce["date"] = pd.to_datetime(vdce["date"], errors="coerce")
    vdce = vdce.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if vdce.empty:
        return VdceContext(stress_score=0.0, state="unknown", asof_date=pd.NaT)
    latest = vdce.iloc[-1]
    state_map = {"GREEN": 0.50, "YELLOW": 0.10, "ORANGE": -0.35, "RED": -0.80}
    score = state_map.get(str(latest.get("state", "unknown")).upper(), 0.0)
    return VdceContext(
        stress_score=float(score),
        state=str(latest.get("state", "unknown")),
        asof_date=pd.to_datetime(latest["date"], errors="coerce"),
    )


def load_vdce_leaders(vdce_leaders_path: Path) -> pd.DataFrame:
    if not vdce_leaders_path.exists():
        return pd.DataFrame(columns=["ticker", "vdce_leader_score"])
    leaders = pd.read_csv(vdce_leaders_path)
    if leaders.empty or "ticker" not in leaders.columns or "leader_score" not in leaders.columns:
        return pd.DataFrame(columns=["ticker", "vdce_leader_score"])
    leaders = leaders[["ticker", "leader_score"]].copy()
    leaders["ticker"] = leaders["ticker"].astype(str).str.upper().str.strip()
    leaders["vdce_leader_score"] = pd.to_numeric(leaders["leader_score"], errors="coerce")
    return leaders.drop(columns=["leader_score"]).dropna(subset=["ticker"]).drop_duplicates("ticker", keep="last")


def pick_driver_phrases(row: pd.Series, horizon: int) -> list[tuple[float, str]]:
    phrases: list[tuple[float, str]] = []

    stock_signal = float(row.get("stock_regime_signal", np.nan))
    if np.isfinite(stock_signal):
        phrase = (
            "stock regime profile is constructive"
            if stock_signal >= 0
            else "stock regime profile is defensive"
        )
        phrases.append((abs(stock_signal), phrase))

    macro_signal = float(row.get("macro_regime_score", np.nan))
    if np.isfinite(macro_signal):
        phrase = (
            f"macro backdrop is {row.get('macro_regime_label', 'mixed')}"
            if macro_signal >= 0
            else f"macro backdrop is {row.get('macro_regime_label', 'defensive')}"
        )
        phrases.append((abs(macro_signal), phrase))

    sector_signal = float(row.get("sector_signal", np.nan))
    if np.isfinite(sector_signal):
        phrase = (
            "sector leadership is supportive"
            if sector_signal >= 0
            else "sector leadership is deteriorating"
        )
        phrases.append((abs(sector_signal), phrase))

    tech_col = {20: "tech_signal_20d", 60: "tech_signal_60d", 90: "tech_signal_90d"}[horizon]
    tech_signal = float(row.get(tech_col, np.nan))
    if np.isfinite(tech_signal):
        phrase = (
            "trend and momentum are aligned"
            if tech_signal >= 0
            else "trend and momentum are soft"
        )
        phrases.append((abs(tech_signal), phrase))

    vdce_signal = float(row.get("vdce_signal", np.nan))
    if np.isfinite(vdce_signal) and abs(vdce_signal) > 0.05:
        phrase = (
            f"market stress is contained ({row.get('vdce_market_state', 'n/a')})"
            if vdce_signal >= 0
            else f"market stress is elevated ({row.get('vdce_market_state', 'n/a')})"
        )
        phrases.append((abs(vdce_signal), phrase))

    phrases.sort(key=lambda item: item[0], reverse=True)
    return phrases[:3]


def build_commentary(row: pd.Series, horizon: int) -> str:
    label = str(row.get(f"regime_{horizon}d_label", "unknown")).replace("_", " ")
    drivers = [text for _, text in pick_driver_phrases(row, horizon)]
    if not drivers:
        return f"{horizon}d regime is {label} with limited supporting context."
    return f"{horizon}d regime is {label}; " + "; ".join(drivers) + "."


def main() -> int:
    args = parse_args()
    root = runtime_project_root(args.project_root)
    data_root = data_dir(root)
    data_root.mkdir(parents=True, exist_ok=True)

    gmm_panel_path = resolve_path(args.gmm_panel, data_root, "cleanroom_gmm_panel.parquet", root_dir=root)
    macro_overlay_path = resolve_path(args.macro_overlay, data_root, "cleanroom_macro_overlay.parquet", root_dir=root)
    macro_fallback_path = resolve_path(args.macro_fallback, data_root, "macro_20d.parquet", root_dir=root)
    sector_panel_path = resolve_path(args.sector_panel, data_root, "panel_with_sector_studies.parquet", root_dir=root)
    technicals_path = resolve_path(args.technicals_parquet, data_root, "20day_technicals.parquet", root_dir=root)
    vdce_csv_path = resolve_path(args.vdce_csv, data_root, "iwm_vdce.csv", root_dir=root)
    vdce_leaders_path = resolve_path(args.vdce_leaders_csv, data_root, "vdce_iwm_leaders.csv", root_dir=root)
    calibration_path = resolve_path(args.calibration_csv, data_root, "calibrated_ou_params.csv", root_dir=root)
    output_path = resolve_path(args.output_parquet, data_root, "cleanroom_regime_summary.parquet", root_dir=root)
    latest_csv_path = resolve_path(args.latest_csv, data_root, "cleanroom_regime_summary_latest.csv", root_dir=root)
    status_path = resolve_path(args.status_output, data_root, "cleanroom_regime_summary_status.json", root_dir=root)

    print(f"Project root: {root}")
    print(f"GMM panel: {gmm_panel_path}")
    print(f"Macro overlay: {macro_overlay_path}")
    print(f"Macro fallback: {macro_fallback_path}")
    print(f"Sector panel: {sector_panel_path}")
    print(f"Technicals parquet: {technicals_path}")
    print(f"Output parquet: {output_path}")

    if args.dry_run:
        print("Dry run complete. No regime summary written.")
        return 0

    if not sector_panel_path.exists():
        raise FileNotFoundError(f"Required sector panel not found: {sector_panel_path}")
    if not technicals_path.exists():
        raise FileNotFoundError(f"Required technicals parquet not found: {technicals_path}")

    sector_panel = pd.read_parquet(sector_panel_path)
    technicals = pd.read_parquet(technicals_path)

    base = pick_latest_per_ticker(technicals)
    keep_cols = [
        "ticker",
        "date",
        "close",
        "gics_sector",
        "gics_industry",
        "rsi_14",
        "price_vs_sma20",
        "sma20_slope_5",
        "ema10_slope_5",
        "realized_vol_20",
        "breakout_20d_high_flag",
        "breakdown_20d_low_flag",
        "dist_from_20d_high",
        "dist_from_20d_low",
        "ret_20d_back",
        "ret_60d_back",
        "ret_90d_back",
    ]
    keep_cols = [col for col in keep_cols if col in base.columns]
    base = base[keep_cols].copy()

    sector_latest = pick_latest_per_ticker(sector_panel)
    sector_keep = [
        "ticker",
        "date",
        "gics_sector",
        "gics_industry",
        "sector_relative_strength_20d",
        "sector_risk_multiplier",
        "sector_avg_20d_vol_prob",
    ]
    sector_keep = [col for col in sector_keep if col in sector_latest.columns]
    sector_latest = sector_latest[sector_keep].copy()
    base = base.merge(sector_latest.drop(columns=["date"], errors="ignore"), on="ticker", how="left")

    stock_regime_map: dict[float, float] = {}
    if gmm_panel_path.exists():
        gmm_panel = pd.read_parquet(gmm_panel_path)
        stock_regime_map = compute_stock_regime_map(gmm_panel)
        gmm_latest = pick_latest_per_ticker(gmm_panel)
        gmm_keep = [
            "ticker",
            "date",
            "gmm_regime_label",
            "gmm_regime_confidence",
            "log_return_1d",
            "drawdown_60",
            "ou_log_close_stretch_63",
        ]
        gmm_keep = [col for col in gmm_keep if col in gmm_latest.columns]
        gmm_latest = gmm_latest[gmm_keep].copy()
        base = base.merge(gmm_latest.drop(columns=["date"], errors="ignore"), on="ticker", how="left")
        base["stock_regime_score"] = (
            pd.to_numeric(base["gmm_regime_label"], errors="coerce").map(stock_regime_map).fillna(0.0)
        )
        base["gmm_regime_confidence"] = pd.to_numeric(base.get("gmm_regime_confidence"), errors="coerce").fillna(0.50)
        base["stock_regime_signal"] = base["stock_regime_score"] * (0.50 + 0.50 * base["gmm_regime_confidence"])
    else:
        base["gmm_regime_label"] = np.nan
        base["gmm_regime_confidence"] = 0.50
        base["stock_regime_score"] = 0.0
        base["stock_regime_signal"] = 0.0

    macro_context = build_macro_context(macro_overlay_path, macro_fallback_path)
    base["macro_regime_score"] = macro_context.score
    base["macro_regime_confidence"] = macro_context.confidence
    base["macro_regime_label"] = macro_context.label
    base["macro_regime_source"] = macro_context.source
    base["macro_regime_date"] = macro_context.asof_date

    vdce_context = build_vdce_context(vdce_csv_path)
    base["vdce_signal"] = vdce_context.stress_score
    base["vdce_market_state"] = vdce_context.state
    base["vdce_market_date"] = vdce_context.asof_date
    vdce_leaders = load_vdce_leaders(vdce_leaders_path)
    if not vdce_leaders.empty:
        base = base.merge(vdce_leaders, on="ticker", how="left")
    else:
        base["vdce_leader_score"] = np.nan

    calibration_quality = compute_calibration_quality(calibration_path)

    sector_rel = pd.to_numeric(
        base.get("sector_relative_strength_20d", pd.Series(np.nan, index=base.index)),
        errors="coerce",
    )
    sector_vol_prob = pd.to_numeric(
        base.get("sector_avg_20d_vol_prob", pd.Series(0.50, index=base.index)),
        errors="coerce",
    )
    sector_risk = pd.to_numeric(
        base.get("sector_risk_multiplier", pd.Series(1.0, index=base.index)),
        errors="coerce",
    )
    base["sector_signal"] = (
        0.45 * sector_rel.apply(lambda x: safe_tanh(float(x), 0.05) if pd.notna(x) else np.nan).fillna(0.0)
        + 0.35 * (0.50 - sector_vol_prob.fillna(0.50))
        + 0.20 * (sector_risk.fillna(1.0) - 1.0)
    )
    base["sector_signal"] = base["sector_signal"].apply(clip)

    price_vs_sma20 = pd.to_numeric(
        base.get("price_vs_sma20", pd.Series(0.0, index=base.index)),
        errors="coerce",
    )
    rsi_14 = pd.to_numeric(
        base.get("rsi_14", pd.Series(50.0, index=base.index)),
        errors="coerce",
    )
    ret_20d_back = pd.to_numeric(
        base.get("ret_20d_back", pd.Series(0.0, index=base.index)),
        errors="coerce",
    )
    ret_60d_back = pd.to_numeric(
        base.get("ret_60d_back", pd.Series(0.0, index=base.index)),
        errors="coerce",
    )
    ret_90d_back = pd.to_numeric(
        base.get("ret_90d_back", pd.Series(0.0, index=base.index)),
        errors="coerce",
    )
    breakout_20d_high_flag = pd.to_numeric(
        base.get("breakout_20d_high_flag", pd.Series(0.0, index=base.index)),
        errors="coerce",
    )
    breakdown_20d_low_flag = pd.to_numeric(
        base.get("breakdown_20d_low_flag", pd.Series(0.0, index=base.index)),
        errors="coerce",
    )
    sma20_slope_5 = pd.to_numeric(
        base.get("sma20_slope_5", pd.Series(0.0, index=base.index)),
        errors="coerce",
    )
    ema10_slope_5 = pd.to_numeric(
        base.get("ema10_slope_5", pd.Series(0.0, index=base.index)),
        errors="coerce",
    )
    realized_vol_20 = pd.to_numeric(
        base.get("realized_vol_20", pd.Series(0.0, index=base.index)),
        errors="coerce",
    )
    dist_from_20d_high = pd.to_numeric(
        base.get("dist_from_20d_high", pd.Series(0.0, index=base.index)),
        errors="coerce",
    )

    base["tech_signal_20d"] = (
        0.30 * price_vs_sma20.apply(lambda x: safe_tanh(float(x), 0.05) if pd.notna(x) else np.nan).fillna(0.0)
        + 0.20 * ((rsi_14.fillna(50.0) - 50.0) / 50.0)
        + 0.20 * ret_20d_back.apply(lambda x: safe_tanh(float(x), 0.12) if pd.notna(x) else np.nan).fillna(0.0)
        + 0.15 * breakout_20d_high_flag.fillna(0.0)
        - 0.15 * breakdown_20d_low_flag.fillna(0.0)
    )
    base["tech_signal_20d"] = base["tech_signal_20d"].apply(clip)

    base["tech_signal_60d"] = (
        0.35 * ret_60d_back.apply(lambda x: safe_tanh(float(x), 0.20) if pd.notna(x) else np.nan).fillna(0.0)
        + 0.20 * sma20_slope_5.apply(lambda x: safe_tanh(float(x), 0.05) if pd.notna(x) else np.nan).fillna(0.0)
        + 0.20 * ema10_slope_5.apply(lambda x: safe_tanh(float(x), 0.05) if pd.notna(x) else np.nan).fillna(0.0)
        + 0.10 * price_vs_sma20.apply(lambda x: safe_tanh(float(x), 0.05) if pd.notna(x) else np.nan).fillna(0.0)
        - 0.15 * realized_vol_20.apply(lambda x: safe_tanh(float(x), 0.08) if pd.notna(x) else np.nan).fillna(0.0)
    )
    base["tech_signal_60d"] = base["tech_signal_60d"].apply(clip)

    base["tech_signal_90d"] = (
        0.45 * ret_90d_back.apply(lambda x: safe_tanh(float(x), 0.25) if pd.notna(x) else np.nan).fillna(0.0)
        + 0.20 * ret_60d_back.apply(lambda x: safe_tanh(float(x), 0.20) if pd.notna(x) else np.nan).fillna(0.0)
        + 0.15 * dist_from_20d_high.apply(lambda x: safe_tanh(float(x), 0.10) if pd.notna(x) else np.nan).fillna(0.0)
        - 0.20 * realized_vol_20.apply(lambda x: safe_tanh(float(x), 0.08) if pd.notna(x) else np.nan).fillna(0.0)
    )
    base["tech_signal_90d"] = base["tech_signal_90d"].apply(clip)

    if "vdce_leader_score" in base.columns:
        base["vdce_signal"] = (
            base["vdce_signal"]
            - pd.to_numeric(base["vdce_leader_score"], errors="coerce").apply(lambda x: safe_tanh(float(x), 0.10) if pd.notna(x) else np.nan).fillna(0.0) * 0.20
        ).apply(clip)

    horizon_weights = {
        20: {"stock": 0.32, "macro": 0.16, "sector": 0.22, "tech": 0.25, "vdce": 0.05},
        60: {"stock": 0.34, "macro": 0.24, "sector": 0.17, "tech": 0.20, "vdce": 0.05},
        90: {"stock": 0.30, "macro": 0.34, "sector": 0.16, "tech": 0.15, "vdce": 0.05},
    }

    for horizon in (20, 60, 90):
        tech_col = f"tech_signal_{horizon}d"
        weights = horizon_weights[horizon]
        score = (
            weights["stock"] * base["stock_regime_signal"].fillna(0.0)
            + weights["macro"] * base["macro_regime_score"].fillna(0.0)
            + weights["sector"] * base["sector_signal"].fillna(0.0)
            + weights["tech"] * base[tech_col].fillna(0.0)
            + weights["vdce"] * base["vdce_signal"].fillna(0.0)
        )
        score = score.apply(clip)
        base[f"regime_{horizon}d_score"] = score
        base[f"regime_{horizon}d_label"] = score.apply(score_to_label)

        coverage = (
            base[["stock_regime_signal", "macro_regime_score", "sector_signal", tech_col]]
            .notna()
            .mean(axis=1)
            .fillna(0.50)
        )
        confidence = (0.38 + 0.42 * score.abs() + 0.20 * coverage) * calibration_quality[horizon]
        base[f"regime_{horizon}d_confidence"] = confidence.clip(0.20, 0.99)

    for horizon in (20, 60, 90):
        base[f"regime_{horizon}d_commentary"] = base.apply(lambda row: build_commentary(row, horizon), axis=1)

    base["summary_date"] = pd.Timestamp.now(tz=timezone.utc).tz_convert(None)
    base = base.sort_values(["ticker"]).reset_index(drop=True)

    output_cols = [
        "ticker",
        "date",
        "gics_sector",
        "gics_industry",
        "close",
        "gmm_regime_label",
        "gmm_regime_confidence",
        "stock_regime_score",
        "macro_regime_label",
        "macro_regime_score",
        "macro_regime_confidence",
        "macro_regime_source",
        "sector_relative_strength_20d",
        "sector_risk_multiplier",
        "sector_avg_20d_vol_prob",
        "rsi_14",
        "price_vs_sma20",
        "ret_20d_back",
        "ret_60d_back",
        "ret_90d_back",
        "vdce_market_state",
        "vdce_signal",
        "vdce_leader_score",
        "regime_20d_score",
        "regime_20d_label",
        "regime_20d_confidence",
        "regime_20d_commentary",
        "regime_60d_score",
        "regime_60d_label",
        "regime_60d_confidence",
        "regime_60d_commentary",
        "regime_90d_score",
        "regime_90d_label",
        "regime_90d_confidence",
        "regime_90d_commentary",
        "summary_date",
    ]
    output_cols = [col for col in output_cols if col in base.columns]
    out_df = base[output_cols].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    out_df.to_csv(latest_csv_path, index=False)

    status = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "output_parquet": str(output_path),
        "latest_csv": str(latest_csv_path),
        "rows_written": int(len(out_df)),
        "macro_source": macro_context.source,
        "macro_date": str(macro_context.asof_date.date()) if pd.notna(macro_context.asof_date) else None,
        "vdce_present": bool(vdce_csv_path.exists()),
        "vdce_date": str(vdce_context.asof_date.date()) if pd.notna(vdce_context.asof_date) else None,
        "stock_regime_rows": int(len(out_df["gmm_regime_label"].dropna())) if "gmm_regime_label" in out_df.columns else 0,
        "tickers": int(out_df["ticker"].nunique()) if "ticker" in out_df.columns else 0,
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    print(f"Saved regime summary: {output_path}")
    print(f"Saved latest CSV: {latest_csv_path}")
    print(f"Saved status file: {status_path}")
    print(f"Rows: {len(out_df):,}")
    for horizon in (20, 60, 90):
        label_col = f"regime_{horizon}d_label"
        if label_col in out_df.columns:
            print(f"{horizon}d labels:", out_df[label_col].value_counts(dropna=False).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
