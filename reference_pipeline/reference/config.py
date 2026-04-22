from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelinePaths:
    project_root: Path = Path(r"Z:\JVAND\QUINTIC_V3")
    raw_prices_dir: Path = Path(r"Z:\JVAND\QUINTIC_V3\DATA\raw\prices")
    reference_dir: Path = Path(r"Z:\JVAND\QUINTIC_V3\DATA\reference")
    research_dir: Path = Path(r"Z:\JVAND\QUINTIC_V3\DATA\research")
    logs_dir: Path = Path(r"Z:\JVAND\QUINTIC_V3\logs")

    @property
    def pipeline_runs_dir(self) -> Path:
        return self.reference_dir / "pipeline_runs"


@dataclass(frozen=True)
class UniverseThresholds:
    min_price: float = 10.0
    min_avg_volume: float = 300_000.0
    min_market_cap: float = 2_000_000_000.0
    prefilter_lookback_days: int = 22


@dataclass(frozen=True)
class RuntimeConfig:
    polygon_api_key: str | None
    as_of: str | None
    include_inactive: bool
    sleep_seconds_between_calls: float
    request_timeout_seconds: float
    user_agent: str = "QuinticV3-ReferencePipeline/4.0"


DEFAULT_PATHS = PipelinePaths()
DEFAULT_THRESHOLDS = UniverseThresholds()
