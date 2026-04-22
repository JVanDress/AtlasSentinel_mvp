from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    env_root = os.getenv("QUINTIC_PROJECT_ROOT")
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.append(Path(r"C:\QUINTIC_V3"))
    candidates.append(Path(__file__).resolve().parents[1])

    for path in candidates:
        if (path / "data").exists() or (path / "scripts").exists():
            return path

    return candidates[0]


def data_dir(root: Path | None = None) -> Path:
    load_simple_dotenv(root, override=False)
    env_data_root = os.getenv("DATA_ROOT", "").strip()
    if env_data_root:
        return Path(env_data_root) / "data" / "stage"
    return (root or project_root()) / "data"


def logs_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "logs"


def cleanroom_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "cleanroom"


def resolve_under_data(name: str, root: Path | None = None) -> Path:
    return data_dir(root) / name


def env_file_candidates(root: Path | None = None) -> list[Path]:
    root = root or project_root()
    return [
        root / ".env",
        root / "data" / ".env",
        root / "scripts" / ".env",
        root / "scripts" / "reference_pipeline" / ".env",
    ]


def load_simple_dotenv(root: Path | None = None, *, override: bool = False) -> list[Path]:
    loaded: list[Path] = []
    for path in env_file_candidates(root):
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and (override or key not in os.environ):
                os.environ[key] = value
        loaded.append(path)
    return loaded
