from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from reference.config import DEFAULT_PATHS
from reference.utils import configure_logging, ensure_dir, utc_now_iso, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Quintic V3 reference pipeline end to end.")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--raw-input", default=str(DEFAULT_PATHS.raw_prices_dir))
    parser.add_argument("--reference-dir", default=str(DEFAULT_PATHS.reference_dir))
    parser.add_argument("--research-dir", default=str(DEFAULT_PATHS.research_dir))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def run_step(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    root = Path(__file__).resolve().parent
    reference_dir = Path(args.reference_dir)
    ensure_dir(reference_dir)

    log_only = ["--log-level", args.log_level]
    with_as_of = ["--log-level", args.log_level]
    if args.as_of:
        with_as_of += ["--as-of", args.as_of]

    run_step([
        args.python_exe,
        str(root / "build_research_prices.py"),
        "--raw-input", args.raw_input,
        "--output-dir", args.research_dir,
        *with_as_of,
    ])

    reference_cmd = [
        args.python_exe,
        str(root / "build_reference_master.py"),
        "--research-prices", str(Path(args.research_dir) / "research_prices_all.parquet"),
        "--output-dir", args.reference_dir,
        "--sleep-seconds", str(args.sleep_seconds),
        "--timeout-seconds", str(args.timeout_seconds),
        *with_as_of,
    ]
    if args.api_key:
        reference_cmd += ["--api-key", args.api_key]
    run_step(reference_cmd)

    run_step([
        args.python_exe,
        str(root / "validate_reference_integrity.py"),
        "--research-prices", str(Path(args.research_dir) / "research_prices_all.parquet"),
        "--reference-master", str(Path(args.reference_dir) / "ticker_reference_master.parquet"),
        "--output-dir", args.reference_dir,
        *with_as_of,
    ])

    run_step([
        args.python_exe,
        str(root / "build_research_universe.py"),
        "--research-prices", str(Path(args.research_dir) / "research_prices_all.parquet"),
        "--reference-master", str(Path(args.reference_dir) / "ticker_reference_master.parquet"),
        "--output-dir", args.research_dir,
        *log_only,
    ])

    write_json(reference_dir / "pipeline_run_summary.json", {
        "run_timestamp_utc": utc_now_iso(),
        "raw_input": args.raw_input,
        "research_dir": args.research_dir,
        "reference_dir": args.reference_dir,
        "as_of": args.as_of,
        "status": "ok",
    })


if __name__ == "__main__":
    main()
