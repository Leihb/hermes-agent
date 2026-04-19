#!/usr/bin/env python3
"""Standalone CLI entrypoint for the Hermes meditation pipeline.

Usage:
    python scripts/run_meditation.py [--lookback-hours 24]

This can be wired into a cron job via:
    hermes cronjob create --schedule="0 2 * * *" --prompt "Run meditation"
    # OR directly:
    python /Users/qiao/.hermes/hermes-agent/scripts/run_meditation.py
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure imports resolve when running from scripts/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_cli.env_loader import load_hermes_dotenv
from hermes_state import SessionDB
from meditation.runner import MeditationPipelineRunner

HERMES_HOME = Path.home() / ".hermes"
load_hermes_dotenv(hermes_home=HERMES_HOME)


def main():
    parser = argparse.ArgumentParser(description="Hermes Meditation Pipeline")
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=24,
        help="How many hours of recent sessions to analyze (default: 24)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the LLM model used for bucket/consolidation analysis",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Override the LLM provider",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    db = SessionDB()
    runner = MeditationPipelineRunner(
        db=db,
        lookback_hours=args.lookback_hours,
        model=args.model,
        provider=args.provider,
    )
    result = runner.run()

    print(f"Meditation complete: {result['status']}")
    print(f"  Episodes: {result['episodes']}")
    print(f"  Buckets:  {result['buckets']}")
    print(f"  Rules:    {result['rules']}")
    if result.get("output_dir"):
        print(f"  Output:   {result['output_dir']}")

    return 0 if result["status"] in ("ok", "silent") else 1


if __name__ == "__main__":
    sys.exit(main())
