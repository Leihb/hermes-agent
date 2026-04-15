#!/usr/bin/env python3
"""
Backfill session message embeddings for semantic search.

Usage:
    python3 scripts/backfill_session_embeddings.py [--limit N]
"""
import argparse
import sys
from pathlib import Path

# Ensure hermes-agent root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_state import SessionDB


def main():
    parser = argparse.ArgumentParser(
        description="Backfill embeddings for historical session messages"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of messages to process (default: all)",
    )
    args = parser.parse_args()

    db = SessionDB()
    stats = db.backfill_message_embeddings(limit=args.limit)
    print(f"Backfill complete: {stats}")


if __name__ == "__main__":
    main()
