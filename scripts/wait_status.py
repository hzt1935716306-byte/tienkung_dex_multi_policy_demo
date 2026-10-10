#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tienkung_demo.command_bus import StatusReader, TERMINAL_STATUSES


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for one multi-policy command result")
    parser.add_argument("--config", default=str(ROOT / "configs" / "multi_evt2.json"))
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--poll", type=float, default=0.05)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    status_path = config.get("status_file")
    if not status_path:
        raise ValueError(f"Config does not define status_file: {config_path}")

    result = StatusReader(status_path, from_start=True).wait_for_terminal(
        args.request_id,
        timeout=args.timeout,
        poll_seconds=args.poll,
    )
    if result is None:
        print(json.dumps({"request_id": args.request_id, "status": "TIMEOUT"}))
        raise SystemExit(2)
    print(json.dumps(result, ensure_ascii=False))
    if result.get("status") not in TERMINAL_STATUSES:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
