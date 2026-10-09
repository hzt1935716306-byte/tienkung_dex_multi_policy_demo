#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tienkung_demo.command_bus import CommandWriter


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one command to the running simulator")
    parser.add_argument("command", help="a, b, r, q, 8, 2, 4, 6, 7, 9, or 5")
    parser.add_argument("--config", default=str(ROOT / "configs" / "demo.json"))
    parser.add_argument("--source", default="manual", help="Command source recorded in the JSONL bus")
    parser.add_argument("--label", default="manual", help="Optional human-readable command label")
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not config.get("command_file"):
        raise ValueError(f"Config does not define command_file: {config_path}")
    writer = CommandWriter(config["command_file"], source=args.source)
    writer.send(args.command, args.label)
    print(f"[INFO] sent {args.command.lower()!r} to {writer.path}")


if __name__ == "__main__":
    main()
