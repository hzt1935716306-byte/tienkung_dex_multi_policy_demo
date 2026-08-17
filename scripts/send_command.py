#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tienkung_demo.command_bus import CommandWriter
from tienkung_demo.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one command to the running simulator")
    parser.add_argument("command", help="a, b, r, q, 8, 2, 4, 6, 7, 9, or 5")
    parser.add_argument("--config", default=str(ROOT / "configs" / "demo.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    writer = CommandWriter(config["command_file"], source="manual")
    writer.send(args.command, "manual")
    print(f"[INFO] sent {args.command.lower()!r} to {writer.path}")


if __name__ == "__main__":
    main()
