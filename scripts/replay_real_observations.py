#!/usr/bin/env python3
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tienkung_demo.real_runner import run


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--offline-observations" not in arguments:
        raise SystemExit("Usage: replay_real_observations.py --offline-observations FILE [run_real options]")
    run(arguments)
