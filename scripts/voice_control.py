#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tienkung_demo.config import load_config
from tienkung_demo.voice import run_text_demo, run_voice_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice/text action controller for the TienKung Dex demo")
    parser.add_argument("--config", default=str(ROOT / "configs" / "demo.json"))
    parser.add_argument("--text", action="store_true", help="Use terminal text instead of DashScope and microphone")
    parser.add_argument("--mic", type=int, default=None, help="PyAudio input device index")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.text:
        run_text_demo(config)
    else:
        run_voice_demo(config, args.mic)


if __name__ == "__main__":
    main()
