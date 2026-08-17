#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tienkung_demo.config import load_config
from tienkung_demo.command_bus import SshCommandWriter
from tienkung_demo.voice import run_text_demo, run_voice_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice/text action controller for the TienKung Dex demo")
    parser.add_argument("--config", default=str(ROOT / "configs" / "demo.json"))
    parser.add_argument("--text", action="store_true", help="Use terminal text instead of DashScope and microphone")
    parser.add_argument("--mic", type=int, default=None, help="PyAudio input device index")
    parser.add_argument("--ssh-target", default="", help="Forward actions to USER@HOST over SSH")
    parser.add_argument("--remote-project", default="", help="Absolute project path on the simulator server")
    parser.add_argument("--remote-python", default="python3", help="Python executable on the simulator server")
    parser.add_argument("--remote-config", default="configs/demo.json", help="Config path on the simulator server")
    args = parser.parse_args()
    config = load_config(args.config)
    writer = None
    if args.ssh_target or args.remote_project:
        if not args.ssh_target or not args.remote_project:
            parser.error("--ssh-target and --remote-project must be used together")
        writer = SshCommandWriter(
            target=args.ssh_target,
            remote_project=args.remote_project,
            remote_python=args.remote_python,
            remote_config=args.remote_config,
        )
    if args.text:
        run_text_demo(config, writer)
    else:
        run_voice_demo(config, args.mic, writer)


if __name__ == "__main__":
    main()
