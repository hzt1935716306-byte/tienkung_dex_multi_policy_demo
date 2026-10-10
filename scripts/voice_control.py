#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tienkung_demo.command_bus import SshCommandWriter, SshStatusReader
from tienkung_demo.voice import run_text_demo, run_voice_demo


def load_voice_config(path: str) -> dict:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not config.get("command_file") or not config.get("status_file"):
        raise ValueError(
            "Voice control requires command_file and status_file; use configs/multi_evt2.json"
        )
    if set(config.get("motions", {})) != {"a", "b"}:
        raise ValueError("Voice whitelist must contain exactly motion keys a and b")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice/text action controller for the TienKung Dex demo")
    parser.add_argument("--config", default=str(ROOT / "configs" / "multi_evt2.json"))
    parser.add_argument("--text", action="store_true", help="Use terminal text instead of DashScope and microphone")
    parser.add_argument("--mic", type=int, default=None, help="PyAudio input device index")
    parser.add_argument("--ssh-target", default="", help="Forward actions to USER@HOST over SSH")
    parser.add_argument("--remote-project", default="", help="Absolute project path on the simulator server")
    parser.add_argument("--remote-python", default="python3", help="Python executable on the simulator server")
    parser.add_argument(
        "--remote-config",
        default="configs/multi_evt2.json",
        help="Config path on the simulator server",
    )
    args = parser.parse_args()
    config = load_voice_config(args.config)
    writer = None
    status_reader = None
    if args.ssh_target or args.remote_project:
        if not args.ssh_target or not args.remote_project:
            parser.error("--ssh-target and --remote-project must be used together")
        writer = SshCommandWriter(
            target=args.ssh_target,
            remote_project=args.remote_project,
            remote_python=args.remote_python,
            remote_config=args.remote_config,
        )
        status_reader = SshStatusReader(
            target=args.ssh_target,
            remote_project=args.remote_project,
            remote_python=args.remote_python,
            remote_config=args.remote_config,
        )
    if args.text:
        run_text_demo(config, writer, status_reader)
    else:
        run_voice_demo(config, args.mic, writer, status_reader)


if __name__ == "__main__":
    main()
