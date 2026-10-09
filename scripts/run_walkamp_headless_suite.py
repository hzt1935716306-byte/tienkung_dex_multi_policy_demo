#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible WALKAMP headless scenarios")
    parser.add_argument("--config", default="configs/walkamp_official.json")
    parser.add_argument("--output-dir", default="artifacts/walkamp_headless_suite")
    parser.add_argument("--scenarios", nargs="+", default=["stand", "low_forward", "stop", "turn"])
    parser.add_argument("--debug-interval", type=int, default=100)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    output_dir = (project / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    failed = False
    for scenario in args.scenarios:
        summary_path = output_dir / f"{scenario}_summary.json"
        log_path = output_dir / f"{scenario}.csv"
        command = [
            sys.executable,
            str(project / "run_walkamp.py"),
            "--config",
            str((project / args.config).resolve()),
            "--scenario",
            scenario,
            "--headless",
            "--no-realtime",
            "--debug-interval",
            str(args.debug_interval),
            "--log",
            str(log_path),
            "--summary",
            str(summary_path),
        ]
        print(f"[SUITE] running {scenario}", flush=True)
        completed = subprocess.run(command, cwd=project, check=False)
        if summary_path.is_file():
            results[scenario] = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            results[scenario] = {"passed": False, "error": "summary not produced"}
        if completed.returncode != 0 or not results[scenario].get("passed", False):
            failed = True

    suite_summary = {
        "passed": not failed,
        "scenarios": results,
    }
    suite_path = output_dir / "suite_summary.json"
    suite_path.write_text(json.dumps(suite_summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"[SUITE] {'PASS' if not failed else 'FAIL'}: {suite_path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
