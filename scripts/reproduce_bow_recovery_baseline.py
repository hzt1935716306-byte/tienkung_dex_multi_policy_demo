#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the Phase 3A bow recovery failure")
    parser.add_argument("--commit", default="6f98d7e")
    parser.add_argument("--config", default="configs/bow_recovery_phase3a_baseline.json")
    parser.add_argument("--output-dir", default="artifacts/bow_recovery_baseline_reproduced")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    config = (project / args.config).resolve()
    output_dir = (project / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix="tienkung_phase3a_"))
    worktree = temporary_root / "repo"
    summary_path = output_dir / "repeat20_summary.json"
    log_path = output_dir / "repeat20.log"

    try:
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), args.commit],
            cwd=project,
            check=True,
        )
        command = [
            sys.executable,
            str(worktree / "run_multi_evt2.py"),
            "--config",
            str(config),
            "--scenario",
            "bow_repeat20",
            "--headless",
            "--no-realtime",
            "--debug-interval",
            "0",
            "--log",
            str(output_dir / "repeat20.csv"),
            "--summary",
            str(summary_path),
        ]
        with log_path.open("w", encoding="utf-8") as output:
            subprocess.run(
                command,
                cwd=worktree,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
    finally:
        if worktree.exists():
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=project,
                check=False,
            )
        shutil.rmtree(temporary_root, ignore_errors=True)

    if not summary_path.is_file():
        raise SystemExit(f"Baseline did not produce a summary; inspect {log_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    commands = summary.get("controller", {}).get("commands", [])
    failed_bows = sum(
        record.get("action_key") == "a" and record.get("status") == "FAILED"
        for record in commands
    )
    reproduced = not summary.get("passed", False) and failed_bows > 0
    print(
        "[BASELINE] "
        f"{'REPRODUCED' if reproduced else 'NOT REPRODUCED'} "
        f"steps={summary.get('executed_steps')}/{summary.get('expected_steps')} "
        f"minimum_root_z={summary.get('minimum_root_z')} failed_bows={failed_bows}"
    )
    print(f"[BASELINE] summary: {summary_path}")
    if not reproduced:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
