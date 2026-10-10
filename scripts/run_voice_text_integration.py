#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tienkung_demo.command_bus import StatusReader
from tienkung_demo.voice import ActionManager, infer_text_actions


CASES = {
    "bow": "请鞠个躬",
    "wave": "向我挥挥手",
}


def absolute_config(
    source: Path,
    destination: Path,
    command_file: Path,
    status_file: Path,
) -> dict[str, Any]:
    config = json.loads(source.read_text(encoding="utf-8"))
    for key in ("model", "walkamp_config"):
        path = Path(config[key]).expanduser()
        config[key] = str((source.parent / path).resolve() if not path.is_absolute() else path)
    for motion in config["motions"].values():
        path = Path(motion["path"]).expanduser()
        motion["path"] = str((source.parent / path).resolve() if not path.is_absolute() else path)
    config["command_file"] = str(command_file)
    config["status_file"] = str(status_file)
    config["scenarios"]["external_forward"]["duration_seconds"] = 18.0
    destination.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return config


def wait_for_startup(log_path: Path, process: subprocess.Popen, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"simulator exited during startup; see {log_path}")
        if log_path.is_file() and "default controller: WALKAMP" in log_path.read_text(
            encoding="utf-8", errors="replace"
        ):
            return
        time.sleep(0.05)
    raise TimeoutError(f"simulator startup timed out; see {log_path}")


def run_case(source: Path, output_dir: Path, action: str, phrase: str) -> dict[str, Any]:
    case_dir = output_dir / action
    case_dir.mkdir(parents=True, exist_ok=True)
    command_file = case_dir / "commands.jsonl"
    status_file = case_dir / "status.jsonl"
    config_path = case_dir / "config.json"
    summary_path = case_dir / "summary.json"
    log_path = case_dir / "simulator.log"
    for path in (command_file, status_file, summary_path, log_path):
        path.unlink(missing_ok=True)
    config = absolute_config(source, config_path, command_file, status_file)

    command = [
        sys.executable,
        "-u",
        str(ROOT / "run_multi_evt2.py"),
        "--config",
        str(config_path),
        "--scenario",
        "external_forward",
        "--headless",
        "--headless-realtime",
        "--no-log",
        "--debug-interval",
        "0",
        "--summary",
        str(summary_path),
    ]
    with log_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        manager: ActionManager | None = None
        try:
            wait_for_startup(log_path, process)
            time.sleep(3.0)
            recognized_at = time.time()
            matched = infer_text_actions(phrase, config)
            if matched != ["a" if action == "bow" else "b"]:
                raise RuntimeError(f"unexpected whitelist result for {phrase!r}: {matched}")
            manager = ActionManager(config)
            request_id = manager.trigger(matched[0])
            if request_id is None:
                raise RuntimeError("ActionManager rejected the integration request")
            terminal = manager.wait(request_id, timeout=30.0)
            if terminal is None:
                raise TimeoutError(f"no terminal status for {request_id}")
            returncode = process.wait(timeout=25.0)
        finally:
            if manager is not None:
                manager.stop()
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5.0)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    statuses = [
        item
        for item in StatusReader(status_file, from_start=True).read()
        if item.get("request_id") == request_id
    ]
    by_status = {item["status"]: item for item in statuses}
    events = summary["controller"]["events"]
    event_by_name = {item["event"]: item for item in events}
    return {
        "passed": (
            returncode == 0
            and summary.get("passed", False)
            and terminal.get("status") == "COMPLETED"
        ),
        "phrase": phrase,
        "action": action,
        "request_id": request_id,
        "recognized_at": recognized_at,
        "issued_at": by_status.get("RECEIVED", {}).get("request_issued_at"),
        "controller_received_at": by_status.get("RECEIVED", {}).get("controller_received_at"),
        "brake_started_sim_time": event_by_name.get("brake_started", {}).get("time"),
        "ready_sim_time": event_by_name.get("ready_for_motion", {}).get("time"),
        "motion_started_sim_time": next(
            (item["time"] for item in events if item["event"] == "entry_transition_completed"),
            None,
        ),
        "completed_at": by_status.get("COMPLETED", {}).get("time"),
        "status_sequence": [item["status"] for item in statuses],
        "terminal": terminal,
        "simulator_passed": summary.get("passed", False),
        "simulator_returncode": returncode,
        "summary": str(summary_path),
        "log": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run text-to-action Phase 3B integration tests")
    parser.add_argument("--config", default="configs/multi_evt2.json")
    parser.add_argument("--output-dir", default="artifacts/voice_text_phase3b")
    args = parser.parse_args()
    source = (ROOT / args.config).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {"passed": True, "cases": {}}
    for action, phrase in CASES.items():
        print(f"[TEXT INTEGRATION] running {action}: {phrase}", flush=True)
        result = run_case(source, output_dir, action, phrase)
        report["cases"][action] = result
        report["passed"] = report["passed"] and result["passed"]
    report_path = output_dir / "results.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[TEXT INTEGRATION] {'PASS' if report['passed'] else 'FAIL'}: {report_path}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
