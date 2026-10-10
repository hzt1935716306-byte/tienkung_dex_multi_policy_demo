#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SHORT_SCENARIOS = (
    "moving_forward_bow",
    "moving_forward_wave",
    "moving_backward_bow",
    "moving_lateral_wave",
    "moving_turn_bow",
    "moving_combined_wave",
    "zero_command_moving_bow",
    "brake_second_command",
    "brake_cancel",
)


def summarize(summary: dict[str, Any]) -> dict[str, Any]:
    events = summary["controller"]["events"]
    ready = [item for item in events if item["event"] == "ready_for_motion"]
    recovery = [item for item in events if item["event"] == "recovery_completed"]
    statuses = [item["status"] for item in summary["controller"]["commands"]]
    stop_times = [float(item.get("request_to_ready_seconds", 0.0)) for item in ready]
    stop_distances = [float(item.get("request_to_ready_distance", 0.0)) for item in ready]
    return {
        "passed": bool(summary.get("passed")),
        "commands": len(statuses),
        "statuses": statuses,
        "completed_actions": statuses.count("COMPLETED"),
        "failed_actions": statuses.count("FAILED"),
        "rejected_actions": statuses.count("REJECTED"),
        "cancelled_actions": statuses.count("CANCELLED"),
        "mean_stop_time_seconds": sum(stop_times) / max(len(stop_times), 1),
        "maximum_stop_time_seconds": max(stop_times, default=0.0),
        "mean_stop_distance_m": sum(stop_distances) / max(len(stop_distances), 1),
        "maximum_stop_distance_m": max(stop_distances, default=0.0),
        "minimum_root_height_m": float(summary["minimum_root_z"]),
        "maximum_abs_roll_pitch_rad": float(summary["maximum_abs_roll_pitch"]),
        "maximum_com_speed_mps": float(
            summary["stand_diagnostics"]["maximum_com_xy_speed"]
        ),
        "complete_steps": int(summary["stand_diagnostics"]["complete_step_count"]),
        "recoveries": len(recovery),
        "maximum_requested_target_delta_rad": float(
            summary["maximum_requested_target_delta"]
        ),
        "maximum_applied_target_delta_rad": float(
            summary["maximum_applied_target_delta"]
        ),
        "maximum_torque_rate_nmps": float(summary["maximum_torque_rate"]),
        "maximum_abs_torque_nm": float(summary["maximum_abs_torque"]),
        "torque_saturation_rate": float(summary["torque_saturation_rate"]),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Phase 3B.1 Walk-Command Suite",
        "",
        f"Overall: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "| Scenario | Result | Stop time [s] | Stop distance [m] | Min root [m] | "
        "Max attitude [rad] | Steps | Statuses |",
        "| --- | :---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, values in report["scenarios"].items():
        lines.append(
            f"| {name} | {'PASS' if values['passed'] else 'FAIL'} | "
            f"{values['maximum_stop_time_seconds']:.3f} | "
            f"{values['maximum_stop_distance_m']:.3f} | "
            f"{values['minimum_root_height_m']:.3f} | "
            f"{values['maximum_abs_roll_pitch_rad']:.3f} | "
            f"{values['complete_steps']} | {', '.join(values['statuses'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3B.1 walking action-trigger tests")
    parser.add_argument("--config", default="configs/multi_evt2.json")
    parser.add_argument("--output-dir", default="artifacts/walk_command_phase3b")
    parser.add_argument("--quick", action="store_true", help="Skip the 20-cycle no-reset scenario")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    config = (project / args.config).resolve()
    output_dir = (project / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = list(SHORT_SCENARIOS)
    if not args.quick:
        scenarios.append("moving_repeat20")

    report: dict[str, Any] = {"passed": True, "scenarios": {}}
    for scenario in scenarios:
        summary_path = output_dir / f"{scenario}_summary.json"
        log_path = output_dir / f"{scenario}.log"
        command = [
            sys.executable,
            str(project / "run_multi_evt2.py"),
            "--config",
            str(config),
            "--scenario",
            scenario,
            "--headless",
            "--no-realtime",
            "--no-log",
            "--debug-interval",
            "0",
            "--summary",
            str(summary_path),
        ]
        print(f"[WALK COMMAND SUITE] running {scenario}", flush=True)
        with log_path.open("w", encoding="utf-8") as output:
            completed = subprocess.run(
                command,
                cwd=project,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            values = summarize(summary)
        else:
            values = {"passed": False, "error": "summary not produced"}
        values["returncode"] = completed.returncode
        values["log"] = str(log_path)
        report["scenarios"][scenario] = values
        if completed.returncode != 0 or not values.get("passed", False):
            report["passed"] = False

    report_path = output_dir / "results.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    markdown_path = output_dir / "report.md"
    write_markdown(markdown_path, report)
    print(f"[WALK COMMAND SUITE] {'PASS' if report['passed'] else 'FAIL'}: {markdown_path}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
