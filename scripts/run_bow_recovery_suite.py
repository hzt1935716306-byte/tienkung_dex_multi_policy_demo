#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


TRANSITION_DURATIONS = (0.2, 0.3, 0.4, 0.6)
REENTRY_PHASE_TIMES = (0.0, 0.2125, 0.425, 0.6375)
RECOVERY_SCENARIOS = (
    "bow",
    "bow_initial_forward",
    "bow_initial_backward",
    "bow_tail_perturbed",
    "bow_repeat20",
)


def _events(summary: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [
        event
        for event in summary.get("controller", {}).get("events", [])
        if event.get("event") == name and event.get("action_key") == "a"
    ]


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def recovery_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    transitions = _events(summary, "transition_out_started")
    acquisitions = _events(summary, "walkamp_control_acquired")
    completions = _events(summary, "recovery_completed")
    by_command: dict[int, dict[str, dict[str, Any]]] = {}
    for label, values in (
        ("transition", transitions),
        ("acquisition", acquisitions),
        ("completion", completions),
    ):
        for event in values:
            command_id = event.get("command_id")
            if command_id is not None:
                by_command.setdefault(int(command_id), {})[label] = event

    forward_displacements: list[float] = []
    planar_displacements: list[float] = []
    recovery_seconds: list[float] = []
    acquisition_pitch: list[float] = []
    acquisition_pitch_rate: list[float] = []
    acquisition_xy_speed: list[float] = []
    acquisition_joint_speed: list[float] = []
    acquisition_contact_losses = 0
    for values in by_command.values():
        transition = values.get("transition")
        acquisition = values.get("acquisition")
        completion = values.get("completion")
        if acquisition:
            metrics = acquisition["metrics"]
            acquisition_pitch.append(abs(float(metrics["pitch"])))
            acquisition_pitch_rate.append(abs(float(metrics["pitch_angular_velocity"])))
            acquisition_xy_speed.append(float(metrics["xy_speed"]))
            acquisition_joint_speed.append(float(metrics["maximum_joint_speed"]))
            if not (
                float(metrics["left_contact_count"]) > 0.0
                and float(metrics["right_contact_count"]) > 0.0
            ):
                acquisition_contact_losses += 1
        if transition and completion:
            start = transition["metrics"]
            end = completion["metrics"]
            dx = float(end["root_x"]) - float(start["root_x"])
            dy = float(end["root_y"]) - float(start["root_y"])
            yaw = float(start["yaw"])
            forward_displacements.append(dx * math.cos(yaw) + dy * math.sin(yaw))
            planar_displacements.append(math.hypot(dx, dy))
            recovery_seconds.append(float(completion["time"]) - float(transition["time"]))

    commands = summary.get("controller", {}).get("commands", [])
    bow_commands = [record for record in commands if record.get("action_key") == "a"]
    transition_metrics = summary.get("transition_out_metrics", {})
    return {
        "passed": bool(summary.get("passed", False)),
        "completed_bows": sum(record.get("status") == "COMPLETED" for record in bow_commands),
        "failed_bows": sum(record.get("status") == "FAILED" for record in bow_commands),
        "handoffs": len(acquisitions),
        "mean_recovery_seconds": _mean(recovery_seconds),
        "mean_forward_displacement": _mean(forward_displacements),
        "mean_planar_displacement": _mean(planar_displacements),
        "maximum_acquisition_abs_pitch": max(acquisition_pitch, default=None),
        "maximum_acquisition_abs_pitch_rate": max(acquisition_pitch_rate, default=None),
        "maximum_acquisition_xy_speed": max(acquisition_xy_speed, default=None),
        "maximum_acquisition_joint_speed": max(acquisition_joint_speed, default=None),
        "acquisition_contact_losses": acquisition_contact_losses,
        "minimum_root_z": summary.get("minimum_root_z"),
        "maximum_abs_torque": summary.get("maximum_abs_torque"),
        "maximum_torque_rate": summary.get("maximum_torque_rate"),
        "torque_saturation_rate": summary.get("torque_saturation_rate"),
        "transition_out_maximum_target_rate": transition_metrics.get(
            "maximum_target_rate"
        ),
        "transition_out_maximum_requested_target_delta": transition_metrics.get(
            "maximum_requested_target_delta"
        ),
        "transition_out_maximum_applied_target_delta": transition_metrics.get(
            "maximum_applied_target_delta"
        ),
        "transition_out_maximum_torque_rate": transition_metrics.get(
            "maximum_torque_rate"
        ),
        "transition_out_maximum_abs_torque": transition_metrics.get(
            "maximum_abs_torque"
        ),
        "transition_out_torque_saturation_rate": transition_metrics.get(
            "torque_saturation_rate"
        ),
        "terminal_failure_reason": summary.get("controller", {}).get(
            "terminal_failure_reason", ""
        ),
    }


def run_case(
    project: Path,
    config: Path,
    output_dir: Path,
    name: str,
    scenario: str,
    extra_args: list[str] | None = None,
) -> tuple[dict[str, Any], int]:
    summary_path = output_dir / f"{name}_summary.json"
    stdout_path = output_dir / f"{name}.log"
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
        *(extra_args or []),
    ]
    print(f"[BOW SUITE] running {name}", flush=True)
    with stdout_path.open("w", encoding="utf-8") as output:
        completed = subprocess.run(
            command,
            cwd=project,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if not summary_path.is_file():
        return {"passed": False, "error": "summary not produced", "log": str(stdout_path)}, completed.returncode
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary, completed.returncode


def _format(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Bow Recovery Reproducible Test Report",
        "",
        f"Overall result: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "## Transition duration sweep",
        "",
        "| duration (s) | pass | pitch rate at takeover | xy speed at takeover | "
        "joint speed at takeover | target delta requested/applied | transition torque rate |",
        "| ---: | :---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in report["duration_sweep"].items():
        lines.append(
            (
                "| {duration} | {passed} | {pitch_rate} | {xy_speed} | {joint_speed} | "
                "{requested}/{applied} | {torque_rate} |"
            ).format(
                duration=name.removeprefix("transition_"),
                passed="PASS" if values["passed"] else "FAIL",
                pitch_rate=_format(values["maximum_acquisition_abs_pitch_rate"]),
                xy_speed=_format(values["maximum_acquisition_xy_speed"]),
                joint_speed=_format(values["maximum_acquisition_joint_speed"]),
                requested=_format(values["transition_out_maximum_requested_target_delta"]),
                applied=_format(values["transition_out_maximum_applied_target_delta"]),
                torque_rate=_format(values["transition_out_maximum_torque_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "## WALKAMP re-entry phase sweep",
            "",
            "| phase time (s) | pass | pitch rate at takeover | xy speed at takeover | "
            "mean forward displacement (m) | contact losses at takeover |",
            "| ---: | :---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, values in report["phase_sweep"].items():
        lines.append(
            "| {phase} | {passed} | {pitch_rate} | {xy_speed} | {forward} | {contact} |".format(
                phase=name.removeprefix("phase_"),
                passed="PASS" if values["passed"] else "FAIL",
                pitch_rate=_format(values["maximum_acquisition_abs_pitch_rate"]),
                xy_speed=_format(values["maximum_acquisition_xy_speed"]),
                forward=_format(values["mean_forward_displacement"]),
                contact=values["acquisition_contact_losses"],
            )
        )
    lines.extend(
        [
            "",
            "## Recovery scenarios",
            "",
            "| scenario | pass | completed bows | failed bows | mean recovery (s) | "
            "mean forward displacement (m) | contact losses at takeover |",
            "| --- | :---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, values in report["recovery_scenarios"].items():
        lines.append(
            "| {name} | {passed} | {completed} | {failed} | {recovery} | {forward} | {contact} |".format(
                name=name,
                passed="PASS" if values["passed"] else "FAIL",
                completed=values["completed_bows"],
                failed=values["failed_bows"],
                recovery=_format(values["mean_recovery_seconds"]),
                forward=_format(values["mean_forward_displacement"]),
                contact=values["acquisition_contact_losses"],
            )
        )
    if report.get("baseline"):
        lines.extend(
            [
                "",
                "## Supplied Phase 3A baseline",
                "",
                "```json",
                json.dumps(report["baseline"], indent=2, ensure_ascii=True),
                "```",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3A.1 bow recovery tests")
    parser.add_argument("--config", default="configs/multi_evt2.json")
    parser.add_argument("--output-dir", default="artifacts/bow_recovery_suite")
    parser.add_argument("--baseline-summary", default="")
    parser.add_argument("--skip-repeat20", action="store_true")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    config = (project / args.config).resolve()
    output_dir = (project / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "config": str(config),
        "duration_sweep": {},
        "phase_sweep": {},
        "recovery_scenarios": {},
    }
    failed = False

    for duration in TRANSITION_DURATIONS:
        name = f"transition_{duration:.1f}"
        summary, returncode = run_case(
            project,
            config,
            output_dir,
            name,
            "bow",
            ["--transition-out", str(duration)],
        )
        report["duration_sweep"][name] = recovery_metrics(summary)
        failed |= returncode != 0 or not summary.get("passed", False)

    for phase_time in REENTRY_PHASE_TIMES:
        name = f"phase_{phase_time:g}"
        summary, returncode = run_case(
            project,
            config,
            output_dir,
            name,
            "bow",
            ["--reentry-phase", str(phase_time)],
        )
        report["phase_sweep"][name] = recovery_metrics(summary)
        failed |= returncode != 0 or not summary.get("passed", False)

    scenarios = RECOVERY_SCENARIOS[:-1] if args.skip_repeat20 else RECOVERY_SCENARIOS
    for scenario in scenarios:
        summary, returncode = run_case(
            project,
            config,
            output_dir,
            scenario,
            scenario,
        )
        report["recovery_scenarios"][scenario] = recovery_metrics(summary)
        failed |= returncode != 0 or not summary.get("passed", False)

    if args.baseline_summary:
        baseline_path = Path(args.baseline_summary).expanduser().resolve()
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        report["baseline"] = {
            "source": str(baseline_path),
            "passed": baseline.get("passed"),
            "executed_steps": baseline.get("executed_steps"),
            "expected_steps": baseline.get("expected_steps"),
            "minimum_root_z": baseline.get("minimum_root_z"),
            "maximum_abs_roll_pitch": baseline.get("maximum_abs_roll_pitch"),
            "maximum_torque_rate": baseline.get("maximum_torque_rate"),
            "completed_bows": sum(
                record.get("action_key") == "a" and record.get("status") == "COMPLETED"
                for record in baseline.get("controller", {}).get("commands", [])
            ),
            "failed_bows": sum(
                record.get("action_key") == "a" and record.get("status") == "FAILED"
                for record in baseline.get("controller", {}).get("commands", [])
            ),
        }

    report["passed"] = not failed
    json_path = output_dir / "suite_summary.json"
    markdown_path = output_dir / "report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    write_markdown(markdown_path, report)
    print(f"[BOW SUITE] {'PASS' if not failed else 'FAIL'}: {markdown_path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
