#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


TYPICAL_JOINTS = (
    "hip_pitch_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "waist_pitch_joint",
    "shoulder_pitch_r_joint",
    "elbow_pitch_r_joint",
)


def _events(summary: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [
        event
        for event in summary.get("controller", {}).get("events", [])
        if event.get("event") == name
    ]


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def summarize_exit(summary: dict[str, Any]) -> dict[str, Any]:
    transitions = _events(summary, "transition_out_started")
    interpolation_starts = _events(summary, "walkamp_interpolation_started")
    acquisitions = _events(summary, "walkamp_control_acquired")
    completions = _events(summary, "recovery_completed")
    by_command: dict[int, dict[str, dict[str, Any]]] = {}
    for label, events in (
        ("transition", transitions),
        ("interpolation", interpolation_starts),
        ("acquisition", acquisitions),
        ("completion", completions),
    ):
        for event in events:
            command_id = event.get("command_id")
            if command_id is not None:
                by_command.setdefault(int(command_id), {})[label] = event

    wait_times: list[float] = []
    interpolation_times: list[float] = []
    recovery_times: list[float] = []
    total_times: list[float] = []
    forward_displacements: list[float] = []
    planar_displacements: list[float] = []
    acquisition_pitch: list[float] = []
    acquisition_pitch_rate: list[float] = []
    acquisition_xy_speed: list[float] = []
    acquisition_joint_speed: list[float] = []
    acquisition_single_support = 0
    for events in by_command.values():
        transition = events.get("transition")
        interpolation = events.get("interpolation")
        acquisition = events.get("acquisition")
        completion = events.get("completion")
        if transition and interpolation:
            wait_times.append(
                float(interpolation["time"]) - float(transition["time"])
            )
        if interpolation and acquisition:
            interpolation_times.append(
                float(acquisition["time"]) - float(interpolation["time"])
            )
        if acquisition:
            metrics = acquisition["metrics"]
            acquisition_pitch.append(abs(float(metrics["pitch"])))
            acquisition_pitch_rate.append(
                abs(float(metrics["pitch_angular_velocity"]))
            )
            acquisition_xy_speed.append(float(metrics["xy_speed"]))
            acquisition_joint_speed.append(float(metrics["maximum_joint_speed"]))
            contacts = (
                float(metrics["left_contact_count"]) > 0.0,
                float(metrics["right_contact_count"]) > 0.0,
            )
            acquisition_single_support += int(sum(contacts) < 2)
        if acquisition and completion:
            recovery_times.append(
                float(completion["time"]) - float(acquisition["time"])
            )
        if transition and completion:
            total_times.append(
                float(completion["time"]) - float(transition["time"])
            )
            start = transition["metrics"]
            end = completion["metrics"]
            dx = float(end["root_x"]) - float(start["root_x"])
            dy = float(end["root_y"]) - float(start["root_y"])
            yaw = float(start["yaw"])
            forward_displacements.append(dx * math.cos(yaw) + dy * math.sin(yaw))
            planar_displacements.append(math.hypot(dx, dy))

    commands = [
        command
        for command in summary.get("controller", {}).get("commands", [])
        if command.get("action_key") in ("a", "b")
    ]
    completed = sum(command.get("status") == "COMPLETED" for command in commands)
    failed = sum(command.get("status") in ("FAILED", "REJECTED") for command in commands)
    exit_metrics = summary.get("exit_metrics", {})
    transition_metrics = exit_metrics.get("TRANSITION_OUT", {})
    recovery_metrics = exit_metrics.get("RECOVER", {})
    boundary = exit_metrics.get("boundary", {})
    selected_previews = [
        event.get("selected_preview", {})
        for event in interpolation_starts
        if event.get("selected_preview")
    ]
    history_modes = sorted(
        {
            str(event.get("history_mode"))
            for event in interpolation_starts
            if event.get("history_mode")
        }
    )
    phase_times = [
        float(preview["phase_time"])
        for preview in selected_previews
        if "phase_time" in preview
    ]

    def boundary_max(field: str) -> float | None:
        values = [
            float(group[field])
            for name, group in boundary.items()
            if name in ("legs", "waist", "arms") and field in group
        ]
        return max(values, default=None)

    def group_max(stage: dict[str, Any], field: str) -> float | None:
        values = [
            float(group.get(field, 0.0))
            for group in stage.get("groups", {}).values()
        ]
        return max(values, default=None)

    return {
        "passed": bool(summary.get("passed", False)),
        "scenario": summary.get("scenario"),
        "exit_mode": summary.get("controller", {}).get("exit_mode"),
        "history_mode": (
            history_modes[0]
            if len(history_modes) == 1
            else history_modes
        ),
        "selected_phase_times": phase_times,
        "commands": len(commands),
        "completed": completed,
        "failed_or_rejected": failed,
        "success_rate": completed / len(commands) if commands else 1.0,
        "mean_wait_seconds": _mean(wait_times),
        "mean_interpolation_seconds": _mean(interpolation_times),
        "mean_recovery_seconds": _mean(recovery_times),
        "mean_total_exit_seconds": _mean(total_times),
        "mean_forward_displacement": _mean(forward_displacements),
        "mean_planar_displacement": _mean(planar_displacements),
        "maximum_acquisition_abs_pitch": max(acquisition_pitch, default=None),
        "maximum_acquisition_abs_pitch_rate": max(
            acquisition_pitch_rate,
            default=None,
        ),
        "maximum_acquisition_xy_speed": max(acquisition_xy_speed, default=None),
        "maximum_acquisition_joint_speed": max(
            acquisition_joint_speed,
            default=None,
        ),
        "acquisition_single_support": acquisition_single_support,
        "boundary_raw_target_jump": boundary_max("raw_target_delta"),
        "boundary_applied_target_jump": boundary_max("applied_target_delta"),
        "boundary_kp_jump": boundary_max("kp_delta"),
        "boundary_kd_jump": boundary_max("kd_delta"),
        "boundary_feedforward_jump": boundary_max("feedforward_delta"),
        "boundary_effort_jump": boundary_max("effort_delta"),
        "boundary_torque_scale_jump": boundary.get("scalar", {}).get(
            "torque_scale_delta"
        ),
        "boundary_torque_jump": boundary.get("scalar", {}).get(
            "maximum_torque_delta"
        ),
        "transition_maximum_joint_speed": transition_metrics.get(
            "maximum_joint_speed"
        ),
        "transition_maximum_joint_acceleration": transition_metrics.get(
            "maximum_joint_acceleration"
        ),
        "transition_maximum_tracking_error": group_max(
            transition_metrics,
            "maximum_tracking_error",
        ),
        "transition_maximum_torque_rate": transition_metrics.get(
            "maximum_torque_rate"
        ),
        "transition_maximum_abs_roll_pitch": transition_metrics.get(
            "maximum_abs_roll_pitch"
        ),
        "transition_maximum_xy_speed": transition_metrics.get(
            "maximum_xy_speed"
        ),
        "exit_single_support_samples": (
            int(transition_metrics.get("single_support_samples", 0))
            + int(recovery_metrics.get("single_support_samples", 0))
        ),
        "transition_single_support_events": int(
            transition_metrics.get("single_support_events", 0)
        ),
        "recovery_single_support_events": int(
            recovery_metrics.get("single_support_events", 0)
        ),
        "extra_recovery_step_events": int(
            recovery_metrics.get("single_support_events", 0)
        ),
        "exit_no_support_samples": (
            int(transition_metrics.get("no_support_samples", 0))
            + int(recovery_metrics.get("no_support_samples", 0))
        ),
        "foot_contact_transitions": (
            int(transition_metrics.get("foot_contact_transitions", 0))
            + int(recovery_metrics.get("foot_contact_transitions", 0))
        ),
        "minimum_root_z": summary.get("minimum_root_z"),
        "torque_saturation_rate": summary.get("torque_saturation_rate"),
        "selected_previews": selected_previews,
    }


def _episodes(modes: list[str]) -> list[tuple[int, int, int]]:
    episodes: list[tuple[int, int, int]] = []
    index = 0
    while index < len(modes):
        if modes[index] != "TRANSITION_OUT":
            index += 1
            continue
        transition_start = index
        while index < len(modes) and modes[index] == "TRANSITION_OUT":
            index += 1
        recovery_start = index
        while index < len(modes) and modes[index] == "RECOVER":
            index += 1
        episodes.append((transition_start, recovery_start, index))
    return episodes


def plot_exit_csv(
    path: str | Path,
    output_dir: str | Path,
    label: str = "",
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = Path(path).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    with csv_path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    episodes = _episodes([row["mode"] for row in rows])
    if not episodes:
        raise ValueError(f"CSV has no TRANSITION_OUT episode: {csv_path}")
    transition_start, recovery_start, recovery_end = episodes[0]
    plot_start = max(0, transition_start - 30)
    plot_end = min(len(rows), recovery_end + 30)
    selected = rows[plot_start:plot_end]
    time = np.asarray([float(row["time"]) for row in selected])
    transition_time = float(rows[transition_start]["time"])
    recovery_time = (
        float(rows[recovery_start]["time"])
        if recovery_start < len(rows)
        else float(rows[-1]["time"])
    )
    complete_time = (
        float(rows[recovery_end]["time"])
        if recovery_end < len(rows)
        else float(rows[-1]["time"])
    )
    markers = (transition_time, recovery_time, complete_time)
    prefix = label or csv_path.stem
    created: list[str] = []

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for axis, joint in zip(axes.flat, TYPICAL_JOINTS):
        axis.plot(
            time,
            [float(row[f"{joint}.raw_target"]) for row in selected],
            label="raw target",
        )
        axis.plot(
            time,
            [float(row[f"{joint}.target"]) for row in selected],
            label="limited target",
        )
        axis.plot(
            time,
            [float(row[f"{joint}.position"]) for row in selected],
            label="actual",
        )
        axis.set_title(joint)
        axis.set_ylabel("angle (rad)")
        axis.grid(alpha=0.25)
        for marker in markers:
            axis.axvline(marker, color="black", linewidth=0.8, alpha=0.45)
    axes.flat[0].legend(loc="best")
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle(f"{prefix}: exit targets and actual positions")
    fig.tight_layout()
    target_path = output_path / f"{prefix}_exit_targets.png"
    fig.savefig(target_path, dpi=160)
    plt.close(fig)
    created.append(str(target_path))

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for axis, joint in zip(axes.flat, TYPICAL_JOINTS):
        axis.plot(
            time,
            [float(row[f"{joint}.torque"]) for row in selected],
            color="tab:red",
        )
        axis.set_title(joint)
        axis.set_ylabel("torque (Nm)")
        axis.grid(alpha=0.25)
        for marker in markers:
            axis.axvline(marker, color="black", linewidth=0.8, alpha=0.45)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle(f"{prefix}: exit joint torques")
    fig.tight_layout()
    torque_path = output_path / f"{prefix}_exit_torques.png"
    fig.savefig(torque_path, dpi=160)
    plt.close(fig)
    created.append(str(torque_path))
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase 3A.3 exit logs")
    parser.add_argument("summary")
    parser.add_argument("--csv", default="")
    parser.add_argument("--output-dir", default="artifacts/smooth_exit_analysis")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    summary_path = Path(args.summary).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = summarize_exit(summary)
    label = args.label or summary_path.stem.removesuffix("_summary")
    if args.csv:
        result["plots"] = plot_exit_csv(args.csv, output_dir, label)
    result_path = output_dir / f"{label}_exit_analysis.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"[EXIT ANALYSIS] {result_path}")


if __name__ == "__main__":
    main()
