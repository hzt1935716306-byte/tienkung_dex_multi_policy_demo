#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def measurement_bounds(summary: dict[str, Any]) -> tuple[int, int, str]:
    measurement = summary.get("measurement", {})
    control_dt = float(summary["control_dt"])
    name = str(measurement.get("name", "whole_run"))
    if "start_time" in measurement:
        start = int(round(float(measurement["start_time"]) / control_dt))
    elif "start_event" in measurement:
        event_name = str(measurement["start_event"])
        events = summary["controller"]["events"]
        matches = [event for event in events if event.get("event") == event_name]
        if not matches:
            raise ValueError(f"Measurement start event {event_name!r} was not emitted")
        start = int(matches[0]["step"])
    else:
        start = 0
    duration = float(
        measurement.get(
            "duration_seconds",
            summary["executed_steps"] * control_dt,
        )
    )
    end = start + int(round(duration / control_dt))
    return start, end, name


def _planar_displacement(
    first: dict[str, str],
    last: dict[str, str],
    prefix: str,
) -> float:
    dx = _float(last, f"{prefix}_x") - _float(first, f"{prefix}_x")
    dy = _float(last, f"{prefix}_y") - _float(first, f"{prefix}_y")
    return math.hypot(dx, dy)


def summarize_window(
    rows: list[dict[str, str]],
    summary: dict[str, Any],
    start_step: int,
    end_step: int,
    name: str,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if start_step <= int(row["step"]) < end_step
    ]
    if not selected:
        raise ValueError(f"No CSV samples inside window {name!r}")
    joint_velocity_columns = [
        key for key in selected[0] if key.endswith(".velocity")
    ]
    torque_columns = [key for key in selected[0] if key.endswith(".torque")]
    target_columns = [key for key in selected[0] if key.endswith(".target")]
    position_columns = [key for key in selected[0] if key.endswith(".position")]
    complete_steps = [
        event
        for event in summary["stand_diagnostics"]["complete_steps"]
        if start_step <= int(event["step"]) < end_step
    ]
    contact_start = _float(selected[0], "contact_edge_count")
    contact_end = _float(selected[-1], "contact_edge_count")
    maximum_tracking_error = 0.0
    maximum_torque_delta = 0.0
    previous_torque: list[float] | None = None
    for row in selected:
        for target_key, position_key in zip(target_columns, position_columns):
            maximum_tracking_error = max(
                maximum_tracking_error,
                abs(_float(row, target_key) - _float(row, position_key)),
            )
        current_torque = [_float(row, key) for key in torque_columns]
        if previous_torque is not None:
            maximum_torque_delta = max(
                maximum_torque_delta,
                max(
                    abs(current - previous)
                    for current, previous in zip(current_torque, previous_torque)
                ),
            )
        previous_torque = current_torque

    maximum_com_speed = max(
        math.hypot(_float(row, "com_vx"), _float(row, "com_vy"))
        for row in selected
    )
    support_codes = [int(round(_float(row, "support_code"))) for row in selected]
    control_dt = float(summary["control_dt"])
    return {
        "name": name,
        "start_step": start_step,
        "end_step": end_step,
        "duration_seconds": len(selected) * control_dt,
        "complete_step_count": len(complete_steps),
        "complete_steps": complete_steps,
        "contact_edge_count": max(0, int(round(contact_end - contact_start))),
        "single_support_seconds": support_codes.count(1) * control_dt,
        "no_support_seconds": support_codes.count(0) * control_dt,
        "left_foot_planar_displacement": _planar_displacement(
            selected[0], selected[-1], "left_foot"
        ),
        "right_foot_planar_displacement": _planar_displacement(
            selected[0], selected[-1], "right_foot"
        ),
        "com_planar_displacement": _planar_displacement(
            selected[0], selected[-1], "com"
        ),
        "root_planar_displacement": math.hypot(
            _float(selected[-1], "root_x") - _float(selected[0], "root_x"),
            _float(selected[-1], "root_y") - _float(selected[0], "root_y"),
        ),
        "maximum_com_xy_speed": maximum_com_speed,
        "maximum_root_xy_speed": max(
            math.hypot(_float(row, "world_vx"), _float(row, "world_vy"))
            for row in selected
        ),
        "maximum_abs_roll": max(abs(_float(row, "roll")) for row in selected),
        "maximum_abs_pitch": max(abs(_float(row, "pitch")) for row in selected),
        "maximum_angular_speed": max(
            math.sqrt(
                _float(row, "body_wx") ** 2
                + _float(row, "body_wy") ** 2
                + _float(row, "body_wz") ** 2
            )
            for row in selected
        ),
        "maximum_joint_speed": max(
            abs(_float(row, key))
            for row in selected
            for key in joint_velocity_columns
        ),
        "maximum_tracking_error": maximum_tracking_error,
        "maximum_abs_torque": max(
            abs(_float(row, key))
            for row in selected
            for key in torque_columns
        ),
        "maximum_torque_delta": maximum_torque_delta,
    }


def plot_window(
    rows: list[dict[str, str]],
    start_step: int,
    end_step: int,
    output: Path,
    title: str,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in rows
        if start_step <= int(row["step"]) < end_step
    ]
    time = [_float(row, "time") - _float(selected[0], "time") for row in selected]
    figure, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(time, [_float(row, "left_foot_x") for row in selected], label="left x")
    axes[0].plot(time, [_float(row, "right_foot_x") for row in selected], label="right x")
    axes[0].plot(time, [_float(row, "com_x") for row in selected], label="CoM x")
    axes[0].set_ylabel("world x [m]")
    axes[0].legend(loc="best")
    axes[1].plot(time, [_float(row, "left_foot_y") for row in selected], label="left y")
    axes[1].plot(time, [_float(row, "right_foot_y") for row in selected], label="right y")
    axes[1].plot(time, [_float(row, "com_y") for row in selected], label="CoM y")
    axes[1].set_ylabel("world y [m]")
    axes[1].legend(loc="best")
    axes[2].plot(time, [_float(row, "roll") for row in selected], label="roll")
    axes[2].plot(time, [_float(row, "pitch") for row in selected], label="pitch")
    axes[2].plot(time, [_float(row, "com_vx") for row in selected], label="CoM vx")
    axes[2].plot(time, [_float(row, "com_vy") for row in selected], label="CoM vy")
    axes[2].set_ylabel("rad / m/s")
    axes[2].legend(loc="best")
    axes[3].step(
        time,
        [_float(row, "support_code") for row in selected],
        where="post",
        label="feet in contact",
    )
    axes[3].step(
        time,
        [_float(row, "complete_step_count") for row in selected],
        where="post",
        label="complete steps",
    )
    axes[3].set_ylabel("count")
    axes[3].set_xlabel("window time [s]")
    axes[3].legend(loc="best")
    figure.suptitle(title)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return str(output)


def analyze_case(
    csv_path: Path,
    summary_path: Path,
    plot_path: Path | None = None,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = read_rows(csv_path)
    start, end, name = measurement_bounds(summary)
    metrics = summarize_window(rows, summary, start, end, name)
    result = {
        "scenario": summary["scenario"],
        "passed": bool(summary["passed"]),
        "control_dt": float(summary["control_dt"]),
        "measurement": metrics,
        "whole_run": summary["stand_diagnostics"],
    }
    events = summary.get("controller", {}).get("events", [])
    transition_events = [
        event for event in events if event.get("event") == "transition_out_started"
    ]
    acquisition_events = [
        event for event in events if event.get("event") == "walkamp_control_acquired"
    ]
    recovery_events = [
        event for event in events if event.get("event") == "recovery_completed"
    ]
    if transition_events and acquisition_events and recovery_events:
        transition_step = int(transition_events[0]["step"])
        acquisition_step = int(acquisition_events[0]["step"])
        recovery_step = int(recovery_events[0]["step"])
        result["transition_out"] = summarize_window(
            rows,
            summary,
            transition_step,
            acquisition_step + 1,
            "transition_out",
        )
        result["walkamp_recover"] = summarize_window(
            rows,
            summary,
            acquisition_step,
            recovery_step + 1,
            "walkamp_recover",
        )
        result["exit_and_recovery"] = summarize_window(
            rows,
            summary,
            transition_step,
            recovery_step + 1,
            "exit_and_recovery",
        )
    if plot_path is not None:
        result["plot"] = plot_window(rows, start, end, plot_path, name)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase 3A.4 standing diagnostics")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plot", default="")
    args = parser.parse_args()
    result = analyze_case(
        Path(args.csv),
        Path(args.summary),
        None if not args.plot else Path(args.plot),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
