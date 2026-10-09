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


def joint_group(name: str) -> str:
    if any(token in name for token in ("hip_", "knee_", "ankle_")):
        return "legs"
    if name.startswith("waist_"):
        return "waist"
    return "arms"


def _maximum_by_group(joint_names: list[str], values: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for group in ("legs", "waist", "arms"):
        indices = [index for index, name in enumerate(joint_names) if joint_group(name) == group]
        result[group] = float(np.max(values[indices])) if indices else 0.0
    return result


def _episodes(modes: list[str]) -> list[tuple[int, int, int]]:
    episodes: list[tuple[int, int, int]] = []
    index = 0
    while index < len(modes):
        if modes[index] != "PRE_ALIGN":
            index += 1
            continue
        pre_start = index
        while index < len(modes) and modes[index] == "PRE_ALIGN":
            index += 1
        if index >= len(modes) or modes[index] != "TRANSITION_IN":
            continue
        transition_start = index
        while index < len(modes) and modes[index] == "TRANSITION_IN":
            index += 1
        episodes.append((pre_start, transition_start, index))
    return episodes


def analyze_csv(path: str | Path) -> dict[str, Any]:
    csv_path = Path(path).expanduser().resolve()
    with csv_path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"CSV has no rows: {csv_path}")
    joint_names = [
        name.removesuffix(".target")
        for name in rows[0]
        if name.endswith(".target") and ".raw_" not in name
    ]
    modes = [row["mode"] for row in rows]
    episodes = _episodes(modes)
    if not episodes:
        raise ValueError(f"CSV has no PRE_ALIGN -> TRANSITION_IN episode: {csv_path}")

    limited = np.asarray(
        [[float(row[f"{name}.target"]) for name in joint_names] for row in rows]
    )
    raw = np.asarray(
        [[float(row[f"{name}.raw_target"]) for name in joint_names] for row in rows]
    )
    actual = np.asarray(
        [[float(row[f"{name}.position"]) for name in joint_names] for row in rows]
    )
    torque = np.asarray(
        [[float(row[f"{name}.torque"]) for name in joint_names] for row in rows]
    )
    raw_kp = np.asarray(
        [[float(row[f"{name}.raw_kp"]) for name in joint_names] for row in rows]
    )
    raw_kd = np.asarray(
        [[float(row[f"{name}.raw_kd"]) for name in joint_names] for row in rows]
    )
    raw_feedforward = np.asarray(
        [[float(row[f"{name}.raw_feedforward"]) for name in joint_names] for row in rows]
    )

    episode_results: list[dict[str, Any]] = []
    for pre_start, transition_start, transition_end in episodes:
        previous = transition_start - 1
        raw_boundary = np.abs(raw[transition_start] - raw[previous])
        limited_boundary = np.abs(limited[transition_start] - limited[previous])
        kp_boundary = np.abs(raw_kp[transition_start] - raw_kp[previous])
        kd_boundary = np.abs(raw_kd[transition_start] - raw_kd[previous])
        feedforward_boundary = np.abs(
            raw_feedforward[transition_start] - raw_feedforward[previous]
        )

        limited_segment = limited[transition_start:transition_end]
        raw_segment = raw[transition_start:transition_end]
        actual_segment = actual[transition_start:transition_end]
        torque_segment = torque[transition_start:transition_end]

        def excess_travel(values: np.ndarray) -> np.ndarray:
            if len(values) < 2:
                return np.zeros(values.shape[1], dtype=np.float64)
            path_length = np.sum(np.abs(np.diff(values, axis=0)), axis=0)
            net = np.abs(values[-1] - values[0])
            return np.maximum(path_length - net, 0.0)

        pre_direction = limited[previous] - limited[pre_start]
        direction_sign = np.sign(pre_direction)
        displacement = limited_segment - limited_segment[0]
        reverse_excursion = np.max(
            np.maximum(-direction_sign * displacement, 0.0),
            axis=0,
        )
        torque_delta = (
            np.max(np.abs(np.diff(torque_segment, axis=0)), axis=0)
            if len(torque_segment) > 1
            else np.zeros(len(joint_names), dtype=np.float64)
        )
        episode_results.append(
            {
                "pre_align_start_step": int(rows[pre_start]["step"]),
                "transition_start_step": int(rows[transition_start]["step"]),
                "motion_start_step": int(rows[transition_end]["step"])
                if transition_end < len(rows)
                else None,
                "boundary": {
                    "raw_target_jump": _maximum_by_group(joint_names, raw_boundary),
                    "limited_target_jump": _maximum_by_group(
                        joint_names, limited_boundary
                    ),
                    "kp_jump": _maximum_by_group(joint_names, kp_boundary),
                    "kd_jump": _maximum_by_group(joint_names, kd_boundary),
                    "feedforward_jump": _maximum_by_group(
                        joint_names, feedforward_boundary
                    ),
                },
                "transition": {
                    "raw_target_excess_travel": _maximum_by_group(
                        joint_names, excess_travel(raw_segment)
                    ),
                    "limited_target_excess_travel": _maximum_by_group(
                        joint_names, excess_travel(limited_segment)
                    ),
                    "actual_position_excess_travel": _maximum_by_group(
                        joint_names, excess_travel(actual_segment)
                    ),
                    "reverse_excursion": _maximum_by_group(
                        joint_names, reverse_excursion
                    ),
                    "maximum_torque_delta": _maximum_by_group(
                        joint_names, torque_delta
                    ),
                },
            }
        )

    def aggregate(path_keys: tuple[str, str]) -> dict[str, float]:
        return {
            group: max(
                episode[path_keys[0]][path_keys[1]][group]
                for episode in episode_results
            )
            for group in ("legs", "waist", "arms")
        }

    return {
        "csv": str(csv_path),
        "episodes": episode_results,
        "maximum": {
            "raw_boundary_jump": aggregate(("boundary", "raw_target_jump")),
            "limited_boundary_jump": aggregate(("boundary", "limited_target_jump")),
            "kp_boundary_jump": aggregate(("boundary", "kp_jump")),
            "kd_boundary_jump": aggregate(("boundary", "kd_jump")),
            "feedforward_boundary_jump": aggregate(
                ("boundary", "feedforward_jump")
            ),
            "limited_target_excess_travel": aggregate(
                ("transition", "limited_target_excess_travel")
            ),
            "actual_position_excess_travel": aggregate(
                ("transition", "actual_position_excess_travel")
            ),
            "reverse_excursion": aggregate(("transition", "reverse_excursion")),
            "maximum_torque_delta": aggregate(
                ("transition", "maximum_torque_delta")
            ),
        },
    }


def plot_csv(path: str | Path, output_dir: str | Path, label: str = "") -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = Path(path).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    with csv_path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    modes = [row["mode"] for row in rows]
    episodes = _episodes(modes)
    if not episodes:
        raise ValueError(f"CSV has no entry episode: {csv_path}")
    pre_start, transition_start, transition_end = episodes[0]
    plot_start = max(0, pre_start - 20)
    plot_end = min(len(rows), transition_end + 30)
    time = np.asarray([float(row["time"]) for row in rows[plot_start:plot_end]])
    pre_time = float(rows[pre_start]["time"])
    transition_time = float(rows[transition_start]["time"])
    motion_time = (
        float(rows[transition_end]["time"])
        if transition_end < len(rows)
        else float(rows[-1]["time"])
    )
    prefix = label or csv_path.stem
    created: list[str] = []

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for axis, joint in zip(axes.flat, TYPICAL_JOINTS):
        selected = rows[plot_start:plot_end]
        axis.plot(time, [float(row[f"{joint}.raw_target"]) for row in selected], label="raw target")
        axis.plot(time, [float(row[f"{joint}.target"]) for row in selected], label="limited target")
        axis.plot(time, [float(row[f"{joint}.position"]) for row in selected], label="actual")
        axis.set_title(joint)
        axis.set_ylabel("angle (rad)")
        axis.grid(alpha=0.25)
        for marker in (pre_time, transition_time, motion_time):
            axis.axvline(marker, color="black", linewidth=0.8, alpha=0.45)
    axes.flat[0].legend(loc="best")
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle(f"{prefix}: entry joint targets and actual positions")
    fig.tight_layout()
    target_path = output_path / f"{prefix}_entry_targets.png"
    fig.savefig(target_path, dpi=160)
    plt.close(fig)
    created.append(str(target_path))

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for axis, joint in zip(axes.flat, TYPICAL_JOINTS):
        selected = rows[plot_start:plot_end]
        axis.plot(time, [float(row[f"{joint}.torque"]) for row in selected], color="tab:red")
        axis.set_title(joint)
        axis.set_ylabel("torque (Nm)")
        axis.grid(alpha=0.25)
        for marker in (pre_time, transition_time, motion_time):
            axis.axvline(marker, color="black", linewidth=0.8, alpha=0.45)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle(f"{prefix}: entry joint torques")
    fig.tight_layout()
    torque_path = output_path / f"{prefix}_entry_torques.png"
    fig.savefig(torque_path, dpi=160)
    plt.close(fig)
    created.append(str(torque_path))
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and plot Phase 3A.2 entry logs")
    parser.add_argument("csv")
    parser.add_argument("--output-dir", default="artifacts/smooth_entry_analysis")
    parser.add_argument("--label", default="")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = analyze_csv(args.csv)
    result["finite"] = all(
        math.isfinite(value)
        for section in result["maximum"].values()
        for value in section.values()
    )
    if not args.no_plots:
        result["plots"] = plot_csv(args.csv, output_dir, args.label)
    name = args.label or Path(args.csv).stem
    summary_path = output_dir / f"{name}_entry_analysis.json"
    summary_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"[ENTRY ANALYSIS] {summary_path}")


if __name__ == "__main__":
    main()
