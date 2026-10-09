#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from analyze_smooth_entry import analyze_csv, plot_csv


MODE_CASES = (
    ("legacy", 0.6, 0.4),
    ("full_body_continuous", 0.6, 0.4),
    ("grouped", 0.6, 0.4),
)
DURATION_CASES = (
    (0.4, 0.4),
    (0.6, 0.2),
    (0.6, 0.3),
    (0.6, 0.4),
    (0.6, 0.6),
    (0.8, 0.4),
)
SEQUENCE_SCENARIOS = ("wave_then_bow", "bow_then_wave")
REPEAT_SCENARIOS = ("wave_repeat20", "bow_repeat20", "alternate_repeat20")


def run_case(
    project: Path,
    config: Path,
    output_dir: Path,
    name: str,
    scenario: str,
    entry_mode: str = "full_body_continuous",
    pre_align: float = 0.6,
    transition_in: float = 0.2,
    write_csv: bool = False,
) -> tuple[dict[str, Any], int, Path | None]:
    summary_path = output_dir / f"{name}_summary.json"
    stdout_path = output_dir / f"{name}.log"
    csv_path = output_dir / f"{name}.csv" if write_csv else None
    command = [
        sys.executable,
        str(project / "run_multi_evt2.py"),
        "--config",
        str(config),
        "--scenario",
        scenario,
        "--entry-mode",
        entry_mode,
        "--pre-align",
        str(pre_align),
        "--transition-in",
        str(transition_in),
        "--headless",
        "--no-realtime",
        "--debug-interval",
        "0",
        "--summary",
        str(summary_path),
    ]
    if csv_path is None:
        command.append("--no-log")
    else:
        command.extend(("--log", str(csv_path)))
    print(f"[ENTRY SUITE] running {name}", flush=True)
    with stdout_path.open("w", encoding="utf-8") as output:
        completed = subprocess.run(
            command,
            cwd=project,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if not summary_path.is_file():
        return {
            "passed": False,
            "error": "summary not produced",
            "log": str(stdout_path),
        }, completed.returncode, csv_path
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary, completed.returncode, csv_path


def compact_metrics(
    summary: dict[str, Any],
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pre_align = summary.get("entry_metrics", {}).get("PRE_ALIGN", {})
    transition = summary.get("entry_metrics", {}).get("TRANSITION_IN", {})
    groups = transition.get("groups", {})
    commands = summary.get("controller", {}).get("commands", [])
    control_dt = float(summary.get("control_dt", 0.01))
    values: dict[str, Any] = {
        "passed": bool(summary.get("passed", False)),
        "entry_mode": summary.get("controller", {}).get("entry_mode"),
        "completed_actions": sum(
            record.get("status") == "COMPLETED" and record.get("action_key") in ("a", "b")
            for record in commands
        ),
        "failed_or_rejected_actions": sum(
            record.get("status") in ("FAILED", "REJECTED")
            and record.get("action_key") in ("a", "b")
            for record in commands
        ),
        "minimum_root_z": summary.get("minimum_root_z"),
        "maximum_abs_roll_pitch": summary.get("maximum_abs_roll_pitch"),
        "torque_saturation_rate": summary.get("torque_saturation_rate"),
        "pre_align_seconds": pre_align.get("samples", 0) * control_dt,
        "transition_in_seconds": transition.get("samples", 0) * control_dt,
        "entry_maximum_abs_roll_pitch": transition.get("maximum_abs_roll_pitch"),
        "entry_maximum_xy_speed": transition.get("maximum_xy_speed"),
        "entry_maximum_angular_speed": transition.get("maximum_angular_speed"),
        "entry_maximum_torque_rate": transition.get("maximum_torque_rate"),
        "entry_maximum_joint_speed": max(
            (group.get("maximum_joint_speed", 0.0) for group in groups.values()),
            default=None,
        ),
        "entry_maximum_tracking_error": max(
            (group.get("maximum_tracking_error", 0.0) for group in groups.values()),
            default=None,
        ),
        "entry_maximum_abs_torque": max(
            (group.get("maximum_abs_torque", 0.0) for group in groups.values()),
            default=None,
        ),
        "entry_boundary": summary.get("entry_metrics", {}).get("boundary", {}),
    }
    if analysis is not None:
        values["continuity_analysis"] = analysis["maximum"]
    return values


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Smooth Entry Phase 3A.2 Test Report",
        "",
        f"Overall result: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "## Entry mode comparison (PRE_ALIGN 0.6 s, TRANSITION_IN 0.4 s)",
        "",
        "| case | pass | raw boundary jump | excess target travel | "
        "entry roll/pitch | entry xy speed | entry joint speed | entry torque rate |",
        "| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in report["mode_comparison"].items():
        continuity = values["continuity_analysis"]
        boundary = max(continuity["raw_boundary_jump"].values())
        excess = max(continuity["limited_target_excess_travel"].values())
        lines.append(
            (
                "| {name} | {passed} | {boundary} | {excess} | {rp} | "
                "{xy} | {jv} | {torque_rate} |"
            ).format(
                name=name,
                passed="PASS" if values["passed"] else "FAIL",
                boundary=_fmt(boundary),
                excess=_fmt(excess),
                rp=_fmt(values["entry_maximum_abs_roll_pitch"]),
                xy=_fmt(values["entry_maximum_xy_speed"]),
                jv=_fmt(values["entry_maximum_joint_speed"]),
                torque_rate=_fmt(values["entry_maximum_torque_rate"]),
            )
        )

    lines.extend(
        [
            "",
            "## Selected configuration (PRE_ALIGN 0.6 s, TRANSITION_IN 0.2 s)",
            "",
            "| case | pass | raw boundary jump | excess target travel | "
            "entry roll/pitch | entry xy speed | entry joint speed | entry torque rate |",
            "| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, values in report["selected"].items():
        continuity = values["continuity_analysis"]
        boundary = max(continuity["raw_boundary_jump"].values())
        excess = max(continuity["limited_target_excess_travel"].values())
        lines.append(
            (
                "| {name} | {passed} | {boundary} | {excess} | {rp} | "
                "{xy} | {jv} | {torque_rate} |"
            ).format(
                name=name,
                passed="PASS" if values["passed"] else "FAIL",
                boundary=_fmt(boundary),
                excess=_fmt(excess),
                rp=_fmt(values["entry_maximum_abs_roll_pitch"]),
                xy=_fmt(values["entry_maximum_xy_speed"]),
                jv=_fmt(values["entry_maximum_joint_speed"]),
                torque_rate=_fmt(values["entry_maximum_torque_rate"]),
            )
        )

    lines.extend(
        [
            "",
            "## Duration sweep (full-body continuous)",
            "",
            "| case | pass | entry roll/pitch | entry xy speed | "
            "entry joint speed | tracking error | entry torque rate |",
            "| --- | :---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, values in report["duration_sweep"].items():
        lines.append(
            "| {name} | {passed} | {rp} | {xy} | {jv} | {tracking} | {torque_rate} |".format(
                name=name,
                passed="PASS" if values["passed"] else "FAIL",
                rp=_fmt(values["entry_maximum_abs_roll_pitch"]),
                xy=_fmt(values["entry_maximum_xy_speed"]),
                jv=_fmt(values["entry_maximum_joint_speed"]),
                tracking=_fmt(values["entry_maximum_tracking_error"]),
                torque_rate=_fmt(values["entry_maximum_torque_rate"]),
            )
        )

    lines.extend(
        [
            "",
            "## Observed entry timing",
            "",
            "| case | PRE_ALIGN observed | TRANSITION_IN observed |",
            "| --- | ---: | ---: |",
        ]
    )
    for name, values in report["selected"].items():
        lines.append(
            "| {name} | {pre} s | {transition} s |".format(
                name=name,
                pre=_fmt(values["pre_align_seconds"]),
                transition=_fmt(values["transition_in_seconds"]),
            )
        )

    lines.extend(
        [
            "",
            "## Continuous-state scenarios",
            "",
            "| scenario | pass | completed actions | failed/rejected | minimum root z | saturation |",
            "| --- | :---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, values in report["continuous_scenarios"].items():
        lines.append(
            "| {name} | {passed} | {completed} | {failed} | {root} | {saturation} |".format(
                name=name,
                passed="PASS" if values["passed"] else "FAIL",
                completed=values["completed_actions"],
                failed=values["failed_or_rejected_actions"],
                root=_fmt(values["minimum_root_z"]),
                saturation=_fmt(values["torque_saturation_rate"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3A.2 smooth-entry tests")
    parser.add_argument("--config", default="configs/multi_evt2.json")
    parser.add_argument("--output-dir", default="artifacts/smooth_entry_phase3a2")
    parser.add_argument("--quick", action="store_true", help="skip duration and repeat tests")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    config = (project / args.config).resolve()
    output_dir = (project / args.output_dir).resolve()
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "config": str(config),
        "mode_comparison": {},
        "selected": {},
        "duration_sweep": {},
        "continuous_scenarios": {},
        "plots": [],
    }
    failed = False

    for scenario in ("bow", "wave"):
        for mode, pre_align, transition_in in MODE_CASES:
            name = f"{scenario}_{mode}"
            summary, returncode, csv_path = run_case(
                project,
                config,
                output_dir,
                name,
                scenario,
                mode,
                pre_align,
                transition_in,
                write_csv=True,
            )
            analysis = analyze_csv(csv_path) if csv_path and csv_path.is_file() else None
            report["mode_comparison"][name] = compact_metrics(summary, analysis)
            if csv_path and csv_path.is_file():
                report["plots"].extend(plot_csv(csv_path, plot_dir, name))
            failed |= returncode != 0 or not summary.get("passed", False)

    for scenario in ("bow", "wave"):
        name = f"{scenario}_selected"
        summary, returncode, csv_path = run_case(
            project,
            config,
            output_dir,
            name,
            scenario,
            write_csv=True,
        )
        analysis = analyze_csv(csv_path) if csv_path and csv_path.is_file() else None
        report["selected"][name] = compact_metrics(summary, analysis)
        if csv_path and csv_path.is_file():
            report["plots"].extend(plot_csv(csv_path, plot_dir, name))
        failed |= returncode != 0 or not summary.get("passed", False)

    if not args.quick:
        for scenario in ("bow", "wave"):
            for pre_align, transition_in in DURATION_CASES:
                name = f"{scenario}_pre{pre_align:g}_transition{transition_in:g}"
                summary, returncode, _ = run_case(
                    project,
                    config,
                    output_dir,
                    name,
                    scenario,
                    pre_align=pre_align,
                    transition_in=transition_in,
                )
                report["duration_sweep"][name] = compact_metrics(summary)
                failed |= returncode != 0 or not summary.get("passed", False)

    scenarios = SEQUENCE_SCENARIOS if args.quick else SEQUENCE_SCENARIOS + REPEAT_SCENARIOS
    for scenario in scenarios:
        summary, returncode, _ = run_case(
            project,
            config,
            output_dir,
            scenario,
            scenario,
        )
        report["continuous_scenarios"][scenario] = compact_metrics(summary)
        failed |= returncode != 0 or not summary.get("passed", False)

    report["passed"] = not failed
    summary_path = output_dir / "suite_summary.json"
    markdown_path = output_dir / "report.md"
    summary_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(markdown_path, report)
    print(f"[ENTRY SUITE] {'PASS' if not failed else 'FAIL'}: {markdown_path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
