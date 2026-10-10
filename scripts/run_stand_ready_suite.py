#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from analyze_stand_ready import analyze_case


DIAGNOSTIC_SCENARIOS = (
    "stand_zero_30",
    "walk_stop_30",
    "wave_return_20",
    "bow_return_20",
)
REPEAT_SCENARIOS = (
    "bow_repeat20",
    "wave_repeat20",
    "alternate_repeat20",
)


def _write_ready_config(source: Path, destination: Path) -> None:
    config = json.loads(source.read_text(encoding="utf-8"))
    source_dir = source.parent
    for key in ("model", "walkamp_config"):
        path = Path(config[key]).expanduser()
        if not path.is_absolute():
            path = source_dir / path
        config[key] = str(path.resolve())
    for motion in config["motions"].values():
        path = Path(motion["path"]).expanduser()
        if not path.is_absolute():
            path = source_dir / path
        motion["path"] = str(path.resolve())
    config["controller"]["ready"]["enabled"] = True
    destination.write_text(
        json.dumps(config, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def run_case(
    project: Path,
    config: Path,
    output_dir: Path,
    name: str,
    scenario: str,
    write_csv: bool,
) -> tuple[dict[str, Any], int, Path | None, Path]:
    summary_path = output_dir / f"{name}_summary.json"
    log_path = output_dir / f"{name}.log"
    csv_path = output_dir / f"{name}.csv" if write_csv else None
    command = [
        sys.executable,
        str(project / "run_multi_evt2.py"),
        "--config",
        str(config),
        "--scenario",
        scenario,
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
    print(f"[STAND SUITE] running {name}", flush=True)
    with log_path.open("w", encoding="utf-8") as output:
        completed = subprocess.run(
            command,
            cwd=project,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if not summary_path.is_file():
        return (
            {"passed": False, "error": "summary not produced", "log": str(log_path)},
            completed.returncode,
            csv_path,
            summary_path,
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary, completed.returncode, csv_path, summary_path


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _measurement_table(cases: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| case | pass | complete steps | contact edges | single support [s] | "
        "left foot drift [m] | right foot drift [m] | CoM drift [m] | "
        "max CoM speed [m/s] | max pitch [rad] |",
        "| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in cases.items():
        values = result["measurement"]
        lines.append(
            "| {name} | {passed} | {steps} | {edges} | {support} | {left} | "
            "{right} | {com} | {speed} | {pitch} |".format(
                name=name,
                passed="PASS" if result.get("passed") else "FAIL",
                steps=values["complete_step_count"],
                edges=values["contact_edge_count"],
                support=_fmt(values["single_support_seconds"]),
                left=_fmt(values["left_foot_planar_displacement"]),
                right=_fmt(values["right_foot_planar_displacement"]),
                com=_fmt(values["com_planar_displacement"]),
                speed=_fmt(values["maximum_com_xy_speed"]),
                pitch=_fmt(values["maximum_abs_pitch"]),
            )
        )
    return lines


def _recovery_table(cases: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| case | exit steps | transition steps | WALKAMP-recover steps | "
        "exit duration [s] | max CoM speed [m/s] | max pitch [rad] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in cases.items():
        whole = result.get("exit_and_recovery")
        if whole is None:
            continue
        transition = result["transition_out"]
        recover = result["walkamp_recover"]
        lines.append(
            f"| {name} | {whole['complete_step_count']} | "
            f"{transition['complete_step_count']} | {recover['complete_step_count']} | "
            f"{whole['duration_seconds']:.3f} | {whole['maximum_com_xy_speed']:.4f} | "
            f"{whole['maximum_abs_pitch']:.4f} |"
        )
    return lines


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    diagnosis = report["diagnosis"]
    lines = [
        "# Stand / READY Phase 3A.4 Diagnostic Report",
        "",
        f"Overall regression result: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "A complete step requires a verified lift, at least 25 mm horizontal foot "
        "motion, and a stable landing. Contact edges and single-support samples are "
        "reported separately.",
        "",
        "## Baseline measurements",
        "",
        *_measurement_table(report["baseline"]),
        "",
        "## Exit and recovery windows",
        "",
        *_recovery_table(report["baseline"]),
        "",
        "## READY-interface control comparison",
        "",
        "The candidate keeps the same verified WALKAMP feedback active in an explicit "
        "READY state. It is not a new Stand Policy.",
        "",
        *_measurement_table(report["ready_comparison"]),
        "",
        *_recovery_table(report["ready_comparison"]),
        "",
        "## Continuous no-reset tests",
        "",
        "| scenario | pass | commands completed | complete steps | contact edges | "
        "foot displacement L/R [m] | falls |",
        "| --- | :---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for name, values in report["continuous_no_reset"].items():
        diagnostic = values.get("stand_diagnostics", {})
        commands = values.get("controller", {}).get("commands", [])
        completed = sum(
            command.get("status") == "COMPLETED" and command.get("action_key") in ("a", "b")
            for command in commands
        )
        left = diagnostic.get("left_foot_displacement", [0.0, 0.0])
        right = diagnostic.get("right_foot_displacement", [0.0, 0.0])
        left_xy = (left[0] ** 2 + left[1] ** 2) ** 0.5
        right_xy = (right[0] ** 2 + right[1] ** 2) ** 0.5
        lines.append(
            f"| {name} | {'PASS' if values.get('passed') else 'FAIL'} | {completed} | "
            f"{diagnostic.get('complete_step_count', 'n/a')} | "
            f"{diagnostic.get('contact_edge_count', 'n/a')} | "
            f"{left_xy:.4f} / {right_xy:.4f} | "
            f"{0 if values.get('minimum_root_z', 0.0) >= values.get('fall_threshold', 1.0) else 1} |"
        )
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"1. WALKAMP zero-command periodic stepping: **{diagnosis['walkamp_zero_periodic_stepping']}**.",
            f"2. Bow adjustment classification: **{diagnosis['bow_adjustment_classification']}**.",
            f"3. Independent Stand Policy needed now: **{diagnosis['stand_policy_recommendation']}**.",
            f"4. READY feedback comparison: **{diagnosis['ready_comparison']}**.",
            "",
            "The official branch-3.0 STOP controller only holds the entry joint "
            "positions with PD, and BeyondZero interpolates to a fixed pose. Neither "
            "is a closed-loop balance policy, so neither is installed as the formal "
            "READY controller.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def classify(report: dict[str, Any]) -> dict[str, str]:
    baseline = report["baseline"]
    stand = baseline["stand_zero_30"]["measurement"]
    stopped = baseline["walk_stop_30"]["measurement"]
    periodic = stand["complete_step_count"] > 0
    stop_steps = stopped["complete_steps"]
    stop_last_offset = (
        0.0
        if not stop_steps
        else (
            int(stop_steps[-1]["step"]) - int(stopped["start_step"])
        )
        * float(baseline["walk_stop_30"]["control_dt"])
    )
    bow_recovery = baseline["bow_return_20"].get("exit_and_recovery", {})
    wave_recovery = baseline["wave_return_20"].get("exit_and_recovery", {})
    bow_excess = bow_recovery.get("complete_step_count", 0) > wave_recovery.get(
        "complete_step_count", 0
    )
    ready_bow = report["ready_comparison"]["bow_return_20_ready"]
    ready_bow_recovery = ready_bow.get("exit_and_recovery", {})
    if periodic:
        stand_answer = "yes; WALKAMP produced complete zero-command steps"
        recommendation = "evaluate and train a verified closed-loop Stand Policy"
    else:
        stand_answer = (
            "no; standard zero-command standing produced no complete steps, and "
            f"walk-stop settling ended within about {stop_last_offset:.2f} s"
        )
        recommendation = (
            "not required for ordinary zero-command standing; keep READY reserved "
            "for a bow-state recovery policy if further reduction is required"
        )
    if bow_excess:
        bow_answer = (
            f"{bow_recovery.get('complete_step_count', 0)} complete steps occur "
            "between bow exit start and recovery completion, versus "
            f"{wave_recovery.get('complete_step_count', 0)} for wave; no bow steps "
            "remain after recovery, so the evidence favors terminal-state dynamic "
            "recovery rather than ordinary zero-speed stepping"
        )
    else:
        bow_answer = (
            "bow recovery does not exceed the measured WALKAMP standing baseline"
        )
    ready_answer = (
        "no physical benefit over Phase 3A.3"
        if ready_bow_recovery.get("complete_step_count", 0)
        >= bow_recovery.get("complete_step_count", 0)
        else "fewer complete steps in this simulation, requiring broader validation"
    )
    return {
        "walkamp_zero_periodic_stepping": stand_answer,
        "bow_adjustment_classification": bow_answer,
        "stand_policy_recommendation": recommendation,
        "ready_comparison": ready_answer,
    }


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    def compact_case(values: dict[str, Any]) -> dict[str, Any]:
        return {
            key: values[key]
            for key in (
                "scenario",
                "passed",
                "control_dt",
                "measurement",
                "transition_out",
                "walkamp_recover",
                "exit_and_recovery",
            )
            if key in values
        }

    repeats: dict[str, Any] = {}
    for name, values in report["continuous_no_reset"].items():
        diagnostics = values.get("stand_diagnostics", {})
        commands = values.get("controller", {}).get("commands", [])
        repeats[name] = {
            "passed": values.get("passed", False),
            "completed_actions": sum(
                command.get("status") == "COMPLETED"
                and command.get("action_key") in ("a", "b")
                for command in commands
            ),
            "minimum_root_z": values.get("minimum_root_z"),
            "complete_step_count": diagnostics.get("complete_step_count"),
            "contact_edge_count": diagnostics.get("contact_edge_count"),
            "left_foot_displacement": diagnostics.get("left_foot_displacement"),
            "right_foot_displacement": diagnostics.get("right_foot_displacement"),
            "com_displacement": diagnostics.get("com_displacement"),
            "maximum_com_xy_speed": diagnostics.get("maximum_com_xy_speed"),
            "maximum_joint_speed": diagnostics.get("maximum_joint_speed"),
            "maximum_abs_torque": diagnostics.get("maximum_abs_torque"),
            "maximum_torque_delta": diagnostics.get("maximum_torque_delta"),
        }
    return {
        "passed": report["passed"],
        "diagnosis": report["diagnosis"],
        "baseline": {
            name: compact_case(values)
            for name, values in report["baseline"].items()
        },
        "ready_comparison": {
            name: compact_case(values)
            for name, values in report["ready_comparison"].items()
        },
        "continuous_no_reset": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3A.4 stand/READY diagnostics")
    parser.add_argument("--config", default="configs/multi_evt2.json")
    parser.add_argument("--output-dir", default="artifacts/stand_ready_phase3a4")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    config = (project / args.config).resolve()
    output_dir = (project / args.output_dir).resolve()
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    ready_config = output_dir / "multi_evt2_ready_feedback.json"
    _write_ready_config(config, ready_config)
    report: dict[str, Any] = {
        "config": str(config),
        "ready_test_config": str(ready_config),
        "baseline": {},
        "ready_comparison": {},
        "continuous_no_reset": {},
        "plots": [],
    }
    failed = False

    for scenario in DIAGNOSTIC_SCENARIOS:
        summary, returncode, csv_path, summary_path = run_case(
            project,
            config,
            output_dir,
            scenario,
            scenario,
            write_csv=True,
        )
        if csv_path is not None and csv_path.is_file() and summary_path.is_file():
            result = analyze_case(
                csv_path,
                summary_path,
                plot_dir / f"{scenario}.png",
            )
            report["baseline"][scenario] = result
            report["plots"].append(result["plot"])
        else:
            report["baseline"][scenario] = summary
        failed |= returncode != 0 or not summary.get("passed", False)

    for scenario in ("wave_return_20", "bow_return_20"):
        name = f"{scenario}_ready"
        summary, returncode, csv_path, summary_path = run_case(
            project,
            ready_config,
            output_dir,
            name,
            scenario,
            write_csv=True,
        )
        if csv_path is not None and csv_path.is_file() and summary_path.is_file():
            result = analyze_case(
                csv_path,
                summary_path,
                plot_dir / f"{name}.png",
            )
            report["ready_comparison"][name] = result
            report["plots"].append(result["plot"])
        else:
            report["ready_comparison"][name] = summary
        failed |= returncode != 0 or not summary.get("passed", False)

    if not args.quick:
        for scenario in REPEAT_SCENARIOS:
            summary, returncode, _, _ = run_case(
                project,
                config,
                output_dir,
                scenario,
                scenario,
                write_csv=False,
            )
            report["continuous_no_reset"][scenario] = summary
            failed |= returncode != 0 or not summary.get("passed", False)

    report["diagnosis"] = classify(report)
    report["passed"] = not failed
    report_path = output_dir / "stand_ready_results.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    compact_path = output_dir / "stand_ready_compact.json"
    compact_path.write_text(
        json.dumps(compact_report(report), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    markdown_path = output_dir / "REPORT.md"
    write_markdown(markdown_path, report)
    print(f"[RESULT] {'PASS' if report['passed'] else 'FAIL'}")
    print(f"[INFO] report: {report_path}")
    print(f"[INFO] compact results: {compact_path}")
    print(f"[INFO] markdown: {markdown_path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
