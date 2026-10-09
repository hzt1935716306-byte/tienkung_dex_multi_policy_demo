#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from analyze_smooth_exit import plot_exit_csv, summarize_exit


PHASES = (0.0, 0.10625, 0.2125, 0.31875, 0.425, 0.53125, 0.6375, 0.74375)
EXIT_DURATIONS = (0.2, 0.3, 0.4, 0.6)
WAVE_WINDOWS = (0.0, 0.4, 0.6, 0.8)


def run_case(
    project: Path,
    config: Path,
    output_dir: Path,
    name: str,
    scenario: str,
    extra_args: tuple[str, ...] = (),
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
        "--headless",
        "--no-realtime",
        "--debug-interval",
        "0",
        "--summary",
        str(summary_path),
        *extra_args,
    ]
    if csv_path is None:
        command.append("--no-log")
    else:
        command.extend(("--log", str(csv_path)))
    print(f"[EXIT SUITE] running {name}", flush=True)
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


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return ", ".join(_fmt(item) for item in value)
    return str(value)


def _physical_score(metrics: dict[str, Any]) -> float:
    if not metrics.get("passed") or metrics.get("success_rate", 0.0) < 1.0:
        return float("inf")

    def value(name: str) -> float:
        raw = metrics.get(name)
        return 0.0 if raw is None else abs(float(raw))

    return (
        4.0 * value("maximum_acquisition_xy_speed")
        + 2.0 * value("maximum_acquisition_abs_pitch_rate")
        + 0.5 * value("maximum_acquisition_joint_speed")
        + 2.0 * value("mean_planar_displacement")
        + 0.2 * value("mean_recovery_seconds")
        + 5.0 * value("acquisition_single_support")
    )


def reference_tail_metrics(
    policy_path: Path,
    handoff_step: int,
    final_step: int,
) -> dict[str, Any]:
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(policy_path),
        providers=["CPUExecutionProvider"],
    )
    output_names = [output.name for output in session.get_outputs()]
    frames: list[dict[str, np.ndarray]] = []
    zero_observation = np.zeros((1, 104), dtype=np.float32)
    for step in range(handoff_step, final_step + 1):
        outputs = session.run(
            output_names,
            {
                "obs": zero_observation,
                "time_step": np.array([[float(step)]], dtype=np.float32),
            },
        )
        frames.append(dict(zip(output_names, outputs)))
    joint_position = np.stack([frame["joint_pos"][0] for frame in frames])
    joint_velocity = np.stack([frame["joint_vel"][0] for frame in frames])
    pelvis_position = np.stack(
        [frame["body_pos_w"][0, 0] for frame in frames]
    )
    return {
        "handoff_step": handoff_step,
        "final_step": final_step,
        "remaining_steps": final_step - handoff_step,
        "maximum_joint_span": float(np.max(np.ptp(joint_position, axis=0))),
        "maximum_endpoint_joint_delta": float(
            np.max(np.abs(joint_position[-1] - joint_position[0]))
        ),
        "maximum_reference_joint_velocity": float(
            np.max(np.abs(joint_velocity))
        ),
        "pelvis_position_span": np.ptp(pelvis_position, axis=0).tolist(),
        "pelvis_endpoint_delta": (
            pelvis_position[-1] - pelvis_position[0]
        ).tolist(),
    }


def _table(lines: list[str], title: str, cases: dict[str, dict[str, Any]]) -> None:
    lines.extend(
        [
            f"## {title}",
            "",
            "| case | pass | history | phase | acquire xy | acquire pitch rate | "
            "acquire joint speed | recover | planar drift | recovery step events | "
            "single support | boundary q |",
            "| --- | :---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: |",
        ]
    )
    for name, values in cases.items():
        phases = values.get("selected_phase_times", [])
        lines.append(
            "| {name} | {passed} | {history} | {phase} | {xy} | {pitch_rate} | "
            "{joint_speed} | {recover} | {drift} | {steps} | {support} | "
            "{boundary} |".format(
                name=name,
                passed="PASS" if values.get("passed") else "FAIL",
                history=_fmt(values.get("history_mode")),
                phase=_fmt(phases[0] if phases else None),
                xy=_fmt(values.get("maximum_acquisition_xy_speed")),
                pitch_rate=_fmt(values.get("maximum_acquisition_abs_pitch_rate")),
                joint_speed=_fmt(values.get("maximum_acquisition_joint_speed")),
                recover=_fmt(values.get("mean_recovery_seconds")),
                drift=_fmt(values.get("mean_planar_displacement")),
                steps=_fmt(values.get("extra_recovery_step_events")),
                support=_fmt(values.get("acquisition_single_support")),
                boundary=_fmt(values.get("boundary_raw_target_jump")),
            )
        )
    lines.append("")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Smooth Exit Phase 3A.3 Test Report",
        "",
        f"Selected-regression result: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "A is the Phase 3A.2 legacy exit, B uses continuous exit with repeated "
        "history, and C uses continuous exit with the per-motion history mode.",
        "",
    ]
    _table(lines, "A/B/C comparison", report["comparison"])
    _table(lines, "Phase sweep", report["phase_sweep"])
    _table(lines, "Transition duration sweep", report["duration_sweep"])
    _table(lines, "Wave early-window sweep", report["wave_window_sweep"])
    _table(lines, "Perturbation tests", report["perturbations"])
    _table(lines, "Continuous-state tests", report["continuous_scenarios"])
    _table(lines, "Expected failure controls", report["failure_controls"])
    lines.extend(
        [
            "## Remaining reference tail at selected handoff",
            "",
            "| motion | remaining steps | maximum joint span | endpoint joint delta | "
            "maximum reference joint speed | pelvis xyz span |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for motion, values in report.get("reference_tail", {}).items():
        lines.append(
            "| {motion} | {steps} | {span} | {delta} | {speed} | {pelvis} |".format(
                motion=motion,
                steps=values["remaining_steps"],
                span=_fmt(values["maximum_joint_span"]),
                delta=_fmt(values["maximum_endpoint_joint_delta"]),
                speed=_fmt(values["maximum_reference_joint_velocity"]),
                pelvis=_fmt(values["pelvis_position_span"]),
            )
        )
    lines.append("")
    lines.extend(
        [
            "## Physical ranking",
            "",
            f"- Bow phase candidate: `{report.get('best_bow_phase')}`",
            f"- Wave phase candidate: `{report.get('best_wave_phase')}`",
            f"- Bow transition duration: `{report.get('best_bow_duration')}`",
            f"- Wave transition duration: `{report.get('best_wave_duration')}`",
            f"- Wave early-exit window: `{report.get('best_wave_window')}`",
            "",
            "The ranking combines acquisition velocity, pitch rate, joint speed, "
            "planar displacement, recovery time, and support state. It is a test "
            "selection aid, not a replacement for the safety gates.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3A.3 smooth-exit tests")
    parser.add_argument("--config", default="configs/multi_evt2.json")
    parser.add_argument("--output-dir", default="artifacts/smooth_exit_phase3a3")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run A/B/C plus short sequence regressions only",
    )
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    config = (project / args.config).resolve()
    output_dir = (project / args.output_dir).resolve()
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "config": str(config),
        "comparison": {},
        "phase_sweep": {},
        "duration_sweep": {},
        "wave_window_sweep": {},
        "perturbations": {},
        "continuous_scenarios": {},
        "failure_controls": {},
        "reference_tail": {},
        "plots": [],
    }
    selected_failed = False

    selected_summaries: dict[str, dict[str, Any]] = {}
    for scenario in ("bow", "wave"):
        baseline_phase = "0.2125" if scenario == "bow" else "0.0"
        baseline_tail_args = (
            ("--wave-exit-window", "0.0") if scenario == "wave" else ()
        )
        variants = {
            "a_legacy_repeated": (
                "--exit-mode",
                "legacy",
                "--exit-history",
                "repeated",
                "--exit-handoff-target",
                "neutral",
                "--reentry-phase",
                baseline_phase,
                "--transition-out",
                "0.6",
                *baseline_tail_args,
            ),
            "b_continuous_repeated": (
                "--exit-mode",
                "continuous",
                "--exit-history",
                "repeated",
                "--exit-handoff-target",
                "neutral",
                "--reentry-phase",
                baseline_phase,
                "--transition-out",
                "0.6",
                *baseline_tail_args,
            ),
            "c_continuous_per_motion": (
                "--exit-mode",
                "continuous",
                "--exit-handoff-target",
                "neutral",
            ),
        }
        for variant, extra_args in variants.items():
            name = f"{scenario}_{variant}"
            summary, returncode, csv_path = run_case(
                project,
                config,
                output_dir,
                name,
                scenario,
                extra_args,
                write_csv=True,
            )
            metrics = summarize_exit(summary)
            report["comparison"][name] = metrics
            if variant == "c_continuous_per_motion":
                selected_summaries[scenario] = summary
            if csv_path and csv_path.is_file():
                report["plots"].extend(plot_exit_csv(csv_path, plot_dir, name))
            selected_failed |= returncode != 0 or not metrics.get("passed", False)

    config_values = json.loads(config.read_text(encoding="utf-8"))
    for scenario, key in (("bow", "a"), ("wave", "b")):
        events = selected_summaries[scenario]["controller"]["events"]
        transition = next(
            event
            for event in events
            if event.get("event") == "transition_out_started"
        )
        motion_config = config_values["motions"][key]
        policy_path = (config.parent / motion_config["path"]).resolve()
        report["reference_tail"][scenario] = reference_tail_metrics(
            policy_path,
            int(transition["frozen_motion_step"]),
            int(motion_config["duration_steps"]) - 1,
        )

    short_sequences = ("bow_then_wave", "wave_then_bow")
    scenarios = short_sequences if args.quick else (
        *short_sequences,
        "bow_repeat20",
        "wave_repeat20",
        "alternate_repeat20",
    )
    for scenario in scenarios:
        summary, returncode, _ = run_case(
            project,
            config,
            output_dir,
            scenario,
            scenario,
        )
        metrics = summarize_exit(summary)
        report["continuous_scenarios"][scenario] = metrics
        selected_failed |= returncode != 0 or not metrics.get("passed", False)

    if not args.quick:
        for scenario, history in (("bow", "measured"), ("wave", "repeated")):
            for phase in PHASES:
                name = f"{scenario}_phase_{phase:g}"
                summary, _, _ = run_case(
                    project,
                    config,
                    output_dir,
                    name,
                    scenario,
                    (
                        "--exit-mode",
                        "continuous",
                        "--exit-history",
                        history,
                        "--reentry-phase",
                        str(phase),
                    ),
                )
                report["phase_sweep"][name] = summarize_exit(summary)

        for scenario in ("bow", "wave"):
            for duration in EXIT_DURATIONS:
                name = f"{scenario}_transition_{duration:g}"
                summary, _, _ = run_case(
                    project,
                    config,
                    output_dir,
                    name,
                    scenario,
                    ("--transition-out", str(duration)),
                )
                report["duration_sweep"][name] = summarize_exit(summary)

        for window in WAVE_WINDOWS:
            name = f"wave_window_{window:g}"
            summary, _, _ = run_case(
                project,
                config,
                output_dir,
                name,
                "wave",
                ("--wave-exit-window", str(window)),
            )
            report["wave_window_sweep"][name] = summarize_exit(summary)

        for scenario in (
            "bow_initial_forward",
            "bow_initial_backward",
            "bow_tail_perturbed",
            "wave_initial_forward",
            "wave_initial_backward",
            "wave_tail_perturbed",
        ):
            summary, returncode, _ = run_case(
                project,
                config,
                output_dir,
                scenario,
                scenario,
            )
            metrics = summarize_exit(summary)
            report["perturbations"][scenario] = metrics
            selected_failed |= returncode != 0 or not metrics.get("passed", False)

        summary, _, _ = run_case(
            project,
            config,
            output_dir,
            "bow_preview_hard_gate",
            "bow",
            ("--exit-handoff-target", "preview"),
        )
        report["failure_controls"]["bow_preview_hard_gate"] = summarize_exit(summary)

    bow_phases = {
        name: values
        for name, values in report["phase_sweep"].items()
        if name.startswith("bow_")
    }
    wave_phases = {
        name: values
        for name, values in report["phase_sweep"].items()
        if name.startswith("wave_")
    }
    if bow_phases:
        report["best_bow_phase"] = min(
            bow_phases,
            key=lambda name: _physical_score(bow_phases[name]),
        )
    if wave_phases:
        report["best_wave_phase"] = min(
            wave_phases,
            key=lambda name: _physical_score(wave_phases[name]),
        )
    bow_durations = {
        name: values
        for name, values in report["duration_sweep"].items()
        if name.startswith("bow_")
    }
    wave_durations = {
        name: values
        for name, values in report["duration_sweep"].items()
        if name.startswith("wave_")
    }
    if bow_durations:
        report["best_bow_duration"] = min(
            bow_durations,
            key=lambda name: _physical_score(bow_durations[name]),
        )
    if wave_durations:
        report["best_wave_duration"] = min(
            wave_durations,
            key=lambda name: _physical_score(wave_durations[name]),
        )
    if report["wave_window_sweep"]:
        report["best_wave_window"] = min(
            report["wave_window_sweep"],
            key=lambda name: _physical_score(report["wave_window_sweep"][name]),
        )
    report["passed"] = not selected_failed
    summary_path = output_dir / "suite_summary.json"
    markdown_path = output_dir / "report.md"
    summary_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(markdown_path, report)
    print(f"[EXIT SUITE] {'PASS' if report['passed'] else 'FAIL'}: {markdown_path}")
    if selected_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
