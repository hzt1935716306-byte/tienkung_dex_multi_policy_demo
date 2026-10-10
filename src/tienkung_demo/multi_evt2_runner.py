from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .command_bus import CommandReader, StatusWriter
from .motion_evt2 import MotionEvt2Policy
from .motion_evt2_runner import FootContactMonitor
from .multi_evt2_controller import CommandStatus, ControllerState, MultiEvt2Controller
from .robot import build_joint_map
from .stand_diagnostics import STAND_DIAGNOSTIC_COLUMNS, StandDiagnostics
from .transition_controller import ConstrainedPDController, copy_target
from .walkamp import WalkAmpPolicy, load_walkamp_config
from .walkamp_runner import CsvStateLogger, quaternion_to_rpy, setup_initial_pose


EXTRA_LOG_COLUMNS = [
    "transition_alpha",
    "left_contact_count",
    "right_contact_count",
    "left_normal_force",
    "right_normal_force",
    "maximum_target_rate",
    "requested_target_delta",
    "applied_target_delta",
    "maximum_torque_rate",
    "torque_saturation_fraction",
    "torque_slew_limited_fraction",
    "alpha_legs",
    "alpha_waist",
    "alpha_arms",
    "raw_torque_scale",
    "target_torque_scale",
]


def entry_joint_log_columns(joint_names: list[str]) -> list[str]:
    columns: list[str] = []
    for name in joint_names:
        columns.extend(
            [
                f"{name}.raw_target",
                f"{name}.raw_kp",
                f"{name}.raw_kd",
                f"{name}.raw_feedforward",
                f"{name}.raw_effort",
                f"{name}.target_kp",
                f"{name}.target_kd",
                f"{name}.target_feedforward",
            ]
        )
    return columns


def empty_entry_metrics() -> dict[str, Any]:
    def stage() -> dict[str, Any]:
        return {
            "samples": 0,
            "maximum_abs_roll_pitch": 0.0,
            "maximum_xy_speed": 0.0,
            "maximum_angular_speed": 0.0,
            "maximum_torque_rate": 0.0,
            "groups": {
                group: {
                    "maximum_raw_target_delta": 0.0,
                    "maximum_applied_target_delta": 0.0,
                    "maximum_kp_delta": 0.0,
                    "maximum_kd_delta": 0.0,
                    "maximum_feedforward_delta": 0.0,
                    "maximum_tracking_error": 0.0,
                    "maximum_joint_speed": 0.0,
                    "maximum_abs_torque": 0.0,
                    "maximum_torque_delta": 0.0,
                }
                for group in ("legs", "waist", "arms")
            },
        }

    return {
        "PRE_ALIGN": stage(),
        "TRANSITION_IN": stage(),
        "boundary": {},
    }


def empty_exit_metrics() -> dict[str, Any]:
    def stage() -> dict[str, Any]:
        return {
            "samples": 0,
            "maximum_abs_roll_pitch": 0.0,
            "maximum_xy_speed": 0.0,
            "maximum_angular_speed": 0.0,
            "maximum_joint_speed": 0.0,
            "maximum_joint_acceleration": 0.0,
            "maximum_torque_rate": 0.0,
            "maximum_abs_torque": 0.0,
            "single_support_samples": 0,
            "single_support_events": 0,
            "no_support_samples": 0,
            "foot_contact_transitions": 0,
            "groups": {
                group: {
                    "maximum_raw_target_delta": 0.0,
                    "maximum_applied_target_delta": 0.0,
                    "maximum_kp_delta": 0.0,
                    "maximum_kd_delta": 0.0,
                    "maximum_feedforward_delta": 0.0,
                    "maximum_effort_delta": 0.0,
                    "maximum_tracking_error": 0.0,
                    "maximum_joint_speed": 0.0,
                    "maximum_joint_acceleration": 0.0,
                    "maximum_abs_torque": 0.0,
                    "maximum_torque_delta": 0.0,
                }
                for group in ("legs", "waist", "arms")
            },
        }

    return {
        "WAIT": stage(),
        "TRANSITION_OUT": stage(),
        "RECOVER": stage(),
        "boundary": {},
    }


def _resolve_path(value: str, config_dir: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return str(path.resolve())


def load_multi_evt2_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a JSON object: {config_path}")
    config["_path"] = str(config_path)
    config_dir = config_path.parent
    config["model"] = _resolve_path(config["model"], config_dir)
    config["walkamp_config"] = _resolve_path(config["walkamp_config"], config_dir)
    if not Path(config["model"]).is_file():
        raise FileNotFoundError(config["model"])
    if not Path(config["walkamp_config"]).is_file():
        raise FileNotFoundError(config["walkamp_config"])
    motions = config.get("motions", {})
    if set(motions) != {"a", "b"}:
        raise ValueError("multi_evt2 config must define exactly actions 'a' and 'b'")
    for motion in motions.values():
        motion["path"] = _resolve_path(motion["path"], config_dir)
        if not Path(motion["path"]).is_file():
            raise FileNotFoundError(motion["path"])
    return config


class ScheduledCommands:
    def __init__(self, config: dict[str, Any], control_dt: float) -> None:
        self.duration_seconds = float(config["duration_seconds"])
        self.events = [
            (int(round(float(item["time"]) / control_dt)), str(item["command"]))
            for item in config.get("events", [])
        ]
        self.next_event = 0
        self.walk_commands = [
            (
                int(round(float(item["time"]) / control_dt)),
                np.asarray(item["command"], dtype=np.float32),
            )
            for item in config.get("walk_commands", [])
        ]
        for _, command in self.walk_commands:
            if command.shape != (3,):
                raise ValueError("walk command must contain [vx, vy, yaw]")
        self.next_walk_command = 0
        self.perturbations = [
            (int(round(float(item["time"]) / control_dt)), item)
            for item in config.get("perturbations", [])
        ]
        self.next_perturbation = 0

    def commands(self, step: int) -> list[str]:
        values: list[str] = []
        while self.next_event < len(self.events) and step >= self.events[self.next_event][0]:
            values.append(self.events[self.next_event][1])
            self.next_event += 1
        return values

    def apply_walk_commands(
        self,
        step: int,
        controller: MultiEvt2Controller,
    ) -> None:
        while (
            self.next_walk_command < len(self.walk_commands)
            and step >= self.walk_commands[self.next_walk_command][0]
        ):
            _, command = self.walk_commands[self.next_walk_command]
            if controller.busy:
                raise RuntimeError(
                    "scheduled WALKAMP command reached while an action is active"
                )
            controller.set_walk_command(command, "scenario")
            print(
                "[SCENARIO] WALKAMP command "
                f"vx={command[0]:.2f} vy={command[1]:.2f} yaw={command[2]:.2f}"
            )
            self.next_walk_command += 1

    def apply_perturbations(self, step: int, data) -> None:
        while (
            self.next_perturbation < len(self.perturbations)
            and step >= self.perturbations[self.next_perturbation][0]
        ):
            _, values = self.perturbations[self.next_perturbation]
            velocity = np.asarray(values.get("delta_world_velocity", [0.0, 0.0]), dtype=np.float64)
            if velocity.shape != (2,):
                raise ValueError("delta_world_velocity must contain [vx, vy]")
            data.qvel[0:2] += velocity
            data.qvel[4] += float(values.get("delta_pitch_angular_velocity", 0.0))
            print(
                f"[PERTURB] step={step} dv=({velocity[0]:.3f},{velocity[1]:.3f}) "
                f"dwy={float(values.get('delta_pitch_angular_velocity', 0.0)):.3f}"
            )
            self.next_perturbation += 1


class KeyboardControl:
    def __init__(self, controller: MultiEvt2Controller, config: dict[str, Any]) -> None:
        self.controller = controller
        self.linear_step = float(config.get("linear_step", 0.05))
        self.lateral_step = float(config.get("lateral_step", 0.05))
        self.yaw_step = float(config.get("yaw_step", 0.1))
        self.quit_requested = False

    def callback(self, keycode: int) -> None:
        key = chr(keycode).upper() if 0 <= keycode < 256 else ""
        if key == "W":
            self.controller.adjust_walk_command(0, self.linear_step)
        elif key == "S":
            self.controller.adjust_walk_command(0, -self.linear_step)
        elif key == "A":
            self.controller.adjust_walk_command(1, self.lateral_step)
        elif key == "D":
            self.controller.adjust_walk_command(1, -self.lateral_step)
        elif key == "Q":
            self.controller.adjust_walk_command(2, self.yaw_step)
        elif key == "E":
            self.controller.adjust_walk_command(2, -self.yaw_step)
        elif key == "J":
            self.controller.request_command("a", "keyboard")
        elif key == "K":
            self.controller.request_command("b", "keyboard")
        elif key == "R":
            self.controller.request_command("r", "keyboard")
        elif keycode == ord(" "):
            self.controller.stop_walk()
        elif keycode == 256:
            self.quit_requested = True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Command-driven WALKAMP + BeyondMimic EVT2 demo")
    parser.add_argument("--config", default="configs/multi_evt2.json")
    parser.add_argument("--scenario", default="manual", help="manual or a scenario from the JSON config")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--headless-realtime",
        action="store_true",
        help="Pace a headless run in wall-clock time so external commands can arrive",
    )
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--command-file", default="")
    parser.add_argument("--status-file", default="")
    parser.add_argument("--transition-in", type=float, default=None)
    parser.add_argument("--transition-out", type=float, default=None)
    parser.add_argument(
        "--exit-mode",
        choices=("legacy", "continuous"),
        default=None,
    )
    parser.add_argument(
        "--exit-history",
        choices=("repeated", "measured"),
        default=None,
    )
    parser.add_argument(
        "--exit-handoff-target",
        choices=("neutral", "preview"),
        default=None,
    )
    parser.add_argument("--wave-exit-window", type=float, default=None)
    parser.add_argument(
        "--entry-mode",
        choices=("grouped", "full_body_continuous", "legacy"),
        default=None,
    )
    parser.add_argument("--pre-align", type=float, default=None)
    parser.add_argument("--reentry-phase", type=float, default=None)
    parser.add_argument("--log", default="")
    parser.add_argument("--summary", default="")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--debug-interval", type=int, default=None)
    return parser.parse_args(argv)


def validate_summary(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    limits = config["validation"]
    checks = {
        "finite": bool(summary["finite"]),
        "root_height": summary["minimum_root_z"] >= summary["fall_threshold"],
        "attitude": summary["maximum_abs_roll_pitch"]
        <= float(limits["maximum_abs_roll_pitch"]),
        "target_rate": summary["maximum_target_rate"]
        <= float(limits["maximum_target_rate"]),
        "torque_rate": summary["maximum_torque_rate"]
        <= float(limits["maximum_torque_rate"]),
        "torque_saturation": summary["torque_saturation_rate"]
        <= float(limits["maximum_torque_saturation_rate"]),
        "controller_not_failed": summary["controller"]["state"]
        != ControllerState.FAILED.value,
    }
    scenario = summary["scenario"]
    commands = summary["controller"]["commands"]
    if scenario == "idle":
        checks["walkamp_only"] = not commands and all(
            event["state"] == ControllerState.WALKAMP_IDLE.value
            for event in summary["controller"]["state_history"]
        )
    elif scenario == "abort":
        checks["abort_completed"] = any(
            record["command"] == "r" and record["status"] == CommandStatus.COMPLETED.value
            for record in commands
        )
        checks["returned_to_walkamp"] = (
            summary["controller"]["state"] == ControllerState.WALKAMP_IDLE.value
        )
    elif scenario != "manual":
        scenario_config = config["scenarios"][scenario]
        expected_statuses = scenario_config.get("expected_command_statuses")
        if expected_statuses is not None:
            actual_statuses = [record["status"] for record in commands]
            checks["expected_command_statuses"] = actual_statuses == list(
                expected_statuses
            )
            checks["returned_to_walkamp"] = summary["controller"]["state"] in (
                ControllerState.WALKAMP_IDLE.value,
                ControllerState.WALKAMP_MOVING.value,
            )
            return checks
        if bool(scenario_config.get("allow_external_actions", False)):
            action_records = [
                record for record in commands if record["action_key"] in ("a", "b")
            ]
            checks["external_action_completed"] = bool(action_records) and all(
                record["status"] == CommandStatus.COMPLETED.value
                for record in action_records
            )
            checks["returned_to_walkamp"] = summary["controller"]["state"] in (
                ControllerState.WALKAMP_IDLE.value,
                ControllerState.WALKAMP_MOVING.value,
            )
            return checks
        aliases = {"a": "a", "bow": "a", "b": "b", "wave": "b"}
        expected_actions = [
            aliases[str(event.get("command", "")).lower()]
            for event in scenario_config.get("events", [])
            if str(event.get("command", "")).lower() in aliases
        ]
        completed_actions = [
            record["action_key"]
            for record in commands
            if record["status"] == CommandStatus.COMPLETED.value
            and record["action_key"] in ("a", "b")
        ]
        checks["all_actions_completed"] = completed_actions == expected_actions
        checks["no_action_failed_or_rejected"] = not any(
            record["action_key"] in ("a", "b")
            and record["status"]
            in (CommandStatus.FAILED.value, CommandStatus.REJECTED.value)
            for record in commands
        )
        checks["returned_to_walkamp"] = (
            summary["controller"]["state"] == ControllerState.WALKAMP_IDLE.value
        )
    return checks


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    config = load_multi_evt2_config(args.config)
    walkamp_config = load_walkamp_config(config["walkamp_config"])
    if Path(config["model"]) != Path(walkamp_config["model"]):
        raise ValueError("Multi-policy and WALKAMP configs must use the same official EVT2 model")
    if args.scenario != "manual" and args.scenario not in config.get("scenarios", {}):
        raise ValueError(f"Unknown scenario {args.scenario!r}")
    if args.transition_in is not None:
        config["controller"]["transition_in_seconds"] = args.transition_in
    if args.pre_align is not None:
        config["controller"]["pre_align"]["duration_seconds"] = args.pre_align
    if args.entry_mode is not None:
        config["controller"]["entry"]["mode"] = args.entry_mode
    if args.transition_out is not None:
        config["controller"]["transition_out_seconds"] = args.transition_out
        for motion in config["motions"].values():
            motion["transition_out_seconds"] = args.transition_out
    if args.exit_mode is not None:
        config["controller"].setdefault("exit", {})["mode"] = args.exit_mode
    if args.exit_history is not None:
        config["controller"].setdefault("exit", {})[
            "history_mode"
        ] = args.exit_history
        for motion in config["motions"].values():
            motion["walkamp_reentry_history_mode"] = args.exit_history
    if args.exit_handoff_target is not None:
        config["controller"].setdefault("exit", {})[
            "handoff_target"
        ] = args.exit_handoff_target
    if args.wave_exit_window is not None:
        window = config["motions"]["b"]["exit_window"]
        window["enabled"] = args.wave_exit_window > 0.0
        window["seconds_before_end"] = max(args.wave_exit_window, 0.0)
    if args.reentry_phase is not None:
        config["controller"]["walkamp_reentry_phase_time"] = args.reentry_phase
        for motion in config["motions"].values():
            motion["walkamp_reentry_phase_time"] = args.reentry_phase

    import mujoco

    model = mujoco.MjModel.from_xml_path(config["model"])
    data = mujoco.MjData(model)
    sim = config["simulation"]
    sim_dt = float(sim["sim_dt"])
    control_dt = float(sim["control_dt"])
    substeps_float = control_dt / sim_dt
    substeps = int(round(substeps_float))
    if not math.isclose(substeps_float, substeps, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("simulation.control_dt must be an integer multiple of simulation.sim_dt")
    model.opt.timestep = sim_dt
    model.opt.iterations = int(sim.get("solver_iterations", 100))
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    if int(model.opt.disableactuator) != 11:
        raise ValueError("Official EVT2 actuator disable bitmask must remain 11")

    joint_map = build_joint_map(model)
    walkamp = WalkAmpPolicy(
        model,
        data,
        joint_map,
        walkamp_config["policy"],
        walkamp_config["walkamp"],
        control_dt,
    )
    setup_initial_pose(
        model,
        data,
        joint_map,
        walkamp,
        float(sim["initial_height"]),
        float(sim["ground_clearance"]),
    )
    scenario_values = (
        {} if args.scenario == "manual" else config["scenarios"][args.scenario]
    )
    initial_state = scenario_values.get("initial_state", {})
    initial_pitch = float(initial_state.get("pitch", 0.0))
    if initial_pitch:
        data.qpos[3:7] = np.array(
            [math.cos(initial_pitch / 2.0), 0.0, math.sin(initial_pitch / 2.0), 0.0],
            dtype=np.float64,
        )
    initial_velocity = np.asarray(
        initial_state.get("world_velocity", [0.0, 0.0]),
        dtype=np.float64,
    )
    if initial_velocity.shape != (2,):
        raise ValueError("initial_state.world_velocity must contain [vx, vy]")
    data.qvel[0:2] = initial_velocity
    data.qvel[4] = float(initial_state.get("pitch_angular_velocity", 0.0))
    if initial_state:
        mujoco.mj_forward(model, data)
        print(f"[INFO] applied initial recovery test state: {json.dumps(initial_state)}")
    hold_target = walkamp.neutral_target("motion_uncontrolled_hold")
    motions: dict[str, MotionEvt2Policy] = {}
    for key, motion_values in config["motions"].items():
        motion_config = dict(config.get("motion_control", {}))
        motion_config.update(motion_values)
        motion_config["control_mode"] = "policy"
        motions[key] = MotionEvt2Policy(
            model,
            data,
            joint_map,
            motion_config["path"],
            motion_config,
            hold_target,
        )
    controller = MultiEvt2Controller(
        data,
        joint_map,
        walkamp,
        motions,
        config["controller"],
        control_dt,
    )
    contact_monitor = FootContactMonitor(model)
    stand_diagnostics = StandDiagnostics(
        model,
        control_dt,
        config.get("stand_diagnostics", {}),
    )
    pd = ConstrainedPDController(
        joint_map,
        sim_dt,
        config["controller"]["constraints"]["maximum_torque_rate"],
    )
    keyboard = KeyboardControl(controller, config.get("keyboard", {}))
    command_path = args.command_file or config.get("command_file", "")
    command_reader = CommandReader(command_path)
    status_path = args.status_file or config.get("status_file", "")
    status_writer = StatusWriter(status_path)
    published_command_history = 0
    request_metadata: dict[str, dict[str, Any]] = {}

    schedule = (
        None
        if args.scenario == "manual"
        else ScheduledCommands(config["scenarios"][args.scenario], control_dt)
    )
    if args.steps is not None:
        max_steps = args.steps
    elif schedule is not None:
        max_steps = int(round(schedule.duration_seconds / control_dt))
    else:
        max_steps = int(sim.get("max_steps", 100000))

    stem = f"multi_evt2_{args.scenario}"
    log_path = Path(args.log).expanduser().resolve() if args.log else Path("artifacts") / f"{stem}.csv"
    summary_path = (
        Path(args.summary).expanduser().resolve()
        if args.summary
        else Path("artifacts") / f"{stem}_summary.json"
    )
    logger = (
        None
        if args.no_log
        else CsvStateLogger(
            log_path,
            joint_map.names,
            extra_columns=(
                EXTRA_LOG_COLUMNS
                + STAND_DIAGNOSTIC_COLUMNS
                + entry_joint_log_columns(joint_map.names)
            ),
        )
    )
    debug_interval = (
        int(sim.get("debug_interval", 25))
        if args.debug_interval is None
        else args.debug_interval
    )
    realtime = bool(sim.get("realtime", True)) and not args.no_realtime
    fall_threshold = float(sim.get("fall_threshold", 0.35))
    minimum_root_z = float(data.qpos[2])
    maximum_abs_roll_pitch = 0.0
    maximum_target_rate = 0.0
    maximum_requested_target_delta = 0.0
    maximum_applied_target_delta = 0.0
    maximum_torque_rate = 0.0
    maximum_abs_torque = 0.0
    saturation_sum = 0.0
    torque_samples = 0
    transition_out_maximum_target_rate = 0.0
    transition_out_maximum_requested_target_delta = 0.0
    transition_out_maximum_applied_target_delta = 0.0
    transition_out_maximum_torque_rate = 0.0
    transition_out_maximum_abs_torque = 0.0
    transition_out_saturation_sum = 0.0
    transition_out_samples = 0
    entry_metrics = empty_entry_metrics()
    exit_metrics = empty_exit_metrics()
    previous_joint_velocity = data.qvel[joint_map.qvel_adr].copy()
    previous_exit_support: tuple[bool, bool] | None = None
    finite = True
    executed_steps = 0

    print(f"[INFO] config: {config['_path']}")
    print(f"[INFO] model: {config['model']}")
    print("[INFO] default controller: WALKAMP; actions are command-triggered only")
    print(f"[INFO] command bus: {command_path or 'disabled'}")
    print(f"[INFO] status bus: {status_path or 'disabled'}")
    if not args.headless:
        print("[INFO] controls: W/S/A/D/Q/E walk, Space stop, J bow, K wave, R recover, Esc quit")

    def step_once(step: int) -> bool:
        nonlocal minimum_root_z, maximum_abs_roll_pitch, maximum_target_rate
        nonlocal maximum_requested_target_delta, maximum_applied_target_delta
        nonlocal maximum_torque_rate, maximum_abs_torque
        nonlocal saturation_sum, torque_samples, finite, executed_steps
        nonlocal transition_out_maximum_target_rate
        nonlocal transition_out_maximum_requested_target_delta
        nonlocal transition_out_maximum_applied_target_delta
        nonlocal transition_out_maximum_torque_rate, transition_out_maximum_abs_torque
        nonlocal transition_out_saturation_sum, transition_out_samples
        nonlocal previous_joint_velocity, previous_exit_support
        nonlocal published_command_history

        def publish_statuses() -> None:
            nonlocal published_command_history
            while published_command_history < len(controller.command_history):
                payload = dict(controller.command_history[published_command_history])
                metadata = request_metadata.get(str(payload.get("request_id", "")), {})
                status_writer.send(
                    {
                        **payload,
                        **metadata,
                        "controller_step": controller.global_step,
                        "simulation_time": controller.global_step * control_dt,
                    }
                )
                published_command_history += 1

        if schedule is not None:
            schedule.apply_walk_commands(step, controller)
            for command in schedule.commands(step):
                controller.request_command(command, f"scenario:{args.scenario}")
            schedule.apply_perturbations(step, data)
        for envelope in command_reader.read_records():
            request_metadata[envelope.request_id] = {
                "label": envelope.label,
                "request_issued_at": envelope.issued_at,
                "controller_received_at": time.time(),
            }
            if envelope.expired:
                controller.reject_command(
                    envelope.command,
                    envelope.source,
                    envelope.request_id,
                    "expired_command",
                )
            else:
                controller.request_command(
                    envelope.command,
                    envelope.source,
                    envelope.request_id,
                )
        publish_statuses()

        contacts = contact_monitor.measure(data)
        state_before_control = controller.state
        state_step_before_control = controller.state_step
        reentry_started_before_control = controller.reentry_started
        reentry_step_before_control = controller.reentry_step
        previous_target = copy_target(controller.last_target, "previous_applied")
        previous_torque = pd.previous_torque.copy()
        output = controller.step(contacts)
        publish_statuses()
        raw_target = controller.last_raw_target
        torque_result = None
        for _ in range(substeps):
            torque_result = pd.apply(data, output.target)
            mujoco.mj_step(model, data)
        assert torque_result is not None
        executed_steps = step + 1
        logged_contacts = contact_monitor.measure(data)
        diagnostic_values = stand_diagnostics.update(
            step,
            data,
            logged_contacts,
            data.qvel[joint_map.qvel_adr],
            torque_result.torque,
        )

        rpy = quaternion_to_rpy(data.sensor("orientation").data.copy())
        minimum_root_z = min(minimum_root_z, float(data.qpos[2]))
        maximum_abs_roll_pitch = max(
            maximum_abs_roll_pitch,
            abs(float(rpy[0])),
            abs(float(rpy[1])),
        )
        maximum_target_rate = max(
            maximum_target_rate,
            controller.target_limiter.last_maximum_rate,
        )
        maximum_requested_target_delta = max(
            maximum_requested_target_delta,
            controller.target_limiter.last_requested_maximum_delta,
        )
        maximum_applied_target_delta = max(
            maximum_applied_target_delta,
            controller.target_limiter.last_applied_maximum_delta,
        )
        maximum_torque_rate = max(maximum_torque_rate, torque_result.maximum_rate)
        maximum_abs_torque = max(maximum_abs_torque, float(np.max(np.abs(torque_result.torque))))
        saturation_sum += torque_result.saturation_fraction
        torque_samples += 1
        if state_before_control.value in ("PRE_ALIGN", "TRANSITION_IN"):
            stage = entry_metrics[state_before_control.value]
            stage["samples"] += 1
            stage["maximum_abs_roll_pitch"] = max(
                stage["maximum_abs_roll_pitch"],
                abs(float(rpy[0])),
                abs(float(rpy[1])),
            )
            stage["maximum_xy_speed"] = max(
                stage["maximum_xy_speed"],
                float(np.linalg.norm(data.qvel[0:2])),
            )
            stage["maximum_angular_speed"] = max(
                stage["maximum_angular_speed"],
                float(np.linalg.norm(data.sensor("angular-velocity").data.copy())),
            )
            stage["maximum_torque_rate"] = max(
                stage["maximum_torque_rate"],
                torque_result.maximum_rate,
            )
            actual_q = data.qpos[joint_map.qpos_adr]
            actual_qd = data.qvel[joint_map.qvel_adr]
            for group, indices in controller.group_indices.items():
                values = stage["groups"][group]
                values["maximum_raw_target_delta"] = max(
                    values["maximum_raw_target_delta"],
                    float(np.max(np.abs(raw_target.q[indices] - previous_target.q[indices]))),
                )
                values["maximum_applied_target_delta"] = max(
                    values["maximum_applied_target_delta"],
                    float(np.max(np.abs(output.target.q[indices] - previous_target.q[indices]))),
                )
                for field, metric in (
                    ("kp", "maximum_kp_delta"),
                    ("kd", "maximum_kd_delta"),
                    ("feedforward", "maximum_feedforward_delta"),
                ):
                    values[metric] = max(
                        values[metric],
                        float(
                            np.max(
                                np.abs(
                                    getattr(raw_target, field)[indices]
                                    - getattr(previous_target, field)[indices]
                                )
                            )
                        ),
                    )
                values["maximum_tracking_error"] = max(
                    values["maximum_tracking_error"],
                    float(np.max(np.abs(output.target.q[indices] - actual_q[indices]))),
                )
                values["maximum_joint_speed"] = max(
                    values["maximum_joint_speed"],
                    float(np.max(np.abs(actual_qd[indices]))),
                )
                values["maximum_abs_torque"] = max(
                    values["maximum_abs_torque"],
                    float(np.max(np.abs(torque_result.torque[indices]))),
                )
                values["maximum_torque_delta"] = max(
                    values["maximum_torque_delta"],
                    float(
                        np.max(
                            np.abs(
                                torque_result.torque[indices]
                                - previous_torque[indices]
                            )
                        )
                    ),
                )
            if (
                state_before_control == ControllerState.TRANSITION_IN
                and state_step_before_control == 0
            ):
                entry_metrics["boundary"] = {
                    group: {
                        "raw_target_delta": float(
                            np.max(
                                np.abs(
                                    raw_target.q[indices]
                                    - previous_target.q[indices]
                                )
                            )
                        ),
                        "applied_target_delta": float(
                            np.max(
                                np.abs(
                                    output.target.q[indices]
                                    - previous_target.q[indices]
                                )
                            )
                        ),
                        "kp_delta": float(
                            np.max(
                                np.abs(
                                    raw_target.kp[indices]
                                    - previous_target.kp[indices]
                                )
                            )
                        ),
                        "kd_delta": float(
                            np.max(
                                np.abs(
                                    raw_target.kd[indices]
                                    - previous_target.kd[indices]
                                )
                            )
                        ),
                        "feedforward_delta": float(
                            np.max(
                                np.abs(
                                    raw_target.feedforward[indices]
                                    - previous_target.feedforward[indices]
                                )
                            )
                        ),
                    }
                    for group, indices in controller.group_indices.items()
                }
        if state_before_control == ControllerState.TRANSITION_OUT:
            transition_out_maximum_target_rate = max(
                transition_out_maximum_target_rate,
                controller.target_limiter.last_maximum_rate,
            )
            transition_out_maximum_requested_target_delta = max(
                transition_out_maximum_requested_target_delta,
                controller.target_limiter.last_requested_maximum_delta,
            )
            transition_out_maximum_applied_target_delta = max(
                transition_out_maximum_applied_target_delta,
                controller.target_limiter.last_applied_maximum_delta,
            )
            transition_out_maximum_torque_rate = max(
                transition_out_maximum_torque_rate,
                torque_result.maximum_rate,
            )
            transition_out_maximum_abs_torque = max(
                transition_out_maximum_abs_torque,
                float(np.max(np.abs(torque_result.torque))),
            )
            transition_out_saturation_sum += torque_result.saturation_fraction
            transition_out_samples += 1
        exit_stage_name: str | None = None
        if state_before_control == ControllerState.TRANSITION_OUT:
            exit_stage_name = (
                "TRANSITION_OUT"
                if reentry_started_before_control
                else "WAIT"
            )
        elif state_before_control == ControllerState.RECOVER:
            exit_stage_name = "RECOVER"
        current_joint_velocity = data.qvel[joint_map.qvel_adr].copy()
        joint_acceleration = (
            current_joint_velocity - previous_joint_velocity
        ) / control_dt
        previous_joint_velocity = current_joint_velocity
        if exit_stage_name is not None:
            stage = exit_metrics[exit_stage_name]
            stage["samples"] += 1
            stage["maximum_abs_roll_pitch"] = max(
                stage["maximum_abs_roll_pitch"],
                abs(float(rpy[0])),
                abs(float(rpy[1])),
            )
            stage["maximum_xy_speed"] = max(
                stage["maximum_xy_speed"],
                float(np.linalg.norm(data.qvel[0:2])),
            )
            stage["maximum_angular_speed"] = max(
                stage["maximum_angular_speed"],
                float(np.linalg.norm(data.sensor("angular-velocity").data.copy())),
            )
            stage["maximum_joint_speed"] = max(
                stage["maximum_joint_speed"],
                float(np.max(np.abs(current_joint_velocity))),
            )
            stage["maximum_joint_acceleration"] = max(
                stage["maximum_joint_acceleration"],
                float(np.max(np.abs(joint_acceleration))),
            )
            stage["maximum_torque_rate"] = max(
                stage["maximum_torque_rate"],
                torque_result.maximum_rate,
            )
            stage["maximum_abs_torque"] = max(
                stage["maximum_abs_torque"],
                float(np.max(np.abs(torque_result.torque))),
            )
            support = (
                contacts["left_contact_count"] > 0.0,
                contacts["right_contact_count"] > 0.0,
            )
            if sum(support) == 1:
                stage["single_support_samples"] += 1
                if previous_exit_support is None or support != previous_exit_support:
                    stage["single_support_events"] += 1
            elif sum(support) == 0:
                stage["no_support_samples"] += 1
            if previous_exit_support is not None:
                stage["foot_contact_transitions"] += sum(
                    current != previous
                    for current, previous in zip(support, previous_exit_support)
                )
            previous_exit_support = support
            actual_q = data.qpos[joint_map.qpos_adr]
            for group, indices in controller.group_indices.items():
                values = stage["groups"][group]
                values["maximum_raw_target_delta"] = max(
                    values["maximum_raw_target_delta"],
                    float(
                        np.max(
                            np.abs(
                                raw_target.q[indices]
                                - previous_target.q[indices]
                            )
                        )
                    ),
                )
                values["maximum_applied_target_delta"] = max(
                    values["maximum_applied_target_delta"],
                    float(
                        np.max(
                            np.abs(
                                output.target.q[indices]
                                - previous_target.q[indices]
                            )
                        )
                    ),
                )
                for field, metric in (
                    ("kp", "maximum_kp_delta"),
                    ("kd", "maximum_kd_delta"),
                    ("feedforward", "maximum_feedforward_delta"),
                    ("effort", "maximum_effort_delta"),
                ):
                    values[metric] = max(
                        values[metric],
                        float(
                            np.max(
                                np.abs(
                                    getattr(raw_target, field)[indices]
                                    - getattr(previous_target, field)[indices]
                                )
                            )
                        ),
                    )
                values["maximum_tracking_error"] = max(
                    values["maximum_tracking_error"],
                    float(
                        np.max(
                            np.abs(output.target.q[indices] - actual_q[indices])
                        )
                    ),
                )
                values["maximum_joint_speed"] = max(
                    values["maximum_joint_speed"],
                    float(np.max(np.abs(current_joint_velocity[indices]))),
                )
                values["maximum_joint_acceleration"] = max(
                    values["maximum_joint_acceleration"],
                    float(np.max(np.abs(joint_acceleration[indices]))),
                )
                values["maximum_abs_torque"] = max(
                    values["maximum_abs_torque"],
                    float(np.max(np.abs(torque_result.torque[indices]))),
                )
                values["maximum_torque_delta"] = max(
                    values["maximum_torque_delta"],
                    float(
                        np.max(
                            np.abs(
                                torque_result.torque[indices]
                                - previous_torque[indices]
                            )
                        )
                    ),
                )
            if (
                exit_stage_name == "TRANSITION_OUT"
                and reentry_step_before_control == 0
            ):
                exit_metrics["boundary"] = {
                    group: {
                        "raw_target_delta": float(
                            np.max(
                                np.abs(
                                    raw_target.q[indices]
                                    - previous_target.q[indices]
                                )
                            )
                        ),
                        "applied_target_delta": float(
                            np.max(
                                np.abs(
                                    output.target.q[indices]
                                    - previous_target.q[indices]
                                )
                            )
                        ),
                        **{
                            f"{field}_delta": float(
                                np.max(
                                    np.abs(
                                        getattr(raw_target, field)[indices]
                                        - getattr(previous_target, field)[indices]
                                    )
                                )
                            )
                            for field in ("kp", "kd", "feedforward", "effort")
                        },
                    }
                    for group, indices in controller.group_indices.items()
                }
                exit_metrics["boundary"]["scalar"] = {
                    "torque_scale_delta": abs(
                        raw_target.torque_scale - previous_target.torque_scale
                    ),
                    "maximum_torque_delta": float(
                        np.max(
                            np.abs(torque_result.torque - previous_torque)
                        )
                    ),
                }
        else:
            previous_exit_support = None
        finite = bool(
            np.all(np.isfinite(data.qpos))
            and np.all(np.isfinite(data.qvel))
            and np.all(np.isfinite(output.target.q))
            and np.all(np.isfinite(torque_result.torque))
        )

        if logger is not None:
            entry_values: dict[str, float] = {
                "alpha_legs": controller.entry_group_alphas["legs"],
                "alpha_waist": controller.entry_group_alphas["waist"],
                "alpha_arms": controller.entry_group_alphas["arms"],
            }
            for index, name in enumerate(joint_map.names):
                entry_values.update(
                    {
                        f"{name}.raw_target": raw_target.q[index],
                        f"{name}.raw_kp": raw_target.kp[index],
                        f"{name}.raw_kd": raw_target.kd[index],
                        f"{name}.raw_feedforward": raw_target.feedforward[index],
                        f"{name}.raw_effort": raw_target.effort[index],
                        f"{name}.target_kp": output.target.kp[index],
                        f"{name}.target_kd": output.target.kd[index],
                        f"{name}.target_feedforward": output.target.feedforward[index],
                    }
                )
            logger.write(
                step,
                data,
                joint_map,
                state_before_control.value,
                walkamp.command,
                output.target,
                torque_result.torque,
                output.observation,
                output.action,
                {
                    "transition_alpha": output.transition_alpha,
                    **logged_contacts,
                    "maximum_target_rate": controller.target_limiter.last_maximum_rate,
                    "requested_target_delta": (
                        controller.target_limiter.last_requested_maximum_delta
                    ),
                    "applied_target_delta": controller.target_limiter.last_applied_maximum_delta,
                    "maximum_torque_rate": torque_result.maximum_rate,
                    "torque_saturation_fraction": torque_result.saturation_fraction,
                    "torque_slew_limited_fraction": torque_result.slew_limited_fraction,
                    "raw_torque_scale": raw_target.torque_scale,
                    "target_torque_scale": output.target.torque_scale,
                    **diagnostic_values,
                    **entry_values,
                },
            )
        if debug_interval > 0 and step % debug_interval == 0:
            print(
                f"[DEBUG] step={step} state={controller.state.value} "
                f"cmd=({walkamp.command[0]:.2f},{walkamp.command[1]:.2f},{walkamp.command[2]:.2f}) "
                f"rpy=({rpy[0]:.3f},{rpy[1]:.3f},{rpy[2]:.3f}) "
                f"root_z={data.qpos[2]:.3f} target_rate={controller.target_limiter.last_maximum_rate:.2f} "
                f"torque_rate={torque_result.maximum_rate:.1f} sat={torque_result.saturation_fraction:.3f}"
            )
        if not finite:
            controller.fail("non_finite_simulation")
            print("[ERROR] simulation contains NaN or Inf")
            return False
        if float(data.qpos[2]) < fall_threshold:
            controller.fail("fall_threshold")
            print(f"[ERROR] robot fell: root_z={data.qpos[2]:.3f} < {fall_threshold:.3f}")
            return False
        if controller.state == ControllerState.FAILED:
            print(f"[ERROR] controller entered FAILED: {controller.terminal_failure_reason}")
            return False
        if keyboard.quit_requested:
            return False
        return True

    try:
        if args.headless:
            for step in range(max_steps):
                started = time.perf_counter()
                if not step_once(step):
                    break
                if args.headless_realtime:
                    delay = control_dt - (time.perf_counter() - started)
                    if delay > 0.0:
                        time.sleep(delay)
        else:
            import mujoco.viewer

            with mujoco.viewer.launch_passive(model, data, key_callback=keyboard.callback) as viewer:
                for step in range(max_steps):
                    if not viewer.is_running() or keyboard.quit_requested:
                        break
                    started = time.perf_counter()
                    if not step_once(step):
                        viewer.sync()
                        break
                    viewer.sync()
                    if realtime:
                        delay = control_dt - (time.perf_counter() - started)
                        if delay > 0.0:
                            time.sleep(delay)
    finally:
        if logger is not None:
            logger.close()

    summary: dict[str, Any] = {
        "scenario": args.scenario,
        "control_dt": control_dt,
        "finite": finite,
        "executed_steps": executed_steps,
        "expected_steps": max_steps,
        "minimum_root_z": minimum_root_z,
        "fall_threshold": fall_threshold,
        "maximum_abs_roll_pitch": maximum_abs_roll_pitch,
        "maximum_target_rate": maximum_target_rate,
        "maximum_requested_target_delta": maximum_requested_target_delta,
        "maximum_applied_target_delta": maximum_applied_target_delta,
        "maximum_torque_rate": maximum_torque_rate,
        "maximum_abs_torque": maximum_abs_torque,
        "torque_saturation_rate": saturation_sum / max(torque_samples, 1),
        "transition_out_metrics": {
            "samples": transition_out_samples,
            "maximum_target_rate": transition_out_maximum_target_rate,
            "maximum_requested_target_delta": (
                transition_out_maximum_requested_target_delta
            ),
            "maximum_applied_target_delta": transition_out_maximum_applied_target_delta,
            "maximum_torque_rate": transition_out_maximum_torque_rate,
            "maximum_abs_torque": transition_out_maximum_abs_torque,
            "torque_saturation_rate": (
                transition_out_saturation_sum / max(transition_out_samples, 1)
            ),
        },
        "entry_metrics": entry_metrics,
        "exit_metrics": exit_metrics,
        "final_command": walkamp.command.tolist(),
        "final_root_x": float(data.qpos[0]),
        "final_root_y": float(data.qpos[1]),
        "controller": controller.summary(),
        "stand_diagnostics": stand_diagnostics.summary(),
        "measurement": scenario_values.get("measurement", {}),
        "model": config["model"],
        "walkamp_policy": walkamp_config["policy"],
    }
    summary["checks"] = validate_summary(summary, config)
    summary["passed"] = all(summary["checks"].values())
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"[RESULT] {'PASS' if summary['passed'] else 'FAIL'} {json.dumps(summary['checks'])}")
    print(f"[INFO] summary: {summary_path}")
    return summary


def main() -> None:
    try:
        summary = run()
    except KeyboardInterrupt:
        print("\n[INFO] stopped by user")
        return
    except Exception as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        raise
    if not summary["passed"]:
        raise SystemExit(1)
