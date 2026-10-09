from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .command_bus import CommandReader
from .motion_evt2 import MotionEvt2Policy
from .motion_evt2_runner import FootContactMonitor
from .multi_evt2_controller import CommandStatus, ControllerState, MultiEvt2Controller
from .robot import build_joint_map
from .transition_controller import ConstrainedPDController
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
]


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
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--command-file", default="")
    parser.add_argument("--transition-in", type=float, default=None)
    parser.add_argument("--transition-out", type=float, default=None)
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
    elif scenario == "wave" or scenario.startswith("bow"):
        action_key = "b" if scenario == "wave" else "a"
        expected = sum(
            str(event.get("command", "")).lower() in (
                action_key,
                "wave" if action_key == "b" else "bow",
            )
            for event in config["scenarios"][scenario].get("events", [])
        )
        checks["action_completed"] = any(
            record["status"] == CommandStatus.COMPLETED.value
            and record["action_key"] == action_key
            for record in commands
        )
        checks["all_actions_completed"] = sum(
            record["status"] == CommandStatus.COMPLETED.value
            and record["action_key"] == action_key
            for record in commands
        ) == expected
        checks["returned_to_walkamp"] = (
            summary["controller"]["state"] == ControllerState.WALKAMP_IDLE.value
        )
    elif scenario == "abort":
        checks["abort_completed"] = any(
            record["command"] == "r" and record["status"] == CommandStatus.COMPLETED.value
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
    if args.transition_out is not None:
        config["controller"]["transition_out_seconds"] = args.transition_out
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
    pd = ConstrainedPDController(
        joint_map,
        sim_dt,
        config["controller"]["constraints"]["maximum_torque_rate"],
    )
    keyboard = KeyboardControl(controller, config.get("keyboard", {}))
    command_path = args.command_file or config.get("command_file", "")
    command_reader = CommandReader(command_path)

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
        else CsvStateLogger(log_path, joint_map.names, extra_columns=EXTRA_LOG_COLUMNS)
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
    finite = True
    executed_steps = 0

    print(f"[INFO] config: {config['_path']}")
    print(f"[INFO] model: {config['model']}")
    print("[INFO] default controller: WALKAMP; actions are command-triggered only")
    print(f"[INFO] command bus: {command_path or 'disabled'}")
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

        if schedule is not None:
            for command in schedule.commands(step):
                controller.request_command(command, f"scenario:{args.scenario}")
            schedule.apply_perturbations(step, data)
        for command, source in command_reader.read():
            controller.request_command(command, source)

        contacts = contact_monitor.measure(data)
        state_before_control = controller.state
        output = controller.step(contacts)
        torque_result = None
        for _ in range(substeps):
            torque_result = pd.apply(data, output.target)
            mujoco.mj_step(model, data)
        assert torque_result is not None
        executed_steps = step + 1

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
        finite = bool(
            np.all(np.isfinite(data.qpos))
            and np.all(np.isfinite(data.qvel))
            and np.all(np.isfinite(output.target.q))
            and np.all(np.isfinite(torque_result.torque))
        )

        if logger is not None:
            logger.write(
                step,
                data,
                joint_map,
                controller.state.value,
                walkamp.command,
                output.target,
                torque_result.torque,
                output.observation,
                output.action,
                {
                    "transition_alpha": output.transition_alpha,
                    **contacts,
                    "maximum_target_rate": controller.target_limiter.last_maximum_rate,
                    "requested_target_delta": (
                        controller.target_limiter.last_requested_maximum_delta
                    ),
                    "applied_target_delta": controller.target_limiter.last_applied_maximum_delta,
                    "maximum_torque_rate": torque_result.maximum_rate,
                    "torque_saturation_fraction": torque_result.saturation_fraction,
                    "torque_slew_limited_fraction": torque_result.slew_limited_fraction,
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
                if not step_once(step):
                    break
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
        "final_command": walkamp.command.tolist(),
        "final_root_x": float(data.qpos[0]),
        "final_root_y": float(data.qpos[1]),
        "controller": controller.summary(),
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
