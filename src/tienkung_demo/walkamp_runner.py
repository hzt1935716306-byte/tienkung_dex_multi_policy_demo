from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .policies import quat_normalize
from .robot import JointMap, build_joint_map
from .simulator import apply_pd, robot_min_geom_z
from .walkamp import WalkAmpPolicy, load_walkamp_config


def quaternion_to_rpy(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(quaternion)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_argument = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    pitch = math.asin(pitch_argument)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def wrapped_angle_difference(first: float, second: float) -> float:
    return (first - second + math.pi) % (2.0 * math.pi) - math.pi


def setup_initial_pose(
    model,
    data,
    joint_map: JointMap,
    policy: WalkAmpPolicy,
    height: float,
    clearance: float,
) -> None:
    import mujoco

    target = policy.neutral_target("initial_hold")
    data.qpos[0:3] = np.array([0.0, 0.0, height], dtype=np.float64)
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qpos[joint_map.qpos_adr] = target.q
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    minimum = robot_min_geom_z(model, data)
    if minimum < clearance:
        lift = clearance - minimum
        data.qpos[2] += lift
        mujoco.mj_forward(model, data)
        print(f"[INFO] auto-lifted official EVT2 root by {lift:.3f} m")
    print(f"[INFO] official EVT2 minimum robot geom z: {robot_min_geom_z(model, data):.3f} m")


class CsvStateLogger:
    def __init__(
        self,
        path: Path,
        joint_names: list[str],
        extra_columns: list[str] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.file = path.open("w", encoding="utf-8", newline="")
        self.writer = csv.writer(self.file)
        columns = [
            "step",
            "time",
            "mode",
            "command_vx",
            "command_vy",
            "command_yaw",
            "root_x",
            "root_y",
            "root_z",
            "root_qw",
            "root_qx",
            "root_qy",
            "root_qz",
            "roll",
            "pitch",
            "yaw",
            "world_vx",
            "world_vy",
            "world_vz",
            "body_vx",
            "body_vy",
            "body_vz",
            "body_wx",
            "body_wy",
            "body_wz",
            "observation_min",
            "observation_max",
            "action_min",
            "action_max",
        ]
        for name in joint_names:
            columns.extend(
                [
                    f"{name}.position",
                    f"{name}.velocity",
                    f"{name}.target",
                    f"{name}.torque",
                    f"{name}.torque_limit",
                ]
            )
        self.extra_columns = list(extra_columns or [])
        columns.extend(self.extra_columns)
        self.writer.writerow(columns)

    def write(
        self,
        step: int,
        data,
        joint_map: JointMap,
        mode: str,
        command: np.ndarray,
        target,
        torque: np.ndarray,
        observation: np.ndarray | None,
        action: np.ndarray | None,
        extra_values: dict[str, float] | None = None,
    ) -> None:
        quaternion = data.sensor("orientation").data.copy()
        rpy = quaternion_to_rpy(quaternion)
        body_velocity = data.sensor("linear-velocity").data.copy()
        body_angular_velocity = data.sensor("angular-velocity").data.copy()
        observation_min = math.nan if observation is None else float(np.min(observation))
        observation_max = math.nan if observation is None else float(np.max(observation))
        action_min = math.nan if action is None else float(np.min(action))
        action_max = math.nan if action is None else float(np.max(action))
        row: list[Any] = [
            step,
            float(data.time),
            mode,
            *command.tolist(),
            *data.qpos[0:3].tolist(),
            *quaternion.tolist(),
            *rpy.tolist(),
            *data.qvel[0:3].tolist(),
            *body_velocity.tolist(),
            *body_angular_velocity.tolist(),
            observation_min,
            observation_max,
            action_min,
            action_max,
        ]
        positions = data.qpos[joint_map.qpos_adr]
        velocities = data.qvel[joint_map.qvel_adr]
        for index in range(len(joint_map.names)):
            row.extend(
                [
                    positions[index],
                    velocities[index],
                    target.q[index],
                    torque[index],
                    target.effort[index],
                ]
            )
        values = extra_values or {}
        row.extend(values.get(name, math.nan) for name in self.extra_columns)
        self.writer.writerow(row)

    def close(self) -> None:
        self.file.flush()
        self.file.close()


class Scenario:
    def __init__(self, name: str, config: dict[str, Any], control_dt: float) -> None:
        self.name = name
        self.duration_seconds = float(config["duration_seconds"])
        self.events = [
            (int(round(float(item["time"]) / control_dt)), np.asarray(item["command"], dtype=np.float32))
            for item in config["commands"]
        ]
        self.next_event = 0

    def update(self, policy_step: int, policy: WalkAmpPolicy) -> None:
        while self.next_event < len(self.events) and policy_step >= self.events[self.next_event][0]:
            _, command = self.events[self.next_event]
            policy.set_command(command)
            print(
                f"[SCENARIO] {self.name}: vx={command[0]:.2f} "
                f"vy={command[1]:.2f} yaw={command[2]:.2f}"
            )
            self.next_event += 1


class InteractiveControl:
    def __init__(self, policy: WalkAmpPolicy, config: dict[str, Any]) -> None:
        self.policy = policy
        self.linear_step = float(config.get("linear_step", 0.05))
        self.lateral_step = float(config.get("lateral_step", 0.05))
        self.yaw_step = float(config.get("yaw_step", 0.1))
        self.quit_requested = False

    def callback(self, keycode: int) -> None:
        key = chr(keycode).upper() if 0 <= keycode < 256 else ""
        if key == "W":
            self.policy.adjust_command(0, self.linear_step)
        elif key == "S":
            self.policy.adjust_command(0, -self.linear_step)
        elif key == "A":
            self.policy.adjust_command(1, self.lateral_step)
        elif key == "D":
            self.policy.adjust_command(1, -self.lateral_step)
        elif key == "Q":
            self.policy.adjust_command(2, self.yaw_step)
        elif key == "E":
            self.policy.adjust_command(2, -self.yaw_step)
        elif keycode == ord(" "):
            self.policy.set_command([0.0, 0.0, 0.0])
            print("[INFO] WALKAMP command stopped")
        elif keycode == 256:
            self.quit_requested = True


def validate_summary(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    limits = config.get("validation", {})
    scenario = summary["scenario"]
    checks = {
        "finite": bool(summary["finite"]),
        "completed": bool(summary["completed"]),
        "root_height": summary["minimum_root_z"] >= summary["fall_threshold"],
        "attitude": summary["maximum_abs_roll_pitch"] <= float(
            limits.get("maximum_abs_roll_pitch", 1.0)
        ),
    }
    if scenario == "stand":
        checks["stand_xy_drift"] = summary["xy_displacement"] <= float(
            limits.get("maximum_stand_xy_drift", 0.5)
        )
    elif scenario == "low_forward":
        checks["forward_displacement"] = summary["x_displacement"] >= float(
            limits.get("minimum_forward_displacement", 0.05)
        )
    elif scenario == "turn":
        checks["turn_yaw"] = abs(summary["yaw_displacement"]) >= float(
            limits.get("minimum_turn_yaw", 0.05)
        )
    elif scenario == "stop":
        checks["stop_speed"] = summary["final_xy_speed"] <= float(
            limits.get("maximum_stop_speed", 0.5)
        )
    return checks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official 23-output WALKAMP policy on xSIM EVT2")
    parser.add_argument("--config", default="configs/walkamp_official.json")
    parser.add_argument("--scenario", default="manual", help="manual, stand, low_forward, stop, or turn")
    parser.add_argument("--headless", action="store_true", help="Run deterministic simulation without a viewer")
    parser.add_argument("--steps", type=int, default=None, help="Override the number of 100 Hz control steps")
    parser.add_argument("--log", default="", help="CSV state log path")
    parser.add_argument("--summary", default="", help="JSON summary path")
    parser.add_argument("--no-log", action="store_true", help="Disable CSV logging")
    parser.add_argument("--debug-interval", type=int, default=None)
    parser.add_argument("--no-realtime", action="store_true")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    config = load_walkamp_config(args.config)
    sim = config["simulation"]
    scenarios = config.get("scenarios", {})
    if args.scenario != "manual" and args.scenario not in scenarios:
        raise ValueError(f"Unknown scenario {args.scenario!r}; available: {sorted(scenarios)}")

    import mujoco

    model = mujoco.MjModel.from_xml_path(config["model"])
    data = mujoco.MjData(model)
    sim_dt = float(sim.get("sim_dt", 0.001))
    control_dt = float(sim.get("control_dt", 0.01))
    substeps_float = control_dt / sim_dt
    substeps = int(round(substeps_float))
    if not math.isclose(substeps_float, substeps, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("simulation.control_dt must be an integer multiple of simulation.sim_dt")
    model.opt.timestep = sim_dt
    model.opt.iterations = int(sim.get("solver_iterations", 100))
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    joint_map = build_joint_map(model)
    policy = WalkAmpPolicy(
        model,
        data,
        joint_map,
        config["policy"],
        config["walkamp"],
        control_dt,
    )
    setup_initial_pose(
        model,
        data,
        joint_map,
        policy,
        float(sim.get("initial_height", 1.0)),
        float(sim.get("ground_clearance", 0.015)),
    )

    initial_hold_steps = int(round(float(sim.get("initial_hold_seconds", 1.0)) / control_dt))
    scenario = Scenario(args.scenario, scenarios[args.scenario], control_dt) if args.scenario != "manual" else None
    if args.steps is not None:
        max_steps = args.steps
    elif scenario is not None:
        max_steps = initial_hold_steps + int(round(scenario.duration_seconds / control_dt))
    else:
        max_steps = int(sim.get("max_steps", 100000))

    artifact_stem = f"walkamp_{args.scenario}"
    log_path = Path(args.log).expanduser().resolve() if args.log else Path("artifacts") / f"{artifact_stem}.csv"
    summary_path = (
        Path(args.summary).expanduser().resolve()
        if args.summary
        else Path("artifacts") / f"{artifact_stem}_summary.json"
    )
    logger = None if args.no_log else CsvStateLogger(log_path, joint_map.names)
    debug_interval = (
        args.debug_interval
        if args.debug_interval is not None
        else int(sim.get("debug_interval", 25))
    )
    fall_threshold = float(sim.get("fall_threshold", 0.45))
    realtime = bool(sim.get("realtime", True)) and not args.no_realtime
    interactive = InteractiveControl(policy, config.get("keyboard", {}))

    initial_xy = data.qpos[0:2].copy()
    initial_yaw = quaternion_to_rpy(data.sensor("orientation").data.copy())[2]
    minimum_root_z = float(data.qpos[2])
    maximum_abs_roll_pitch = 0.0
    maximum_torque_ratio = 0.0
    finite = True
    completed = True
    policy_started = False
    executed_steps = 0

    print(f"[INFO] config: {config['_path']}")
    print(f"[INFO] model: {config['model']}")
    print(f"[INFO] policy: {config['policy']} (840 -> 23)")
    print(f"[INFO] joints: {len(joint_map.names)} (23 policy + 6 explicit hold)")
    print(f"[INFO] timing: sim={sim_dt:.3f}s control/policy={control_dt:.3f}s")
    print(f"[INFO] scenario: {args.scenario}; initial hold: {initial_hold_steps} steps")
    if not args.headless:
        print("[INFO] controls: W/S forward/back, A/D lateral, Q/E turn, Space stop, Esc quit")
    if logger is not None:
        print(f"[INFO] CSV log: {logger.path}")

    def step_once(step: int) -> bool:
        nonlocal policy_started, minimum_root_z, maximum_abs_roll_pitch
        nonlocal maximum_torque_ratio, finite, completed, executed_steps

        observation = None
        action = None
        if step < initial_hold_steps:
            mode = "initial_hold"
            target = policy.neutral_target(mode)
        else:
            mode = "policy"
            if not policy_started:
                policy.reset()
                policy_started = True
            policy_step = step - initial_hold_steps
            if scenario is not None:
                scenario.update(policy_step, policy)
            target, observation, action = policy.step()

        torque = np.zeros(len(joint_map.names), dtype=np.float64)
        for _ in range(substeps):
            torque = apply_pd(data, joint_map, target)
            mujoco.mj_step(model, data)
        executed_steps = step + 1

        quaternion = data.sensor("orientation").data.copy()
        rpy = quaternion_to_rpy(quaternion)
        minimum_root_z = min(minimum_root_z, float(data.qpos[2]))
        maximum_abs_roll_pitch = max(maximum_abs_roll_pitch, abs(float(rpy[0])), abs(float(rpy[1])))
        torque_ratio = np.max(np.abs(torque) / np.maximum(target.effort, 1.0e-9))
        maximum_torque_ratio = max(maximum_torque_ratio, float(torque_ratio))
        finite = bool(
            np.all(np.isfinite(data.qpos))
            and np.all(np.isfinite(data.qvel))
            and np.all(np.isfinite(torque))
        )

        if logger is not None:
            logger.write(
                step,
                data,
                joint_map,
                mode,
                policy.command,
                target,
                torque,
                observation,
                action,
            )
        if debug_interval > 0 and step % debug_interval == 0:
            observation_range = (
                "n/a" if observation is None else f"({observation.min():.3f},{observation.max():.3f})"
            )
            action_range = "n/a" if action is None else f"({action.min():.3f},{action.max():.3f})"
            print(
                f"[DEBUG] step={step} mode={mode} "
                f"cmd=({policy.command[0]:.2f},{policy.command[1]:.2f},{policy.command[2]:.2f}) "
                f"rpy=({rpy[0]:.3f},{rpy[1]:.3f},{rpy[2]:.3f}) "
                f"vel=({data.qvel[0]:.3f},{data.qvel[1]:.3f},{data.qvel[2]:.3f}) "
                f"target=({target.q.min():.3f},{target.q.max():.3f}) "
                f"obs={observation_range} act={action_range} "
                f"tau=({torque.min():.2f},{torque.max():.2f}) root_z={data.qpos[2]:.3f}"
            )
        if not finite:
            print("[ERROR] simulation contains NaN or Inf")
            completed = False
            return False
        if data.qpos[2] < fall_threshold:
            print(f"[ERROR] robot fell: root_z={data.qpos[2]:.3f} < {fall_threshold:.3f}")
            completed = False
            return False
        if interactive.quit_requested:
            completed = False
            return False
        return True

    try:
        if args.headless:
            for step in range(max_steps):
                if not step_once(step):
                    break
        else:
            import mujoco.viewer

            with mujoco.viewer.launch_passive(model, data, key_callback=interactive.callback) as viewer:
                for step in range(max_steps):
                    if not viewer.is_running():
                        completed = False
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

    final_rpy = quaternion_to_rpy(data.sensor("orientation").data.copy())
    displacement = data.qpos[0:2] - initial_xy
    completed = completed and executed_steps == max_steps
    summary: dict[str, Any] = {
        "scenario": args.scenario,
        "completed": completed,
        "finite": finite,
        "executed_steps": executed_steps,
        "expected_steps": max_steps,
        "control_dt": control_dt,
        "fall_threshold": fall_threshold,
        "minimum_root_z": minimum_root_z,
        "maximum_abs_roll_pitch": maximum_abs_roll_pitch,
        "maximum_torque_ratio": maximum_torque_ratio,
        "x_displacement": float(displacement[0]),
        "y_displacement": float(displacement[1]),
        "xy_displacement": float(np.linalg.norm(displacement)),
        "yaw_displacement": wrapped_angle_difference(float(final_rpy[2]), float(initial_yaw)),
        "final_xy_speed": float(np.linalg.norm(data.qvel[0:2])),
        "final_root_z": float(data.qpos[2]),
        "final_command": policy.command.tolist(),
        "model": config["model"],
        "policy": config["policy"],
    }
    if args.scenario == "manual":
        summary["checks"] = {"finite": finite}
        summary["passed"] = finite
    else:
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


if __name__ == "__main__":
    main()
