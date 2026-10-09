from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .motion_evt2 import MotionEvt2Policy, load_motion_evt2_config
from .robot import JointMap, build_joint_map
from .simulator import apply_pd, robot_min_geom_z
from .walkamp import WalkAmpPolicy, load_walkamp_config
from .walkamp_runner import CsvStateLogger, quaternion_to_rpy


MOTION_LOG_COLUMNS = [
    "left_contact_count",
    "right_contact_count",
    "left_normal_force",
    "right_normal_force",
    "tracking_error_mean",
    "tracking_error_max",
    "torque_saturation_fraction",
    "target_clipped_fraction",
    "feedforward_min",
    "feedforward_max",
    "feedforward_nonzero_fraction",
]


class FootContactMonitor:
    def __init__(self, model) -> None:
        import mujoco

        self.model = model
        self.ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.left_ids: set[int] = set()
        self.right_ids: set[int] = set()
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name.startswith("foot_left_"):
                self.left_ids.add(geom_id)
            elif name.startswith("foot_right_"):
                self.right_ids.add(geom_id)
        if self.ground_id < 0 or not self.left_ids or not self.right_ids:
            raise ValueError("Official EVT2 foot or ground collision geometries are missing")

    def measure(self, data) -> dict[str, float]:
        import mujoco

        values = {
            "left_contact_count": 0.0,
            "right_contact_count": 0.0,
            "left_normal_force": 0.0,
            "right_normal_force": 0.0,
        }
        contact_force = np.zeros(6, dtype=np.float64)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.ground_id not in pair:
                continue
            side = None
            if pair & self.left_ids:
                side = "left"
            elif pair & self.right_ids:
                side = "right"
            if side is None:
                continue
            contact_force[:] = 0.0
            mujoco.mj_contactForce(self.model, data, contact_index, contact_force)
            values[f"{side}_contact_count"] += 1.0
            values[f"{side}_normal_force"] += abs(float(contact_force[0]))
        return values


def ensure_episode_ground_clearance(model, data, clearance: float) -> float:
    import mujoco

    minimum = robot_min_geom_z(model, data)
    if minimum < clearance:
        lift = clearance - minimum
        data.qpos[2] += lift
        mujoco.mj_forward(model, data)
        print(f"[INFO] lifted reference-start root by {lift:.3f} m")
    minimum = robot_min_geom_z(model, data)
    print(f"[INFO] reference-start minimum robot geom z: {minimum:.3f} m")
    return minimum


def validate_summary(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    limits = config.get("validation", {})
    return {
        "finite": bool(summary["finite"]),
        "full_trajectory": bool(summary["full_trajectory_completed"]),
        "root_height": summary["minimum_root_z"] >= summary["fall_threshold"],
        "attitude": summary["maximum_abs_roll_pitch"]
        <= float(limits.get("maximum_abs_roll_pitch", 1.2)),
        "tracking": summary["tracking_error_mean"]
        <= float(limits.get("maximum_mean_tracking_error", 0.6)),
        "torque_saturation": summary["torque_saturation_rate"]
        <= float(limits.get("maximum_torque_saturation_rate", 0.75)),
        "foot_contact": summary["any_foot_contact_ratio"]
        >= float(limits.get("minimum_foot_contact_ratio", 0.5)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 19-output BeyondMimic motion on the official 29-joint EVT2 model"
    )
    parser.add_argument("--config", default="configs/motion_evt2.json")
    parser.add_argument("--motion", choices=("a", "b"), required=True)
    parser.add_argument("--mode", choices=("reference", "policy"), default="policy")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--steps", type=int, default=None, help="Diagnostic truncation of the trajectory")
    parser.add_argument("--log", default="")
    parser.add_argument("--summary", default="")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--debug-interval", type=int, default=None)
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    config = load_motion_evt2_config(args.config)
    walkamp_config = load_walkamp_config(config["walkamp_config"])
    if Path(config["model"]) != Path(walkamp_config["model"]):
        raise ValueError("Motion EVT2 and WALKAMP hold configuration must use the same MuJoCo model")

    import mujoco

    model = mujoco.MjModel.from_xml_path(config["model"])
    data = mujoco.MjData(model)
    sim = config["simulation"]
    sim_dt = float(sim.get("sim_dt", 0.001))
    control_dt = float(sim.get("control_dt", 0.01))
    substeps_float = control_dt / sim_dt
    substeps = int(round(substeps_float))
    if not math.isclose(substeps_float, substeps, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("simulation.control_dt must be an integer multiple of simulation.sim_dt")
    model.opt.timestep = sim_dt
    model.opt.iterations = int(sim.get("solver_iterations", 100))
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    if int(model.opt.disableactuator) != 11:
        raise ValueError(
            "Official EVT2 must retain actuatorgroupdisable='0 1 3' (disableactuator bitmask 11)"
        )

    joint_map = build_joint_map(model)
    walkamp = WalkAmpPolicy(
        model,
        data,
        joint_map,
        walkamp_config["policy"],
        walkamp_config["walkamp"],
        control_dt,
    )
    hold_target = walkamp.neutral_target("motion_evt2_hold")
    motion_config = dict(config.get("motion_control", {}))
    motion_config.update(config["motions"][args.motion])
    motion_config["control_mode"] = args.mode
    motion = MotionEvt2Policy(
        model,
        data,
        joint_map,
        motion_config["path"],
        motion_config,
        hold_target,
    )
    motion.reset_episode_to_reference()
    ensure_episode_ground_clearance(
        model,
        data,
        float(sim.get("ground_clearance", 0.015)),
    )
    motion.start()

    duration_steps = motion.duration_steps
    max_steps = duration_steps if args.steps is None else min(max(0, args.steps), duration_steps)
    artifact_stem = f"motion_evt2_{args.motion}_{args.mode}"
    log_path = Path(args.log).expanduser().resolve() if args.log else Path("artifacts") / f"{artifact_stem}.csv"
    summary_path = (
        Path(args.summary).expanduser().resolve()
        if args.summary
        else Path("artifacts") / f"{artifact_stem}_summary.json"
    )
    logger = (
        None
        if args.no_log
        else CsvStateLogger(log_path, joint_map.names, extra_columns=MOTION_LOG_COLUMNS)
    )
    contact_monitor = FootContactMonitor(model)
    debug_interval = (
        int(sim.get("debug_interval", 25))
        if args.debug_interval is None
        else args.debug_interval
    )
    fall_threshold = float(sim.get("fall_threshold", 0.35))
    realtime = bool(sim.get("realtime", True)) and not args.no_realtime
    quit_requested = False

    def key_callback(keycode: int) -> None:
        nonlocal quit_requested
        if keycode == 256:
            quit_requested = True

    controlled_indices = motion.model_indices
    initial_root = data.qpos[0:3].copy()
    minimum_root_z = float(data.qpos[2])
    maximum_abs_roll_pitch = 0.0
    tracking_error_sum = 0.0
    tracking_error_samples = 0
    tracking_error_max = 0.0
    saturated_samples = 0
    torque_samples = 0
    saturated_by_joint = np.zeros(len(joint_map.names), dtype=np.int64)
    torque_samples_by_joint = np.zeros(len(joint_map.names), dtype=np.int64)
    clipped_target_samples = 0
    target_samples = 0
    clipped_by_joint = np.zeros(len(joint_map.names), dtype=np.int64)
    maximum_abs_feedforward = 0.0
    left_contact_steps = 0
    right_contact_steps = 0
    any_contact_steps = 0
    double_contact_steps = 0
    left_force_sum = 0.0
    right_force_sum = 0.0
    left_force_max = 0.0
    right_force_max = 0.0
    finite = True
    completed = True
    executed_steps = 0

    print(f"[INFO] config: {config['_path']}")
    print(f"[INFO] model: {config['model']}")
    print(f"[INFO] motion: {motion.name} ({args.motion}), mode={args.mode}")
    print(f"[INFO] policy: {motion.path} (104 + time_step -> 19)")
    print(f"[INFO] joints: 29 = 19 motion + 10 WALKAMP-neutral hold")
    print(
        f"[INFO] control: gains={motion.gain_profile}, "
        f"preserve_clipped_effort={motion.preserve_clipped_target_effort}"
    )
    print(f"[INFO] actuator disable bitmask: {int(model.opt.disableactuator)}")
    print(f"[INFO] steps: {max_steps}/{duration_steps}, control_dt={control_dt:.3f}s")
    if logger is not None:
        print(f"[INFO] CSV log: {logger.path}")

    def step_once(step: int) -> bool:
        nonlocal minimum_root_z, maximum_abs_roll_pitch, tracking_error_sum
        nonlocal tracking_error_samples, tracking_error_max, saturated_samples
        nonlocal torque_samples, clipped_target_samples, target_samples
        nonlocal left_contact_steps, right_contact_steps, any_contact_steps
        nonlocal double_contact_steps, left_force_sum, right_force_sum
        nonlocal left_force_max, right_force_max, finite, completed, executed_steps
        nonlocal maximum_abs_feedforward

        target, observation, action = motion.step(step)
        step_saturated = 0
        step_torque_samples = 0
        torque = np.zeros(len(joint_map.names), dtype=np.float64)
        for _ in range(substeps):
            torque = apply_pd(data, joint_map, target)
            torque_limits = target.effort * target.torque_scale
            saturation_mask = np.abs(torque) >= torque_limits * (1.0 - 1.0e-6)
            step_saturated += int(np.count_nonzero(saturation_mask))
            step_torque_samples += len(joint_map.names)
            saturated_by_joint[:] += saturation_mask
            torque_samples_by_joint[:] += 1
            mujoco.mj_step(model, data)
        executed_steps = step + 1

        positions = data.qpos[joint_map.qpos_adr]
        tracking_error = np.abs(target.q[controlled_indices] - positions[controlled_indices])
        tracking_error_sum += float(np.sum(tracking_error))
        tracking_error_samples += tracking_error.size
        tracking_error_max = max(tracking_error_max, float(np.max(tracking_error)))
        saturated_samples += step_saturated
        torque_samples += step_torque_samples
        clipped_target_samples += int(np.count_nonzero(motion.last_target_clipped))
        target_samples += len(joint_map.names)
        clipped_by_joint[:] += motion.last_target_clipped
        maximum_abs_feedforward = max(
            maximum_abs_feedforward, float(np.max(np.abs(target.feedforward)))
        )

        contacts = contact_monitor.measure(data)
        left_contact = contacts["left_contact_count"] > 0.0
        right_contact = contacts["right_contact_count"] > 0.0
        left_contact_steps += int(left_contact)
        right_contact_steps += int(right_contact)
        any_contact_steps += int(left_contact or right_contact)
        double_contact_steps += int(left_contact and right_contact)
        left_force_sum += contacts["left_normal_force"]
        right_force_sum += contacts["right_normal_force"]
        left_force_max = max(left_force_max, contacts["left_normal_force"])
        right_force_max = max(right_force_max, contacts["right_normal_force"])

        quaternion = data.sensor("orientation").data.copy()
        rpy = quaternion_to_rpy(quaternion)
        minimum_root_z = min(minimum_root_z, float(data.qpos[2]))
        maximum_abs_roll_pitch = max(
            maximum_abs_roll_pitch,
            abs(float(rpy[0])),
            abs(float(rpy[1])),
        )
        finite = bool(
            np.all(np.isfinite(data.qpos))
            and np.all(np.isfinite(data.qvel))
            and np.all(np.isfinite(target.q))
            and np.all(np.isfinite(target.feedforward))
            and np.all(np.isfinite(torque))
        )
        step_tracking_mean = float(np.mean(tracking_error))
        step_saturation_fraction = step_saturated / max(step_torque_samples, 1)
        extra_values = {
            **contacts,
            "tracking_error_mean": step_tracking_mean,
            "tracking_error_max": float(np.max(tracking_error)),
            "torque_saturation_fraction": step_saturation_fraction,
            "target_clipped_fraction": float(np.mean(motion.last_target_clipped)),
            "feedforward_min": float(np.min(target.feedforward)),
            "feedforward_max": float(np.max(target.feedforward)),
            "feedforward_nonzero_fraction": float(
                np.mean(np.abs(target.feedforward) > 1.0e-12)
            ),
        }
        if logger is not None:
            logger.write(
                step,
                data,
                joint_map,
                args.mode,
                np.zeros(3, dtype=np.float32),
                target,
                torque,
                observation,
                action,
                extra_values,
            )
        if debug_interval > 0 and step % debug_interval == 0:
            print(
                f"[DEBUG] step={step} mode={args.mode} root_z={data.qpos[2]:.3f} "
                f"rpy=({rpy[0]:.3f},{rpy[1]:.3f},{rpy[2]:.3f}) "
                f"track_mean={step_tracking_mean:.3f} track_max={np.max(tracking_error):.3f} "
                f"sat={step_saturation_fraction:.3f} "
                f"contact=({int(left_contact)},{int(right_contact)}) "
                f"force=({contacts['left_normal_force']:.1f},{contacts['right_normal_force']:.1f}) "
                f"tau=({torque.min():.1f},{torque.max():.1f})"
            )
        if not finite:
            print("[ERROR] simulation contains NaN or Inf")
            completed = False
            return False
        if data.qpos[2] < fall_threshold:
            print(f"[ERROR] robot fell: root_z={data.qpos[2]:.3f} < {fall_threshold:.3f}")
            completed = False
            return False
        if quit_requested:
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

            with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
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

    completed = completed and executed_steps == max_steps
    full_trajectory_completed = completed and executed_steps == duration_steps
    denominator = max(executed_steps, 1)
    displacement = data.qpos[0:3] - initial_root
    summary: dict[str, Any] = {
        "motion_key": args.motion,
        "motion_name": motion.name,
        "mode": args.mode,
        "completed": completed,
        "full_trajectory_completed": full_trajectory_completed,
        "finite": finite,
        "executed_steps": executed_steps,
        "duration_steps": duration_steps,
        "control_dt": control_dt,
        "fall_threshold": fall_threshold,
        "minimum_root_z": minimum_root_z,
        "final_root_z": float(data.qpos[2]),
        "maximum_abs_roll_pitch": maximum_abs_roll_pitch,
        "root_displacement": displacement.tolist(),
        "tracking_error_mean": tracking_error_sum / max(tracking_error_samples, 1),
        "tracking_error_max": tracking_error_max,
        "torque_saturation_rate": saturated_samples / max(torque_samples, 1),
        "target_clipped_rate": clipped_target_samples / max(target_samples, 1),
        "maximum_abs_feedforward": maximum_abs_feedforward,
        "gain_profile": motion.gain_profile,
        "preserve_clipped_target_effort": motion.preserve_clipped_target_effort,
        "torque_saturation_by_joint": {
            name: float(saturated_by_joint[index] / max(torque_samples_by_joint[index], 1))
            for index, name in enumerate(joint_map.names)
        },
        "target_clipped_by_joint": {
            name: float(clipped_by_joint[index] / denominator)
            for index, name in enumerate(joint_map.names)
        },
        "left_foot_contact_ratio": left_contact_steps / denominator,
        "right_foot_contact_ratio": right_contact_steps / denominator,
        "any_foot_contact_ratio": any_contact_steps / denominator,
        "double_support_ratio": double_contact_steps / denominator,
        "left_normal_force_mean": left_force_sum / denominator,
        "right_normal_force_mean": right_force_sum / denominator,
        "left_normal_force_max": left_force_max,
        "right_normal_force_max": right_force_max,
        "actuator_disable_bitmask": int(model.opt.disableactuator),
        "controlled_joints": motion.joint_names,
        "held_joints": motion.uncontrolled_joint_names,
        "model": config["model"],
        "policy": str(motion.path),
    }
    summary["checks"] = validate_summary(summary, config)
    summary["passed"] = all(summary["checks"].values())
    summary["policy_success"] = args.mode == "policy" and summary["passed"]
    summary["reference_diagnostic_success"] = args.mode == "reference" and summary["passed"]

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    result_kind = "POLICY" if args.mode == "policy" else "REFERENCE DIAGNOSTIC"
    print(f"[RESULT] {result_kind} {'PASS' if summary['passed'] else 'FAIL'} {summary['checks']}")
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
