from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .command_bus import CommandReader
from .config import load_config
from .policies import MotionPolicy, WalkPolicy
from .robot import ControlTarget, JointMap, base_target, build_joint_map


def robot_min_geom_z(model, data) -> float:
    import mujoco

    min_z = np.inf
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == 0:
            continue
        geom_type = model.geom_type[geom_id]
        # geom_rbound is orientation independent and greatly overestimates the
        # vertical extent of the EVT2 foot cylinders after they rotate sideways.
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = model.geom_dataid[geom_id]
            start = model.mesh_vertadr[mesh_id]
            count = model.mesh_vertnum[mesh_id]
            vertices = model.mesh_vert[start : start + count]
            matrix = data.geom_xmat[geom_id].reshape(3, 3)
            minimum = float((data.geom_xpos[geom_id] + vertices @ matrix.T)[:, 2].min())
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            matrix = data.geom_xmat[geom_id].reshape(3, 3)
            extent = float(np.sum(np.abs(matrix[2]) * model.geom_size[geom_id, :3]))
            minimum = float(data.geom_xpos[geom_id, 2] - extent)
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            minimum = float(data.geom_xpos[geom_id, 2] - model.geom_size[geom_id, 0])
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            matrix = data.geom_xmat[geom_id].reshape(3, 3)
            radius, half_length = model.geom_size[geom_id, :2]
            extent = float(radius + abs(matrix[2, 2]) * half_length)
            minimum = float(data.geom_xpos[geom_id, 2] - extent)
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            matrix = data.geom_xmat[geom_id].reshape(3, 3)
            radius, half_length = model.geom_size[geom_id, :2]
            extent = float(
                abs(matrix[2, 2]) * half_length
                + radius * np.linalg.norm(matrix[2, :2])
            )
            minimum = float(data.geom_xpos[geom_id, 2] - extent)
        elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            matrix = data.geom_xmat[geom_id].reshape(3, 3)
            extent = float(np.linalg.norm(matrix[2] * model.geom_size[geom_id, :3]))
            minimum = float(data.geom_xpos[geom_id, 2] - extent)
        else:
            minimum = float(data.geom_xpos[geom_id, 2] - model.geom_rbound[geom_id])
        min_z = min(min_z, minimum)
    return min_z


def setup_initial_pose(model, data, joint_map: JointMap, height: float, clearance: float) -> None:
    import mujoco

    target = base_target(joint_map, "initial")
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
        print(f"[INFO] auto-lifted root by {lift:.3f} m")
    print(f"[INFO] robot min geom z: {robot_min_geom_z(model, data):.3f} m")


def ensure_ground_clearance(model, data, clearance: float) -> None:
    import mujoco

    minimum = robot_min_geom_z(model, data)
    if minimum < clearance:
        lift = clearance - minimum
        data.qpos[2] += lift
        mujoco.mj_forward(model, data)
        print(f"[INFO] auto-lifted reference pose by {lift:.3f} m")
    print(f"[INFO] reference-pose min geom z: {robot_min_geom_z(model, data):.3f} m")


def blend_targets(first: ControlTarget, second: ControlTarget, alpha: float, label: str) -> ControlTarget:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return ControlTarget(
        q=(1.0 - alpha) * first.q + alpha * second.q,
        kp=(1.0 - alpha) * first.kp + alpha * second.kp,
        kd=(1.0 - alpha) * first.kd + alpha * second.kd,
        effort=(1.0 - alpha) * first.effort + alpha * second.effort,
        torque_scale=(1.0 - alpha) * first.torque_scale + alpha * second.torque_scale,
        label=label,
    )


def apply_pd(data, joint_map: JointMap, target: ControlTarget) -> np.ndarray:
    q = data.qpos[joint_map.qpos_adr]
    qd = data.qvel[joint_map.qvel_adr]
    torque = target.torque_scale * (target.kp * (target.q - q) - target.kd * qd)
    torque = np.clip(torque, -target.effort, target.effort)
    data.ctrl[:] = 0.0
    data.ctrl[joint_map.actuator_ids] = torque
    return torque


class DisabledWalkPolicy:
    """Minimal baseline interface used by the 19-DOF motion-only model."""

    def __init__(self, joint_map: JointMap):
        self.command = np.zeros(3, dtype=np.float32)
        self.policy_command = self.command.copy()
        self.target = base_target(joint_map, "stand")

    def set_zero_command(self) -> None:
        self.command[:] = 0.0

    def adjust_command(self, index: int, amount: float) -> None:
        del index, amount
        print("[WARN] walk policy is disabled for this robot model")

    def is_stationary(self) -> bool:
        return True

    def reset_history(self, fill_current: bool = True) -> None:
        del fill_current

    def step(self, force: bool = False) -> ControlTarget:
        del force
        return self.target


class MultiPolicyController:
    def __init__(
        self,
        walk: WalkPolicy | DisabledWalkPolicy,
        motions: dict[str, MotionPolicy],
        control_dt: float,
        transition_time: float,
        baseline_mode: str = "walk",
        return_neutral_time: float = 0.5,
        return_neutral_hold: float = 0.5,
        return_walk_time: float = 0.5,
        return_via_motion_start: bool = False,
        idle_motion: MotionPolicy | None = None,
    ):
        self.walk = walk
        self.motions = motions
        self.transition_steps = max(1, int(round(transition_time / control_dt)))
        self.return_neutral_steps = max(1, int(round(return_neutral_time / control_dt)))
        self.return_neutral_hold_steps = max(0, int(round(return_neutral_hold / control_dt)))
        self.return_walk_steps = max(1, int(round(return_walk_time / control_dt)))
        self.return_via_motion_start = return_via_motion_start
        self.idle_motion = idle_motion if baseline_mode == "walk" else None
        if baseline_mode not in ("walk", "stand"):
            raise ValueError(f"Unknown baseline mode: {baseline_mode}")
        self.baseline_mode = baseline_mode
        self.stand_motion = next(iter(motions.values())) if motions else None
        if baseline_mode == "stand" and self.stand_motion is not None:
            self.stand_motion.start()
            self.stand_target = self.stand_motion.first_target()
        else:
            self.stand_target = walk.step(force=True)
        self.stand_target.label = "stand"
        self.state = baseline_mode
        self.active_key: str | None = None
        self.active_motion: MotionPolicy | None = None
        self.motion_step = 0
        self.transition_step = 0
        self.transition_from: ControlTarget | None = None
        self.transition_to: ControlTarget | None = None
        self.quit_requested = False
        if baseline_mode == "walk" and self.idle_motion is not None and self.walk.is_stationary():
            self.idle_motion.start()
            self.last_target, _, _ = self.idle_motion.step(0)
            self.last_target.label = "idle"
        else:
            self.last_target = self.walk.step(force=True) if baseline_mode == "walk" else self.stand_target

    def _prepare_baseline(self) -> None:
        if self.baseline_mode == "walk":
            if self.idle_motion is not None and self.walk.is_stationary():
                self.idle_motion.start()
            else:
                self.walk.reset_history(fill_current=True)

    def _baseline_target(self) -> tuple[ControlTarget, np.ndarray | None, np.ndarray | None]:
        if self.baseline_mode == "stand":
            if self.stand_motion is None:
                return self.stand_target, None, None
            target, observation, action = self.stand_motion.step(0)
            target.label = "stand"
            return target, observation, action
        if self.idle_motion is not None and self.walk.is_stationary():
            target, observation, action = self.idle_motion.step(0)
            target.label = "idle"
            return target, observation, action
        return self.walk.step(), None, None

    def _begin_baseline_transition(self, source: str) -> None:
        self.transition_from = self.last_target
        self._prepare_baseline()
        self.transition_to, _, _ = self._baseline_target()
        self.transition_step = 0
        self.state = "to_walk"
        target_name = "idle" if self.idle_motion is not None and self.walk.is_stationary() else self.baseline_mode
        print(f"[INFO] switching to {target_name} ({source})")

    def adjust_walk_command(self, index: int, amount: float, source: str) -> None:
        was_stationary = self.walk.is_stationary()
        self.walk.adjust_command(index, amount)
        if was_stationary != self.walk.is_stationary() and self.state in ("walk", "to_walk"):
            self._begin_baseline_transition(source)

    def stop_walk(self, source: str) -> None:
        was_stationary = self.walk.is_stationary()
        self.walk.set_zero_command()
        print(f"[INFO] walk command cleared ({source})")
        if not was_stationary and self.state in ("walk", "to_walk"):
            self._begin_baseline_transition(source)

    def request_motion(self, key: str, source: str = "keyboard") -> None:
        key = key.lower()
        if key not in self.motions:
            print(f"[WARN] no motion registered for key {key!r}")
            return
        if self.state != self.baseline_mode:
            print(f"[WARN] motion {key!r} ignored while state={self.state}")
            return
        self.walk.set_zero_command()
        self.active_key = key
        self.active_motion = self.motions[key]
        self.active_motion.start()
        self.transition_from = self.last_target
        self.transition_to = self.active_motion.first_target()
        self.transition_step = 0
        self.motion_step = 0
        self.state = "to_motion"
        print(f"[INFO] switching to {self.active_motion.name} ({source}, key={key})")

    def request_walk(self, source: str = "keyboard") -> None:
        self.walk.set_zero_command()
        if self.state == self.baseline_mode:
            print(f"[INFO] already in baseline mode {self.baseline_mode} ({source})")
            return
        self.transition_from = self.last_target
        if self.baseline_mode == "walk":
            if self.active_motion is None or not self.return_via_motion_start:
                self._prepare_baseline()
                self.transition_to, _, _ = self._baseline_target()
                self.state = "to_walk"
            else:
                self.active_motion.start()
                self.transition_to = self.active_motion.first_target()
                self.state = "to_neutral"
        else:
            self.stand_motion.start()
            self.transition_to = self.stand_motion.first_target()
            self.transition_to.label = "stand"
            self.state = "to_walk"
        self.transition_step = 0
        target_name = "idle" if self.idle_motion is not None and self.walk.is_stationary() else self.baseline_mode
        print(f"[INFO] switching back to {target_name} ({source})")

    def step(self) -> tuple[ControlTarget, np.ndarray | None, np.ndarray | None]:
        observation = None
        action = None
        if self.state == "walk":
            target, observation, action = self._baseline_target()
        elif self.state == "stand":
            if self.stand_motion is None:
                target = self.stand_target
            else:
                target, observation, action = self.stand_motion.step(0)
                target.label = "stand"
        elif self.state == "to_motion":
            alpha = (self.transition_step + 1) / self.transition_steps
            target = blend_targets(self.transition_from, self.transition_to, alpha, "blend:to_motion")
            self.transition_step += 1
            if self.transition_step >= self.transition_steps:
                self.state = "motion"
                self.motion_step = 0
        elif self.state == "motion":
            target, observation, action = self.active_motion.step(self.motion_step)
            self.motion_step += 1
            if self.motion_step >= self.active_motion.duration_steps:
                self.request_walk(source="motion_complete")
        elif self.state == "to_neutral":
            self.transition_to, observation, action = self.active_motion.step(0)
            alpha = (self.transition_step + 1) / self.return_neutral_steps
            target = blend_targets(self.transition_from, self.transition_to, alpha, "blend:to_neutral")
            self.transition_step += 1
            if self.transition_step >= self.return_neutral_steps:
                self.state = "neutral_hold"
                self.transition_step = 0
        elif self.state == "neutral_hold":
            target, observation, action = self.active_motion.step(0)
            target.label = "motion:neutral"
            self.transition_step += 1
            if self.transition_step >= self.return_neutral_hold_steps:
                self.walk.reset_history(fill_current=True)
                self.transition_from = target
                self.transition_to = self.walk.step(force=True)
                self.transition_step = 0
                self.state = "to_walk"
        elif self.state == "to_walk":
            self.transition_to, observation, action = self._baseline_target()
            transition_steps = self.transition_steps if self.baseline_mode == "stand" else self.return_walk_steps
            alpha = (self.transition_step + 1) / transition_steps
            target = blend_targets(self.transition_from, self.transition_to, alpha, "blend:to_walk")
            self.transition_step += 1
            if self.transition_step >= transition_steps:
                self.state = self.baseline_mode
                self.active_key = None
                self.active_motion = None
        else:
            raise RuntimeError(f"Unknown controller state: {self.state}")
        self.last_target = target
        return target, observation, action


def handle_command(controller: MultiPolicyController, command: str, source: str) -> None:
    command = command.strip().lower()
    if command in ("8", "2", "4", "6", "7", "9", "5") and controller.state not in ("walk", "to_walk"):
        print(f"[WARN] velocity command ignored while state={controller.state}")
        return
    if command == "8":
        controller.adjust_walk_command(0, 0.2, source)
    elif command == "2":
        controller.adjust_walk_command(0, -0.2, source)
    elif command == "4":
        controller.adjust_walk_command(1, 0.2, source)
    elif command == "6":
        controller.adjust_walk_command(1, -0.2, source)
    elif command == "7":
        controller.adjust_walk_command(2, -0.2, source)
    elif command == "9":
        controller.adjust_walk_command(2, 0.2, source)
    elif command == "5":
        controller.stop_walk(source)
    elif command == "r":
        controller.request_walk(source)
    elif command == "q":
        controller.quit_requested = True
    elif command in controller.motions:
        controller.request_motion(command, source)


GAME_KEY_COMMANDS = {
    "w": "8",
    "s": "2",
    "a": "4",
    "d": "6",
    "q": "7",
    "e": "9",
    " ": "5",
    "j": "a",
    "k": "b",
}


def command_from_keycode(keycode: int) -> str | None:
    if 320 <= keycode <= 329:
        return str(keycode - 320)
    if keycode == 256:  # GLFW_KEY_ESCAPE
        return "q"
    if 0 <= keycode < 256:
        character = chr(keycode).lower()
        return GAME_KEY_COMMANDS.get(character, character)
    return None


def key_callback(controller: MultiPolicyController):

    def callback(keycode: int) -> None:
        command = command_from_keycode(keycode)
        if command is not None:
            handle_command(controller, command, "keyboard")

    return callback


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TienKung Dex walk + motion policy MuJoCo demo")
    parser.add_argument("--config", default="configs/demo.json", help="Demo JSON configuration")
    parser.add_argument("--model", default="", help="Override the MuJoCo XML path")
    parser.add_argument("--motion-mode", choices=("policy", "reference"), default=None)
    parser.add_argument("--no-viewer", action="store_true", help="Run without the MuJoCo window")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max control steps")
    parser.add_argument("--auto-motion", default="", help="Trigger one motion key in headless mode")
    parser.add_argument("--auto-motion-after", type=int, default=100)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    if config.get("experimental"):
        warning = config.get("compatibility_warning", "This configuration is experimental.")
        print(f"[WARN] {warning}")
    if args.model:
        config["model"] = str(Path(args.model).expanduser().resolve())
    if args.motion_mode:
        for motion in config["motions"].values():
            motion["control_mode"] = args.motion_mode

    import mujoco

    model = mujoco.MjModel.from_xml_path(config["model"])
    data = mujoco.MjData(model)
    sim = config.get("simulation", {})
    control_dt = float(sim.get("control_dt", 0.01))
    sim_dt = float(sim.get("sim_dt", 0.001))
    substeps = max(1, int(round(control_dt / sim_dt)))
    model.opt.timestep = sim_dt
    model.opt.iterations = int(sim.get("solver_iterations", 100))
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    joint_map = build_joint_map(model)
    model.dof_damping[joint_map.qvel_adr] = 0.0
    setup_initial_pose(
        model,
        data,
        joint_map,
        float(sim.get("initial_height", 0.95)),
        float(sim.get("ground_clearance", 0.015)),
    )

    baseline_mode = str(sim.get("baseline_mode", "walk")).lower()
    if baseline_mode == "walk":
        walk = WalkPolicy(model, data, joint_map, config["walk_policy"], config.get("walk", {}), control_dt)
        walk.reset_history(fill_current=False)
    else:
        walk = DisabledWalkPolicy(joint_map)
    motions = {
        key.lower(): MotionPolicy(model, data, joint_map, motion["path"], motion)
        for key, motion in config["motions"].items()
    }
    idle_motion_key = str(sim.get("idle_motion_key", "")).lower()
    idle_motion = motions.get(idle_motion_key) if idle_motion_key else None
    if idle_motion_key and idle_motion is None:
        raise ValueError(f"simulation.idle_motion_key {idle_motion_key!r} is not registered in motions.")
    if "joint_armature" in sim:
        model.dof_armature[joint_map.qvel_adr] = float(sim["joint_armature"])
    if "joint_frictionloss" in sim:
        model.dof_frictionloss[joint_map.qvel_adr] = float(sim["joint_frictionloss"])
    if baseline_mode == "stand" and motions:
        next(iter(motions.values())).reset_robot_to_reference()
        ensure_ground_clearance(model, data, float(sim.get("ground_clearance", 0.015)))
        walk.reset_history(fill_current=False)
    controller = MultiPolicyController(
        walk,
        motions,
        control_dt,
        float(sim.get("transition_time", 0.5)),
        baseline_mode,
        float(sim.get("return_neutral_time", 0.5)),
        float(sim.get("return_neutral_hold", 0.5)),
        float(sim.get("return_walk_time", 0.5)),
        bool(sim.get("return_via_motion_start", False)),
        idle_motion,
    )
    command_file = config.get("command_file", "/tmp/tienkung_dex_commands.jsonl")
    command_reader = CommandReader(command_file)
    max_steps = args.max_steps if args.max_steps is not None else int(sim.get("max_steps", 100000))
    debug = bool(sim.get("debug", True)) if args.debug is None else args.debug
    debug_interval = int(sim.get("debug_interval", 25))
    fall_threshold = float(sim.get("fall_threshold", 0.2))
    realtime = bool(sim.get("realtime", True))
    no_viewer = args.no_viewer or bool(sim.get("no_viewer", False))
    initial_xy = data.qpos[0:2].copy()

    print(f"[INFO] config: {config['_path']}")
    print(f"[INFO] model: {config['model']}")
    print(f"[INFO] joints: {len(joint_map.names)}, sim_dt={sim_dt}, control_dt={control_dt}")
    print(f"[INFO] command file: {command_file}")
    print(f"[INFO] baseline mode: {baseline_mode}")
    if baseline_mode == "walk":
        print("[INFO] controls: W/S forward/back, A/D lateral, Q/E yaw, Space stop")
        if motions:
            print("[INFO] actions: J=bow, K=wave, R=return, Esc=quit")
        else:
            print("[INFO] Esc=quit")
        if idle_motion is not None:
            print(f"[INFO] zero-speed stabilizer: {idle_motion.name} frame 0")
    print(f"[INFO] motions: {', '.join(f'{key}={policy.name}' for key, policy in motions.items())}")
    print(f"[INFO] external commands: r=return to {baseline_mode}, q=quit")

    def step_once(step: int) -> bool:
        if no_viewer and args.auto_motion and step == args.auto_motion_after:
            controller.request_motion(args.auto_motion, "auto")
        for command, source in command_reader.read():
            handle_command(controller, command, source)

        target, observation, action = controller.step()
        torque = np.zeros(len(joint_map.names), dtype=np.float64)
        for _ in range(substeps):
            torque = apply_pd(data, joint_map, target)
            mujoco.mj_step(model, data)

        if debug and step % max(1, debug_interval) == 0:
            obs_range = "n/a" if observation is None else f"({observation.min():.3f},{observation.max():.3f})"
            act_range = "n/a" if action is None else f"({action.min():.3f},{action.max():.3f})"
            print(
                f"[DEBUG] state={controller.state} step={step} target={target.label} "
                f"cmd=({walk.command[0]:.2f},{walk.command[1]:.2f},{walk.command[2]:.2f}) "
                f"policy_cmd=({walk.policy_command[0]:.2f},{walk.policy_command[1]:.2f},"
                f"{walk.policy_command[2]:.2f}) "
                f"obs={obs_range} act={act_range} tau=({torque.min():.2f},{torque.max():.2f}) "
                f"root_z={data.qpos[2]:.3f} "
                f"drift=({data.qpos[0] - initial_xy[0]:.3f},{data.qpos[1] - initial_xy[1]:.3f})"
            )
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            print("[ERROR] simulation contains NaN/Inf")
            return False
        if data.qpos[2] < fall_threshold:
            print(f"[WARN] stopping: root_z={data.qpos[2]:.3f} below {fall_threshold:.3f}")
            return False
        return not controller.quit_requested

    if no_viewer:
        for step in range(max_steps):
            if not step_once(step):
                break
        return

    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback(controller)) as viewer:
        for step in range(max_steps):
            if not viewer.is_running() or controller.quit_requested:
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


def main() -> None:
    try:
        run()
    except KeyboardInterrupt:
        print("\n[INFO] stopped by user")


if __name__ == "__main__":
    main()
