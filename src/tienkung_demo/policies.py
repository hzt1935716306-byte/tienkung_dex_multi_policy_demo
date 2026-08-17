from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .robot import (
    ALL_JOINT_NAMES,
    HOLD_KD,
    HOLD_KP,
    MOTION_BODY_NAMES,
    WALK_DEFAULTS,
    WALK_EFFORT,
    WALK_KD,
    WALK_KP,
    WALK_MUJOCO_NAMES,
    WALK_POLICY_NAMES,
    ControlTarget,
    JointMap,
    base_target,
)


def csv_values(value: str, cast=float) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def quat_normalize(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1.0e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(quat)
    return np.array([w, -x, -y, -z], dtype=np.float64)


def quat_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def yaw_from_quat(quat: np.ndarray) -> float:
    w, x, y, z = quat_normalize(quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_quat(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def inverse_rotate(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return quat_to_matrix(quat).T @ np.asarray(vector, dtype=np.float64)


def body_local_angular_velocity(model, data, body_id: int) -> np.ndarray:
    import mujoco

    velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 1)
    return velocity[0:3].copy()


class WalkPolicy:
    observation_dim = 75
    history_length = 10
    action_scale = 0.25
    clip = 100.0

    def __init__(self, model, data, joint_map: JointMap, path: str, config: dict, control_dt: float):
        import mujoco
        import onnxruntime as ort

        self.model = model
        self.data = data
        self.joint_map = joint_map
        self.path = Path(path)
        self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        if input_shape[-1] != self.observation_dim * self.history_length:
            raise RuntimeError(f"Walk policy expects {input_shape}, expected [1, 750].")

        self.policy_dt = float(config.get("policy_dt", 0.02))
        self.repeat = max(1, int(round(self.policy_dt / control_dt)))
        self.command = np.asarray(config.get("initial_command", [0.0, 0.0, 0.0]), dtype=np.float32)
        if self.command.shape != (3,):
            raise ValueError("walk.initial_command must contain [vx, vy, yaw].")
        self.policy_command = self.command.copy()
        self.station_keep = bool(config.get("station_keep", True))
        self.station_keep_kp = np.asarray(config.get("station_keep_kp", [1.0, 1.0, 1.0]), dtype=np.float64)
        self.station_keep_ki = np.asarray(config.get("station_keep_ki", [0.1, 0.1, 0.1]), dtype=np.float64)
        self.station_keep_limit = np.asarray(
            config.get("station_keep_command_limit", [0.5, 0.35, 0.6]), dtype=np.float64
        )
        if self.station_keep_kp.shape != (3,) or self.station_keep_ki.shape != (3,):
            raise ValueError("walk station-keeping gains must contain [x, y, yaw].")
        if self.station_keep_limit.shape != (3,) or np.any(self.station_keep_limit <= 0.0):
            raise ValueError("walk.station_keep_command_limit must contain three positive values.")
        self.command_deadband = float(config.get("command_deadband", 1.0e-4))
        self.station_integral_limit = float(config.get("station_keep_integral_limit", 1.0))
        self.station_anchor_xy = self.data.qpos[0:2].copy()
        self.station_anchor_yaw = yaw_from_quat(self.data.qpos[3:7])
        self.station_integral = np.zeros(3, dtype=np.float64)
        self.action = np.zeros(len(WALK_POLICY_NAMES), dtype=np.float32)
        self.history = np.zeros(self.observation_dim * self.history_length, dtype=np.float32)
        self.policy_steps = 0
        self.control_ticks = 0
        self.phase = np.zeros(2, dtype=np.float64)
        self.gait_cycle = float(config.get("gait_cycle", 0.85))
        self.phase_ratio = np.asarray(config.get("phase_ratio", [0.38, 0.38]), dtype=np.float64)
        self.phase_offset = np.asarray(config.get("phase_offset", [0.38, 0.88]), dtype=np.float64)
        self.torque_scale = float(config.get("torque_scale", 1.0))
        self.effort_scale = float(config.get("effort_scale", 1.0))
        self.cached_target: ControlTarget | None = None

        present_pairs = [
            (policy_index, joint_map.index[name])
            for policy_index, name in enumerate(WALK_POLICY_NAMES)
            if name in joint_map.index
        ]
        self.policy_indices = np.asarray([pair[0] for pair in present_pairs], dtype=np.int32)
        self.model_indices = np.asarray([pair[1] for pair in present_pairs], dtype=np.int32)
        self.default_policy = np.asarray([WALK_DEFAULTS[name] for name in WALK_POLICY_NAMES], dtype=np.float64)
        self.missing_joint_names = [name for name in WALK_POLICY_NAMES if name not in joint_map.index]

        self.has_imu_sensors = all(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name) >= 0
            for name in ("orientation", "angular-velocity")
        )
        self.pelvis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self.pelvis_body_id < 0:
            self.pelvis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Base_link")
        if not self.has_imu_sensors and self.pelvis_body_id < 0:
            raise ValueError("MuJoCo model has no IMU sensors and no 'pelvis' or 'Base_link' body.")
        print(f"[INFO] walk policy loaded: {self.path} (750 -> 20, {1 / self.policy_dt:.0f} Hz)")
        if self.missing_joint_names:
            print(f"[INFO] walk outputs ignored for fixed joints: {self.missing_joint_names}")

    def set_zero_command(self) -> None:
        self.command[:] = 0.0
        self.reset_station_anchor()

    def is_stationary(self) -> bool:
        return np.linalg.norm(self.command) <= self.command_deadband

    def reset_station_anchor(self) -> None:
        self.station_anchor_xy = self.data.qpos[0:2].copy()
        self.station_anchor_yaw = yaw_from_quat(self.data.qpos[3:7])
        self.station_integral[:] = 0.0

    def adjust_command(self, index: int, amount: float) -> None:
        limits = np.array([1.0, 0.5, 1.57], dtype=np.float32)
        was_stationary = self.is_stationary()
        self.command[index] = np.clip(self.command[index] + amount, -limits[index], limits[index])
        is_stationary = self.is_stationary()
        if is_stationary and not was_stationary:
            self.reset_station_anchor()
        elif not is_stationary:
            self.station_integral[:] = 0.0
        print(f"[INFO] walk command vx={self.command[0]:.2f} vy={self.command[1]:.2f} yaw={self.command[2]:.2f}")

    def _station_keeping_command(self, advance_integral: bool) -> np.ndarray:
        if not self.station_keep or np.linalg.norm(self.command) > self.command_deadband:
            return self.command.astype(np.float64)

        yaw = yaw_from_quat(self.data.qpos[3:7])
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        position_error_w = self.station_anchor_xy - self.data.qpos[0:2]
        yaw_error = (self.station_anchor_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
        if advance_integral:
            self.station_integral[0:2] += position_error_w * self.policy_dt
            self.station_integral[2] += yaw_error * self.policy_dt
            self.station_integral = np.clip(
                self.station_integral,
                -self.station_integral_limit,
                self.station_integral_limit,
            )

        world_to_body = np.array([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]], dtype=np.float64)
        position_error_b = world_to_body @ position_error_w
        integral_error_b = world_to_body @ self.station_integral[0:2]
        correction = np.empty(3, dtype=np.float64)
        correction[0:2] = (
            self.station_keep_kp[0:2] * position_error_b
            + self.station_keep_ki[0:2] * integral_error_b
        )
        correction[2] = self.station_keep_kp[2] * yaw_error + self.station_keep_ki[2] * self.station_integral[2]
        return np.clip(correction, -self.station_keep_limit, self.station_keep_limit)

    def reset_history(self, fill_current: bool = True) -> None:
        self.action[:] = 0.0
        self.control_ticks = 0
        if fill_current:
            observation = self._single_observation()
            self.history = np.tile(observation, self.history_length).astype(np.float32)
        else:
            self.history[:] = 0.0
        self.cached_target = None

    def _single_observation(self, advance_station_integral: bool = False) -> np.ndarray:
        q_policy = self.default_policy.copy()
        qd_policy = np.zeros(len(WALK_POLICY_NAMES), dtype=np.float64)
        q_policy[self.policy_indices] = self.data.qpos[self.joint_map.qpos_adr[self.model_indices]]
        qd_policy[self.policy_indices] = self.data.qvel[self.joint_map.qvel_adr[self.model_indices]]
        if self.has_imu_sensors:
            orientation = quat_normalize(self.data.sensor("orientation").data.copy())
            angular_velocity = self.data.sensor("angular-velocity").data.copy()
        else:
            orientation = quat_normalize(self.data.xquat[self.pelvis_body_id].copy())
            angular_velocity = body_local_angular_velocity(self.model, self.data, self.pelvis_body_id)
        self.policy_command = self._station_keeping_command(advance_station_integral).astype(np.float32)
        observation = np.concatenate(
            [
                angular_velocity,
                inverse_rotate(orientation, np.array([0.0, 0.0, -1.0])),
                self.policy_command,
                q_policy - self.default_policy,
                qd_policy,
                self.action,
                np.sin(2.0 * np.pi * self.phase),
                np.cos(2.0 * np.pi * self.phase),
                self.phase_ratio,
            ]
        ).astype(np.float32)
        if observation.shape != (self.observation_dim,):
            raise RuntimeError(f"Walk observation has shape {observation.shape}, expected (75,).")
        return observation

    def _update_phase(self) -> None:
        time_ratio = self.policy_steps * self.policy_dt / self.gait_cycle
        self.phase = (time_ratio + self.phase_offset) % 1.0

    def _infer(self) -> ControlTarget:
        observation = self._single_observation(advance_station_integral=True)
        self.history = np.roll(self.history, -self.observation_dim)
        self.history[-self.observation_dim :] = observation
        output = self.session.run(None, {self.input_name: self.history[None, :]})[0]
        self.action = np.clip(output[0], -self.clip, self.clip).astype(np.float32)

        target = base_target(self.joint_map, "walk")
        target.q[self.model_indices] = (
            self.default_policy[self.policy_indices] + self.action[self.policy_indices] * self.action_scale
        )
        for name in WALK_MUJOCO_NAMES:
            if name not in self.joint_map.index:
                continue
            index = self.joint_map.index[name]
            target.kp[index] = WALK_KP[name]
            target.kd[index] = WALK_KD[name]
            target.effort[index] = min(self.joint_map.efforts[index], WALK_EFFORT[name] * self.effort_scale)
        for name in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"):
            if name in self.joint_map.index:
                target.effort[self.joint_map.index[name]] = self.joint_map.efforts[self.joint_map.index[name]]
        target.q = np.clip(target.q, self.joint_map.ranges[:, 0], self.joint_map.ranges[:, 1])
        target.torque_scale = self.torque_scale
        self.policy_steps += 1
        self._update_phase()
        self.cached_target = target
        return target

    def step(self, force: bool = False) -> ControlTarget:
        should_infer = force or self.cached_target is None or self.control_ticks % self.repeat == 0
        target = self._infer() if should_infer else self.cached_target
        self.control_ticks += 1
        return target


class MotionPolicy:
    def __init__(self, model, data, joint_map: JointMap, path: str, config: dict):
        import mujoco
        import onnxruntime as ort

        self.model = model
        self.data = data
        self.joint_map = joint_map
        self.path = Path(path)
        self.name = str(config.get("name", self.path.stem))
        self.duration_steps = int(config["duration_steps"])
        self.start_step = int(config.get("start_step", 0))
        self.control_mode = str(config.get("control_mode", "policy")).lower()
        if self.control_mode not in ("policy", "reference"):
            raise ValueError(f"Unknown motion control_mode {self.control_mode!r}")
        self.torque_scale = float(config.get("torque_scale", 1.0))
        self.effort_scale = float(config.get("effort_scale", 1.0))
        self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.output_names = [item.name for item in self.session.get_outputs()]
        metadata = self.session.get_modelmeta().custom_metadata_map

        required = ["joint_names", "joint_stiffness", "joint_damping", "default_joint_pos", "action_scale"]
        missing = [key for key in required if key not in metadata]
        if missing:
            raise RuntimeError(f"Motion ONNX metadata is missing {missing}: {self.path}")
        self.joint_names = csv_values(metadata["joint_names"], str)
        self.kp = np.asarray(csv_values(metadata["joint_stiffness"]), dtype=np.float64)
        self.kd = np.asarray(csv_values(metadata["joint_damping"]), dtype=np.float64)
        self.default_q = np.asarray(csv_values(metadata["default_joint_pos"]), dtype=np.float64)
        self.action_scale = np.asarray(csv_values(metadata["action_scale"]), dtype=np.float64)
        present_pairs = [
            (policy_index, joint_map.index[name])
            for policy_index, name in enumerate(self.joint_names)
            if name in joint_map.index
        ]
        self.policy_indices = np.asarray([pair[0] for pair in present_pairs], dtype=np.int32)
        self.model_indices = np.asarray([pair[1] for pair in present_pairs], dtype=np.int32)
        self.missing_joint_names = [name for name in self.joint_names if name not in joint_map.index]
        self.last_action = np.zeros(len(self.joint_names), dtype=np.float64)
        self.zero_observation = np.zeros((1, 104), dtype=np.float32)
        self.anchor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self.anchor_body_id < 0:
            self.anchor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Base_link")
        if self.anchor_body_id < 0:
            raise ValueError("MuJoCo model has no 'pelvis' or 'Base_link' body.")
        self.anchor_motion_index = MOTION_BODY_NAMES.index("pelvis")
        self.yaw_delta = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        print(
            f"[INFO] motion loaded: {self.name} key policy={self.path.name} "
            f"(104 -> {len(self.joint_names)}, steps={self.duration_steps}, mode={self.control_mode})"
        )
        if self.missing_joint_names:
            print(f"[INFO] motion joints simulated at defaults (not in walk model): {self.missing_joint_names}")

    def _run(self, observation: np.ndarray, time_step: int) -> dict[str, np.ndarray]:
        inputs = {
            "obs": observation.astype(np.float32),
            "time_step": np.array([[float(time_step)]], dtype=np.float32),
        }
        unknown = set(inputs) - self.input_names
        if unknown:
            raise RuntimeError(f"Unexpected ONNX inputs {unknown} for {self.path}")
        return dict(zip(self.output_names, self.session.run(self.output_names, inputs)))

    def start(self) -> None:
        reference = self._run(self.zero_observation, self.start_step)
        reference_quat = quat_normalize(reference["body_quat_w"][0, self.anchor_motion_index])
        robot_quat = quat_normalize(self.data.xquat[self.anchor_body_id].copy())
        self.yaw_delta = yaw_quat(yaw_from_quat(robot_quat) - yaw_from_quat(reference_quat))
        self.last_action[:] = 0.0

    def reset_robot_to_reference(self) -> None:
        """Place MuJoCo at the first reference frame, matching IsaacLab playback."""
        import mujoco

        reference = self._run(self.zero_observation, self.start_step)
        reference_position = reference["body_pos_w"][0, self.anchor_motion_index].astype(np.float64)
        reference_quat = quat_normalize(reference["body_quat_w"][0, self.anchor_motion_index])

        self.data.qpos[self.joint_map.qpos_adr] = 0.0
        self.data.qpos[self.joint_map.qpos_adr[self.model_indices]] = reference["joint_pos"][
            0, self.policy_indices
        ]
        self.data.qpos[0:3] = reference_position
        self.data.qpos[3:7] = reference_quat
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        delta_quat = quat_multiply(
            reference_quat,
            quat_conjugate(quat_normalize(self.data.xquat[self.anchor_body_id].copy())),
        )
        self.data.qpos[3:7] = quat_normalize(quat_multiply(delta_quat, self.data.qpos[3:7]))
        mujoco.mj_forward(self.model, self.data)
        self.data.qpos[0:3] += reference_position - self.data.xpos[self.anchor_body_id]
        mujoco.mj_forward(self.model, self.data)

    def _reference(self, policy_step: int) -> dict[str, np.ndarray]:
        time_step = min(self.start_step + policy_step, self.start_step + self.duration_steps - 1)
        reference = self._run(self.zero_observation, time_step)
        quaternions = reference["body_quat_w"].astype(np.float64).copy()
        for index in range(quaternions.shape[1]):
            quaternions[0, index] = quat_normalize(quat_multiply(self.yaw_delta, quaternions[0, index]))
        reference["body_quat_w"] = quaternions.astype(np.float32)
        return reference

    def _observation(self, reference: dict[str, np.ndarray]) -> np.ndarray:
        q = self.default_q.copy()
        qd = np.zeros(len(self.joint_names), dtype=np.float64)
        q[self.policy_indices] = self.data.qpos[self.joint_map.qpos_adr[self.model_indices]]
        qd[self.policy_indices] = self.data.qvel[self.joint_map.qvel_adr[self.model_indices]]
        robot_quat = quat_normalize(self.data.xquat[self.anchor_body_id].copy())
        reference_quat = quat_normalize(reference["body_quat_w"][0, self.anchor_motion_index])
        relative_quat = quat_multiply(quat_conjugate(robot_quat), reference_quat)
        anchor_orientation = quat_to_matrix(relative_quat)[:, :2].reshape(-1)
        observation = np.concatenate(
            [
                reference["joint_pos"][0],
                reference["joint_vel"][0],
                anchor_orientation,
                body_local_angular_velocity(self.model, self.data, self.anchor_body_id),
                q - self.default_q,
                qd,
                self.last_action,
            ]
        ).astype(np.float32)
        if observation.shape != (104,):
            raise RuntimeError(f"Motion observation has shape {observation.shape}, expected (104,).")
        return observation[None, :]

    def _target(self, action: np.ndarray) -> ControlTarget:
        policy_target = self.default_q + self.action_scale * action
        return self._target_from_positions(policy_target)

    def _target_from_positions(self, policy_target: np.ndarray) -> ControlTarget:
        target = base_target(self.joint_map, f"motion:{self.name}")
        target.q[:] = 0.0
        target.q[self.model_indices] = policy_target[self.policy_indices]
        target.kp[self.model_indices] = self.kp[self.policy_indices]
        target.kd[self.model_indices] = self.kd[self.policy_indices]
        target.effort[:] = np.minimum(target.effort, self.joint_map.efforts)
        target.effort[self.model_indices] = self.joint_map.efforts[self.model_indices] * self.effort_scale
        target.q = np.clip(target.q, self.joint_map.ranges[:, 0], self.joint_map.ranges[:, 1])
        target.torque_scale = self.torque_scale
        return target

    def first_target(self) -> ControlTarget:
        reference = self._reference(0)
        if self.control_mode == "reference":
            return self._target_from_positions(reference["joint_pos"][0].astype(np.float64))
        observation = self._observation(reference)
        outputs = self._run(observation, self.start_step)
        return self._target(outputs["actions"][0].astype(np.float64))

    def neutral_target(self) -> ControlTarget:
        return self._target(np.zeros(len(self.joint_names), dtype=np.float64))

    def step(self, policy_step: int) -> tuple[ControlTarget, np.ndarray, np.ndarray]:
        reference = self._reference(policy_step)
        observation = self._observation(reference)
        if self.control_mode == "reference":
            reference_position = reference["joint_pos"][0].astype(np.float64)
            action = (reference_position - self.default_q) / np.maximum(np.abs(self.action_scale), 1.0e-8)
            self.last_action = action.copy()
            return self._target_from_positions(reference_position), observation, action
        time_step = min(self.start_step + policy_step, self.start_step + self.duration_steps - 1)
        outputs = self._run(observation, time_step)
        action = outputs["actions"][0].astype(np.float64)
        self.last_action = action.copy()
        return self._target(action), observation, action
