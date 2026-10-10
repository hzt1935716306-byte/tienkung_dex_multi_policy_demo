from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .control_target import ControlTarget, copy_control_target
from .onnx_registry import (
    MOTION_CONTRACT,
    WALKAMP_CONTRACT,
    create_cpu_session,
    validate_session,
)
from .robot_state import RobotStateSnapshot


WALKAMP_RUNTIME_JOINT_NAMES = (
    "hip_pitch_l_joint",
    "hip_pitch_r_joint",
    "waist_yaw_joint",
    "hip_roll_l_joint",
    "hip_roll_r_joint",
    "waist_roll_joint",
    "hip_yaw_l_joint",
    "hip_yaw_r_joint",
    "waist_pitch_joint",
    "knee_pitch_l_joint",
    "knee_pitch_r_joint",
    "shoulder_pitch_l_joint",
    "shoulder_pitch_r_joint",
    "ankle_pitch_l_joint",
    "ankle_pitch_r_joint",
    "shoulder_roll_l_joint",
    "shoulder_roll_r_joint",
    "ankle_roll_l_joint",
    "ankle_roll_r_joint",
    "shoulder_yaw_l_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_l_joint",
    "elbow_pitch_r_joint",
)

MOTION_RUNTIME_JOINT_NAMES = (
    "hip_pitch_l_joint",
    "hip_pitch_r_joint",
    "waist_yaw_joint",
    "hip_roll_l_joint",
    "hip_roll_r_joint",
    "waist_roll_joint",
    "hip_yaw_l_joint",
    "hip_yaw_r_joint",
    "waist_pitch_joint",
    "knee_pitch_l_joint",
    "knee_pitch_r_joint",
    "shoulder_pitch_l_joint",
    "shoulder_pitch_r_joint",
    "ankle_pitch_l_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_l_joint",
    "ankle_roll_r_joint",
    "elbow_pitch_l_joint",
    "elbow_pitch_r_joint",
)

MOTION_OBSERVATION_NAMES = (
    "command",
    "motion_anchor_ori_b",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
)


def _csv(value: str, converter=float) -> list[Any]:
    return [converter(item.strip()) for item in value.split(",") if item.strip()]


def _quat_normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm < 1.0e-8:
        raise ValueError("quaternion has zero norm")
    return value / norm


def _quat_conjugate(value: np.ndarray) -> np.ndarray:
    w, x, y, z = value
    return np.array([w, -x, -y, -z], dtype=np.float64)


def _quat_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
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


def _quat_matrix(value: np.ndarray) -> np.ndarray:
    w, x, y, z = _quat_normalize(value)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _yaw(value: np.ndarray) -> float:
    w, x, y, z = _quat_normalize(value)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_quat(angle: float) -> np.ndarray:
    return np.array([math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)], dtype=np.float64)


def _inverse_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return _quat_matrix(quaternion).T @ np.asarray(vector, dtype=np.float64)


def _gait_phase(
    timer: float,
    cycle: float,
    offset: np.ndarray,
    ratio: np.ndarray,
) -> np.ndarray:
    left = (timer / cycle + float(offset[0])) % 1.0
    right = (timer / cycle + float(offset[1])) % 1.0
    return np.asarray(
        [
            math.sin(2.0 * math.pi * left),
            math.sin(2.0 * math.pi * right),
            math.cos(2.0 * math.pi * left),
            math.cos(2.0 * math.pi * right),
            ratio[0],
            ratio[1],
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class RuntimeJointLayout:
    names: tuple[str, ...]
    ranges: np.ndarray
    efforts: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.names)
        if count != 29 or len(set(self.names)) != count:
            raise ValueError("RuntimeJointLayout requires 29 unique joint names")
        ranges = np.asarray(self.ranges, dtype=np.float64).copy()
        efforts = np.asarray(self.efforts, dtype=np.float64).copy()
        if ranges.shape != (count, 2) or efforts.shape != (count,):
            raise ValueError("RuntimeJointLayout ranges/efforts have invalid shapes")
        if not np.all(np.isfinite(ranges)) or not np.all(np.isfinite(efforts)):
            raise ValueError("RuntimeJointLayout contains NaN or Inf")
        if np.any(ranges[:, 0] >= ranges[:, 1]) or np.any(efforts <= 0.0):
            raise ValueError("RuntimeJointLayout has invalid limits")
        object.__setattr__(self, "ranges", ranges)
        object.__setattr__(self, "efforts", efforts)

    @property
    def index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.names)}

    @classmethod
    def from_joint_map(cls, joint_map: Any) -> "RuntimeJointLayout":
        return cls(tuple(joint_map.names), joint_map.ranges, joint_map.efforts)


class WalkAmpRuntime:
    """Official 840 -> 23 WALKAMP inference without simulator access."""

    observation_size = 84
    history_length = 10
    action_size = 23

    def __init__(
        self,
        policy_path: str | Path,
        config: Mapping[str, Any],
        layout: RuntimeJointLayout,
    ) -> None:
        self.config = dict(config)
        self.layout = layout
        if tuple(config["joint_names"]) != WALKAMP_RUNTIME_JOINT_NAMES:
            raise ValueError("WALKAMP joint order differs from the official contract")
        hold = config["uncontrolled_joints"]
        self.uncontrolled_names = tuple(hold["joint_names"])
        if len(self.uncontrolled_names) != 6:
            raise ValueError("WALKAMP must define six held joints")
        if set(WALKAMP_RUNTIME_JOINT_NAMES) | set(self.uncontrolled_names) != set(layout.names):
            raise ValueError("WALKAMP 23+6 partition does not match the 29-joint layout")

        index = layout.index
        self.policy_indices = np.asarray([index[name] for name in WALKAMP_RUNTIME_JOINT_NAMES], dtype=np.int32)
        self.uncontrolled_indices = np.asarray([index[name] for name in self.uncontrolled_names], dtype=np.int32)
        self.default_angles = np.asarray(config["default_joint_angles"], dtype=np.float64)
        self.kp = np.asarray(config["kp"], dtype=np.float64)
        self.kd = np.asarray(config["kd"], dtype=np.float64)
        self.hold_positions = np.asarray(hold["positions"], dtype=np.float64)
        self.hold_kp = np.asarray(hold["kp"], dtype=np.float64)
        self.hold_kd = np.asarray(hold["kd"], dtype=np.float64)
        self.hold_effort = np.asarray(hold["effort_limits"], dtype=np.float64)
        if any(value.shape != (23,) for value in (self.default_angles, self.kp, self.kd)):
            raise ValueError("WALKAMP policy vectors must contain 23 values")
        if any(value.shape != (6,) for value in (self.hold_positions, self.hold_kp, self.hold_kd, self.hold_effort)):
            raise ValueError("WALKAMP held-joint vectors must contain six values")

        self.action_scale = float(config["action_scale"])
        self.clip_observations = float(config["clip_observations"])
        self.clip_actions = float(config["clip_actions"])
        self.policy_dt = float(config["policy_dt"])
        self.gait_cycle = float(config["gait_cycle"])
        self.phase_ratio = np.asarray(config["phase_ratio"], dtype=np.float64)
        self.phase_offset = np.asarray(config["phase_offset"], dtype=np.float64)
        self.command_limits = np.asarray(config["command_limits"], dtype=np.float32)
        self.session = create_cpu_session(policy_path)
        validate_session(self.session, WALKAMP_CONTRACT)
        self.input_name = self.session.get_inputs()[0].name
        self.history = np.zeros(self.observation_size * self.history_length, dtype=np.float32)
        self.last_action = np.zeros(self.action_size, dtype=np.float32)
        self.phase_time = 0.0
        self.first_observation = True

    def reset(self, phase_time: float = 0.0) -> None:
        self.history.fill(0.0)
        self.last_action.fill(0.0)
        self.phase_time = float(phase_time) % self.gait_cycle
        self.first_observation = True

    def observation(
        self,
        state: RobotStateSnapshot,
        command: np.ndarray,
        executed_action: np.ndarray | None = None,
        phase_time: float | None = None,
    ) -> np.ndarray:
        if tuple(state.joint_names) != self.layout.names:
            raise ValueError("Robot state joint order differs from the runtime layout")
        command = np.asarray(command, dtype=np.float32)
        if command.shape != (3,):
            raise ValueError("WALKAMP command must contain [vx, vy, yaw_rate]")
        previous = self.last_action if executed_action is None else np.asarray(executed_action, dtype=np.float32)
        if previous.shape != (23,):
            raise ValueError("WALKAMP previous action must contain 23 values")
        phase = _gait_phase(
            self.phase_time if phase_time is None else phase_time,
            self.gait_cycle,
            self.phase_offset,
            self.phase_ratio,
        )
        observation = np.concatenate(
            [
                state.angular_velocity_body,
                _inverse_rotate(state.orientation_wxyz, np.array([0.0, 0.0, -1.0])),
                np.clip(command, -self.command_limits, self.command_limits),
                state.joint_position[self.policy_indices] - self.default_angles,
                state.joint_velocity[self.policy_indices],
                previous,
                phase,
            ]
        ).astype(np.float32)
        if observation.shape != (84,) or not np.all(np.isfinite(observation)):
            raise RuntimeError("WALKAMP observation contract failed")
        return observation

    def target_from_action(self, action: np.ndarray, label: str = "walkamp_runtime") -> ControlTarget:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (23,):
            raise ValueError("WALKAMP action must contain 23 values")
        count = len(self.layout.names)
        q = np.zeros(count, dtype=np.float64)
        kp = np.zeros(count, dtype=np.float64)
        kd = np.zeros(count, dtype=np.float64)
        q[self.policy_indices] = self.default_angles + self.action_scale * action
        kp[self.policy_indices] = self.kp
        kd[self.policy_indices] = self.kd
        q[self.uncontrolled_indices] = self.hold_positions
        kp[self.uncontrolled_indices] = self.hold_kp
        kd[self.uncontrolled_indices] = self.hold_kd
        effort = self.layout.efforts.copy()
        effort[self.uncontrolled_indices] = np.minimum(effort[self.uncontrolled_indices], self.hold_effort)
        target = ControlTarget(
            q=np.clip(q, self.layout.ranges[:, 0], self.layout.ranges[:, 1]),
            kp=kp,
            kd=kd,
            feedforward=np.zeros(count, dtype=np.float64),
            effort=effort,
            torque_scale=1.0,
            label=label,
        )
        target.validate(count)
        return target

    def neutral_target(self) -> ControlTarget:
        return self.target_from_action(np.zeros(23, dtype=np.float64), "walkamp_neutral")

    def step(
        self,
        state: RobotStateSnapshot,
        command: np.ndarray | list[float] | tuple[float, float, float],
    ) -> tuple[ControlTarget, np.ndarray, np.ndarray]:
        command_array = np.clip(np.asarray(command, dtype=np.float32), -self.command_limits, self.command_limits)
        observation = self.observation(state, command_array)
        if self.first_observation:
            self.history[:] = np.tile(observation, self.history_length)
            self.first_observation = False
        else:
            self.history[:-self.observation_size] = self.history[self.observation_size :]
            self.history[-self.observation_size :] = observation
        clipped = np.clip(self.history, -self.clip_observations, self.clip_observations).astype(np.float32)
        output = self.session.run(None, {self.input_name: clipped.reshape(1, -1)})[0][0]
        action = np.clip(output, -self.clip_actions, self.clip_actions).astype(np.float32)
        target = self.target_from_action(action)
        self.last_action[:] = action
        self.phase_time += self.policy_dt
        return target, clipped.copy(), action.copy()


class MotionRuntime:
    """BeyondMimic 104 + time -> 19 inference without simulator access."""

    observation_size = 104
    action_size = 19

    def __init__(
        self,
        policy_path: str | Path,
        config: Mapping[str, Any],
        layout: RuntimeJointLayout,
        hold_target: ControlTarget,
        gain_override: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        self.path = Path(policy_path)
        self.name = str(config.get("name", self.path.stem))
        self.config = dict(config)
        self.layout = layout
        self.duration_steps = int(config["duration_steps"])
        self.start_step = int(config.get("start_step", 0))
        self.torque_scale = float(config.get("torque_scale", 1.0))
        self.effort_scale = float(config.get("effort_scale", 1.0))
        self.preserve_clipped_target_effort = bool(
            config.get("preserve_clipped_target_effort", False)
        )
        self.previous_action_clip = float(config.get("previous_action_clip", 100.0))
        self.session = create_cpu_session(self.path)
        metadata = validate_session(self.session, MOTION_CONTRACT)
        self.output_names = [item.name for item in self.session.get_outputs()]
        self.joint_names = tuple(_csv(metadata["joint_names"], str))
        if self.joint_names != MOTION_RUNTIME_JOINT_NAMES:
            raise ValueError("BeyondMimic joint_names metadata differs from the 19-joint contract")
        observation_names = tuple(_csv(metadata["observation_names"], str))
        if observation_names != MOTION_OBSERVATION_NAMES or metadata["command_names"] != "motion":
            raise ValueError("BeyondMimic observation metadata differs from the 104-value contract")
        self.default_q = np.asarray(_csv(metadata["default_joint_pos"]), dtype=np.float64)
        self.action_scale = np.asarray(_csv(metadata["action_scale"]), dtype=np.float64)
        self.kp = np.asarray(_csv(metadata["joint_stiffness"]), dtype=np.float64)
        self.kd = np.asarray(_csv(metadata["joint_damping"]), dtype=np.float64)
        if gain_override is not None:
            self.kp = np.asarray([gain_override[name][0] for name in self.joint_names], dtype=np.float64)
            self.kd = np.asarray([gain_override[name][1] for name in self.joint_names], dtype=np.float64)
        index = layout.index
        self.model_indices = np.asarray([index[name] for name in self.joint_names], dtype=np.int32)
        self.hold_target = copy_control_target(hold_target, f"motion_hold:{self.name}")
        self.hold_target.validate(len(layout.names))
        self.last_action = np.zeros(19, dtype=np.float64)
        self.yaw_delta = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.started = False

    def _run(self, observation: np.ndarray, time_step: int) -> dict[str, np.ndarray]:
        inputs = {
            "obs": np.asarray(observation, dtype=np.float32).reshape(1, 104),
            "time_step": np.asarray([[float(time_step)]], dtype=np.float32),
        }
        return dict(zip(self.output_names, self.session.run(self.output_names, inputs)))

    def _reference(self, policy_step: int) -> dict[str, np.ndarray]:
        time_step = min(self.start_step + policy_step, self.start_step + self.duration_steps - 1)
        outputs = self._run(np.zeros((1, 104), dtype=np.float32), time_step)
        quaternions = outputs["body_quat_w"].astype(np.float64).copy()
        for index in range(quaternions.shape[1]):
            quaternions[0, index] = _quat_normalize(_quat_multiply(self.yaw_delta, quaternions[0, index]))
        outputs["body_quat_w"] = quaternions.astype(np.float32)
        return outputs

    def begin_from_live_state(self, state: RobotStateSnapshot) -> dict[str, float]:
        reference = self._run(np.zeros((1, 104), dtype=np.float32), self.start_step)
        reference_quat = _quat_normalize(reference["body_quat_w"][0, 0])
        self.yaw_delta = _yaw_quat(_yaw(state.orientation_wxyz) - _yaw(reference_quat))
        current_q = state.select_position(self.joint_names)
        self.last_action[:] = np.clip(
            (current_q - self.default_q) / self.action_scale,
            -self.previous_action_clip,
            self.previous_action_clip,
        )
        self.started = True
        return {
            "maximum_abs_joint_delta": float(
                np.max(np.abs(reference["joint_pos"][0].astype(np.float64) - current_q))
            ),
            "maximum_abs_previous_action": float(np.max(np.abs(self.last_action))),
            "yaw_delta": float(_yaw(self.yaw_delta)),
        }

    def observation(self, state: RobotStateSnapshot, reference: Mapping[str, np.ndarray]) -> np.ndarray:
        if tuple(state.joint_names) != self.layout.names:
            raise ValueError("Robot state joint order differs from the runtime layout")
        robot_quat = state.orientation_wxyz
        reference_quat = _quat_normalize(reference["body_quat_w"][0, 0])
        relative = _quat_multiply(_quat_conjugate(robot_quat), reference_quat)
        anchor_orientation = _quat_matrix(relative)[:, :2].reshape(-1)
        q = state.joint_position[self.model_indices]
        qd = state.joint_velocity[self.model_indices]
        observation = np.concatenate(
            [
                reference["joint_pos"][0],
                reference["joint_vel"][0],
                anchor_orientation,
                state.angular_velocity_body,
                q - self.default_q,
                qd,
                self.last_action,
            ]
        ).astype(np.float32)
        if observation.shape != (104,) or not np.all(np.isfinite(observation)):
            raise RuntimeError("BeyondMimic observation contract failed")
        return observation[None, :]

    def target_from_action(self, action: np.ndarray) -> ControlTarget:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (19,):
            raise ValueError("BeyondMimic action must contain 19 values")
        target = copy_control_target(self.hold_target, f"motion_runtime:{self.name}")
        desired = self.default_q + self.action_scale * action
        target.q[self.model_indices] = desired
        target.kp[self.model_indices] = self.kp
        target.kd[self.model_indices] = self.kd
        target.effort[self.model_indices] = self.layout.efforts[self.model_indices] * self.effort_scale
        unclipped = target.q.copy()
        target.q = np.clip(unclipped, self.layout.ranges[:, 0], self.layout.ranges[:, 1])
        if self.preserve_clipped_target_effort:
            target.feedforward += target.kp * (unclipped - target.q)
        target.torque_scale = self.torque_scale
        target.validate(len(self.layout.names))
        return target

    def step(
        self,
        state: RobotStateSnapshot,
        policy_step: int,
    ) -> tuple[ControlTarget, np.ndarray, np.ndarray]:
        if not self.started:
            raise RuntimeError("Call begin_from_live_state() before MotionRuntime.step()")
        reference = self._reference(policy_step)
        observation = self.observation(state, reference)
        time_step = min(self.start_step + policy_step, self.start_step + self.duration_steps - 1)
        outputs = self._run(observation, time_step)
        action = outputs["actions"][0].astype(np.float64)
        target = self.target_from_action(action)
        self.last_action[:] = action
        return target, observation.copy(), action.copy()
