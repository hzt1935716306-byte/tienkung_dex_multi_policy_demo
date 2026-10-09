from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .policies import (
    MotionPolicy,
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    yaw_from_quat,
    yaw_quat,
)
from .robot import ControlTarget, JointMap


MOTION_EVT2_JOINT_NAMES = [
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
]

MOTION_OBSERVATION_NAMES = [
    "command",
    "motion_anchor_ori_b",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
]

TRAINING_NOMINAL_GAINS = {
    "hip_pitch_l_joint": (300.0, 15.0),
    "hip_pitch_r_joint": (300.0, 15.0),
    "waist_yaw_joint": (400.0, 5.0),
    "hip_roll_l_joint": (300.0, 15.0),
    "hip_roll_r_joint": (300.0, 15.0),
    "waist_roll_joint": (400.0, 15.0),
    "hip_yaw_l_joint": (150.0, 7.5),
    "hip_yaw_r_joint": (150.0, 7.5),
    "waist_pitch_joint": (400.0, 10.0),
    "knee_pitch_l_joint": (350.0, 15.0),
    "knee_pitch_r_joint": (350.0, 15.0),
    "shoulder_pitch_l_joint": (150.0, 5.0),
    "shoulder_pitch_r_joint": (150.0, 5.0),
    "ankle_pitch_l_joint": (30.0, 3.75),
    "ankle_pitch_r_joint": (30.0, 3.75),
    "ankle_roll_l_joint": (16.8, 2.1),
    "ankle_roll_r_joint": (16.8, 2.1),
    "elbow_pitch_l_joint": (150.0, 5.0),
    "elbow_pitch_r_joint": (150.0, 5.0),
}


def _resolve_path(value: str, config_dir: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return str(path.resolve())


def load_motion_evt2_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a JSON object: {config_path}")

    config["_path"] = str(config_path)
    config_dir = config_path.parent
    for field in ("model", "walkamp_config"):
        config[field] = _resolve_path(config[field], config_dir)
        if not Path(config[field]).is_file():
            raise FileNotFoundError(config[field])
    motions = config.get("motions", {})
    if set(motions) != {"a", "b"}:
        raise ValueError("motion_evt2 config must define exactly motions 'a' and 'b'")
    for motion in motions.values():
        motion["path"] = _resolve_path(motion["path"], config_dir)
        if not Path(motion["path"]).is_file():
            raise FileNotFoundError(motion["path"])
    return config


def copy_target(target: ControlTarget, label: str) -> ControlTarget:
    return ControlTarget(
        q=target.q.copy(),
        kp=target.kp.copy(),
        kd=target.kd.copy(),
        feedforward=target.feedforward.copy(),
        effort=target.effort.copy(),
        torque_scale=float(target.torque_scale),
        label=label,
    )


class MotionEvt2Policy(MotionPolicy):
    """BeyondMimic 19-DOF motion adapter for the official 29-joint EVT2 model."""

    observation_size = 104
    action_size = 19

    def __init__(
        self,
        model,
        data,
        joint_map: JointMap,
        path: str,
        config: dict[str, Any],
        hold_target: ControlTarget,
    ) -> None:
        super().__init__(model, data, joint_map, path, config)
        self.gain_profile = str(config.get("gain_profile", "onnx_metadata"))
        if self.gain_profile == "training_nominal":
            self.kp = np.asarray(
                [TRAINING_NOMINAL_GAINS[name][0] for name in self.joint_names], dtype=np.float64
            )
            self.kd = np.asarray(
                [TRAINING_NOMINAL_GAINS[name][1] for name in self.joint_names], dtype=np.float64
            )
        elif self.gain_profile != "onnx_metadata":
            raise ValueError(f"Unknown motion gain_profile {self.gain_profile!r}")
        self.preserve_clipped_target_effort = bool(
            config.get("preserve_clipped_target_effort", False)
        )
        self.hold_target = copy_target(hold_target, "motion_evt2_hold")
        self.episode_initialized = False
        self.last_unclipped_target = self.hold_target.q.copy()
        self.last_target_clipped = np.zeros(len(joint_map.names), dtype=bool)
        self.initialization_mode = "uninitialized"
        self.live_start_count = 0
        self.previous_action_clip = float(config.get("previous_action_clip", 100.0))
        self._validate_contract()

        controlled = set(self.joint_names)
        self.uncontrolled_joint_names = [name for name in joint_map.names if name not in controlled]
        self.uncontrolled_indices = np.asarray(
            [joint_map.index[name] for name in self.uncontrolled_joint_names], dtype=np.int32
        )
        if len(self.uncontrolled_joint_names) != 10:
            raise ValueError(
                f"Motion/EVT2 partition must be 19 controlled + 10 held, got "
                f"{len(self.joint_names)} + {len(self.uncontrolled_joint_names)}"
            )

    def _validate_contract(self) -> None:
        if len(self.joint_map.names) != 29:
            raise ValueError(f"Motion EVT2 requires 29 joints, found {len(self.joint_map.names)}")
        if self.joint_names != MOTION_EVT2_JOINT_NAMES:
            raise ValueError("Motion ONNX joint_names metadata does not match the expected 19-joint order")
        if self.missing_joint_names:
            raise ValueError(f"Official EVT2 is missing motion joints: {self.missing_joint_names}")

        metadata = self.session.get_modelmeta().custom_metadata_map
        observation_names = [item.strip() for item in metadata.get("observation_names", "").split(",")]
        if observation_names != MOTION_OBSERVATION_NAMES:
            raise ValueError(
                f"Motion observation_names metadata is {observation_names}, "
                f"expected {MOTION_OBSERVATION_NAMES}"
            )
        if metadata.get("command_names") != "motion":
            raise ValueError("Motion ONNX command_names metadata must equal 'motion'")

        expected_inputs = {
            "obs": [1, self.observation_size],
            "time_step": [1, 1],
        }
        actual_inputs = {item.name: item.shape for item in self.session.get_inputs()}
        if actual_inputs != expected_inputs:
            raise ValueError(f"Motion ONNX inputs are {actual_inputs}, expected {expected_inputs}")
        expected_outputs = {
            "actions": [1, self.action_size],
            "joint_pos": [1, self.action_size],
            "joint_vel": [1, self.action_size],
            "body_pos_w": [1, 24, 3],
            "body_quat_w": [1, 24, 4],
            "body_lin_vel_w": [1, 24, 3],
            "body_ang_vel_w": [1, 24, 3],
        }
        actual_outputs = {item.name: item.shape for item in self.session.get_outputs()}
        if actual_outputs != expected_outputs:
            raise ValueError(f"Motion ONNX outputs are {actual_outputs}, expected {expected_outputs}")

        arrays = (self.default_q, self.kp, self.kd, self.action_scale)
        if any(array.shape != (self.action_size,) for array in arrays):
            raise ValueError("Motion metadata vectors must all contain 19 values")
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("Motion metadata contains NaN or Inf")
        if np.any(self.kp < 0.0) or np.any(self.kd < 0.0):
            raise ValueError("Motion metadata gains must be non-negative")
        if np.any(np.abs(self.action_scale) < 1.0e-8):
            raise ValueError("Motion action_scale must be non-zero")
        if not (0.0 < self.torque_scale <= 1.0):
            raise ValueError("Motion torque_scale must be in (0, 1]")
        if not (0.0 < self.effort_scale <= 1.0):
            raise ValueError("Motion effort_scale must be in (0, 1]")

    def reset_episode_to_reference(self) -> None:
        """Initialize one standalone episode from reference frame zero exactly once."""
        import mujoco

        if self.episode_initialized:
            raise RuntimeError("Reference-state reset is only permitted once at episode start")
        reference = self._run(self.zero_observation, self.start_step)
        reference_position = reference["body_pos_w"][0, self.anchor_motion_index].astype(np.float64)
        reference_quat = quat_normalize(reference["body_quat_w"][0, self.anchor_motion_index])

        self.data.qpos[self.joint_map.qpos_adr] = self.hold_target.q
        self.data.qpos[self.joint_map.qpos_adr[self.model_indices]] = reference["joint_pos"][
            0, self.policy_indices
        ]
        self.data.qpos[0:3] = reference_position
        self.data.qpos[3:7] = reference_quat
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        actual_quat = quat_normalize(self.data.xquat[self.anchor_body_id].copy())
        delta_quat = quat_multiply(reference_quat, quat_conjugate(actual_quat))
        self.data.qpos[3:7] = quat_normalize(quat_multiply(delta_quat, self.data.qpos[3:7]))
        mujoco.mj_forward(self.model, self.data)
        self.data.qpos[0:3] += reference_position - self.data.xpos[self.anchor_body_id]
        mujoco.mj_forward(self.model, self.data)

        self.last_action[:] = 0.0
        self.episode_initialized = True
        self.initialization_mode = "reference_reset"

    def begin_from_live_state(self) -> dict[str, Any]:
        """Initialize policy history from the current robot without changing simulation state."""
        reference = self._run(self.zero_observation, self.start_step)
        reference_quat = quat_normalize(reference["body_quat_w"][0, self.anchor_motion_index])
        robot_quat = quat_normalize(self.data.xquat[self.anchor_body_id].copy())
        self.yaw_delta = yaw_quat(yaw_from_quat(robot_quat) - yaw_from_quat(reference_quat))

        current_q = self.default_q.copy()
        current_q[self.policy_indices] = self.data.qpos[
            self.joint_map.qpos_adr[self.model_indices]
        ]
        previous_action = (current_q - self.default_q) / self.action_scale
        self.last_action[:] = np.clip(
            previous_action,
            -self.previous_action_clip,
            self.previous_action_clip,
        )
        self.episode_initialized = True
        self.initialization_mode = "live_state"
        self.live_start_count += 1

        reference_q = reference["joint_pos"][0].astype(np.float64)
        delta = reference_q - current_q
        return {
            "live_start_count": self.live_start_count,
            "maximum_abs_joint_delta": float(np.max(np.abs(delta))),
            "maximum_abs_previous_action": float(np.max(np.abs(self.last_action))),
            "yaw_delta": float(yaw_from_quat(self.yaw_delta)),
        }

    def _target_from_positions(self, policy_target: np.ndarray) -> ControlTarget:
        policy_target = np.asarray(policy_target, dtype=np.float64)
        if policy_target.shape != (self.action_size,):
            raise ValueError(f"Motion target must have shape (19,), got {policy_target.shape}")
        if not np.all(np.isfinite(policy_target)):
            raise ValueError("Motion target contains NaN or Inf")

        target = copy_target(self.hold_target, f"motion_evt2:{self.name}:{self.control_mode}")
        target.q[self.model_indices] = policy_target[self.policy_indices]
        target.kp[self.model_indices] = self.kp[self.policy_indices]
        target.kd[self.model_indices] = self.kd[self.policy_indices]
        target.effort[self.model_indices] = (
            self.joint_map.efforts[self.model_indices] * self.effort_scale
        )
        self.last_unclipped_target = target.q.copy()
        target.q = np.clip(target.q, self.joint_map.ranges[:, 0], self.joint_map.ranges[:, 1])
        self.last_target_clipped = target.q != self.last_unclipped_target
        if self.preserve_clipped_target_effort:
            target.feedforward += target.kp * (self.last_unclipped_target - target.q)
        target.torque_scale = self.torque_scale
        return target

    def first_target(self) -> ControlTarget:
        if not self.episode_initialized:
            raise RuntimeError(
                "Call reset_episode_to_reference() or begin_from_live_state() before requesting targets"
            )
        return super().first_target()

    def reference_start_target(self) -> ControlTarget:
        if not self.episode_initialized:
            raise RuntimeError(
                "Call reset_episode_to_reference() or begin_from_live_state() before requesting targets"
            )
        reference = self._reference(0)
        return self._target_from_positions(reference["joint_pos"][0].astype(np.float64))

    def step(self, policy_step: int) -> tuple[ControlTarget, np.ndarray, np.ndarray]:
        if not self.episode_initialized:
            raise RuntimeError(
                "Call reset_episode_to_reference() or begin_from_live_state() before stepping the motion"
            )
        target, observation, action = super().step(policy_step)
        if not (
            np.all(np.isfinite(target.q))
            and np.all(np.isfinite(observation))
            and np.all(np.isfinite(action))
        ):
            raise RuntimeError("Motion inference produced NaN or Inf")
        return target, observation, action
