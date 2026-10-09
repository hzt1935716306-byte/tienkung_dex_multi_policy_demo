from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .policies import inverse_rotate, quat_normalize
from .robot import ControlTarget, JointMap


WALKAMP_JOINT_NAMES = [
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
]

WALKAMP_UNCONTROLLED_JOINT_NAMES = [
    "elbow_yaw_l_joint",
    "wrist_pitch_l_joint",
    "wrist_roll_l_joint",
    "elbow_yaw_r_joint",
    "wrist_pitch_r_joint",
    "wrist_roll_r_joint",
]

OFFICIAL_DEFAULT_ANGLES = np.array(
    [
        -0.15,
        -0.15,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.3,
        0.3,
        0.2,
        0.2,
        -0.15,
        -0.15,
        0.1,
        -0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.5,
        -0.5,
    ],
    dtype=np.float64,
)

OFFICIAL_KP = np.array(
    [
        300.0,
        300.0,
        400.0,
        300.0,
        300.0,
        400.0,
        150.0,
        150.0,
        400.0,
        350.0,
        350.0,
        150.0,
        150.0,
        30.0,
        30.0,
        50.0,
        50.0,
        16.8,
        16.8,
        50.0,
        50.0,
        150.0,
        150.0,
    ],
    dtype=np.float64,
)

OFFICIAL_KD = np.array(
    [
        10.0,
        10.0,
        5.0,
        10.0,
        10.0,
        10.0,
        5.0,
        5.0,
        10.0,
        10.0,
        10.0,
        7.5,
        7.5,
        2.5,
        2.5,
        2.5,
        2.5,
        1.4,
        1.4,
        2.5,
        2.5,
        5.0,
        5.0,
    ],
    dtype=np.float64,
)


def _resolve_path(value: str, config_dir: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return str(path.resolve())


def load_walkamp_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a JSON object: {config_path}")

    config["_path"] = str(config_path)
    config_dir = config_path.parent
    for field in ("model", "policy"):
        config[field] = _resolve_path(config[field], config_dir)
        if not Path(config[field]).is_file():
            raise FileNotFoundError(config[field])
    return config


def gait_phase(
    timer: float,
    gait_cycle: float,
    left_offset: float,
    right_offset: float,
    left_ratio: float,
    right_ratio: float,
) -> np.ndarray:
    left_phase = (timer / gait_cycle + left_offset) % 1.0
    right_phase = (timer / gait_cycle + right_offset) % 1.0
    return np.array(
        [
            math.sin(2.0 * math.pi * left_phase),
            math.sin(2.0 * math.pi * right_phase),
            math.cos(2.0 * math.pi * left_phase),
            math.cos(2.0 * math.pi * right_phase),
            left_ratio,
            right_ratio,
        ],
        dtype=np.float32,
    )


class WalkAmpPolicy:
    """Official WALKAMP inference adapter for the 29-joint EVT2 MuJoCo model."""

    observation_size = 84
    history_length = 10
    action_size = 23

    def __init__(
        self,
        model,
        data,
        joint_map: JointMap,
        policy_path: str | Path,
        config: dict[str, Any],
        control_dt: float,
    ) -> None:
        import mujoco
        import onnxruntime as ort

        self.model = model
        self.data = data
        self.joint_map = joint_map
        self.config = config
        self.policy_path = Path(policy_path)

        configured_names = list(config.get("joint_names", WALKAMP_JOINT_NAMES))
        if configured_names != WALKAMP_JOINT_NAMES:
            raise ValueError("walkamp.joint_names must match the official WALKAMP order exactly")
        missing = [name for name in WALKAMP_JOINT_NAMES if name not in joint_map.index]
        if missing:
            raise ValueError(f"MuJoCo model is missing WALKAMP joints: {missing}")
        uncontrolled = list(config.get("uncontrolled_joints", {}).get("joint_names", []))
        if uncontrolled != WALKAMP_UNCONTROLLED_JOINT_NAMES:
            raise ValueError("walkamp.uncontrolled_joints.joint_names must list the six non-policy joints")
        if len(joint_map.names) != 29:
            raise ValueError(f"WALKAMP requires the complete 29-joint model, found {len(joint_map.names)}")

        self.policy_indices = np.array([joint_map.index[name] for name in WALKAMP_JOINT_NAMES], dtype=np.int32)
        self.uncontrolled_indices = np.array(
            [joint_map.index[name] for name in WALKAMP_UNCONTROLLED_JOINT_NAMES], dtype=np.int32
        )
        self.default_angles = np.asarray(config.get("default_joint_angles", []), dtype=np.float64)
        self.kp = np.asarray(config.get("kp", []), dtype=np.float64)
        self.kd = np.asarray(config.get("kd", []), dtype=np.float64)
        if not np.array_equal(self.default_angles, OFFICIAL_DEFAULT_ANGLES):
            raise ValueError("walkamp.default_joint_angles differs from official walk_amp.yaml")
        if not np.array_equal(self.kp, OFFICIAL_KP) or not np.array_equal(self.kd, OFFICIAL_KD):
            raise ValueError("walkamp kp/kd differs from official walk_amp.yaml")

        exact_integer_values = {
            "motor_num": 29,
            "action_size": self.action_size,
            "observation_size": self.observation_size,
            "history_length": self.history_length,
            "decimation": 1,
        }
        for name, expected in exact_integer_values.items():
            if int(config.get(name, -1)) != expected:
                raise ValueError(f"walkamp.{name} must equal the official value {expected}")
        exact_float_values = {
            "policy_dt": 0.01,
            "warm_start_time": 0.0,
            "action_scale": 0.25,
            "clip_observations": 100.0,
            "clip_actions": 100.0,
            "gait_cycle": 0.85,
        }
        for name, expected in exact_float_values.items():
            if not math.isclose(float(config.get(name, math.nan)), expected, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(f"walkamp.{name} must equal the official value {expected}")
        scales = config.get("observation_scales", {})
        expected_scale_names = (
            "linear_velocity",
            "angular_velocity",
            "joint_position",
            "joint_velocity",
        )
        if any(float(scales.get(name, math.nan)) != 1.0 for name in expected_scale_names):
            raise ValueError("all official WALKAMP observation scales must equal 1.0")

        self.policy_dt = float(config.get("policy_dt", 0.01))
        self.decimation = int(config.get("decimation", 1))
        if not math.isclose(control_dt, self.policy_dt * self.decimation, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                f"control_dt={control_dt} must equal policy_dt*decimation="
                f"{self.policy_dt * self.decimation}"
            )
        self.action_scale = float(config.get("action_scale", 0.25))
        self.clip_observations = float(config.get("clip_observations", 100.0))
        self.clip_actions = float(config.get("clip_actions", 100.0))
        self.gait_cycle = float(config.get("gait_cycle", 0.85))
        self.phase_ratio = np.asarray(config.get("phase_ratio", [0.38, 0.38]), dtype=np.float64)
        self.phase_offset = np.asarray(config.get("phase_offset", [0.38, 0.88]), dtype=np.float64)
        if self.phase_ratio.shape != (2,) or self.phase_offset.shape != (2,):
            raise ValueError("phase_ratio and phase_offset must each contain two values")
        if not np.array_equal(self.phase_ratio, np.array([0.38, 0.38])):
            raise ValueError("walkamp.phase_ratio differs from official fsm_walkamp.py")
        if not np.array_equal(self.phase_offset, np.array([0.38, 0.88])):
            raise ValueError("walkamp.phase_offset differs from official fsm_walkamp.py")

        hold = config["uncontrolled_joints"]
        self.hold_positions = np.asarray(hold["positions"], dtype=np.float64)
        self.hold_kp = np.asarray(hold["kp"], dtype=np.float64)
        self.hold_kd = np.asarray(hold["kd"], dtype=np.float64)
        self.hold_effort = np.asarray(hold["effort_limits"], dtype=np.float64)
        for name, values in (
            ("positions", self.hold_positions),
            ("kp", self.hold_kp),
            ("kd", self.hold_kd),
            ("effort_limits", self.hold_effort),
        ):
            if values.shape != (6,):
                raise ValueError(f"uncontrolled_joints.{name} must contain six values")
        if np.any(self.hold_effort <= 0.0):
            raise ValueError("uncontrolled joint effort limits must be positive")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.enable_mem_pattern = False
        options.enable_mem_reuse = True
        self.session = ort.InferenceSession(
            str(self.policy_path), options, providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 840]:
            raise RuntimeError(f"Official WALKAMP input must be [1, 840], got {[item.shape for item in inputs]}")
        if not outputs or outputs[0].shape != [1, 23]:
            raise RuntimeError(f"Official WALKAMP output must be [1, 23], got {[item.shape for item in outputs]}")
        self.input_name = inputs[0].name

        orientation_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation")
        angular_velocity_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity"
        )
        if orientation_id < 0 or angular_velocity_id < 0:
            raise ValueError("Official WALKAMP requires orientation and angular-velocity sensors")

        self.command = np.zeros(3, dtype=np.float32)
        self.last_actions = np.zeros(self.action_size, dtype=np.float32)
        self.actions = np.zeros(self.action_size, dtype=np.float32)
        self.history = np.zeros(self.observation_size * self.history_length, dtype=np.float32)
        self.timer_gait = 0.0
        self.first_observation = True
        self.last_observation = np.zeros(self.observation_size, dtype=np.float32)
        self.last_target: ControlTarget | None = None

    def set_command(self, command: np.ndarray | list[float] | tuple[float, float, float]) -> None:
        command_array = np.asarray(command, dtype=np.float32)
        if command_array.shape != (3,):
            raise ValueError("WALKAMP command must be [vx, vy, yaw_rate]")
        limits = np.asarray(self.config.get("command_limits", [1.0, 0.5, 1.57]), dtype=np.float32)
        self.command[:] = np.clip(command_array, -limits, limits)

    def adjust_command(self, index: int, amount: float) -> None:
        command = self.command.copy()
        command[index] += amount
        self.set_command(command)
        print(
            f"[INFO] WALKAMP command vx={self.command[0]:.2f} "
            f"vy={self.command[1]:.2f} yaw={self.command[2]:.2f}"
        )

    def reset(self) -> None:
        self.command[:] = 0.0
        self.last_actions[:] = 0.0
        self.actions[:] = 0.0
        self.history[:] = 0.0
        self.timer_gait = 0.0
        self.first_observation = True
        self.last_target = None

    def neutral_target(self, label: str = "walkamp_hold") -> ControlTarget:
        q = np.zeros(len(self.joint_map.names), dtype=np.float64)
        kp = np.zeros_like(q)
        kd = np.zeros_like(q)
        effort = self.joint_map.efforts.copy()
        q[self.policy_indices] = self.default_angles
        kp[self.policy_indices] = self.kp
        kd[self.policy_indices] = self.kd
        q[self.uncontrolled_indices] = self.hold_positions
        kp[self.uncontrolled_indices] = self.hold_kp
        kd[self.uncontrolled_indices] = self.hold_kd
        effort[self.uncontrolled_indices] = np.minimum(
            effort[self.uncontrolled_indices], self.hold_effort
        )
        q = np.clip(q, self.joint_map.ranges[:, 0], self.joint_map.ranges[:, 1])
        return ControlTarget(
            q=q,
            kp=kp,
            kd=kd,
            feedforward=np.zeros(len(self.joint_map.names), dtype=np.float64),
            effort=effort,
            torque_scale=1.0,
            label=label,
        )

    def _single_observation(self) -> np.ndarray:
        orientation = quat_normalize(self.data.sensor("orientation").data.copy())
        angular_velocity = self.data.sensor("angular-velocity").data.copy()
        q = self.data.qpos[self.joint_map.qpos_adr[self.policy_indices]]
        qd = self.data.qvel[self.joint_map.qvel_adr[self.policy_indices]]
        phase = gait_phase(
            self.timer_gait,
            self.gait_cycle,
            self.phase_offset[0],
            self.phase_offset[1],
            self.phase_ratio[0],
            self.phase_ratio[1],
        )
        observation = np.concatenate(
            [
                angular_velocity,
                inverse_rotate(orientation, np.array([0.0, 0.0, -1.0])),
                self.command,
                q - self.default_angles,
                qd,
                self.last_actions,
                phase,
            ]
        ).astype(np.float32)
        if observation.shape != (self.observation_size,):
            raise RuntimeError(
                f"WALKAMP observation has shape {observation.shape}, expected ({self.observation_size},)"
            )
        return observation

    def _update_history(self, observation: np.ndarray) -> None:
        if self.first_observation:
            self.history[:] = np.tile(observation, self.history_length)
            self.first_observation = False
        else:
            self.history[:-self.observation_size] = self.history[self.observation_size :]
            self.history[-self.observation_size :] = observation

    def step(self) -> tuple[ControlTarget, np.ndarray, np.ndarray]:
        observation = self._single_observation()
        self._update_history(observation)
        clipped_history = np.clip(
            self.history, -self.clip_observations, self.clip_observations
        ).astype(np.float32)
        output = self.session.run(None, {self.input_name: clipped_history.reshape(1, -1)})[0][0]
        self.actions[:] = np.clip(output, -self.clip_actions, self.clip_actions)

        target = self.neutral_target("walkamp_policy")
        target.q[self.policy_indices] = self.default_angles + self.actions * self.action_scale
        target.q = np.clip(target.q, self.joint_map.ranges[:, 0], self.joint_map.ranges[:, 1])

        self.last_actions[:] = self.actions
        self.last_observation[:] = observation
        self.timer_gait += self.policy_dt
        self.last_target = target
        return target, clipped_history, self.actions.copy()
