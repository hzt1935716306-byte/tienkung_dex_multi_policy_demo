from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from ..robot import ALL_JOINT_NAMES


@dataclass(frozen=True)
class HardwareJoint:
    official_index: int
    can_id: int
    official_name: str
    runtime_name: str
    group: str


# Open-X-Humanoid/Deploy_Tienkung 3.0, common/body_id_map.py.
OFFICIAL_HARDWARE_JOINTS = (
    HardwareJoint(0, 51, "l_hip_pitch", "hip_pitch_l_joint", "leg"),
    HardwareJoint(1, 52, "l_hip_roll", "hip_roll_l_joint", "leg"),
    HardwareJoint(2, 53, "l_hip_yaw", "hip_yaw_l_joint", "leg"),
    HardwareJoint(3, 54, "l_knee", "knee_pitch_l_joint", "leg"),
    HardwareJoint(4, 55, "l_ankle_pitch", "ankle_pitch_l_joint", "leg"),
    HardwareJoint(5, 56, "l_ankle_roll", "ankle_roll_l_joint", "leg"),
    HardwareJoint(6, 61, "r_hip_pitch", "hip_pitch_r_joint", "leg"),
    HardwareJoint(7, 62, "r_hip_roll", "hip_roll_r_joint", "leg"),
    HardwareJoint(8, 63, "r_hip_yaw", "hip_yaw_r_joint", "leg"),
    HardwareJoint(9, 64, "r_knee", "knee_pitch_r_joint", "leg"),
    HardwareJoint(10, 65, "r_ankle_pitch", "ankle_pitch_r_joint", "leg"),
    HardwareJoint(11, 66, "r_ankle_roll", "ankle_roll_r_joint", "leg"),
    HardwareJoint(12, 33, "waist_yaw", "waist_yaw_joint", "waist"),
    HardwareJoint(13, 32, "waist_roll", "waist_roll_joint", "waist"),
    HardwareJoint(14, 31, "waist_pitch", "waist_pitch_joint", "waist"),
    HardwareJoint(15, 11, "l_shoulder_pitch", "shoulder_pitch_l_joint", "arm"),
    HardwareJoint(16, 12, "l_shoulder_roll", "shoulder_roll_l_joint", "arm"),
    HardwareJoint(17, 13, "l_shoulder_yaw", "shoulder_yaw_l_joint", "arm"),
    HardwareJoint(18, 14, "l_elbow", "elbow_pitch_l_joint", "arm"),
    HardwareJoint(19, 15, "l_wrist_yaw", "elbow_yaw_l_joint", "arm"),
    HardwareJoint(20, 16, "l_wrist_pitch", "wrist_pitch_l_joint", "arm"),
    HardwareJoint(21, 17, "l_wrist_roll", "wrist_roll_l_joint", "arm"),
    HardwareJoint(22, 21, "r_shoulder_pitch", "shoulder_pitch_r_joint", "arm"),
    HardwareJoint(23, 22, "r_shoulder_roll", "shoulder_roll_r_joint", "arm"),
    HardwareJoint(24, 23, "r_shoulder_yaw", "shoulder_yaw_r_joint", "arm"),
    HardwareJoint(25, 24, "r_elbow", "elbow_pitch_r_joint", "arm"),
    HardwareJoint(26, 25, "r_wrist_yaw", "elbow_yaw_r_joint", "arm"),
    HardwareJoint(27, 26, "r_wrist_pitch", "wrist_pitch_r_joint", "arm"),
    HardwareJoint(28, 27, "r_wrist_roll", "wrist_roll_r_joint", "arm"),
)


@dataclass(frozen=True)
class MappedMotorState:
    position: np.ndarray
    velocity: np.ndarray
    torque: np.ndarray
    temperature: np.ndarray
    error: np.ndarray


class HardwareJointMapper:
    """Maps official CAN IDs to the repository's canonical 29-joint order."""

    def __init__(
        self,
        *,
        zero_position: Iterable[float],
        direction: Iterable[float],
        semantic_offset: Iterable[float],
        current_to_torque: Iterable[float],
    ) -> None:
        self.runtime_names = tuple(ALL_JOINT_NAMES)
        self.runtime_index = {name: index for index, name in enumerate(self.runtime_names)}
        self.by_can_id = {item.can_id: item for item in OFFICIAL_HARDWARE_JOINTS}
        self.by_runtime_name = {item.runtime_name: item for item in OFFICIAL_HARDWARE_JOINTS}
        if len(self.by_can_id) != 29 or set(self.by_runtime_name) != set(self.runtime_names):
            raise RuntimeError("Official hardware map does not cover the canonical 29 joints")
        self.zero_position = self._official_vector(zero_position, "zero_position")
        self.direction = self._official_vector(direction, "direction")
        self.semantic_offset = self._official_vector(semantic_offset, "semantic_offset")
        self.current_to_torque = self._official_vector(current_to_torque, "current_to_torque")
        if not np.all(np.isin(self.direction, (-1.0, 1.0))):
            raise ValueError("Each hardware direction must be -1 or +1")
        if np.any(self.current_to_torque <= 0.0):
            raise ValueError("current_to_torque values must be positive")

    @staticmethod
    def _official_vector(value: Iterable[float], name: str) -> np.ndarray:
        result = np.asarray(list(value), dtype=np.float64)
        if result.shape != (29,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain 29 finite values in official index order")
        return result

    @property
    def ankle_runtime_indices(self) -> np.ndarray:
        return np.asarray(
            [
                self.runtime_index["ankle_pitch_l_joint"],
                self.runtime_index["ankle_roll_l_joint"],
                self.runtime_index["ankle_pitch_r_joint"],
                self.runtime_index["ankle_roll_r_joint"],
            ],
            dtype=np.int32,
        )

    def map_statuses(self, statuses: Iterable[object]) -> MappedMotorState:
        values = list(statuses)
        seen: set[int] = set()
        position = np.full(29, np.nan, dtype=np.float64)
        velocity = np.full(29, np.nan, dtype=np.float64)
        torque = np.full(29, np.nan, dtype=np.float64)
        temperature = np.full(29, np.nan, dtype=np.float64)
        error = np.full(29, np.nan, dtype=np.float64)
        for status in values:
            can_id = int(status.name)
            if can_id not in self.by_can_id:
                raise ValueError(f"Unknown motor CAN ID {can_id}")
            if can_id in seen:
                raise ValueError(f"Duplicate motor CAN ID {can_id}")
            seen.add(can_id)
            item = self.by_can_id[can_id]
            official = item.official_index
            runtime = self.runtime_index[item.runtime_name]
            direction = self.direction[official]
            position[runtime] = (
                (float(status.pos) - self.zero_position[official]) * direction
                + self.semantic_offset[official]
            )
            velocity[runtime] = float(status.speed) * direction
            torque[runtime] = float(status.current) * self.current_to_torque[official] * direction
            temperature[runtime] = float(status.temperature)
            error[runtime] = float(status.error)
        missing = sorted(set(self.by_can_id) - seen)
        if missing:
            raise ValueError(f"Missing motor CAN IDs: {missing}")
        return MappedMotorState(position, velocity, torque, temperature, error)

    def runtime_to_official(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (29,):
            raise ValueError("runtime vector must contain 29 values")
        result = np.zeros(29, dtype=np.float64)
        for item in OFFICIAL_HARDWARE_JOINTS:
            result[item.official_index] = values[self.runtime_index[item.runtime_name]]
        return result

    def command_to_hardware_position(self, runtime_position: np.ndarray) -> np.ndarray:
        semantic = self.runtime_to_official(runtime_position)
        return (semantic - self.semantic_offset) * self.direction + self.zero_position

    def command_to_hardware_signed(self, runtime_value: np.ndarray) -> np.ndarray:
        return self.runtime_to_official(runtime_value) * self.direction

    def split_official(self, values: np.ndarray) -> Mapping[str, np.ndarray]:
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (29,):
            raise ValueError("official vector must contain 29 values")
        return {"leg": values[:12], "waist": values[12:15], "arm": values[15:]}
