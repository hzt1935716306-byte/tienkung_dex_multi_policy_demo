from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


def _vector(value: Any, size: int, name: str, *, finite: bool = True) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
    if finite and not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or Inf")
    return result


@dataclass(frozen=True)
class RobotStateSnapshot:
    """Timestamped state with explicit validity for measured and estimated fields.

    Joint values use ``joint_names`` order. Quaternion values are scalar-first
    ``[w, x, y, z]`` and angular velocity is expressed in the pelvis/body frame.
    Optional fields remain ``None`` when the real interface cannot measure or
    estimate them; callers must never replace them with invented zeroes.
    """

    timestamp: float
    received_monotonic: float
    source: str
    joint_names: tuple[str, ...]
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    orientation_wxyz: np.ndarray
    angular_velocity_body: np.ndarray
    joint_torque: np.ndarray | None = None
    joint_temperature: np.ndarray | None = None
    motor_error: np.ndarray | None = None
    linear_acceleration_body: np.ndarray | None = None
    base_position_world: np.ndarray | None = None
    base_linear_velocity_world: np.ndarray | None = None
    foot_contact: np.ndarray | None = None
    foot_normal_force: np.ndarray | None = None
    center_of_mass_world: np.ndarray | None = None
    validity: Mapping[str, bool] = field(default_factory=dict)
    source_timestamps: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        count = len(self.joint_names)
        if count != 29 or len(set(self.joint_names)) != count:
            raise ValueError("RobotStateSnapshot requires 29 unique joint names")
        if not np.isfinite(self.timestamp) or not np.isfinite(self.received_monotonic):
            raise ValueError("snapshot timestamps must be finite")
        if not self.source:
            raise ValueError("snapshot source must not be empty")
        object.__setattr__(self, "joint_position", _vector(self.joint_position, count, "joint_position"))
        object.__setattr__(self, "joint_velocity", _vector(self.joint_velocity, count, "joint_velocity"))
        quaternion = _vector(self.orientation_wxyz, 4, "orientation_wxyz")
        norm = float(np.linalg.norm(quaternion))
        if norm < 1.0e-8:
            raise ValueError("orientation_wxyz has zero norm")
        object.__setattr__(self, "orientation_wxyz", quaternion / norm)
        object.__setattr__(
            self,
            "angular_velocity_body",
            _vector(self.angular_velocity_body, 3, "angular_velocity_body"),
        )
        optional_vectors = {
            "joint_torque": count,
            "joint_temperature": count,
            "motor_error": count,
            "linear_acceleration_body": 3,
            "base_position_world": 3,
            "base_linear_velocity_world": 3,
            "foot_contact": 2,
            "foot_normal_force": 2,
            "center_of_mass_world": 3,
        }
        for name, size in optional_vectors.items():
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _vector(value, size, name))
        object.__setattr__(self, "validity", dict(self.validity))
        object.__setattr__(self, "source_timestamps", dict(self.source_timestamps))

    @property
    def index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.joint_names)}

    def select_position(self, names: list[str] | tuple[str, ...]) -> np.ndarray:
        index = self.index
        return self.joint_position[[index[name] for name in names]].copy()

    def select_velocity(self, names: list[str] | tuple[str, ...]) -> np.ndarray:
        index = self.index
        return self.joint_velocity[[index[name] for name in names]].copy()

    def is_valid(self, name: str) -> bool:
        return bool(self.validity.get(name, getattr(self, name, None) is not None))

    def age(self, now_monotonic: float) -> float:
        return max(0.0, float(now_monotonic) - self.received_monotonic)

    def to_json_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": "tienkung.robot_state.v1",
            "timestamp": self.timestamp,
            "received_monotonic": self.received_monotonic,
            "source": self.source,
            "joint_names": list(self.joint_names),
            "joint_position": self.joint_position.tolist(),
            "joint_velocity": self.joint_velocity.tolist(),
            "orientation_wxyz": self.orientation_wxyz.tolist(),
            "angular_velocity_body": self.angular_velocity_body.tolist(),
            "validity": dict(self.validity),
            "source_timestamps": dict(self.source_timestamps),
        }
        for name in (
            "joint_torque",
            "joint_temperature",
            "motor_error",
            "linear_acceleration_body",
            "base_position_world",
            "base_linear_velocity_world",
            "foot_contact",
            "foot_normal_force",
            "center_of_mass_world",
        ):
            value = getattr(self, name)
            result[name] = None if value is None else value.tolist()
        return result

    @classmethod
    def from_json_dict(cls, value: Mapping[str, Any]) -> "RobotStateSnapshot":
        if value.get("schema") not in (None, "tienkung.robot_state.v1"):
            raise ValueError(f"Unsupported robot state schema {value.get('schema')!r}")
        fields = dict(value)
        fields.pop("schema", None)
        fields["joint_names"] = tuple(fields["joint_names"])
        return cls(**fields)

    @classmethod
    def from_mujoco(
        cls,
        data: Any,
        joint_map: Any,
        timestamp: float,
        *,
        received_monotonic: float | None = None,
        foot_contact: np.ndarray | None = None,
        foot_normal_force: np.ndarray | None = None,
    ) -> "RobotStateSnapshot":
        orientation = np.asarray(data.sensor("orientation").data, dtype=np.float64)
        angular_velocity = np.asarray(data.sensor("angular-velocity").data, dtype=np.float64)
        return cls(
            timestamp=float(timestamp),
            received_monotonic=float(timestamp if received_monotonic is None else received_monotonic),
            source="mujoco",
            joint_names=tuple(joint_map.names),
            joint_position=data.qpos[joint_map.qpos_adr],
            joint_velocity=data.qvel[joint_map.qvel_adr],
            orientation_wxyz=orientation,
            angular_velocity_body=angular_velocity,
            joint_torque=np.zeros(len(joint_map.names), dtype=np.float64),
            base_position_world=data.qpos[:3],
            base_linear_velocity_world=data.qvel[:3],
            foot_contact=foot_contact,
            foot_normal_force=foot_normal_force,
            validity={
                "joint_position": True,
                "joint_velocity": True,
                "orientation_wxyz": True,
                "angular_velocity_body": True,
                "base_position_world": True,
                "base_linear_velocity_world": True,
                "foot_contact": foot_contact is not None,
                "foot_normal_force": foot_normal_force is not None,
            },
        )
