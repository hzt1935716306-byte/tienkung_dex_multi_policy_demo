from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


def euler_xyz_to_quaternion_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quaternion_matrix(value: np.ndarray) -> np.ndarray:
    w, x, y, z = value / np.linalg.norm(value)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class ImuAdapter:
    """Converts official bodyctrl_msgs/Imu while preserving calibration status."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.source = str(config.get("orientation_source", "euler"))
        if self.source not in ("euler", "quaternion"):
            raise ValueError("orientation_source must be euler or quaternion")
        self.mount_quaternion = np.asarray(
            config.get("sensor_to_pelvis_wxyz", [1.0, 0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        if self.mount_quaternion.shape != (4,) or np.linalg.norm(self.mount_quaternion) < 1.0e-8:
            raise ValueError("sensor_to_pelvis_wxyz must be a valid quaternion")
        self.mount_quaternion /= np.linalg.norm(self.mount_quaternion)
        self.transform_confirmed = bool(config.get("transform_confirmed", False))
        self.allow_unconfirmed_shadow = bool(config.get("allow_unconfirmed_shadow", True))

    def convert(self, message: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, bool]]:
        if int(getattr(message, "error", 0)) != 0:
            raise ValueError(f"IMU reports error={int(message.error)}")
        if self.source == "euler":
            sensor_quaternion = euler_xyz_to_quaternion_wxyz(
                float(message.euler.roll),
                float(message.euler.pitch),
                float(message.euler.yaw),
            )
        else:
            sensor_quaternion = np.asarray(
                [
                    message.orientation.w,
                    message.orientation.x,
                    message.orientation.y,
                    message.orientation.z,
                ],
                dtype=np.float64,
            )
            if np.linalg.norm(sensor_quaternion) < 1.0e-8:
                raise ValueError("IMU quaternion has zero norm")
            sensor_quaternion /= np.linalg.norm(sensor_quaternion)
        orientation = quaternion_multiply(self.mount_quaternion, sensor_quaternion)
        orientation /= np.linalg.norm(orientation)
        angular_velocity_sensor = np.asarray(
            [message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z],
            dtype=np.float64,
        )
        acceleration_sensor = np.asarray(
            [message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z],
            dtype=np.float64,
        )
        sensor_to_pelvis = quaternion_matrix(self.mount_quaternion)
        angular_velocity = sensor_to_pelvis @ angular_velocity_sensor
        acceleration = sensor_to_pelvis @ acceleration_sensor
        if not all(np.all(np.isfinite(value)) for value in (orientation, angular_velocity, acceleration)):
            raise ValueError("IMU message contains NaN or Inf")
        usable = self.transform_confirmed or self.allow_unconfirmed_shadow
        return orientation, angular_velocity, acceleration, {
            "orientation_wxyz": usable,
            "angular_velocity_body": usable,
            "imu_transform_confirmed": self.transform_confirmed,
        }
