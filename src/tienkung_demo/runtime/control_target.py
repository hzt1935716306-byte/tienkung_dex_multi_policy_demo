from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ControlTarget:
    """One complete joint-space command before hardware-specific conversion."""

    q: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    feedforward: np.ndarray
    effort: np.ndarray
    torque_scale: float
    label: str

    def validate(self, joint_count: int = 29) -> None:
        for name in ("q", "kp", "kd", "feedforward", "effort"):
            value = np.asarray(getattr(self, name))
            if value.shape != (joint_count,):
                raise ValueError(f"ControlTarget.{name} must have shape ({joint_count},), got {value.shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"ControlTarget.{name} contains NaN or Inf")
        if np.any(self.kp < 0.0) or np.any(self.kd < 0.0):
            raise ValueError("ControlTarget gains must be non-negative")
        if np.any(self.effort <= 0.0):
            raise ValueError("ControlTarget effort limits must be positive")
        if not np.isfinite(self.torque_scale) or self.torque_scale <= 0.0:
            raise ValueError("ControlTarget torque_scale must be finite and positive")


def copy_control_target(target: ControlTarget, label: str | None = None) -> ControlTarget:
    return ControlTarget(
        q=np.asarray(target.q, dtype=np.float64).copy(),
        kp=np.asarray(target.kp, dtype=np.float64).copy(),
        kd=np.asarray(target.kd, dtype=np.float64).copy(),
        feedforward=np.asarray(target.feedforward, dtype=np.float64).copy(),
        effort=np.asarray(target.effort, dtype=np.float64).copy(),
        torque_scale=float(target.torque_scale),
        label=target.label if label is None else label,
    )
