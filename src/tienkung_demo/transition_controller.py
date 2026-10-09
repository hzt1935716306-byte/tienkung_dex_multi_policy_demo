from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .robot import ControlTarget, JointMap


def quintic_alpha(progress: float) -> float:
    """Minimum-jerk interpolation with zero endpoint velocity and acceleration."""
    s = float(np.clip(progress, 0.0, 1.0))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def copy_target(target: ControlTarget, label: str | None = None) -> ControlTarget:
    return ControlTarget(
        q=target.q.copy(),
        kp=target.kp.copy(),
        kd=target.kd.copy(),
        feedforward=target.feedforward.copy(),
        effort=target.effort.copy(),
        torque_scale=float(target.torque_scale),
        label=target.label if label is None else label,
    )


def blend_targets(
    first: ControlTarget,
    second: ControlTarget,
    progress: float,
    label: str,
) -> ControlTarget:
    """Blend two complete physical targets after both policies have been evaluated."""
    alpha = quintic_alpha(progress)
    return ControlTarget(
        q=(1.0 - alpha) * first.q + alpha * second.q,
        kp=(1.0 - alpha) * first.kp + alpha * second.kp,
        kd=(1.0 - alpha) * first.kd + alpha * second.kd,
        feedforward=(1.0 - alpha) * first.feedforward + alpha * second.feedforward,
        effort=(1.0 - alpha) * first.effort + alpha * second.effort,
        torque_scale=(1.0 - alpha) * first.torque_scale + alpha * second.torque_scale,
        label=label,
    )


def prealign_target(
    balance_target: ControlTarget,
    motion_target: ControlTarget,
    progress: float,
) -> ControlTarget:
    """Move desired posture while retaining WALKAMP feedback and effort settings."""
    target = copy_target(balance_target, "pre_align")
    alpha = quintic_alpha(progress)
    target.q = (1.0 - alpha) * balance_target.q + alpha * motion_target.q
    return target


def values_by_joint_group(
    joint_map: JointMap,
    values: dict[str, float],
) -> np.ndarray:
    def group(name: str) -> str:
        if any(token in name for token in ("hip_", "knee_", "ankle_")):
            return "legs"
        if name.startswith("waist_"):
            return "waist"
        return "arms"

    fallback = float(values.get("default", np.inf))
    return np.asarray(
        [float(values.get(group(name), fallback)) for name in joint_map.names],
        dtype=np.float64,
    )


class TargetRateLimiter:
    def __init__(self, joint_map: JointMap, control_dt: float, rates: dict[str, float]) -> None:
        self.control_dt = float(control_dt)
        self.maximum_rates = values_by_joint_group(joint_map, rates)
        if np.any(self.maximum_rates <= 0.0):
            raise ValueError("All target rate limits must be positive")
        self.previous_q: np.ndarray | None = None
        self.last_maximum_rate = 0.0

    def reset(self, positions: np.ndarray) -> None:
        positions = np.asarray(positions, dtype=np.float64)
        self.previous_q = positions.copy()
        self.last_maximum_rate = 0.0

    def apply(self, target: ControlTarget) -> ControlTarget:
        limited = copy_target(target)
        if self.previous_q is None:
            self.previous_q = limited.q.copy()
            return limited
        requested_delta = limited.q - self.previous_q
        maximum_delta = self.maximum_rates * self.control_dt
        applied_delta = np.clip(requested_delta, -maximum_delta, maximum_delta)
        limited.q = self.previous_q + applied_delta
        self.last_maximum_rate = float(np.max(np.abs(applied_delta) / self.control_dt))
        self.previous_q = limited.q.copy()
        return limited


@dataclass
class TorqueResult:
    torque: np.ndarray
    unconstrained_torque: np.ndarray
    saturation_fraction: float
    slew_limited_fraction: float
    maximum_rate: float


class ConstrainedPDController:
    def __init__(
        self,
        joint_map: JointMap,
        control_dt: float,
        torque_rates: dict[str, float],
    ) -> None:
        self.joint_map = joint_map
        self.control_dt = float(control_dt)
        self.maximum_rates = values_by_joint_group(joint_map, torque_rates)
        if np.any(self.maximum_rates <= 0.0):
            raise ValueError("All torque rate limits must be positive")
        self.previous_torque = np.zeros(len(joint_map.names), dtype=np.float64)

    def reset(self, torque: np.ndarray | None = None) -> None:
        if torque is None:
            self.previous_torque[:] = 0.0
        else:
            self.previous_torque[:] = np.asarray(torque, dtype=np.float64)

    def apply(self, data, target: ControlTarget) -> TorqueResult:
        q = data.qpos[self.joint_map.qpos_adr]
        qd = data.qvel[self.joint_map.qvel_adr]
        unconstrained = target.torque_scale * (
            target.kp * (target.q - q) - target.kd * qd + target.feedforward
        )
        effort_limited = np.clip(unconstrained, -target.effort, target.effort)
        maximum_delta = self.maximum_rates * self.control_dt
        torque_delta = np.clip(
            effort_limited - self.previous_torque,
            -maximum_delta,
            maximum_delta,
        )
        torque = self.previous_torque + torque_delta
        saturation = np.abs(unconstrained) > target.effort + 1.0e-9
        slew_limited = np.abs(effort_limited - self.previous_torque) > maximum_delta + 1.0e-9
        maximum_rate = float(np.max(np.abs(torque - self.previous_torque) / self.control_dt))
        self.previous_torque[:] = torque
        data.ctrl[:] = 0.0
        data.ctrl[self.joint_map.actuator_ids] = torque
        return TorqueResult(
            torque=torque.copy(),
            unconstrained_torque=unconstrained,
            saturation_fraction=float(np.mean(saturation)),
            slew_limited_fraction=float(np.mean(slew_limited)),
            maximum_rate=maximum_rate,
        )
