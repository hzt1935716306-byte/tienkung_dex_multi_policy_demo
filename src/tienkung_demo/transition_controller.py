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


def endpoint_progress(step: int, duration_steps: int) -> float:
    """Return a stage progress with exact zero and one endpoints."""
    if duration_steps <= 1:
        return 1.0
    return float(np.clip(step / (duration_steps - 1), 0.0, 1.0))


def joint_group_indices(joint_map: JointMap) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {"legs": [], "waist": [], "arms": []}
    for index, name in enumerate(joint_map.names):
        if any(token in name for token in ("hip_", "knee_", "ankle_")):
            groups["legs"].append(index)
        elif name.startswith("waist_"):
            groups["waist"].append(index)
        else:
            groups["arms"].append(index)
    return {
        name: np.asarray(indices, dtype=np.int32)
        for name, indices in groups.items()
    }


def joint_group_weights(
    joint_map: JointMap,
    group_weights: dict[str, float],
) -> np.ndarray:
    unknown = set(group_weights) - {"legs", "waist", "arms"}
    if unknown:
        raise ValueError(f"Unknown joint groups: {sorted(unknown)}")
    if set(group_weights) != {"legs", "waist", "arms"}:
        raise ValueError("Joint group weights must define legs, waist, and arms")
    groups = joint_group_indices(joint_map)
    weights = np.zeros(len(joint_map.names), dtype=np.float64)
    for group, indices in groups.items():
        value = float(group_weights[group])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Joint group weight {group}={value} is outside [0, 1]")
        weights[indices] = value
    return weights


def blend_targets_with_joint_weights(
    first: ControlTarget,
    second: ControlTarget,
    weights: np.ndarray,
    label: str,
) -> ControlTarget:
    """Blend complete physical targets with one interpolation weight per joint."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != first.q.shape or weights.shape != second.q.shape:
        raise ValueError("Joint weights must match both control targets")
    if np.any(weights < 0.0) or np.any(weights > 1.0):
        raise ValueError("Joint weights must stay inside [0, 1]")
    inverse = 1.0 - weights
    scalar_weight = float(np.mean(weights))
    return ControlTarget(
        q=inverse * first.q + weights * second.q,
        kp=inverse * first.kp + weights * second.kp,
        kd=inverse * first.kd + weights * second.kd,
        feedforward=inverse * first.feedforward + weights * second.feedforward,
        effort=inverse * first.effort + weights * second.effort,
        torque_scale=(
            (1.0 - scalar_weight) * first.torque_scale
            + scalar_weight * second.torque_scale
        ),
        label=label,
    )


def prealign_group_target(
    balance_target: ControlTarget,
    motion_target: ControlTarget,
    group_alphas: dict[str, float],
    joint_map: JointMap,
) -> ControlTarget:
    """Pre-align selected joint positions while retaining balance-controller gains."""
    target = copy_target(balance_target, "pre_align_grouped")
    weights = joint_group_weights(joint_map, group_alphas)
    target.q = (1.0 - weights) * balance_target.q + weights * motion_target.q
    return target


def continuous_group_transition_target(
    anchor_target: ControlTarget,
    balance_start: ControlTarget,
    balance_target: ControlTarget,
    motion_target: ControlTarget,
    group_alphas: dict[str, float],
    joint_map: JointMap,
    anchored_groups: set[str] | None = None,
) -> ControlTarget:
    """Blend from the last applied target without resetting to a policy endpoint.

    Non-anchored groups retain live balance-policy changes. Their initial source
    offset is decayed with progress so the first sample is exactly the anchor.
    Anchored groups use the anchor directly, which prevents already pre-aligned
    joints from following the balance policy backwards.
    """
    anchored_groups = set(anchored_groups or ())
    unknown = anchored_groups - {"legs", "waist", "arms"}
    if unknown:
        raise ValueError(f"Unknown anchored joint groups: {sorted(unknown)}")
    weights = joint_group_weights(joint_map, group_alphas)
    source = copy_target(balance_target, "continuous_entry_source")
    groups = joint_group_indices(joint_map)
    vector_fields = ("q", "kp", "kd", "feedforward", "effort")
    for group, indices in groups.items():
        alpha = float(group_alphas[group])
        if group in anchored_groups:
            for field in vector_fields:
                getattr(source, field)[indices] = getattr(anchor_target, field)[indices]
        else:
            correction = 1.0 - alpha
            for field in vector_fields:
                current = getattr(source, field)
                current[indices] += correction * (
                    getattr(anchor_target, field)[indices]
                    - getattr(balance_start, field)[indices]
                )

    scalar_alpha = float(np.mean(weights))
    source.torque_scale = (
        anchor_target.torque_scale
        if anchored_groups
        else balance_target.torque_scale
        + (1.0 - scalar_alpha)
        * (anchor_target.torque_scale - balance_start.torque_scale)
    )
    return blend_targets_with_joint_weights(
        source,
        motion_target,
        weights,
        "transition_in_continuous",
    )


def continuous_transition_target(
    anchor_target: ControlTarget,
    source_start: ControlTarget,
    source_target: ControlTarget,
    destination_target: ControlTarget,
    progress: float,
    label: str,
) -> ControlTarget:
    """Blend live targets while exactly continuing the last applied target.

    The source policy remains live during the transition. Its change from the
    first source sample is retained and decays as the destination acquires
    control. At progress zero the result is exactly ``anchor_target``; at one
    it is exactly ``destination_target``.
    """
    alpha = quintic_alpha(progress)
    source = copy_target(anchor_target, f"{label}_live_source")
    for field in ("q", "kp", "kd", "feedforward", "effort"):
        getattr(source, field)[:] += alpha * (
            getattr(source_target, field) - getattr(source_start, field)
        )
    source.torque_scale += alpha * (
        source_target.torque_scale - source_start.torque_scale
    )
    return blend_targets(source, destination_target, progress, label)


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
        self.last_requested_maximum_delta = 0.0
        self.last_applied_maximum_delta = 0.0

    def reset(self, positions: np.ndarray) -> None:
        positions = np.asarray(positions, dtype=np.float64)
        self.previous_q = positions.copy()
        self.last_maximum_rate = 0.0
        self.last_requested_maximum_delta = 0.0
        self.last_applied_maximum_delta = 0.0

    def apply(self, target: ControlTarget) -> ControlTarget:
        limited = copy_target(target)
        if self.previous_q is None:
            self.previous_q = limited.q.copy()
            return limited
        requested_delta = limited.q - self.previous_q
        maximum_delta = self.maximum_rates * self.control_dt
        applied_delta = np.clip(requested_delta, -maximum_delta, maximum_delta)
        limited.q = self.previous_q + applied_delta
        self.last_requested_maximum_delta = float(np.max(np.abs(requested_delta)))
        self.last_applied_maximum_delta = float(np.max(np.abs(applied_delta)))
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
