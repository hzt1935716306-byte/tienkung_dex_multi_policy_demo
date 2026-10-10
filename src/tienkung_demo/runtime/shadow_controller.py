from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping

import numpy as np

from .control_target import ControlTarget, copy_control_target
from .policy_runtime import MotionRuntime, WalkAmpRuntime
from .robot_state import RobotStateSnapshot


class ShadowControllerState(str, Enum):
    WALKAMP_IDLE = "WALKAMP_IDLE"
    BRAKE = "BRAKE"
    READY_CHECK = "READY_CHECK"
    PRE_ALIGN = "PRE_ALIGN"
    TRANSITION_IN = "TRANSITION_IN"
    MOTION = "MOTION"
    TRANSITION_OUT = "TRANSITION_OUT"
    RECOVER = "RECOVER"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ShadowStep:
    target: ControlTarget
    state: ShadowControllerState
    observation: np.ndarray | None
    action: np.ndarray | None
    event: str | None
    readiness_blockers: tuple[str, ...]


def _smoothstep5(progress: float) -> float:
    value = float(np.clip(progress, 0.0, 1.0))
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def _blend(first: ControlTarget, second: ControlTarget, alpha: float, label: str) -> ControlTarget:
    value = float(np.clip(alpha, 0.0, 1.0))
    return ControlTarget(
        q=(1.0 - value) * first.q + value * second.q,
        kp=(1.0 - value) * first.kp + value * second.kp,
        kd=(1.0 - value) * first.kd + value * second.kd,
        feedforward=(1.0 - value) * first.feedforward + value * second.feedforward,
        effort=np.minimum(first.effort, second.effort),
        torque_scale=(1.0 - value) * first.torque_scale + value * second.torque_scale,
        label=label,
    )


def _roll_pitch(quaternion: np.ndarray) -> tuple[float, float]:
    w, x, y, z = quaternion
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_value = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    return roll, math.asin(float(pitch_value))


class ShadowMultiPolicyController:
    """Hardware-neutral Phase 3B flow used only to calculate Shadow targets."""

    def __init__(
        self,
        walkamp: WalkAmpRuntime,
        motions: Mapping[str, MotionRuntime],
        config: Mapping[str, object],
        control_dt: float,
    ) -> None:
        if set(motions) != {"a", "b"}:
            raise ValueError("Shadow controller requires motions a and b")
        self.walkamp = walkamp
        self.motions = dict(motions)
        self.config = dict(config)
        self.control_dt = float(control_dt)
        self.state = ShadowControllerState.WALKAMP_IDLE
        self.walk_command = np.zeros(3, dtype=np.float32)
        self.requested_command = np.zeros(3, dtype=np.float32)
        self.active_key: str | None = None
        self.active_motion: MotionRuntime | None = None
        self.state_steps = 0
        self.stable_steps = 0
        self.motion_step = 0
        self.entry_start: ControlTarget | None = None
        self.exit_start: ControlTarget | None = None
        self.last_target = walkamp.neutral_target()
        self.command_events: list[dict[str, object]] = []
        names = walkamp.layout.names
        self.arm_indices = np.asarray(
            [index for index, name in enumerate(names) if "shoulder" in name or "elbow" in name or "wrist" in name],
            dtype=np.int32,
        )

    def set_walk_command(self, command: np.ndarray | list[float]) -> None:
        value = np.asarray(command, dtype=np.float32)
        if value.shape != (3,):
            raise ValueError("walk command must contain [vx, vy, yaw_rate]")
        self.requested_command[:] = value
        if self.state == ShadowControllerState.WALKAMP_IDLE:
            self.walk_command[:] = value

    def request_motion(self, key: str) -> bool:
        if key not in self.motions:
            self.command_events.append({"key": key, "status": "REJECTED", "reason": "unsupported"})
            return False
        if self.state != ShadowControllerState.WALKAMP_IDLE:
            self.command_events.append({"key": key, "status": "REJECTED", "reason": "busy"})
            return False
        self.active_key = key
        self.active_motion = self.motions[key]
        self.command_events.append({"key": key, "status": "ACCEPTED"})
        self._enter(
            ShadowControllerState.BRAKE
            if np.max(np.abs(self.walk_command)) > 1.0e-6
            else ShadowControllerState.READY_CHECK
        )
        return True

    def request_recovery(self) -> bool:
        if self.state == ShadowControllerState.WALKAMP_IDLE:
            return False
        if self.state in (ShadowControllerState.BRAKE, ShadowControllerState.READY_CHECK):
            self.active_key = None
            self.active_motion = None
            self.walk_command.fill(0.0)
            self.requested_command.fill(0.0)
            self._enter(ShadowControllerState.WALKAMP_IDLE)
            return True
        self.exit_start = copy_control_target(self.last_target, "shadow_abort_exit_start")
        self.walkamp.reset()
        self._enter(ShadowControllerState.TRANSITION_OUT)
        return True

    def _enter(self, state: ShadowControllerState) -> None:
        self.state = state
        self.state_steps = 0
        self.stable_steps = 0

    def _duration_steps(self, name: str) -> int:
        return max(1, int(round(float(self.config[name]) / self.control_dt)))

    def _readiness(self, state: RobotStateSnapshot) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []
        for name in ("base_linear_velocity_world", "foot_contact", "foot_normal_force"):
            if not state.is_valid(name):
                reasons.append(f"unavailable:{name}")
        roll, pitch = _roll_pitch(state.orientation_wxyz)
        if abs(roll) > 0.35 or abs(pitch) > 0.35:
            reasons.append("base_attitude")
        if np.linalg.norm(state.angular_velocity_body) > 0.5:
            reasons.append("base_angular_velocity")
        if np.max(np.abs(state.joint_velocity)) > 2.5:
            reasons.append("joint_velocity")
        if state.base_linear_velocity_world is not None:
            if np.linalg.norm(state.base_linear_velocity_world[:2]) > float(self.config["maximum_ready_xy_speed"]):
                reasons.append("base_linear_velocity")
        if state.foot_contact is not None and not bool(np.all(state.foot_contact > 0.5)):
            reasons.append("double_support")
        if state.foot_normal_force is not None:
            if np.any(state.foot_normal_force < float(self.config["minimum_support_force"])):
                reasons.append("support_force")
        return not reasons, tuple(reasons)

    def _brake(self) -> None:
        rates = np.asarray(self.config["brake_deceleration"], dtype=np.float32) * self.control_dt
        self.walk_command[:] = np.sign(self.walk_command) * np.maximum(np.abs(self.walk_command) - rates, 0.0)

    def _motion_target(self, state: RobotStateSnapshot, step: int) -> tuple[ControlTarget, np.ndarray, np.ndarray]:
        assert self.active_motion is not None
        return self.active_motion.step(state, step)

    def step(self, state: RobotStateSnapshot) -> ShadowStep:
        event: str | None = None
        readiness_blockers: tuple[str, ...] = ()
        walk_target, walk_observation, walk_action = self.walkamp.step(state, self.walk_command)
        observation: np.ndarray | None = walk_observation
        action: np.ndarray | None = walk_action

        if self.state == ShadowControllerState.WALKAMP_IDLE:
            target = walk_target
        elif self.state == ShadowControllerState.BRAKE:
            self._brake()
            target = walk_target
            if np.max(np.abs(self.walk_command)) <= 1.0e-6:
                self._enter(ShadowControllerState.READY_CHECK)
                event = "brake_command_zero"
        elif self.state == ShadowControllerState.READY_CHECK:
            target = walk_target
            ready, readiness_blockers = self._readiness(state)
            self.stable_steps = self.stable_steps + 1 if ready else 0
            if self.stable_steps >= self._duration_steps("ready_stable_seconds"):
                assert self.active_motion is not None
                self.active_motion.begin_from_live_state(state)
                motion_target, observation, action = self._motion_target(state, 0)
                self.entry_start = copy_control_target(self.last_target, "shadow_entry_start")
                self.last_motion_target = motion_target
                self._enter(ShadowControllerState.PRE_ALIGN)
                event = "ready_confirmed"
            elif self.state_steps + 1 >= self._duration_steps("ready_timeout_seconds"):
                if self.active_key is not None:
                    self.command_events.append(
                        {
                            "key": self.active_key,
                            "status": "FAILED",
                            "reason": "ready_timeout",
                            "readiness_blockers": list(readiness_blockers),
                        }
                    )
                self.active_key = None
                self.active_motion = None
                self.walk_command.fill(0.0)
                self.requested_command.fill(0.0)
                self._enter(ShadowControllerState.WALKAMP_IDLE)
                event = "ready_failed"
        elif self.state == ShadowControllerState.PRE_ALIGN:
            motion_target, observation, action = self._motion_target(state, 0)
            alpha = _smoothstep5((self.state_steps + 1) / self._duration_steps("pre_align_seconds"))
            target = copy_control_target(walk_target, "shadow_pre_align")
            target.q[self.arm_indices] = (1.0 - alpha) * walk_target.q[self.arm_indices] + alpha * motion_target.q[self.arm_indices]
            target.kp[self.arm_indices] = (1.0 - alpha) * walk_target.kp[self.arm_indices] + alpha * motion_target.kp[self.arm_indices]
            target.kd[self.arm_indices] = (1.0 - alpha) * walk_target.kd[self.arm_indices] + alpha * motion_target.kd[self.arm_indices]
            self.last_motion_target = motion_target
            if self.state_steps + 1 >= self._duration_steps("pre_align_seconds"):
                self.entry_start = copy_control_target(target, "shadow_transition_in_start")
                self._enter(ShadowControllerState.TRANSITION_IN)
                event = "pre_align_complete"
        elif self.state == ShadowControllerState.TRANSITION_IN:
            motion_target, observation, action = self._motion_target(state, 0)
            alpha = _smoothstep5((self.state_steps + 1) / self._duration_steps("transition_in_seconds"))
            assert self.entry_start is not None
            target = _blend(self.entry_start, motion_target, alpha, "shadow_transition_in")
            self.last_motion_target = motion_target
            if self.state_steps + 1 >= self._duration_steps("transition_in_seconds"):
                self.motion_step = 0
                self._enter(ShadowControllerState.MOTION)
                event = "motion_started"
        elif self.state == ShadowControllerState.MOTION:
            target, observation, action = self._motion_target(state, self.motion_step)
            self.last_motion_target = target
            self.motion_step += 1
            assert self.active_motion is not None
            if self.motion_step >= self.active_motion.duration_steps:
                self.exit_start = copy_control_target(target, "shadow_transition_out_start")
                self.walkamp.reset()
                self._enter(ShadowControllerState.TRANSITION_OUT)
                event = "motion_complete"
        elif self.state == ShadowControllerState.TRANSITION_OUT:
            alpha = _smoothstep5((self.state_steps + 1) / self._duration_steps("transition_out_seconds"))
            assert self.exit_start is not None
            target = _blend(self.exit_start, walk_target, alpha, "shadow_transition_out")
            if self.state_steps + 1 >= self._duration_steps("transition_out_seconds"):
                self._enter(ShadowControllerState.RECOVER)
                event = "walkamp_acquired"
        elif self.state == ShadowControllerState.RECOVER:
            target = walk_target
            ready, readiness_blockers = self._readiness(state)
            self.stable_steps = self.stable_steps + 1 if ready else 0
            if self.stable_steps >= self._duration_steps("recover_stable_seconds"):
                if self.active_key is not None:
                    self.command_events.append({"key": self.active_key, "status": "COMPLETED"})
                self.active_key = None
                self.active_motion = None
                self.requested_command.fill(0.0)
                self.walk_command.fill(0.0)
                self._enter(ShadowControllerState.WALKAMP_IDLE)
                event = "recovery_complete"
        else:
            target = walk_target

        target.validate(29)
        self.last_target = copy_control_target(target)
        self.state_steps += 1
        return ShadowStep(target, self.state, observation, action, event, readiness_blockers)
