from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np

from .motion_evt2 import MotionEvt2Policy
from .robot import ControlTarget, JointMap
from .transition_controller import (
    TargetRateLimiter,
    blend_targets,
    prealign_target,
)
from .walkamp import WalkAmpPolicy
from .walkamp_runner import quaternion_to_rpy


class ControllerState(str, Enum):
    WALKAMP_IDLE = "WALKAMP_IDLE"
    READY_CHECK = "READY_CHECK"
    PRE_ALIGN = "PRE_ALIGN"
    TRANSITION_IN = "TRANSITION_IN"
    MOTION = "MOTION"
    TRANSITION_OUT = "TRANSITION_OUT"
    RECOVER = "RECOVER"


class CommandStatus(str, Enum):
    RECEIVED = "RECEIVED"
    ACCEPTED = "ACCEPTED"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


@dataclass
class CommandRecord:
    command_id: int
    command: str
    action_key: str | None
    source: str
    status: str
    received_step: int
    updated_step: int
    reason: str = ""


@dataclass
class ControllerOutput:
    target: ControlTarget
    observation: np.ndarray | None
    action: np.ndarray | None
    transition_alpha: float


ACTION_ALIASES = {
    "a": "a",
    "bow": "a",
    "b": "b",
    "wave": "b",
}


class MultiEvt2Controller:
    """Command-driven WALKAMP/BeyondMimic state machine for the official EVT2 model."""

    def __init__(
        self,
        data,
        joint_map: JointMap,
        walkamp: WalkAmpPolicy,
        motions: dict[str, MotionEvt2Policy],
        config: dict[str, Any],
        control_dt: float,
    ) -> None:
        self.data = data
        self.joint_map = joint_map
        self.walkamp = walkamp
        self.motions = motions
        self.config = config
        self.control_dt = float(control_dt)
        self.state = ControllerState.WALKAMP_IDLE
        self.global_step = 0
        self.state_step = 0
        self.motion_step = 0
        self.frozen_motion_step = 0
        self.active_key: str | None = None
        self.active_motion: MotionEvt2Policy | None = None
        self.active_record: CommandRecord | None = None
        self.abort_record: CommandRecord | None = None
        self.command_records: list[CommandRecord] = []
        self.next_command_id = 1
        self.ready_stable_steps = 0
        self.recover_stable_steps = 0
        self.reentry_started = False
        self.reentry_step = 0
        self.abort_pending = False
        self.abort_safe_steps = 0
        self.recovery_failed = False
        self.last_rejection_reason = ""
        self.last_start_delta: dict[str, float] = {}
        self.last_readiness: dict[str, Any] = {}
        self.state_history: list[dict[str, Any]] = [
            {"step": 0, "state": self.state.value, "reason": "startup"}
        ]

        constraints = config["constraints"]
        self.target_limiter = TargetRateLimiter(
            joint_map,
            self.control_dt,
            constraints["maximum_target_rate"],
        )
        current_q = self.data.qpos[self.joint_map.qpos_adr].copy()
        self.target_limiter.reset(current_q)
        self.walkamp.begin_from_live_state((0.0, 0.0, 0.0))
        self.last_target = self.walkamp.neutral_target("walkamp_idle_start")

        self.group_indices = {
            "legs": np.asarray(
                [
                    index
                    for index, name in enumerate(joint_map.names)
                    if any(token in name for token in ("hip_", "knee_", "ankle_"))
                ],
                dtype=np.int32,
            ),
            "waist": np.asarray(
                [index for index, name in enumerate(joint_map.names) if name.startswith("waist_")],
                dtype=np.int32,
            ),
            "arms": np.asarray(
                [
                    index
                    for index, name in enumerate(joint_map.names)
                    if not any(
                        token in name
                        for token in ("hip_", "knee_", "ankle_", "waist_")
                    )
                ],
                dtype=np.int32,
            ),
        }

    @property
    def busy(self) -> bool:
        return self.state != ControllerState.WALKAMP_IDLE

    def _duration_steps(self, seconds: float) -> int:
        return max(1, int(round(float(seconds) / self.control_dt)))

    def _log_state(self, reason: str) -> None:
        payload = {"step": self.global_step, "state": self.state.value, "reason": reason}
        self.state_history.append(payload)
        print(f"[STATE] {json.dumps(payload, ensure_ascii=True)}")

    def _set_state(self, state: ControllerState, reason: str) -> None:
        self.state = state
        self.state_step = 0
        self._log_state(reason)

    def _new_record(self, command: str, action_key: str | None, source: str) -> CommandRecord:
        record = CommandRecord(
            command_id=self.next_command_id,
            command=command,
            action_key=action_key,
            source=source,
            status=CommandStatus.RECEIVED.value,
            received_step=self.global_step,
            updated_step=self.global_step,
        )
        self.next_command_id += 1
        self.command_records.append(record)
        self._log_command(record)
        return record

    def _log_command(self, record: CommandRecord) -> None:
        print(f"[COMMAND] {json.dumps(asdict(record), ensure_ascii=True)}")

    def _update_record(
        self,
        record: CommandRecord | None,
        status: CommandStatus,
        reason: str = "",
    ) -> None:
        if record is None:
            return
        record.status = status.value
        record.updated_step = self.global_step
        record.reason = reason
        self._log_command(record)

    def request_command(self, command: str, source: str = "external") -> CommandRecord:
        normalized = command.strip().lower()
        if normalized == "r":
            return self._request_abort(source)

        action_key = ACTION_ALIASES.get(normalized)
        record = self._new_record(normalized, action_key, source)
        if action_key is None or action_key not in self.motions:
            self._update_record(record, CommandStatus.REJECTED, "unsupported_action")
            return record
        if self.busy:
            self._update_record(record, CommandStatus.REJECTED, f"controller_busy:{self.state.value}")
            return record

        self.walkamp.set_command((0.0, 0.0, 0.0))
        self.active_key = action_key
        self.active_motion = self.motions[action_key]
        self.active_record = record
        self.ready_stable_steps = 0
        self.recovery_failed = False
        self.abort_pending = False
        self.abort_safe_steps = 0
        self._update_record(record, CommandStatus.ACCEPTED)
        self._set_state(ControllerState.READY_CHECK, f"accepted:{action_key}")
        return record

    def _request_abort(self, source: str) -> CommandRecord:
        record = self._new_record("r", None, source)
        if not self.busy or self.state in (
            ControllerState.TRANSITION_OUT,
            ControllerState.RECOVER,
        ):
            self._update_record(record, CommandStatus.REJECTED, f"nothing_abortable:{self.state.value}")
            return record
        self.abort_record = record
        self._update_record(record, CommandStatus.ACCEPTED)
        self._update_record(record, CommandStatus.EXECUTING, "safe_recovery_requested")
        self._update_record(self.active_record, CommandStatus.FAILED, "aborted_by_command")
        if self.state == ControllerState.MOTION:
            self.abort_pending = True
            self.abort_safe_steps = 0
            print("[INFO] safe abort queued; waiting for a verified support and posture window")
        else:
            self._begin_transition_out("abort_requested")
        return record

    def adjust_walk_command(self, index: int, amount: float, source: str = "keyboard") -> None:
        if self.busy:
            print(f"[WARN] walk command ignored while state={self.state.value} source={source}")
            return
        self.walkamp.adjust_command(index, amount)

    def stop_walk(self, source: str = "keyboard") -> None:
        if self.busy:
            print(f"[WARN] stop command ignored while state={self.state.value} source={source}")
            return
        self.walkamp.set_command((0.0, 0.0, 0.0))
        print(f"[INFO] WALKAMP command stopped ({source})")

    def _state_metrics(self, contacts: dict[str, float]) -> dict[str, float]:
        orientation = self.data.sensor("orientation").data.copy()
        rpy = quaternion_to_rpy(orientation)
        joint_velocity = self.data.qvel[self.joint_map.qvel_adr]
        return {
            "root_z": float(self.data.qpos[2]),
            "abs_roll_pitch": float(max(abs(rpy[0]), abs(rpy[1]))),
            "xy_speed": float(np.linalg.norm(self.data.qvel[0:2])),
            "angular_speed": float(
                np.linalg.norm(self.data.sensor("angular-velocity").data.copy())
            ),
            "maximum_joint_speed": float(np.max(np.abs(joint_velocity))),
            "left_contact_count": float(contacts.get("left_contact_count", 0.0)),
            "right_contact_count": float(contacts.get("right_contact_count", 0.0)),
            "left_normal_force": float(contacts.get("left_normal_force", 0.0)),
            "right_normal_force": float(contacts.get("right_normal_force", 0.0)),
        }

    def _check_state(
        self,
        contacts: dict[str, float],
        limits: dict[str, Any],
        require_neutral: bool = False,
    ) -> tuple[bool, list[str], dict[str, float]]:
        metrics = self._state_metrics(contacts)
        reasons: list[str] = []
        checks = (
            (metrics["root_z"] >= float(limits["minimum_root_height"]), "root_height"),
            (
                metrics["abs_roll_pitch"] <= float(limits["maximum_abs_roll_pitch"]),
                "attitude",
            ),
            (metrics["xy_speed"] <= float(limits["maximum_xy_speed"]), "xy_speed"),
            (
                metrics["angular_speed"] <= float(limits["maximum_angular_speed"]),
                "angular_speed",
            ),
            (
                metrics["maximum_joint_speed"] <= float(limits["maximum_joint_speed"]),
                "joint_speed",
            ),
            (metrics["left_contact_count"] > 0.0, "left_contact"),
            (metrics["right_contact_count"] > 0.0, "right_contact"),
            (
                metrics["left_normal_force"] >= float(limits["minimum_normal_force"]),
                "left_support_force",
            ),
            (
                metrics["right_normal_force"] >= float(limits["minimum_normal_force"]),
                "right_support_force",
            ),
        )
        reasons.extend(name for passed, name in checks if not passed)
        if require_neutral:
            neutral = self.walkamp.neutral_target("recover_check")
            actual_q = self.data.qpos[self.joint_map.qpos_adr]
            configured = limits["maximum_neutral_joint_error"]
            group_limits = (
                {name: float(configured) for name in self.group_indices}
                if isinstance(configured, (int, float))
                else configured
            )
            for group, indices in self.group_indices.items():
                neutral_error = float(np.max(np.abs(actual_q[indices] - neutral.q[indices])))
                metrics[f"neutral_{group}_error"] = neutral_error
                if neutral_error > float(group_limits[group]):
                    reasons.append(f"neutral_{group}_error")
        self.last_readiness = {"metrics": metrics, "reasons": reasons}
        return not reasons, reasons, metrics

    def _start_delta(self, reference_target: ControlTarget) -> dict[str, float]:
        if self.active_motion is None:
            return {}
        actual_q = self.data.qpos[self.joint_map.qpos_adr]
        controlled = set(self.active_motion.model_indices.tolist())
        values: dict[str, float] = {}
        for group, indices in self.group_indices.items():
            selected = np.asarray([index for index in indices if index in controlled], dtype=np.int32)
            values[group] = (
                0.0
                if selected.size == 0
                else float(np.max(np.abs(reference_target.q[selected] - actual_q[selected])))
            )
        return values

    def _validate_start_delta(self, values: dict[str, float]) -> list[str]:
        limits = self.config["pre_align"]["maximum_start_delta"]
        return [
            f"{group}_delta={value:.3f}>{float(limits[group]):.3f}"
            for group, value in values.items()
            if value > float(limits[group])
        ]

    def _tracking_errors(self, target: ControlTarget) -> dict[str, float]:
        actual_q = self.data.qpos[self.joint_map.qpos_adr]
        return {
            group: float(np.max(np.abs(target.q[indices] - actual_q[indices])))
            for group, indices in self.group_indices.items()
        }

    def _emergency_reasons(self) -> list[str]:
        limits = self.config["emergency"]
        rpy = quaternion_to_rpy(self.data.sensor("orientation").data.copy())
        reasons: list[str] = []
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            reasons.append("non_finite_state")
        if float(self.data.qpos[2]) < float(limits["minimum_root_height"]):
            reasons.append("root_height")
        if max(abs(float(rpy[0])), abs(float(rpy[1]))) > float(
            limits["maximum_abs_roll_pitch"]
        ):
            reasons.append("attitude")
        return reasons

    def _reject_active(self, reason: str) -> None:
        self.last_rejection_reason = reason
        self._update_record(self.active_record, CommandStatus.REJECTED, reason)
        self.active_record = None
        self.active_motion = None
        self.active_key = None
        self.walkamp.begin_from_live_state((0.0, 0.0, 0.0))
        self._set_state(ControllerState.WALKAMP_IDLE, f"rejected:{reason}")

    def _fail_active(self, reason: str) -> None:
        if self.active_record is not None and self.active_record.status not in (
            CommandStatus.FAILED.value,
            CommandStatus.REJECTED.value,
        ):
            self._update_record(self.active_record, CommandStatus.FAILED, reason)
        self._begin_transition_out(f"failure:{reason}")

    def _begin_transition_out(self, reason: str) -> None:
        self.walkamp.set_command((0.0, 0.0, 0.0))
        if self.active_motion is not None:
            self.frozen_motion_step = min(
                max(self.motion_step - 1, 0),
                self.active_motion.duration_steps - 1,
            )
        self.recover_stable_steps = 0
        self.reentry_started = False
        self.reentry_step = 0
        self._set_state(ControllerState.TRANSITION_OUT, reason)

    def _finish_recovery(self) -> None:
        if self.active_record is not None and self.active_record.status == CommandStatus.EXECUTING.value:
            self._update_record(self.active_record, CommandStatus.COMPLETED)
        if self.abort_record is not None:
            self._update_record(self.abort_record, CommandStatus.COMPLETED)
        self.active_record = None
        self.abort_record = None
        self.active_motion = None
        self.active_key = None
        self.motion_step = 0
        self.reentry_started = False
        self.reentry_step = 0
        self.abort_pending = False
        self.abort_safe_steps = 0
        self.recover_stable_steps = 0
        self._set_state(ControllerState.WALKAMP_IDLE, "recovery_complete")

    def fail(self, reason: str) -> None:
        if self.active_record is not None and self.active_record.status not in (
            CommandStatus.COMPLETED.value,
            CommandStatus.REJECTED.value,
            CommandStatus.FAILED.value,
        ):
            self._update_record(self.active_record, CommandStatus.FAILED, reason)
        if self.abort_record is not None and self.abort_record.status != CommandStatus.COMPLETED.value:
            self._update_record(self.abort_record, CommandStatus.FAILED, reason)

    def _step_idle(self) -> ControllerOutput:
        target, observation, action = self.walkamp.step()
        return ControllerOutput(target, observation, action, 0.0)

    def _step_ready_check(self, contacts: dict[str, float]) -> ControllerOutput:
        target, observation, action = self.walkamp.step()
        settings = self.config["ready_check"]
        ready, reasons, _ = self._check_state(contacts, settings)
        self.ready_stable_steps = self.ready_stable_steps + 1 if ready else 0
        required = self._duration_steps(settings["stable_seconds"])
        timeout = self._duration_steps(settings["timeout_seconds"])
        self.state_step += 1
        if self.ready_stable_steps >= required:
            assert self.active_motion is not None
            diagnostics = self.active_motion.begin_from_live_state()
            reference_target = self.active_motion.reference_start_target()
            self.last_start_delta = self._start_delta(reference_target)
            delta_reasons = self._validate_start_delta(self.last_start_delta)
            print(
                f"[INFO] live motion start {json.dumps({'diagnostics': diagnostics, 'delta': self.last_start_delta})}"
            )
            if delta_reasons:
                self._reject_active(";".join(delta_reasons))
            else:
                self._update_record(self.active_record, CommandStatus.EXECUTING)
                self._set_state(ControllerState.PRE_ALIGN, "ready_and_start_delta_valid")
        elif self.state_step >= timeout:
            self._reject_active("ready_timeout:" + ",".join(reasons))
        return ControllerOutput(target, observation, action, 0.0)

    def _step_pre_align(self, contacts: dict[str, float]) -> ControllerOutput:
        assert self.active_motion is not None
        walk_target, _, _ = self.walkamp.step()
        motion_target, observation, action = self.active_motion.step(0)
        settings = self.config["pre_align"]
        duration = self._duration_steps(settings["duration_seconds"])
        progress = min((self.state_step + 1) / duration, 1.0)
        target = prealign_target(walk_target, motion_target, progress)
        self.state_step += 1

        if progress >= 1.0:
            safe, reasons, _ = self._check_state(contacts, settings["handoff_limits"])
            errors = self._tracking_errors(motion_target)
            tracking_limits = settings["maximum_tracking_error"]
            tracking_reasons = [
                f"{group}_tracking={value:.3f}>{float(tracking_limits[group]):.3f}"
                for group, value in errors.items()
                if value > float(tracking_limits[group])
            ]
            if safe and not tracking_reasons:
                self._set_state(ControllerState.TRANSITION_IN, "pre_align_complete")
            elif self.state_step >= duration + self._duration_steps(settings["timeout_seconds"]):
                self._fail_active("pre_align_timeout:" + ",".join(reasons + tracking_reasons))
        return ControllerOutput(target, observation, action, progress)

    def _step_transition_in(self) -> ControllerOutput:
        assert self.active_motion is not None
        walk_target, _, _ = self.walkamp.step()
        motion_target, observation, action = self.active_motion.step(0)
        duration = self._duration_steps(self.config["transition_in_seconds"])
        progress = min((self.state_step + 1) / duration, 1.0)
        target = blend_targets(walk_target, motion_target, progress, "transition_in")
        self.state_step += 1
        if progress >= 1.0:
            self.motion_step = 0
            self._set_state(ControllerState.MOTION, "motion_control_acquired")
        return ControllerOutput(target, observation, action, progress)

    def _step_motion(self, contacts: dict[str, float]) -> ControllerOutput:
        assert self.active_motion is not None
        target, observation, action = self.active_motion.step(self.motion_step)
        self.motion_step += 1
        if self.abort_pending:
            settings = self.config["recover"]
            safe, _, _ = self._check_state(contacts, settings, require_neutral=True)
            self.abort_safe_steps = self.abort_safe_steps + 1 if safe else 0
            if self.abort_safe_steps >= self._duration_steps(settings["abort_stable_seconds"]):
                self._begin_transition_out("safe_abort_window")
        if (
            self.state == ControllerState.MOTION
            and self.motion_step >= self.active_motion.duration_steps
        ):
            reason = "safe_abort_terminal" if self.abort_pending else "motion_complete"
            self._begin_transition_out(reason)
        return ControllerOutput(target, observation, action, 1.0)

    def _step_transition_out(self, contacts: dict[str, float]) -> ControllerOutput:
        observation = None
        action = None
        if self.active_motion is not None and self.active_motion.episode_initialized:
            source, observation, action = self.active_motion.step(self.frozen_motion_step)
        else:
            source = self.last_target

        settings = self.config["recover"]
        if not self.reentry_started:
            safe, reasons, _ = self._check_state(contacts, settings, require_neutral=True)
            self.recover_stable_steps = self.recover_stable_steps + 1 if safe else 0
            target = source
            progress = 0.0
            required = self._duration_steps(settings["takeover_stable_seconds"])
            if self.recover_stable_steps >= required:
                self.walkamp.begin_from_live_state((0.0, 0.0, 0.0))
                self.reentry_started = True
                self.reentry_step = 0
                print("[INFO] WALKAMP live history initialized at verified takeover state")
            elif self.state_step >= self._duration_steps(settings["takeover_timeout_seconds"]):
                if not self.recovery_failed:
                    reason = "takeover_timeout:" + ",".join(reasons)
                    self.recovery_failed = True
                    self._update_record(self.active_record, CommandStatus.FAILED, reason)
                    self._update_record(self.abort_record, CommandStatus.FAILED, reason)
                    print(f"[ERROR] {reason}; retaining motion feedback target")
        else:
            walk_target, _, _ = self.walkamp.step()
            duration = self._duration_steps(self.config["transition_out_seconds"])
            progress = min((self.reentry_step + 1) / duration, 1.0)
            target = blend_targets(source, walk_target, progress, "transition_out")
            self.reentry_step += 1
            if progress >= 1.0:
                self.recover_stable_steps = 0
                self._set_state(ControllerState.RECOVER, "walkamp_control_acquired")
                return ControllerOutput(target, observation, action, progress)
        self.state_step += 1
        return ControllerOutput(target, observation, action, progress)

    def _step_recover(self, contacts: dict[str, float]) -> ControllerOutput:
        settings = self.config["recover"]
        target, observation, action = self.walkamp.step()
        safe, reasons, _ = self._check_state(contacts, settings, require_neutral=False)
        self.recover_stable_steps = self.recover_stable_steps + 1 if safe else 0
        required = self._duration_steps(settings["stable_seconds"])
        if self.recover_stable_steps >= required:
            self._finish_recovery()
            return ControllerOutput(target, observation, action, 0.0)
        elif self.state_step >= self._duration_steps(settings["timeout_seconds"]):
            if not self.recovery_failed:
                reason = "recover_timeout:" + ",".join(reasons)
                self.recovery_failed = True
                self._update_record(self.active_record, CommandStatus.FAILED, reason)
                self._update_record(self.abort_record, CommandStatus.FAILED, reason)
                print(f"[ERROR] {reason}; retaining WALKAMP zero-command feedback")
        self.state_step += 1
        return ControllerOutput(target, observation, action, 0.0)

    def step(self, contacts: dict[str, float]) -> ControllerOutput:
        emergency = self._emergency_reasons()
        if emergency and self.state not in (
            ControllerState.WALKAMP_IDLE,
            ControllerState.TRANSITION_OUT,
            ControllerState.RECOVER,
        ):
            self._fail_active("emergency:" + ",".join(emergency))

        if self.state == ControllerState.WALKAMP_IDLE:
            output = self._step_idle()
        elif self.state == ControllerState.READY_CHECK:
            output = self._step_ready_check(contacts)
        elif self.state == ControllerState.PRE_ALIGN:
            output = self._step_pre_align(contacts)
        elif self.state == ControllerState.TRANSITION_IN:
            output = self._step_transition_in()
        elif self.state == ControllerState.MOTION:
            output = self._step_motion(contacts)
        elif self.state == ControllerState.TRANSITION_OUT:
            output = self._step_transition_out(contacts)
        elif self.state == ControllerState.RECOVER:
            output = self._step_recover(contacts)
        else:
            raise RuntimeError(f"Unknown controller state {self.state!r}")

        output.target = self.target_limiter.apply(output.target)
        self.last_target = output.target
        self.global_step += 1
        return output

    def summary(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "busy": self.busy,
            "recovery_failed": self.recovery_failed,
            "last_rejection_reason": self.last_rejection_reason,
            "last_start_delta": self.last_start_delta,
            "last_readiness": self.last_readiness,
            "state_history": self.state_history,
            "commands": [asdict(record) for record in self.command_records],
        }
