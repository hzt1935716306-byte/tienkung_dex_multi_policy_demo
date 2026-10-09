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
    continuous_group_transition_target,
    copy_target,
    endpoint_progress,
    joint_group_indices,
    prealign_group_target,
    prealign_target,
    quintic_alpha,
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
    FAILED = "FAILED"


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
        self.handoff_stable_steps = 0
        self.exit_window_opened = False
        self.recovery_failed = False
        self.terminal_failure_reason = ""
        self.last_rejection_reason = ""
        self.last_start_delta: dict[str, float] = {}
        self.last_readiness: dict[str, Any] = {}
        self.last_handoff: dict[str, Any] = {}
        self.last_recovery: dict[str, Any] = {}
        self.last_contacts: dict[str, float] = {}
        entry_config = config.get("entry", {})
        self.entry_mode = str(entry_config.get("mode", "full_body_continuous"))
        if self.entry_mode not in ("grouped", "full_body_continuous", "legacy"):
            raise ValueError(f"Unknown entry mode {self.entry_mode!r}")
        self.entry_anchor_target: ControlTarget | None = None
        self.entry_balance_start: ControlTarget | None = None
        self.entry_group_alphas = {"legs": 0.0, "waist": 0.0, "arms": 0.0}
        self.event_history: list[dict[str, Any]] = []
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
        self.last_raw_target = copy_target(self.last_target, "raw_start")

        self.group_indices = joint_group_indices(joint_map)

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

    def _log_event(self, name: str, **details: Any) -> None:
        payload = {
            "step": self.global_step,
            "time": self.global_step * self.control_dt,
            "event": name,
            "state": self.state.value,
            "action_key": self.active_key,
            "command_id": None if self.active_record is None else self.active_record.command_id,
            **details,
        }
        self.event_history.append(payload)
        print(f"[EVENT] {json.dumps(payload, ensure_ascii=True)}")

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
        self.handoff_stable_steps = 0
        self.exit_window_opened = False
        self.terminal_failure_reason = ""
        self.entry_anchor_target = None
        self.entry_balance_start = None
        self.entry_group_alphas = {"legs": 0.0, "waist": 0.0, "arms": 0.0}
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
        angular_velocity = self.data.sensor("angular-velocity").data.copy()
        joint_velocity = self.data.qvel[self.joint_map.qvel_adr]
        return {
            "root_x": float(self.data.qpos[0]),
            "root_y": float(self.data.qpos[1]),
            "root_z": float(self.data.qpos[2]),
            "roll": float(rpy[0]),
            "pitch": float(rpy[1]),
            "yaw": float(rpy[2]),
            "abs_roll_pitch": float(max(abs(rpy[0]), abs(rpy[1]))),
            "world_vx": float(self.data.qvel[0]),
            "world_vy": float(self.data.qvel[1]),
            "xy_speed": float(np.linalg.norm(self.data.qvel[0:2])),
            "pitch_angular_velocity": float(angular_velocity[1]),
            "angular_speed": float(np.linalg.norm(angular_velocity)),
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
        self.last_readiness = {"metrics": metrics, "reasons": reasons}
        return not reasons, reasons, metrics

    def handoff_feasible(
        self,
        contacts: dict[str, float],
        motion_target: ControlTarget,
    ) -> tuple[bool, list[str], dict[str, float]]:
        """Check whether WALKAMP may safely take over the current live state."""
        limits = self.config["handoff"]
        metrics = self._state_metrics(contacts)
        checks = (
            (metrics["root_z"] >= float(limits["minimum_root_height"]), "root_height"),
            (abs(metrics["roll"]) <= float(limits["maximum_abs_roll"]), "roll"),
            (abs(metrics["pitch"]) <= float(limits["maximum_abs_pitch"]), "pitch"),
            (
                abs(metrics["pitch_angular_velocity"])
                <= float(limits["maximum_abs_pitch_angular_velocity"]),
                "pitch_angular_velocity",
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
        reasons = [name for passed, name in checks if not passed]
        neutral = self.walkamp.neutral_target("handoff_check")
        actual_q = self.data.qpos[self.joint_map.qpos_adr]
        for group, indices in self.group_indices.items():
            posture_error = float(np.max(np.abs(actual_q[indices] - neutral.q[indices])))
            target_delta = float(np.max(np.abs(motion_target.q[indices] - neutral.q[indices])))
            metrics[f"{group}_posture_error"] = posture_error
            metrics[f"{group}_target_delta"] = target_delta
            if posture_error > float(limits["maximum_posture_error"][group]):
                reasons.append(f"{group}_posture_error")
            if target_delta > float(limits["maximum_target_delta"][group]):
                reasons.append(f"{group}_target_delta")
        self.last_handoff = {"metrics": metrics, "reasons": reasons}
        return not reasons, reasons, metrics

    def recovery_complete(
        self,
        contacts: dict[str, float],
    ) -> tuple[bool, list[str], dict[str, float]]:
        """Check stability after WALKAMP has fully acquired control."""
        complete, reasons, metrics = self._check_state(
            contacts,
            self.config["recover"]["completion_limits"],
        )
        self.last_recovery = {"metrics": metrics, "reasons": reasons}
        return complete, reasons, metrics

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

    def _entry_alphas(self, progress: float, stage: str) -> dict[str, float]:
        settings = self.config.get("entry", {})
        if stage == "pre_align":
            if self.entry_mode == "grouped":
                enabled = settings.get(
                    "pre_align_group_weights",
                    {"legs": 0.0, "waist": 0.0, "arms": 1.0},
                )
            else:
                enabled = {"legs": 1.0, "waist": 1.0, "arms": 1.0}
            alpha = quintic_alpha(progress)
            return {
                group: alpha * float(enabled[group])
                for group in ("legs", "waist", "arms")
            }
        if stage != "transition_in":
            raise ValueError(f"Unknown entry stage {stage!r}")
        scales = settings.get(
            "transition_duration_scale",
            {"legs": 1.0, "waist": 1.0, "arms": 1.0},
        )
        alphas: dict[str, float] = {}
        for group in ("legs", "waist", "arms"):
            scale = float(scales[group])
            if not 0.0 < scale <= 1.0:
                raise ValueError(
                    "Entry transition duration scales must be inside (0, 1]"
                )
            alphas[group] = quintic_alpha(min(progress / scale, 1.0))
        return alphas

    def _target_group_deltas(
        self,
        target: ControlTarget,
        reference: ControlTarget,
    ) -> dict[str, dict[str, float]]:
        values: dict[str, dict[str, float]] = {}
        for group, indices in self.group_indices.items():
            values[group] = {
                field: float(
                    np.max(
                        np.abs(
                            getattr(target, field)[indices]
                            - getattr(reference, field)[indices]
                        )
                    )
                )
                for field in ("q", "kp", "kd", "feedforward", "effort")
            }
        values["scalar"] = {
            "torque_scale": abs(target.torque_scale - reference.torque_scale)
        }
        return values

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

    def _start_walkamp_interpolation(self, reason: str) -> None:
        phase_time = (
            self.active_motion.walkamp_reentry_phase_time
            if self.active_motion is not None
            else float(self.config.get("walkamp_reentry_phase_time", 0.0))
        )
        self.walkamp.begin_from_live_state((0.0, 0.0, 0.0), phase_time=phase_time)
        self.reentry_started = True
        self.reentry_step = 0
        self.recover_stable_steps = 0
        self._log_event(
            "walkamp_interpolation_started",
            reason=reason,
            frozen_motion_step=self.frozen_motion_step,
            exit_wait_seconds=self.state_step * self.control_dt,
            walkamp_phase_time=phase_time,
            metrics=self._state_metrics(self.last_contacts),
        )

    def _begin_transition_out(self, reason: str, handoff_prequalified: bool = False) -> None:
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
        self._log_event(
            "transition_out_started",
            reason=reason,
            frozen_motion_step=self.frozen_motion_step,
            handoff_prequalified=handoff_prequalified,
            metrics=self._state_metrics(self.last_contacts),
        )
        if handoff_prequalified:
            self._start_walkamp_interpolation("prequalified_tail_window")

    def _enter_terminal_failure(
        self,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.recovery_failed = True
        self.terminal_failure_reason = reason
        if self.active_record is not None and self.active_record.status not in (
            CommandStatus.COMPLETED.value,
            CommandStatus.REJECTED.value,
            CommandStatus.FAILED.value,
        ):
            self._update_record(self.active_record, CommandStatus.FAILED, reason)
        if self.abort_record is not None and self.abort_record.status not in (
            CommandStatus.COMPLETED.value,
            CommandStatus.FAILED.value,
        ):
            self._update_record(self.abort_record, CommandStatus.FAILED, reason)
        self._log_event(
            "terminal_failure",
            reason=reason,
            metrics=self._state_metrics(self.last_contacts),
            **(details or {}),
        )
        self._set_state(ControllerState.FAILED, reason)

    def _finish_recovery(self) -> None:
        self._log_event(
            "recovery_completed",
            recovery_seconds=self.state_step * self.control_dt,
            metrics=self._state_metrics(self.last_contacts),
        )
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
        self.handoff_stable_steps = 0
        self.exit_window_opened = False
        self.recover_stable_steps = 0
        self.entry_anchor_target = None
        self.entry_balance_start = None
        self.entry_group_alphas = {"legs": 0.0, "waist": 0.0, "arms": 0.0}
        self._set_state(ControllerState.WALKAMP_IDLE, "recovery_complete")

    def fail(self, reason: str) -> None:
        self._enter_terminal_failure(reason)

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
        if self.entry_mode == "legacy":
            progress = min((self.state_step + 1) / duration, 1.0)
            target = prealign_target(walk_target, motion_target, progress)
            alpha = quintic_alpha(progress)
            self.entry_group_alphas = {
                "legs": alpha,
                "waist": alpha,
                "arms": alpha,
            }
        else:
            progress = endpoint_progress(self.state_step, duration)
            self.entry_group_alphas = self._entry_alphas(progress, "pre_align")
            target = prealign_group_target(
                walk_target,
                motion_target,
                self.entry_group_alphas,
                self.joint_map,
            )
        self.state_step += 1

        if progress >= 1.0:
            safe, reasons, _ = self._check_state(contacts, settings["handoff_limits"])
            errors = self._tracking_errors(motion_target)
            tracking_limits = settings["maximum_tracking_error"]
            checked_groups = (
                set(errors)
                if self.entry_mode != "grouped"
                else {
                    group
                    for group, weight in self.config["entry"][
                        "pre_align_group_weights"
                    ].items()
                    if float(weight) > 0.0
                }
            )
            tracking_reasons = [
                f"{group}_tracking={value:.3f}>{float(tracking_limits[group]):.3f}"
                for group, value in errors.items()
                if group in checked_groups and value > float(tracking_limits[group])
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
        if self.entry_mode == "legacy":
            progress = min((self.state_step + 1) / duration, 1.0)
            alpha = quintic_alpha(progress)
            self.entry_group_alphas = {
                "legs": alpha,
                "waist": alpha,
                "arms": alpha,
            }
            target = blend_targets(walk_target, motion_target, progress, "transition_in")
        else:
            progress = endpoint_progress(self.state_step, duration)
            if self.entry_anchor_target is None:
                self.entry_anchor_target = copy_target(
                    self.last_target,
                    "entry_anchor_applied",
                )
                self.entry_balance_start = copy_target(
                    walk_target,
                    "entry_balance_start",
                )
            assert self.entry_balance_start is not None
            self.entry_group_alphas = self._entry_alphas(progress, "transition_in")
            anchored_groups = (
                {"arms"}
                if self.entry_mode == "grouped"
                else {"legs", "waist", "arms"}
            )
            target = continuous_group_transition_target(
                self.entry_anchor_target,
                self.entry_balance_start,
                walk_target,
                motion_target,
                self.entry_group_alphas,
                self.joint_map,
                anchored_groups=anchored_groups,
            )
            if self.state_step == 0:
                self._log_event(
                    "entry_transition_started",
                    entry_mode=self.entry_mode,
                    group_alphas=dict(self.entry_group_alphas),
                    raw_target_delta=self._target_group_deltas(
                        target,
                        self.entry_anchor_target,
                    ),
                    metrics=self._state_metrics(self.last_contacts),
                )
        self.state_step += 1
        if progress >= 1.0:
            self.motion_step = 0
            self._log_event(
                "entry_transition_completed",
                entry_mode=self.entry_mode,
                transition_seconds=self.state_step * self.control_dt,
                group_alphas=dict(self.entry_group_alphas),
                metrics=self._state_metrics(self.last_contacts),
            )
            self._set_state(ControllerState.MOTION, "motion_control_acquired")
        return ControllerOutput(target, observation, action, progress)

    def _step_motion(self, contacts: dict[str, float]) -> ControllerOutput:
        assert self.active_motion is not None
        policy_step = self.motion_step
        target, observation, action = self.active_motion.step(policy_step)
        self.motion_step += 1
        exit_window = self.active_motion.exit_window
        window_steps = self._duration_steps(float(exit_window.get("seconds_before_end", 0.0)))
        in_exit_window = bool(exit_window.get("enabled", False)) and (
            self.active_motion.duration_steps - self.motion_step <= window_steps
        )
        if in_exit_window and not self.exit_window_opened:
            self.exit_window_opened = True
            self.handoff_stable_steps = 0
            self._log_event(
                "motion_exit_window_opened",
                motion_step=policy_step,
                remaining_steps=self.active_motion.duration_steps - self.motion_step,
                metrics=self._state_metrics(contacts),
            )

        if in_exit_window:
            feasible, reasons, metrics = self.handoff_feasible(contacts, target)
            self.handoff_stable_steps = self.handoff_stable_steps + 1 if feasible else 0
            required = self._duration_steps(float(exit_window["stable_seconds"]))
            if self.handoff_stable_steps >= required:
                self._log_event(
                    "handoff_condition_met",
                    source="motion_tail",
                    motion_step=policy_step,
                    stable_seconds=self.handoff_stable_steps * self.control_dt,
                    metrics=metrics,
                )
                self._log_event(
                    "motion_execution_ended",
                    reason="early_handoff_window",
                    motion_step=policy_step,
                    reference_steps_remaining=(
                        self.active_motion.duration_steps - self.motion_step
                    ),
                    metrics=metrics,
                )
                self._begin_transition_out("early_handoff_window", handoff_prequalified=True)

        if self.state == ControllerState.MOTION and self.abort_pending:
            feasible, _, metrics = self.handoff_feasible(contacts, target)
            self.abort_safe_steps = self.abort_safe_steps + 1 if feasible else 0
            if self.abort_safe_steps >= self._duration_steps(
                self.config["handoff"]["abort_stable_seconds"]
            ):
                self._log_event(
                    "handoff_condition_met",
                    source="safe_abort",
                    motion_step=policy_step,
                    stable_seconds=self.abort_safe_steps * self.control_dt,
                    metrics=metrics,
                )
                self._begin_transition_out("safe_abort_window", handoff_prequalified=True)
        if (
            self.state == ControllerState.MOTION
            and self.motion_step >= self.active_motion.duration_steps
        ):
            self._log_event(
                "motion_timeline_ended",
                motion_step=policy_step,
                metrics=self._state_metrics(contacts),
            )
            reason = "safe_abort_terminal" if self.abort_pending else "motion_complete"
            self._log_event(
                "motion_execution_ended",
                reason=reason,
                motion_step=policy_step,
                reference_steps_remaining=0,
                metrics=self._state_metrics(contacts),
            )
            self._begin_transition_out(reason, handoff_prequalified=False)
        return ControllerOutput(target, observation, action, 1.0)

    def _step_transition_out(self, contacts: dict[str, float]) -> ControllerOutput:
        observation = None
        action = None
        if self.active_motion is not None and self.active_motion.episode_initialized:
            source, observation, action = self.active_motion.step(self.frozen_motion_step)
        else:
            source = self.last_target

        if not self.reentry_started:
            feasible, reasons, metrics = self.handoff_feasible(contacts, source)
            self.handoff_stable_steps = self.handoff_stable_steps + 1 if feasible else 0
            target = source
            progress = 0.0
            settings = self.config["handoff"]
            required = self._duration_steps(settings["stable_seconds"])
            if self.handoff_stable_steps >= required:
                self._log_event(
                    "handoff_condition_met",
                    source="terminal_wait",
                    motion_step=self.frozen_motion_step,
                    stable_seconds=self.handoff_stable_steps * self.control_dt,
                    metrics=metrics,
                )
                self._start_walkamp_interpolation("terminal_wait_verified")
            elif self.state_step >= self._duration_steps(settings["terminal_wait_timeout_seconds"]):
                reason = "handoff_timeout:" + ",".join(reasons)
                self._enter_terminal_failure(
                    reason,
                    {
                        "exit_wait_seconds": self.state_step * self.control_dt,
                        "handoff": self.last_handoff,
                    },
                )
        else:
            walk_target, _, _ = self.walkamp.step()
            duration = self._duration_steps(self.config["transition_out_seconds"])
            progress = min((self.reentry_step + 1) / duration, 1.0)
            target = blend_targets(source, walk_target, progress, "transition_out")
            self.reentry_step += 1
            if progress >= 1.0:
                self.recover_stable_steps = 0
                self._set_state(ControllerState.RECOVER, "walkamp_control_acquired")
                self._log_event(
                    "walkamp_control_acquired",
                    transition_seconds=self.reentry_step * self.control_dt,
                    metrics=self._state_metrics(contacts),
                )
                return ControllerOutput(target, observation, action, progress)
        self.state_step += 1
        return ControllerOutput(target, observation, action, progress)

    def _step_recover(self, contacts: dict[str, float]) -> ControllerOutput:
        settings = self.config["recover"]
        target, observation, action = self.walkamp.step()
        safe, reasons, metrics = self.recovery_complete(contacts)
        self.recover_stable_steps = self.recover_stable_steps + 1 if safe else 0
        required = self._duration_steps(settings["stable_seconds"])
        if self.recover_stable_steps >= required:
            self._finish_recovery()
            return ControllerOutput(target, observation, action, 0.0)
        elif self.state_step >= self._duration_steps(settings["timeout_seconds"]):
            reason = "recover_timeout:" + ",".join(reasons)
            self._enter_terminal_failure(
                reason,
                {
                    "recovery_seconds": self.state_step * self.control_dt,
                    "recovery": {"metrics": metrics, "reasons": reasons},
                },
            )
        self.state_step += 1
        return ControllerOutput(target, observation, action, 0.0)

    def step(self, contacts: dict[str, float]) -> ControllerOutput:
        self.last_contacts = dict(contacts)
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
        elif self.state == ControllerState.FAILED:
            output = ControllerOutput(self.last_target, None, None, 0.0)
        else:
            raise RuntimeError(f"Unknown controller state {self.state!r}")

        self.last_raw_target = copy_target(output.target, "raw_requested")
        output.target = self.target_limiter.apply(output.target)
        self.last_target = output.target
        self.global_step += 1
        return output

    def summary(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "busy": self.busy,
            "entry_mode": self.entry_mode,
            "recovery_failed": self.recovery_failed,
            "last_rejection_reason": self.last_rejection_reason,
            "last_start_delta": self.last_start_delta,
            "last_readiness": self.last_readiness,
            "last_handoff": self.last_handoff,
            "last_recovery": self.last_recovery,
            "terminal_failure_reason": self.terminal_failure_reason,
            "state_history": self.state_history,
            "events": self.event_history,
            "commands": [asdict(record) for record in self.command_records],
        }
