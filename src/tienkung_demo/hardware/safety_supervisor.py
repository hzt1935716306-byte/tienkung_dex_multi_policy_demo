from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..runtime.control_target import ControlTarget
from ..runtime.robot_state import RobotStateSnapshot


@dataclass(frozen=True)
class SafetyDecision:
    shadow_inference_allowed: bool
    transition_evaluation_allowed: bool
    motor_output_allowed: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]


class HardwareSafetySupervisor:
    """Fail-closed software checks around the read-only Phase 4 runtime."""

    def __init__(self, config: Mapping[str, Any], hardware_contract: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.hardware_contract = dict(hardware_contract)
        self.emergency_stop_latched = False
        if not bool(config.get("shadow_only", True)):
            raise ValueError("Phase 4 implementation only supports shadow_only=true")
        if bool(config.get("publish_motor_commands", False)):
            raise ValueError("Phase 4 config must keep publish_motor_commands=false")
        limits = config["limits"]
        self.maximum_state_age = float(limits["maximum_state_age_seconds"])
        self.maximum_source_timestamp_skew = float(
            limits["maximum_source_timestamp_skew_seconds"]
        )
        self.maximum_temperature = float(limits["maximum_temperature_celsius"])
        self.maximum_joint_speed = float(limits["maximum_joint_speed"])
        self.maximum_measured_torque = float(limits["maximum_abs_measured_torque"])
        self.maximum_feedforward = float(limits["maximum_abs_feedforward_torque"])
        self.joint_lower = np.asarray(limits["joint_lower"], dtype=np.float64)
        self.joint_upper = np.asarray(limits["joint_upper"], dtype=np.float64)
        if self.joint_lower.shape != (29,) or self.joint_upper.shape != (29,):
            raise ValueError("Safety joint limits must contain 29 values")
        if np.any(self.joint_lower >= self.joint_upper):
            raise ValueError("Safety joint limits are invalid")

    @classmethod
    def from_files(cls, real_config: Mapping[str, Any], contract_path: str | Path) -> "HardwareSafetySupervisor":
        with Path(contract_path).open("r", encoding="utf-8") as file:
            contract = json.load(file)
        return cls(real_config["safety"], contract)

    def latch_emergency_stop(self, reason: str) -> None:
        self.emergency_stop_latched = True

    def contract_blockers(self) -> list[str]:
        required = self.hardware_contract.get("required_confirmations", {})
        return [name for name, value in required.items() if value.get("status") != "confirmed"]

    def evaluate(
        self,
        state: RobotStateSnapshot,
        target: ControlTarget | None,
        now_monotonic: float,
    ) -> SafetyDecision:
        blockers: list[str] = []
        warnings: list[str] = []
        if self.emergency_stop_latched:
            blockers.append("emergency_stop_latched")
        if state.age(now_monotonic) > self.maximum_state_age:
            blockers.append("state_stale")
        if state.source_timestamps:
            timestamp_values = list(state.source_timestamps.values())
            if max(timestamp_values) - min(timestamp_values) > self.maximum_source_timestamp_skew:
                blockers.append("source_timestamp_skew")
        required_inference = (
            "joint_position",
            "joint_velocity",
            "orientation_wxyz",
            "angular_velocity_body",
        )
        blockers.extend(f"invalid_{name}" for name in required_inference if not state.is_valid(name))
        if state.motor_error is None or np.any(state.motor_error != 0.0):
            blockers.append("motor_error")
        if state.joint_temperature is None:
            blockers.append("temperature_missing")
        elif np.max(state.joint_temperature) > self.maximum_temperature:
            blockers.append("temperature_limit")
        if state.joint_torque is None:
            blockers.append("measured_torque_missing")
        elif np.max(np.abs(state.joint_torque)) > self.maximum_measured_torque:
            blockers.append("measured_torque_limit")
        if np.max(np.abs(state.joint_velocity)) > self.maximum_joint_speed:
            blockers.append("joint_speed_limit")
        if np.any(state.joint_position < self.joint_lower) or np.any(state.joint_position > self.joint_upper):
            blockers.append("measured_joint_limit")
        if target is not None:
            try:
                target.validate(29)
            except ValueError as exc:
                blockers.append(f"invalid_target:{exc}")
            else:
                if np.any(target.q < self.joint_lower) or np.any(target.q > self.joint_upper):
                    blockers.append("target_joint_limit")
                if np.max(np.abs(target.feedforward * target.torque_scale)) > self.maximum_feedforward:
                    blockers.append("feedforward_limit")

        transition_fields = (
            "base_linear_velocity_world",
            "foot_contact",
            "foot_normal_force",
        )
        transition_missing = [name for name in transition_fields if not state.is_valid(name)]
        if transition_missing:
            warnings.extend(f"transition_state_unavailable:{name}" for name in transition_missing)
        contract_blockers = self.contract_blockers()
        warnings.extend(f"hardware_contract_unconfirmed:{name}" for name in contract_blockers)
        shadow_allowed = not blockers
        return SafetyDecision(
            shadow_inference_allowed=shadow_allowed,
            transition_evaluation_allowed=shadow_allowed and not transition_missing,
            motor_output_allowed=False,
            blockers=tuple(sorted(set(blockers))),
            warnings=tuple(sorted(set(warnings))),
        )

    def request_motor_enable(self) -> None:
        raise PermissionError(
            "Motor output is intentionally unavailable on the Phase 4 software-acceptance branch"
        )
