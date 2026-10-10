from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


STAND_DIAGNOSTIC_COLUMNS = [
    "left_foot_x",
    "left_foot_y",
    "left_foot_z",
    "right_foot_x",
    "right_foot_y",
    "right_foot_z",
    "com_x",
    "com_y",
    "com_z",
    "com_vx",
    "com_vy",
    "com_vz",
    "support_code",
    "contact_edge_count",
    "complete_step_count",
    "stance_slip_maximum",
]


@dataclass
class StepEvent:
    step: int
    time: float
    foot: str
    flight_seconds: float
    horizontal_displacement: float
    takeoff_xy: list[float]
    landing_xy: list[float]


@dataclass
class _FootState:
    raw_contact: bool | None = None
    airborne: bool = False
    takeoff_xy: np.ndarray | None = None
    flight_steps: int = 0
    landing_steps: int = 0
    landing_xy: np.ndarray | None = None
    stance_anchor_xy: np.ndarray | None = None
    last_contact_xy: np.ndarray | None = None
    maximum_stance_slip: float = 0.0


class CompleteStepDetector:
    """Separate raw contact edges from lift-displace-land step events."""

    def __init__(
        self,
        control_dt: float,
        minimum_flight_seconds: float = 0.04,
        minimum_horizontal_displacement: float = 0.025,
        landing_stable_seconds: float = 0.04,
    ) -> None:
        self.control_dt = float(control_dt)
        self.minimum_flight_steps = max(
            1, int(round(float(minimum_flight_seconds) / self.control_dt))
        )
        self.minimum_horizontal_displacement = float(
            minimum_horizontal_displacement
        )
        self.landing_stable_steps = max(
            1, int(round(float(landing_stable_seconds) / self.control_dt))
        )
        self.feet = {"left": _FootState(), "right": _FootState()}
        self.contact_edge_count = 0
        self.events: list[StepEvent] = []

    def _update_foot(
        self,
        foot: str,
        step: int,
        contact: bool,
        position: np.ndarray,
    ) -> None:
        state = self.feet[foot]
        xy = np.asarray(position[0:2], dtype=np.float64)
        if state.raw_contact is None:
            state.raw_contact = contact
            if contact:
                state.stance_anchor_xy = xy.copy()
                state.last_contact_xy = xy.copy()
            return
        if contact != state.raw_contact:
            self.contact_edge_count += 1
        state.raw_contact = contact

        if not state.airborne:
            if contact:
                if state.stance_anchor_xy is None:
                    state.stance_anchor_xy = xy.copy()
                state.maximum_stance_slip = max(
                    state.maximum_stance_slip,
                    float(np.linalg.norm(xy - state.stance_anchor_xy)),
                )
                state.last_contact_xy = xy.copy()
                return
            state.airborne = True
            state.takeoff_xy = (
                xy.copy()
                if state.last_contact_xy is None
                else state.last_contact_xy.copy()
            )
            state.flight_steps = 1
            state.landing_steps = 0
            state.landing_xy = None
            return

        if not contact:
            state.flight_steps += 1
            state.landing_steps = 0
            state.landing_xy = None
            return

        state.landing_steps += 1
        state.landing_xy = xy.copy()
        if state.landing_steps < self.landing_stable_steps:
            return

        assert state.takeoff_xy is not None
        displacement = float(np.linalg.norm(state.landing_xy - state.takeoff_xy))
        if (
            state.flight_steps >= self.minimum_flight_steps
            and displacement >= self.minimum_horizontal_displacement
        ):
            self.events.append(
                StepEvent(
                    step=step,
                    time=step * self.control_dt,
                    foot=foot,
                    flight_seconds=state.flight_steps * self.control_dt,
                    horizontal_displacement=displacement,
                    takeoff_xy=state.takeoff_xy.tolist(),
                    landing_xy=state.landing_xy.tolist(),
                )
            )
        state.airborne = False
        state.takeoff_xy = None
        state.flight_steps = 0
        state.landing_steps = 0
        state.stance_anchor_xy = state.landing_xy.copy()
        state.last_contact_xy = state.landing_xy.copy()
        state.landing_xy = None

    def update(
        self,
        step: int,
        contacts: dict[str, float],
        left_position: np.ndarray,
        right_position: np.ndarray,
    ) -> None:
        self._update_foot(
            "left",
            step,
            contacts.get("left_contact_count", 0.0) > 0.0,
            left_position,
        )
        self._update_foot(
            "right",
            step,
            contacts.get("right_contact_count", 0.0) > 0.0,
            right_position,
        )

    @property
    def maximum_stance_slip(self) -> float:
        return max(state.maximum_stance_slip for state in self.feet.values())

    def summary(self) -> dict[str, Any]:
        return {
            "contact_edge_count": self.contact_edge_count,
            "complete_step_count": len(self.events),
            "complete_steps": [asdict(event) for event in self.events],
            "maximum_stance_slip": self.maximum_stance_slip,
            "thresholds": {
                "minimum_flight_seconds": (
                    self.minimum_flight_steps * self.control_dt
                ),
                "minimum_horizontal_displacement": (
                    self.minimum_horizontal_displacement
                ),
                "landing_stable_seconds": (
                    self.landing_stable_steps * self.control_dt
                ),
            },
        }


@dataclass
class _Range:
    minimum: np.ndarray = field(
        default_factory=lambda: np.full(3, np.inf, dtype=np.float64)
    )
    maximum: np.ndarray = field(
        default_factory=lambda: np.full(3, -np.inf, dtype=np.float64)
    )

    def update(self, value: np.ndarray) -> None:
        self.minimum = np.minimum(self.minimum, value)
        self.maximum = np.maximum(self.maximum, value)

    def span(self) -> list[float]:
        if not np.all(np.isfinite(self.minimum)):
            return [0.0, 0.0, 0.0]
        return (self.maximum - self.minimum).tolist()


class StandDiagnostics:
    """Online kinematic diagnostics without changing MuJoCo state or control."""

    def __init__(
        self,
        model,
        control_dt: float,
        step_detection: dict[str, Any] | None = None,
    ) -> None:
        import mujoco

        self.model = model
        self.control_dt = float(control_dt)
        self.foot_geom_ids: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            ids = [
                geom_id
                for geom_id in range(model.ngeom)
                if (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                    or ""
                ).startswith(f"foot_{side}_")
            ]
            if not ids:
                raise ValueError(f"Official EVT2 {side} foot geoms are missing")
            self.foot_geom_ids[side] = np.asarray(ids, dtype=np.int32)
        self.body_masses = np.asarray(model.body_mass, dtype=np.float64)
        self.total_mass = float(np.sum(self.body_masses))
        if self.total_mass <= 0.0:
            raise ValueError("MuJoCo model has no positive body mass")
        self.step_detector = CompleteStepDetector(
            control_dt,
            **(step_detection or {}),
        )
        self.previous_com: np.ndarray | None = None
        self.initial_positions: dict[str, np.ndarray] = {}
        self.latest_positions: dict[str, np.ndarray] = {}
        self.ranges = {
            "left_foot": _Range(),
            "right_foot": _Range(),
            "com": _Range(),
        }
        self.maximum_com_speed = 0.0
        self.maximum_joint_speed = 0.0
        self.maximum_abs_torque = 0.0
        self.maximum_torque_delta = 0.0
        self.previous_torque: np.ndarray | None = None
        self.single_support_samples = 0
        self.no_support_samples = 0
        self.samples = 0

    def _foot_position(self, data, side: str) -> np.ndarray:
        return np.mean(data.geom_xpos[self.foot_geom_ids[side]], axis=0)

    def _center_of_mass(self, data) -> np.ndarray:
        return np.sum(
            np.asarray(data.xipos) * self.body_masses[:, None], axis=0
        ) / self.total_mass

    def update(
        self,
        step: int,
        data,
        contacts: dict[str, float],
        joint_velocity: np.ndarray,
        torque: np.ndarray,
    ) -> dict[str, float]:
        left = self._foot_position(data, "left")
        right = self._foot_position(data, "right")
        com = self._center_of_mass(data)
        com_velocity = (
            np.zeros(3, dtype=np.float64)
            if self.previous_com is None
            else (com - self.previous_com) / self.control_dt
        )
        self.previous_com = com.copy()
        positions = {"left_foot": left, "right_foot": right, "com": com}
        if not self.initial_positions:
            self.initial_positions = {
                name: value.copy() for name, value in positions.items()
            }
        self.latest_positions = {
            name: value.copy() for name, value in positions.items()
        }
        for name, value in positions.items():
            self.ranges[name].update(value)
        self.step_detector.update(step, contacts, left, right)

        support = (
            contacts.get("left_contact_count", 0.0) > 0.0,
            contacts.get("right_contact_count", 0.0) > 0.0,
        )
        support_count = int(support[0]) + int(support[1])
        self.single_support_samples += int(support_count == 1)
        self.no_support_samples += int(support_count == 0)
        self.maximum_com_speed = max(
            self.maximum_com_speed,
            float(np.linalg.norm(com_velocity[0:2])),
        )
        self.maximum_joint_speed = max(
            self.maximum_joint_speed,
            float(np.max(np.abs(joint_velocity))),
        )
        self.maximum_abs_torque = max(
            self.maximum_abs_torque,
            float(np.max(np.abs(torque))),
        )
        if self.previous_torque is not None:
            self.maximum_torque_delta = max(
                self.maximum_torque_delta,
                float(np.max(np.abs(torque - self.previous_torque))),
            )
        self.previous_torque = torque.copy()
        self.samples += 1
        return {
            "left_foot_x": float(left[0]),
            "left_foot_y": float(left[1]),
            "left_foot_z": float(left[2]),
            "right_foot_x": float(right[0]),
            "right_foot_y": float(right[1]),
            "right_foot_z": float(right[2]),
            "com_x": float(com[0]),
            "com_y": float(com[1]),
            "com_z": float(com[2]),
            "com_vx": float(com_velocity[0]),
            "com_vy": float(com_velocity[1]),
            "com_vz": float(com_velocity[2]),
            "support_code": float(support_count),
            "contact_edge_count": float(self.step_detector.contact_edge_count),
            "complete_step_count": float(len(self.step_detector.events)),
            "stance_slip_maximum": self.step_detector.maximum_stance_slip,
        }

    def summary(self) -> dict[str, Any]:
        def displacement(name: str) -> list[float]:
            if name not in self.initial_positions:
                return [0.0, 0.0, 0.0]
            return (
                self.latest_positions[name] - self.initial_positions[name]
            ).tolist()

        return {
            **self.step_detector.summary(),
            "samples": self.samples,
            "single_support_samples": self.single_support_samples,
            "single_support_seconds": self.single_support_samples * self.control_dt,
            "no_support_samples": self.no_support_samples,
            "no_support_seconds": self.no_support_samples * self.control_dt,
            "maximum_com_xy_speed": self.maximum_com_speed,
            "maximum_joint_speed": self.maximum_joint_speed,
            "maximum_abs_torque": self.maximum_abs_torque,
            "maximum_torque_delta": self.maximum_torque_delta,
            "left_foot_span": self.ranges["left_foot"].span(),
            "right_foot_span": self.ranges["right_foot"].span(),
            "com_span": self.ranges["com"].span(),
            "left_foot_displacement": displacement("left_foot"),
            "right_foot_displacement": displacement("right_foot"),
            "com_displacement": displacement("com"),
        }
