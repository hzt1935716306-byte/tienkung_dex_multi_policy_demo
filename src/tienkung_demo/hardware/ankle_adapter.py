from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class AnkleConversionError(RuntimeError):
    pass


@dataclass(frozen=True)
class AnkleState:
    position: np.ndarray
    velocity: np.ndarray
    torque: np.ndarray


@dataclass(frozen=True)
class ParallelAnkleCommand:
    position: np.ndarray
    velocity: np.ndarray
    feedforward: np.ndarray
    kp: np.ndarray
    kd: np.ndarray


class SptlibAnkleAdapter:
    """Thin, checked wrapper around the official precompiled ``sptlib_python``."""

    def __init__(
        self,
        parallel_kp: np.ndarray | list[float],
        parallel_kd: np.ndarray | list[float],
        library: Any | None = None,
    ) -> None:
        if library is None:
            try:
                from sptlib_python import funcSPTrans
            except ImportError as exc:
                raise AnkleConversionError(
                    "official sptlib_python is not installed; ankle conversion is unavailable"
                ) from exc
            library = funcSPTrans()
        self.library = library
        self.parallel_kp = self._four(parallel_kp, "parallel_kp")
        self.parallel_kd = self._four(parallel_kd, "parallel_kd")
        if np.any(self.parallel_kp <= 0.0) or np.any(self.parallel_kd < 0.0):
            raise ValueError("parallel ankle gains are invalid")
        self.last_parallel_state: AnkleState | None = None

    @staticmethod
    def _four(value: Any, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64).copy()
        if result.shape != (4,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain four finite values")
        return result

    def parallel_to_serial(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        torque: np.ndarray,
    ) -> AnkleState:
        parallel = AnkleState(
            self._four(position, "parallel position"),
            self._four(velocity, "parallel velocity"),
            self._four(torque, "parallel torque"),
        )
        self.library.set_p_est(parallel.position, parallel.velocity, parallel.torque)
        self.library.calcFK()
        self.library.calcIK()
        success, serial_position, serial_velocity, serial_torque = self.library.get_s_state()
        if not bool(success):
            raise AnkleConversionError("official parallel-to-serial ankle conversion failed")
        serial = AnkleState(
            self._four(serial_position, "serial position"),
            self._four(serial_velocity, "serial velocity"),
            self._four(serial_torque, "serial torque"),
        )
        self.last_parallel_state = parallel
        return serial

    def serial_to_parallel_command(
        self,
        desired_position: np.ndarray,
        desired_velocity: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        feedforward: np.ndarray,
        measured_serial_position: np.ndarray,
        measured_serial_velocity: np.ndarray,
    ) -> ParallelAnkleCommand:
        if self.last_parallel_state is None:
            raise AnkleConversionError("parallel ankle state must be converted before encoding a command")
        desired_position = self._four(desired_position, "serial desired position")
        desired_velocity = self._four(desired_velocity, "serial desired velocity")
        kp = self._four(kp, "serial kp")
        kd = self._four(kd, "serial kd")
        feedforward = self._four(feedforward, "serial feedforward")
        measured_position = self._four(measured_serial_position, "serial measured position")
        measured_velocity = self._four(measured_serial_velocity, "serial measured velocity")

        serial_torque = (
            kp * (desired_position - measured_position)
            + kd * (desired_velocity - measured_velocity)
            + feedforward
        )
        self.library.set_s_des(desired_position, desired_velocity, serial_torque)
        self.library.calc_joint_pos_ref()
        self.library.calc_joint_tor_des()
        success, parallel_position, parallel_velocity, parallel_torque = self.library.get_p_des()
        if not bool(success):
            raise AnkleConversionError("official serial-to-parallel ankle conversion failed")
        parallel_position = self._four(parallel_position, "parallel desired position")
        parallel_velocity = self._four(parallel_velocity, "parallel desired velocity")
        parallel_torque = self._four(parallel_torque, "parallel desired torque")
        state = self.last_parallel_state
        # This is the position-equivalent command used by Deploy_Tienkung 3.0.
        position_equivalent = (
            parallel_torque
            - self.parallel_kd * (parallel_velocity - state.velocity)
        ) / self.parallel_kp + state.position
        return ParallelAnkleCommand(
            position=self._four(position_equivalent, "parallel equivalent position"),
            velocity=parallel_velocity,
            feedforward=np.zeros(4, dtype=np.float64),
            kp=self.parallel_kp.copy(),
            kd=self.parallel_kd.copy(),
        )
