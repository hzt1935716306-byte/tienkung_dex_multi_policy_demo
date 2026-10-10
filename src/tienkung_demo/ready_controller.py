from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .robot import ControlTarget
from .walkamp import WalkAmpPolicy


@dataclass
class ReadyOutput:
    target: ControlTarget
    observation: np.ndarray | None
    action: np.ndarray | None


class ReadyController(Protocol):
    """Interface reserved for a verified full-body READY controller."""

    name: str
    verified: bool

    def begin_from_live_state(self) -> dict[str, object]: ...

    def step(self) -> ReadyOutput: ...


class WalkAmpReadyController:
    """READY baseline that keeps the already-tested WALKAMP feedback loop active."""

    name = "walkamp_feedback"
    verified = True

    def __init__(self, policy: WalkAmpPolicy) -> None:
        self.policy = policy

    def begin_from_live_state(self) -> dict[str, object]:
        self.policy.set_command((0.0, 0.0, 0.0))
        return {
            "controller": self.name,
            "verified_feedback": self.verified,
            "state_reset": False,
        }

    def step(self) -> ReadyOutput:
        target, observation, action = self.policy.step()
        return ReadyOutput(target, observation, action)


def build_ready_controller(
    config: dict[str, object],
    walkamp: WalkAmpPolicy,
) -> ReadyController:
    controller_type = str(config.get("controller", "walkamp_feedback"))
    if controller_type != "walkamp_feedback":
        raise ValueError(
            f"READY controller {controller_type!r} is not verified; "
            "only 'walkamp_feedback' is currently available"
        )
    return WalkAmpReadyController(walkamp)
