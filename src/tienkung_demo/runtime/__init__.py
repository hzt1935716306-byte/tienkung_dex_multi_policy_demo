"""MuJoCo-independent policy runtime primitives."""

from .control_target import ControlTarget, copy_control_target
from .robot_state import RobotStateSnapshot

__all__ = ["ControlTarget", "RobotStateSnapshot", "copy_control_target"]
