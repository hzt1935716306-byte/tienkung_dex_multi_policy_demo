from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Mapping

import numpy as np

from ..runtime.control_target import ControlTarget
from ..runtime.robot_state import RobotStateSnapshot
from .ankle_adapter import SptlibAnkleAdapter
from .joint_mapper import HardwareJointMapper, OFFICIAL_HARDWARE_JOINTS
from .state_estimator import ImuAdapter


OFFICIAL_STATE_TOPICS = {
    "leg": "/leg/status",
    "arm": "/arm/status",
    "waist": "/waist/status",
    "imu": "/imu/status",
}

OFFICIAL_COMMAND_TOPICS = {
    "leg": "/leg/cmd_ctrl",
    "arm": "/arm/cmd_ctrl",
    "waist": "/waist/cmd_ctrl",
}

OFFICIAL_MESSAGE_TYPES = {
    "motor_status": "bodyctrl_msgs/msg/MotorStatusMsg",
    "imu": "bodyctrl_msgs/msg/Imu",
    "motor_command": "bodyctrl_msgs/msg/CmdMotorCtrl",
}


def ros_stamp_seconds(message: Any) -> float:
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


@dataclass(frozen=True)
class HardwareMotorCommand:
    name: int
    kp: float
    kd: float
    pos: float
    spd: float
    tor: float


@dataclass(frozen=True)
class HardwareCommandFrame:
    timestamp: float
    label: str
    groups: Mapping[str, tuple[HardwareMotorCommand, ...]]

    @property
    def command_count(self) -> int:
        return sum(len(values) for values in self.groups.values())


class OfficialRosStateBuffer:
    """Pure-Python parser for official message-shaped objects.

    This class has no ROS dependency and is used by unit tests and offline replay.
    """

    expected_ids = {
        "leg": {item.can_id for item in OFFICIAL_HARDWARE_JOINTS if item.group == "leg"},
        "waist": {item.can_id for item in OFFICIAL_HARDWARE_JOINTS if item.group == "waist"},
        "arm": {item.can_id for item in OFFICIAL_HARDWARE_JOINTS if item.group == "arm"},
    }

    def __init__(
        self,
        mapper: HardwareJointMapper,
        imu_adapter: ImuAdapter,
        ankle_adapter: SptlibAnkleAdapter | None,
        *,
        ankle_state_space: str = "parallel",
    ) -> None:
        if ankle_state_space not in ("parallel", "serial"):
            raise ValueError("ankle_state_space must be parallel or serial")
        self.mapper = mapper
        self.imu_adapter = imu_adapter
        self.ankle_adapter = ankle_adapter
        self.ankle_state_space = ankle_state_space
        self._motor_messages: dict[str, tuple[Any, float, float]] = {}
        self._imu_message: tuple[Any, float, float] | None = None
        self._lock = threading.Lock()

    def ingest_motor(self, group: str, message: Any, received_monotonic: float | None = None) -> None:
        if group not in self.expected_ids:
            raise ValueError(f"Unknown motor group {group!r}")
        statuses = list(message.status)
        ids = [int(status.name) for status in statuses]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate CAN ID in {group} status message")
        if set(ids) != self.expected_ids[group]:
            missing = sorted(self.expected_ids[group] - set(ids))
            extra = sorted(set(ids) - self.expected_ids[group])
            raise ValueError(f"{group} status IDs mismatch: missing={missing}, extra={extra}")
        received = time.monotonic() if received_monotonic is None else float(received_monotonic)
        with self._lock:
            self._motor_messages[group] = (message, ros_stamp_seconds(message), received)

    def ingest_imu(self, message: Any, received_monotonic: float | None = None) -> None:
        received = time.monotonic() if received_monotonic is None else float(received_monotonic)
        self.imu_adapter.convert(message)
        with self._lock:
            self._imu_message = (message, ros_stamp_seconds(message), received)

    def snapshot(self, now_monotonic: float | None = None) -> RobotStateSnapshot:
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        with self._lock:
            missing = [name for name in ("leg", "waist", "arm") if name not in self._motor_messages]
            if self._imu_message is None:
                missing.append("imu")
            if missing:
                raise RuntimeError(f"Cannot build robot state; missing topics: {missing}")
            motor_messages = dict(self._motor_messages)
            imu_message = self._imu_message
        assert imu_message is not None
        statuses = []
        for group in ("leg", "waist", "arm"):
            statuses.extend(motor_messages[group][0].status)
        mapped = self.mapper.map_statuses(statuses)
        position = mapped.position.copy()
        velocity = mapped.velocity.copy()
        torque = mapped.torque.copy()
        if self.ankle_state_space == "parallel":
            if self.ankle_adapter is None:
                raise RuntimeError("Official parallel ankle state requires sptlib_python conversion")
            indices = self.mapper.ankle_runtime_indices
            serial = self.ankle_adapter.parallel_to_serial(
                position[indices], velocity[indices], torque[indices]
            )
            position[indices] = serial.position
            velocity[indices] = serial.velocity
            torque[indices] = serial.torque

        orientation, angular_velocity, acceleration, imu_validity = self.imu_adapter.convert(imu_message[0])
        source_timestamps = {
            name: values[1] for name, values in motor_messages.items()
        }
        source_timestamps["imu"] = imu_message[1]
        received_times = [values[2] for values in motor_messages.values()] + [imu_message[2]]
        return RobotStateSnapshot(
            timestamp=max(source_timestamps.values()),
            # The snapshot is only as fresh as its oldest required topic.
            received_monotonic=min(received_times),
            source="tiangong_ros2",
            joint_names=self.mapper.runtime_names,
            joint_position=position,
            joint_velocity=velocity,
            orientation_wxyz=orientation,
            angular_velocity_body=angular_velocity,
            joint_torque=torque,
            joint_temperature=mapped.temperature,
            motor_error=mapped.error,
            linear_acceleration_body=acceleration,
            validity={
                "joint_position": True,
                "joint_velocity": True,
                "joint_torque": True,
                "joint_temperature": True,
                "motor_error": True,
                "linear_acceleration_body": True,
                "base_position_world": False,
                "base_linear_velocity_world": False,
                "foot_contact": False,
                "foot_normal_force": False,
                **imu_validity,
            },
            source_timestamps=source_timestamps,
        )

    def topic_ages(self, now_monotonic: float | None = None) -> dict[str, float]:
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        with self._lock:
            ages = {name: now - values[2] for name, values in self._motor_messages.items()}
            if self._imu_message is not None:
                ages["imu"] = now - self._imu_message[2]
        return ages


class TiangongRos2StateSource:
    """Read-only ROS2 source. It deliberately creates no publishers."""

    def __init__(self, buffer: OfficialRosStateBuffer, node_name: str = "tienkung_shadow_state") -> None:
        self.buffer = buffer
        self.node_name = node_name
        self.node = None
        self._owns_rclpy = False

    def start(self) -> None:
        try:
            import rclpy
            from bodyctrl_msgs.msg import Imu, MotorStatusMsg
        except ImportError as exc:
            raise RuntimeError(
                "ROS2 Python or bodyctrl_msgs is unavailable. Source the official ROS2 environment first."
            ) from exc
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True
        self.node = rclpy.create_node(self.node_name)
        qos_depth = 10
        self.node.create_subscription(
            MotorStatusMsg,
            OFFICIAL_STATE_TOPICS["leg"],
            lambda message: self.buffer.ingest_motor("leg", message),
            qos_depth,
        )
        self.node.create_subscription(
            MotorStatusMsg,
            OFFICIAL_STATE_TOPICS["arm"],
            lambda message: self.buffer.ingest_motor("arm", message),
            qos_depth,
        )
        self.node.create_subscription(
            MotorStatusMsg,
            OFFICIAL_STATE_TOPICS["waist"],
            lambda message: self.buffer.ingest_motor("waist", message),
            qos_depth,
        )
        self.node.create_subscription(Imu, OFFICIAL_STATE_TOPICS["imu"], self.buffer.ingest_imu, qos_depth)

    def spin_once(self, timeout_seconds: float = 0.01) -> None:
        if self.node is None:
            raise RuntimeError("ROS2 state source has not been started")
        import rclpy

        rclpy.spin_once(self.node, timeout_sec=timeout_seconds)

    def close(self) -> None:
        if self.node is None:
            return
        import rclpy

        self.node.destroy_node()
        self.node = None
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()


class TiangongCommandEncoder:
    """Encodes an audited candidate frame; it never publishes ROS messages."""

    def __init__(self, mapper: HardwareJointMapper, ankle_adapter: SptlibAnkleAdapter | None) -> None:
        self.mapper = mapper
        self.ankle_adapter = ankle_adapter

    def encode(
        self,
        target: ControlTarget,
        state: RobotStateSnapshot,
        timestamp: float,
        desired_velocity: np.ndarray | None = None,
    ) -> HardwareCommandFrame:
        target.validate(29)
        if tuple(state.joint_names) != self.mapper.runtime_names:
            raise ValueError("State joint order differs from hardware mapper")
        q = target.q.copy()
        velocity = np.zeros(29, dtype=np.float64) if desired_velocity is None else np.asarray(desired_velocity, dtype=np.float64).copy()
        if velocity.shape != (29,):
            raise ValueError("desired_velocity must contain 29 values")
        kp = target.kp * target.torque_scale
        kd = target.kd * target.torque_scale
        feedforward = target.feedforward * target.torque_scale

        ankle_indices = self.mapper.ankle_runtime_indices
        if self.ankle_adapter is None:
            raise RuntimeError("Cannot encode official DEX command without sptlib ankle conversion")
        parallel = self.ankle_adapter.serial_to_parallel_command(
            q[ankle_indices],
            velocity[ankle_indices],
            kp[ankle_indices],
            kd[ankle_indices],
            feedforward[ankle_indices],
            state.joint_position[ankle_indices],
            state.joint_velocity[ankle_indices],
        )
        q[ankle_indices] = parallel.position
        velocity[ankle_indices] = parallel.velocity
        kp[ankle_indices] = parallel.kp
        kd[ankle_indices] = parallel.kd
        feedforward[ankle_indices] = parallel.feedforward

        official_position = self.mapper.command_to_hardware_position(q)
        official_velocity = self.mapper.command_to_hardware_signed(velocity)
        official_feedforward = self.mapper.command_to_hardware_signed(feedforward)
        official_kp = self.mapper.runtime_to_official(kp)
        official_kd = self.mapper.runtime_to_official(kd)
        groups: dict[str, list[HardwareMotorCommand]] = {"leg": [], "waist": [], "arm": []}
        for item in OFFICIAL_HARDWARE_JOINTS:
            index = item.official_index
            groups[item.group].append(
                HardwareMotorCommand(
                    name=item.can_id,
                    kp=float(official_kp[index]),
                    kd=float(official_kd[index]),
                    pos=float(official_position[index]),
                    spd=float(official_velocity[index]),
                    tor=float(official_feedforward[index]),
                )
            )
        return HardwareCommandFrame(
            timestamp=float(timestamp),
            label=target.label,
            groups={name: tuple(values) for name, values in groups.items()},
        )


class DisabledMotorCommandSink:
    """Phase 4 software gate: motor command publication is unavailable by design."""

    publish_count = 0

    def publish(self, frame: HardwareCommandFrame) -> None:
        raise PermissionError(
            "Motor publication is disabled in Phase 4 Shadow Mode; no ROS command publishers exist"
        )
