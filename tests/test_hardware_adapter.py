from __future__ import annotations

from dataclasses import dataclass, field, replace
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tienkung_demo.hardware.ankle_adapter import AnkleConversionError, SptlibAnkleAdapter
from tienkung_demo.hardware.joint_mapper import HardwareJointMapper, OFFICIAL_HARDWARE_JOINTS
from tienkung_demo.hardware.safety_supervisor import HardwareSafetySupervisor
from tienkung_demo.hardware.state_estimator import ImuAdapter
from tienkung_demo.hardware.tiangong_ros2 import (
    DisabledMotorCommandSink,
    OfficialRosStateBuffer,
    TiangongCommandEncoder,
    TiangongRos2StateSource,
)
from tienkung_demo.motion_evt2 import MotionEvt2Policy, TRAINING_NOMINAL_GAINS
from tienkung_demo.motion_evt2_runner import ensure_episode_ground_clearance
from tienkung_demo.multi_evt2_runner import load_multi_evt2_config
from tienkung_demo.real_runner import build_mapper, build_runtimes, load_real_config
from tienkung_demo.robot import build_joint_map
from tienkung_demo.runtime.policy_runtime import MotionRuntime, RuntimeJointLayout, WalkAmpRuntime
from tienkung_demo.runtime.robot_state import RobotStateSnapshot
from tienkung_demo.walkamp import WalkAmpPolicy, load_walkamp_config
from tienkung_demo.walkamp_runner import setup_initial_pose


@dataclass
class FakeStamp:
    sec: int = 10
    nanosec: int = 20


@dataclass
class FakeHeader:
    stamp: FakeStamp = field(default_factory=FakeStamp)


@dataclass
class FakeStatus:
    name: int
    pos: float
    speed: float
    current: float
    temperature: float = 35.0
    error: int = 0


@dataclass
class FakeMotorMessage:
    status: list[FakeStatus]
    header: FakeHeader = field(default_factory=FakeHeader)


@dataclass
class FakeVector:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class FakeQuaternion:
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class FakeEuler:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass
class FakeImu:
    header: FakeHeader = field(default_factory=FakeHeader)
    orientation: FakeQuaternion = field(default_factory=FakeQuaternion)
    angular_velocity: FakeVector = field(default_factory=FakeVector)
    linear_acceleration: FakeVector = field(default_factory=lambda: FakeVector(0.0, 0.0, 9.81))
    euler: FakeEuler = field(default_factory=FakeEuler)
    error: int = 0


class FakeSpt:
    def __init__(self, success: bool = True) -> None:
        self.success = success
        self.parallel = None
        self.serial_desired = None

    def set_p_est(self, q, qd, tau):
        self.parallel = (np.asarray(q), np.asarray(qd), np.asarray(tau))

    def calcFK(self):
        pass

    def calcIK(self):
        pass

    def get_s_state(self):
        q, qd, tau = self.parallel
        return self.success, q.copy(), qd.copy(), tau.copy()

    def set_s_des(self, q, qd, tau):
        self.serial_desired = (np.asarray(q), np.asarray(qd), np.asarray(tau))

    def calc_joint_pos_ref(self):
        pass

    def calc_joint_tor_des(self):
        pass

    def get_p_des(self):
        q, qd, tau = self.serial_desired
        return self.success, q.copy(), qd.copy(), tau.copy()


class HardwareAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.real_config = load_real_config(ROOT / "configs" / "real_evt2.json")
        cls.mapper = build_mapper(cls.real_config)

    def messages(self):
        grouped = {"leg": [], "waist": [], "arm": []}
        for item in OFFICIAL_HARDWARE_JOINTS:
            grouped[item.group].append(
                FakeStatus(
                    name=item.can_id,
                    pos=item.official_index * 0.01,
                    speed=item.official_index * 0.001,
                    current=0.1,
                )
            )
        return {name: FakeMotorMessage(values) for name, values in grouped.items()}

    def state_buffer(self, *, conversion_success: bool = True):
        ankle = SptlibAnkleAdapter([15.0] * 4, [1.25] * 4, FakeSpt(conversion_success))
        buffer = OfficialRosStateBuffer(
            self.mapper,
            ImuAdapter(self.real_config["imu"]),
            ankle,
            ankle_state_space="parallel",
        )
        for group, message in self.messages().items():
            buffer.ingest_motor(group, message, received_monotonic=20.0)
        buffer.ingest_imu(FakeImu(), received_monotonic=20.0)
        return buffer, ankle

    def test_official_mapping_is_complete_and_bijective(self) -> None:
        self.assertEqual(len(OFFICIAL_HARDWARE_JOINTS), 29)
        self.assertEqual(len({item.can_id for item in OFFICIAL_HARDWARE_JOINTS}), 29)
        self.assertEqual(len({item.runtime_name for item in OFFICIAL_HARDWARE_JOINTS}), 29)
        self.assertEqual([item.can_id for item in OFFICIAL_HARDWARE_JOINTS[:12]], [51, 52, 53, 54, 55, 56, 61, 62, 63, 64, 65, 66])
        self.assertEqual([item.can_id for item in OFFICIAL_HARDWARE_JOINTS[12:15]], [33, 32, 31])
        self.assertEqual([item.can_id for item in OFFICIAL_HARDWARE_JOINTS[15:]], [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27])

    def test_official_message_shapes_build_one_snapshot(self) -> None:
        buffer, _ = self.state_buffer()
        snapshot = buffer.snapshot(now_monotonic=20.01)
        self.assertEqual(snapshot.joint_position.shape, (29,))
        self.assertEqual(snapshot.joint_velocity.shape, (29,))
        self.assertEqual(snapshot.source, "tiangong_ros2")
        self.assertFalse(snapshot.is_valid("base_linear_velocity_world"))
        self.assertFalse(snapshot.is_valid("foot_contact"))
        self.assertFalse(snapshot.is_valid("imu_transform_confirmed"))
        left_pitch_index = snapshot.index["hip_pitch_l_joint"]
        self.assertAlmostEqual(snapshot.joint_position[left_pitch_index], 0.0)
        left_roll_index = snapshot.index["hip_roll_l_joint"]
        self.assertAlmostEqual(snapshot.joint_position[left_roll_index], 0.01)

    def test_missing_duplicate_and_failed_ankle_data_are_rejected(self) -> None:
        messages = self.messages()
        duplicate = FakeMotorMessage(messages["waist"].status + [messages["waist"].status[0]])
        buffer, _ = self.state_buffer()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            buffer.ingest_motor("waist", duplicate)
        messages["leg"].status.pop()
        with self.assertRaisesRegex(ValueError, "missing"):
            buffer.ingest_motor("leg", messages["leg"])
        failed, _ = self.state_buffer(conversion_success=False)
        with self.assertRaises(AnkleConversionError):
            failed.snapshot()

    def test_command_encoder_is_complete_but_sink_is_disabled(self) -> None:
        buffer, ankle = self.state_buffer()
        state = buffer.snapshot(now_monotonic=20.01)
        _, walkamp, _, _ = build_runtimes(self.real_config)
        target = walkamp.neutral_target()
        frame = TiangongCommandEncoder(self.mapper, ankle).encode(target, state, 10.0)
        self.assertEqual(frame.command_count, 29)
        self.assertEqual([len(frame.groups[name]) for name in ("leg", "waist", "arm")], [12, 3, 14])
        self.assertEqual({command.name for values in frame.groups.values() for command in values}, {item.can_id for item in OFFICIAL_HARDWARE_JOINTS})
        sink = DisabledMotorCommandSink()
        with self.assertRaises(PermissionError):
            sink.publish(frame)
        self.assertEqual(sink.publish_count, 0)

    def test_ros_state_source_contains_no_publisher_creation(self) -> None:
        source = inspect.getsource(TiangongRos2StateSource)
        self.assertNotIn("create_publisher", source)
        self.assertIn("create_subscription", source)

    def test_safety_is_fail_closed_for_stale_and_incomplete_contract(self) -> None:
        buffer, _ = self.state_buffer()
        state = buffer.snapshot(now_monotonic=20.0)
        supervisor = HardwareSafetySupervisor.from_files(
            self.real_config, self.real_config["hardware_contract"]
        )
        _, walkamp, _, _ = build_runtimes(self.real_config)
        decision = supervisor.evaluate(state, walkamp.neutral_target(), 20.2)
        self.assertIn("state_stale", decision.blockers)
        self.assertFalse(decision.motor_output_allowed)
        self.assertFalse(decision.transition_evaluation_allowed)
        self.assertTrue(supervisor.contract_blockers())
        with self.assertRaises(PermissionError):
            supervisor.request_motor_enable()

    def test_safety_rejects_sensor_and_candidate_limit_violations(self) -> None:
        buffer, _ = self.state_buffer()
        state = buffer.snapshot(now_monotonic=20.0)
        supervisor = HardwareSafetySupervisor.from_files(
            self.real_config, self.real_config["hardware_contract"]
        )
        _, walkamp, _, _ = build_runtimes(self.real_config)

        unsafe_state = replace(
            state,
            joint_velocity=np.full(29, 21.0),
            joint_torque=np.full(29, 331.0),
            joint_temperature=np.full(29, 76.0),
            motor_error=np.concatenate(([1.0], np.zeros(28))),
            source_timestamps={"leg": 10.0, "waist": 10.0, "arm": 10.0, "imu": 10.04},
        )
        decision = supervisor.evaluate(unsafe_state, walkamp.neutral_target(), 20.0)
        self.assertTrue(
            {
                "source_timestamp_skew",
                "motor_error",
                "temperature_limit",
                "measured_torque_limit",
                "joint_speed_limit",
            }.issubset(decision.blockers)
        )

        unsafe_target = walkamp.neutral_target()
        unsafe_target.q[0] = supervisor.joint_upper[0] + 0.1
        unsafe_target.feedforward[1] = supervisor.maximum_feedforward + 1.0
        decision = supervisor.evaluate(state, unsafe_target, 20.0)
        self.assertIn("target_joint_limit", decision.blockers)
        self.assertIn("feedforward_limit", decision.blockers)
        self.assertFalse(decision.motor_output_allowed)


class PolicyRuntimeConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import mujoco

        cls.walk_config = load_walkamp_config(ROOT / "configs" / "walkamp_official.json")
        cls.multi_config = load_multi_evt2_config(ROOT / "configs" / "multi_evt2.json")
        cls.model = mujoco.MjModel.from_xml_path(cls.walk_config["model"])
        cls.joint_map = build_joint_map(cls.model)
        cls.layout = RuntimeJointLayout.from_joint_map(cls.joint_map)

    def test_walkamp_original_and_decoupled_runtime_match(self) -> None:
        import mujoco

        data = mujoco.MjData(self.model)
        original = WalkAmpPolicy(
            self.model,
            data,
            self.joint_map,
            self.walk_config["policy"],
            self.walk_config["walkamp"],
            self.walk_config["simulation"]["control_dt"],
        )
        setup_initial_pose(self.model, data, self.joint_map, original, 1.0, 0.015)
        snapshot = RobotStateSnapshot.from_mujoco(data, self.joint_map, 0.0)
        runtime = WalkAmpRuntime(
            self.walk_config["policy"], self.walk_config["walkamp"], self.layout
        )
        original_target, original_observation, original_action = original.step()
        runtime_target, runtime_observation, runtime_action = runtime.step(snapshot, np.zeros(3))
        np.testing.assert_allclose(runtime_observation, original_observation, atol=1.0e-6, rtol=0.0)
        np.testing.assert_allclose(runtime_action, original_action, atol=1.0e-6, rtol=0.0)
        np.testing.assert_allclose(runtime_target.q, original_target.q, atol=1.0e-8, rtol=0.0)

    def test_motion_original_and_decoupled_runtime_match(self) -> None:
        import mujoco

        for key in ("a", "b"):
            with self.subTest(key=key):
                data = mujoco.MjData(self.model)
                walk = WalkAmpPolicy(
                    self.model,
                    data,
                    self.joint_map,
                    self.walk_config["policy"],
                    self.walk_config["walkamp"],
                    self.walk_config["simulation"]["control_dt"],
                )
                config = dict(self.multi_config["motion_control"])
                config.update(self.multi_config["motions"][key])
                original = MotionEvt2Policy(
                    self.model,
                    data,
                    self.joint_map,
                    config["path"],
                    config,
                    walk.neutral_target(),
                )
                original.reset_episode_to_reference()
                ensure_episode_ground_clearance(self.model, data, 0.015)
                snapshot = RobotStateSnapshot.from_mujoco(data, self.joint_map, 0.0)
                original.begin_from_live_state()
                runtime = MotionRuntime(
                    config["path"],
                    config,
                    self.layout,
                    walk.neutral_target(),
                    gain_override=TRAINING_NOMINAL_GAINS,
                )
                runtime.begin_from_live_state(snapshot)
                original_target, original_observation, original_action = original.step(0)
                runtime_target, runtime_observation, runtime_action = runtime.step(snapshot, 0)
                np.testing.assert_allclose(runtime_observation, original_observation, atol=1.0e-5, rtol=0.0)
                np.testing.assert_allclose(runtime_action, original_action, atol=1.0e-5, rtol=0.0)
                np.testing.assert_allclose(runtime_target.q, original_target.q, atol=1.0e-7, rtol=0.0)
                np.testing.assert_allclose(runtime_target.feedforward, original_target.feedforward, atol=1.0e-6, rtol=0.0)

    def test_real_config_and_all_onnx_contracts_load(self) -> None:
        config = load_real_config(ROOT / "configs" / "real_evt2.json")
        layout, walkamp, motions, controller = build_runtimes(config)
        self.assertEqual(len(layout.names), 29)
        self.assertEqual(walkamp.history.shape, (840,))
        self.assertEqual(set(motions), {"a", "b"})
        self.assertEqual({motion.action_size for motion in motions.values()}, {19})
        self.assertEqual(controller.state.value, "WALKAMP_IDLE")

    def test_all_policies_infer_from_official_ros_message_shape(self) -> None:
        config = load_real_config(ROOT / "configs" / "real_evt2.json")
        _, walkamp, motions, _ = build_runtimes(config)
        mapper = build_mapper(config)
        ankle = SptlibAnkleAdapter([15.0] * 4, [1.25] * 4, FakeSpt())
        buffer = OfficialRosStateBuffer(
            mapper,
            ImuAdapter(config["imu"]),
            ankle,
            ankle_state_space="parallel",
        )
        grouped = {"leg": [], "waist": [], "arm": []}
        for item in OFFICIAL_HARDWARE_JOINTS:
            grouped[item.group].append(FakeStatus(item.can_id, 0.0, 0.0, 0.0))
        for group, statuses in grouped.items():
            buffer.ingest_motor(group, FakeMotorMessage(statuses), received_monotonic=20.0)
        buffer.ingest_imu(FakeImu(), received_monotonic=20.0)
        state = buffer.snapshot(now_monotonic=20.0)

        walk_target, walk_observation, walk_action = walkamp.step(state, np.zeros(3))
        self.assertEqual(walk_observation.shape, (840,))
        self.assertEqual(walk_action.shape, (23,))
        walk_target.validate(29)

        for key, motion in motions.items():
            with self.subTest(key=key):
                motion.begin_from_live_state(state)
                target, observation, action = motion.step(state, 0)
                self.assertEqual(observation.shape, (1, 104))
                self.assertEqual(action.shape, (19,))
                self.assertTrue(np.all(np.isfinite(observation)))
                self.assertTrue(np.all(np.isfinite(action)))
                target.validate(29)

    def test_live_format_without_estimators_cannot_enter_motion(self) -> None:
        config = load_real_config(ROOT / "configs" / "real_evt2.json")
        _, _, _, controller = build_runtimes(config)
        joint_names = tuple(config["joint_layout"]["names"])
        state = RobotStateSnapshot(
            timestamp=0.0,
            received_monotonic=0.0,
            source="ros_format_test",
            joint_names=joint_names,
            joint_position=np.zeros(29),
            joint_velocity=np.zeros(29),
            orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity_body=np.zeros(3),
            joint_torque=np.zeros(29),
            joint_temperature=np.full(29, 30.0),
            motor_error=np.zeros(29),
            validity={
                "joint_position": True,
                "joint_velocity": True,
                "orientation_wxyz": True,
                "angular_velocity_body": True,
                "base_linear_velocity_world": False,
                "foot_contact": False,
                "foot_normal_force": False,
            },
        )
        self.assertTrue(controller.request_motion("a"))
        events = []
        for _ in range(510):
            result = controller.step(state)
            if result.event:
                events.append(result.event)
        self.assertEqual(result.state.value, "WALKAMP_IDLE")
        self.assertIn("ready_failed", events)
        self.assertIn(
            {
                "key": "a",
                "status": "FAILED",
                "reason": "ready_timeout",
                "readiness_blockers": [
                    "unavailable:base_linear_velocity_world",
                    "unavailable:foot_contact",
                    "unavailable:foot_normal_force",
                ],
            },
            controller.command_events,
        )


if __name__ == "__main__":
    unittest.main()
