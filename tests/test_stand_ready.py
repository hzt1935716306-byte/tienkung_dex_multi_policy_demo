from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tienkung_demo.motion_evt2 import MotionEvt2Policy
from tienkung_demo.multi_evt2_controller import ControllerState, MultiEvt2Controller
from tienkung_demo.multi_evt2_runner import load_multi_evt2_config
from tienkung_demo.ready_controller import build_ready_controller
from tienkung_demo.robot import build_joint_map
from tienkung_demo.stand_diagnostics import (
    STAND_DIAGNOSTIC_COLUMNS,
    CompleteStepDetector,
    StandDiagnostics,
)
from tienkung_demo.walkamp import WalkAmpPolicy, load_walkamp_config
from tienkung_demo.walkamp_runner import setup_initial_pose


class CompleteStepDetectorTests(unittest.TestCase):
    def test_contact_edge_without_foot_displacement_is_not_a_step(self) -> None:
        detector = CompleteStepDetector(0.01)
        left = np.array([0.0, 0.1, 0.0])
        right = np.array([0.0, -0.1, 0.0])
        both = {"left_contact_count": 1.0, "right_contact_count": 1.0}
        left_air = {"left_contact_count": 0.0, "right_contact_count": 1.0}
        detector.update(0, both, left, right)
        for step in range(1, 8):
            detector.update(step, left_air, left, right)
        for step in range(8, 13):
            detector.update(step, both, left, right)
        self.assertGreaterEqual(detector.contact_edge_count, 2)
        self.assertEqual(len(detector.events), 0)

    def test_lift_displace_and_stable_landing_is_one_complete_step(self) -> None:
        detector = CompleteStepDetector(0.01)
        left = np.array([0.0, 0.1, 0.0])
        right = np.array([0.0, -0.1, 0.0])
        both = {"left_contact_count": 1.0, "right_contact_count": 1.0}
        left_air = {"left_contact_count": 0.0, "right_contact_count": 1.0}
        detector.update(0, both, left, right)
        for step in range(1, 8):
            moved = left + np.array([0.04 * step / 7.0, 0.0, 0.03])
            detector.update(step, left_air, moved, right)
        landed = left + np.array([0.04, 0.0, 0.0])
        for step in range(8, 13):
            detector.update(step, both, landed, right)
        self.assertEqual(len(detector.events), 1)
        self.assertEqual(detector.events[0].foot, "left")
        self.assertGreaterEqual(
            detector.events[0].horizontal_displacement,
            0.025,
        )


class StandReadyIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import mujoco

        cls.config = load_multi_evt2_config(ROOT / "configs" / "multi_evt2.json")
        cls.walkamp_config = load_walkamp_config(cls.config["walkamp_config"])
        cls.model = mujoco.MjModel.from_xml_path(cls.config["model"])
        cls.joint_map = build_joint_map(cls.model)

    def build_stack(self, ready_enabled: bool = False):
        import mujoco

        data = mujoco.MjData(self.model)
        walkamp = WalkAmpPolicy(
            self.model,
            data,
            self.joint_map,
            self.walkamp_config["policy"],
            self.walkamp_config["walkamp"],
            self.config["simulation"]["control_dt"],
        )
        setup_initial_pose(
            self.model,
            data,
            self.joint_map,
            walkamp,
            self.config["simulation"]["initial_height"],
            self.config["simulation"]["ground_clearance"],
        )
        hold = walkamp.neutral_target("test_hold")
        motions = {}
        for key, values in self.config["motions"].items():
            motion_config = dict(self.config["motion_control"])
            motion_config.update(values)
            motion_config["control_mode"] = "policy"
            motions[key] = MotionEvt2Policy(
                self.model,
                data,
                self.joint_map,
                motion_config["path"],
                motion_config,
                hold,
            )
        controller_config = copy.deepcopy(self.config["controller"])
        controller_config["ready"]["enabled"] = ready_enabled
        controller = MultiEvt2Controller(
            data,
            self.joint_map,
            walkamp,
            motions,
            controller_config,
            self.config["simulation"]["control_dt"],
        )
        return data, walkamp, controller

    def test_default_ready_interface_is_disabled(self) -> None:
        self.assertFalse(self.config["controller"]["ready"]["enabled"])
        _, _, controller = self.build_stack()
        controller.request_command("bow", "test")
        self.assertEqual(controller.state, ControllerState.READY_CHECK)
        self.assertFalse(controller.ready_enabled)

    def test_enabled_ready_uses_walkamp_feedback_and_full_target(self) -> None:
        data, _, controller = self.build_stack(ready_enabled=True)
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        controller.request_command("bow", "test")
        self.assertEqual(controller.state, ControllerState.READY)
        self.assertEqual(controller.ready_phase, "before_motion")
        output = controller.ready_controller.step()
        self.assertEqual(output.target.q.shape, (29,))
        self.assertTrue(np.all(np.isfinite(output.target.q)))
        np.testing.assert_array_equal(data.qpos, qpos)
        np.testing.assert_array_equal(data.qvel, qvel)

    def test_unverified_ready_controller_is_rejected(self) -> None:
        _, walkamp, _ = self.build_stack()
        with self.assertRaisesRegex(ValueError, "not verified"):
            build_ready_controller({"controller": "fixed_pose_pd"}, walkamp)

    def test_stand_diagnostics_outputs_finite_evt2_metrics(self) -> None:
        import mujoco

        data, _, _ = self.build_stack()
        mujoco.mj_forward(self.model, data)
        diagnostics = StandDiagnostics(self.model, 0.01)
        values = diagnostics.update(
            0,
            data,
            {
                "left_contact_count": 1.0,
                "right_contact_count": 1.0,
            },
            data.qvel[self.joint_map.qvel_adr],
            np.zeros(29, dtype=np.float64),
        )
        self.assertEqual(set(values), set(STAND_DIAGNOSTIC_COLUMNS))
        self.assertTrue(np.all(np.isfinite(list(values.values()))))
        self.assertEqual(diagnostics.summary()["complete_step_count"], 0)


if __name__ == "__main__":
    unittest.main()
