from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tienkung_demo.motion_evt2 import MotionEvt2Policy
from tienkung_demo.multi_evt2_controller import (
    CommandStatus,
    ControllerState,
    MultiEvt2Controller,
)
from tienkung_demo.multi_evt2_runner import load_multi_evt2_config
from tienkung_demo.robot import build_joint_map
from tienkung_demo.transition_controller import (
    TargetRateLimiter,
    blend_targets,
    quintic_alpha,
)
from tienkung_demo.walkamp import WalkAmpPolicy, load_walkamp_config
from tienkung_demo.walkamp_runner import setup_initial_pose


class MultiEvt2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import mujoco

        cls.config = load_multi_evt2_config(ROOT / "configs" / "multi_evt2.json")
        cls.walkamp_config = load_walkamp_config(cls.config["walkamp_config"])
        cls.model = mujoco.MjModel.from_xml_path(cls.config["model"])
        cls.joint_map = build_joint_map(cls.model)

    def build_stack(self):
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
        hold_target = walkamp.neutral_target("test_hold")
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
                hold_target,
            )
        controller = MultiEvt2Controller(
            data,
            self.joint_map,
            walkamp,
            motions,
            self.config["controller"],
            self.config["simulation"]["control_dt"],
        )
        return data, walkamp, motions, controller

    def test_quintic_has_smooth_endpoints(self) -> None:
        self.assertEqual(quintic_alpha(-1.0), 0.0)
        self.assertEqual(quintic_alpha(0.0), 0.0)
        self.assertEqual(quintic_alpha(1.0), 1.0)
        self.assertEqual(quintic_alpha(2.0), 1.0)
        epsilon = 1.0e-5
        self.assertAlmostEqual(quintic_alpha(epsilon) / epsilon, 0.0, places=7)
        self.assertAlmostEqual((1.0 - quintic_alpha(1.0 - epsilon)) / epsilon, 0.0, places=7)

    def test_blend_operates_on_complete_29_joint_targets(self) -> None:
        _, walkamp, _, _ = self.build_stack()
        first = walkamp.neutral_target("first")
        second = walkamp.neutral_target("second")
        second.q += 0.2
        second.kp *= 0.5
        second.kd *= 0.75
        second.feedforward += 1.0
        second.effort *= 0.8
        second.torque_scale = 0.6

        midpoint = blend_targets(first, second, 0.5, "midpoint")
        self.assertEqual(midpoint.q.shape, (29,))
        np.testing.assert_allclose(midpoint.q, 0.5 * (first.q + second.q))
        np.testing.assert_allclose(midpoint.kp, 0.5 * (first.kp + second.kp))
        np.testing.assert_allclose(midpoint.kd, 0.5 * (first.kd + second.kd))
        np.testing.assert_allclose(midpoint.feedforward, 0.5)
        np.testing.assert_allclose(midpoint.effort, 0.5 * (first.effort + second.effort))
        self.assertAlmostEqual(midpoint.torque_scale, 0.8)

    def test_target_rate_limiter_respects_joint_groups(self) -> None:
        _, walkamp, _, _ = self.build_stack()
        initial = walkamp.neutral_target("initial")
        requested = walkamp.neutral_target("requested")
        requested.q[:] = initial.q + 10.0
        limiter = TargetRateLimiter(
            self.joint_map,
            0.01,
            {"legs": 1.0, "waist": 2.0, "arms": 3.0},
        )
        limiter.reset(initial.q)
        limited = limiter.apply(requested)
        for index, name in enumerate(self.joint_map.names):
            expected = 0.01
            if name.startswith("waist_"):
                expected = 0.02
            elif not any(token in name for token in ("hip_", "knee_", "ankle_")):
                expected = 0.03
            self.assertAlmostEqual(limited.q[index] - initial.q[index], expected)

    def test_live_initialization_never_changes_simulation_state(self) -> None:
        data, walkamp, motions, _ = self.build_stack()
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        first = motions["a"].begin_from_live_state()
        second = motions["a"].begin_from_live_state()
        walkamp.begin_from_live_state((0.0, 0.0, 0.0))

        np.testing.assert_array_equal(data.qpos, qpos)
        np.testing.assert_array_equal(data.qvel, qvel)
        self.assertEqual(first["live_start_count"], 1)
        self.assertEqual(second["live_start_count"], 2)
        self.assertEqual(motions["a"].initialization_mode, "live_state")
        self.assertTrue(np.all(np.isfinite(motions["a"].last_action)))
        self.assertEqual(walkamp.history.shape, (840,))
        history = walkamp.history.reshape(10, 84)
        for frame in history[1:]:
            np.testing.assert_array_equal(frame, history[0])

    def test_command_lifecycle_and_busy_rejection(self) -> None:
        _, _, _, controller = self.build_stack()
        self.assertEqual(controller.state, ControllerState.WALKAMP_IDLE)

        unsupported = controller.request_command("dance", "test")
        self.assertEqual(unsupported.status, CommandStatus.REJECTED.value)
        self.assertEqual(unsupported.reason, "unsupported_action")
        self.assertEqual(controller.state, ControllerState.WALKAMP_IDLE)

        accepted = controller.request_command("bow", "test")
        self.assertEqual(accepted.status, CommandStatus.ACCEPTED.value)
        self.assertEqual(controller.state, ControllerState.READY_CHECK)

        busy = controller.request_command("wave", "test")
        self.assertEqual(busy.status, CommandStatus.REJECTED.value)
        self.assertIn("controller_busy", busy.reason)

    def test_each_cycle_returns_one_finite_29_joint_target(self) -> None:
        _, _, _, controller = self.build_stack()
        contacts = {
            "left_contact_count": 1.0,
            "right_contact_count": 1.0,
            "left_normal_force": 100.0,
            "right_normal_force": 100.0,
        }
        output = controller.step(contacts)
        self.assertEqual(controller.state, ControllerState.WALKAMP_IDLE)
        self.assertEqual(output.target.q.shape, (29,))
        self.assertTrue(np.all(np.isfinite(output.target.q)))
        self.assertTrue(np.all(np.isfinite(output.target.kp)))
        self.assertTrue(np.all(np.isfinite(output.target.kd)))


if __name__ == "__main__":
    unittest.main()
