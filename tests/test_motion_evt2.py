from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tienkung_demo.motion_evt2 import (
    MOTION_EVT2_JOINT_NAMES,
    MotionEvt2Policy,
    load_motion_evt2_config,
)
from tienkung_demo.motion_evt2_runner import ensure_episode_ground_clearance
from tienkung_demo.robot import build_joint_map
from tienkung_demo.simulator import apply_pd
from tienkung_demo.walkamp import WalkAmpPolicy, load_walkamp_config


class MotionEvt2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import mujoco

        cls.config = load_motion_evt2_config(ROOT / "configs" / "motion_evt2.json")
        cls.walkamp_config = load_walkamp_config(cls.config["walkamp_config"])
        cls.model = mujoco.MjModel.from_xml_path(cls.config["model"])
        cls.joint_map = build_joint_map(cls.model)

    def build_policy(self, key: str = "a", mode: str = "policy"):
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
        hold_target = walkamp.neutral_target("test_hold")
        motion_config = dict(self.config["motions"][key])
        motion_config["control_mode"] = mode
        motion = MotionEvt2Policy(
            self.model,
            data,
            self.joint_map,
            motion_config["path"],
            motion_config,
            hold_target,
        )
        return data, motion, hold_target

    def test_official_actuator_groups_remain_disabled(self) -> None:
        self.assertEqual(int(self.model.opt.disableactuator), 11)

    def test_19_controlled_plus_10_held_partition(self) -> None:
        _, motion, _ = self.build_policy()
        self.assertEqual(motion.joint_names, MOTION_EVT2_JOINT_NAMES)
        self.assertEqual(len(motion.joint_names), 19)
        self.assertEqual(len(motion.uncontrolled_joint_names), 10)
        self.assertFalse(set(motion.joint_names) & set(motion.uncontrolled_joint_names))
        self.assertEqual(
            set(motion.joint_names) | set(motion.uncontrolled_joint_names),
            set(self.joint_map.names),
        )

    def test_onnx_contracts_for_both_motions(self) -> None:
        for key in ("a", "b"):
            with self.subTest(key=key):
                _, motion, _ = self.build_policy(key)
                self.assertEqual(
                    {item.name: item.shape for item in motion.session.get_inputs()},
                    {"obs": [1, 104], "time_step": [1, 1]},
                )
                self.assertEqual(motion.session.get_outputs()[0].shape, [1, 19])
                self.assertEqual(motion.default_q.shape, (19,))
                self.assertEqual(motion.kp.shape, (19,))
                self.assertEqual(motion.kd.shape, (19,))
                self.assertEqual(motion.action_scale.shape, (19,))

    def test_uncontrolled_joints_preserve_walkamp_hold(self) -> None:
        _, motion, hold_target = self.build_policy()
        target = motion._target_from_positions(motion.default_q)
        indices = motion.uncontrolled_indices
        np.testing.assert_array_equal(target.q[indices], hold_target.q[indices])
        np.testing.assert_array_equal(target.kp[indices], hold_target.kp[indices])
        np.testing.assert_array_equal(target.kd[indices], hold_target.kd[indices])
        np.testing.assert_array_equal(target.effort[indices], hold_target.effort[indices])
        self.assertTrue(np.all(target.effort[indices] > 0.0))

    def test_target_observation_action_and_step_are_finite(self) -> None:
        import mujoco

        for key in ("a", "b"):
            with self.subTest(key=key):
                data, motion, _ = self.build_policy(key)
                motion.reset_episode_to_reference()
                ensure_episode_ground_clearance(self.model, data, 0.015)
                motion.start()
                target, observation, action = motion.step(0)
                self.assertEqual(observation.shape, (1, 104))
                self.assertEqual(action.shape, (19,))
                self.assertTrue(np.all(np.isfinite(observation)))
                self.assertTrue(np.all(np.isfinite(action)))
                self.assertTrue(np.all(np.isfinite(target.q)))
                self.assertTrue(np.all(target.q >= self.joint_map.ranges[:, 0]))
                self.assertTrue(np.all(target.q <= self.joint_map.ranges[:, 1]))
                torque = apply_pd(data, self.joint_map, target)
                mujoco.mj_step(self.model, data)
                self.assertTrue(np.all(np.isfinite(torque)))
                self.assertTrue(np.all(np.isfinite(data.qpos)))
                self.assertTrue(np.all(np.isfinite(data.qvel)))

    def test_reference_reset_is_episode_start_only(self) -> None:
        _, motion, _ = self.build_policy()
        motion.reset_episode_to_reference()
        with self.assertRaisesRegex(RuntimeError, "only permitted once"):
            motion.reset_episode_to_reference()


if __name__ == "__main__":
    unittest.main()
