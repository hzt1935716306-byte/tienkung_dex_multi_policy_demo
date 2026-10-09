from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tienkung_demo.model_audit import compare_models
from tienkung_demo.robot import build_joint_map
from tienkung_demo.walkamp import (
    WALKAMP_JOINT_NAMES,
    WALKAMP_UNCONTROLLED_JOINT_NAMES,
    WalkAmpPolicy,
    gait_phase,
    load_walkamp_config,
)
from tienkung_demo.walkamp_runner import setup_initial_pose


class WalkAmpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import mujoco

        cls.config = load_walkamp_config(ROOT / "configs" / "walkamp_official.json")
        cls.model = mujoco.MjModel.from_xml_path(cls.config["model"])

    def test_vendored_files_match_official_hashes(self) -> None:
        expected = {}
        checksum_path = ROOT / "third_party" / "OPEN_X_SHA256SUMS"
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            digest, relative_path = line.split(maxsplit=1)
            expected[ROOT / relative_path] = digest
        for path, digest in expected.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_official_model_shape(self) -> None:
        self.assertEqual((self.model.nq, self.model.nv, self.model.nu), (36, 35, 58))
        joint_map = build_joint_map(self.model)
        self.assertEqual(len(joint_map.names), 29)
        self.assertEqual(set(WALKAMP_JOINT_NAMES) | set(WALKAMP_UNCONTROLLED_JOINT_NAMES), set(joint_map.names))

    def test_gait_phase_matches_official_formula(self) -> None:
        phase = gait_phase(0.0, 0.85, 0.38, 0.88, 0.38, 0.38)
        expected = np.array(
            [
                np.sin(2.0 * np.pi * 0.38),
                np.sin(2.0 * np.pi * 0.88),
                np.cos(2.0 * np.pi * 0.38),
                np.cos(2.0 * np.pi * 0.88),
                0.38,
                0.38,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(phase, expected, atol=1.0e-7, rtol=0.0)

    def test_history_action_and_uncontrolled_hold(self) -> None:
        import mujoco

        data = mujoco.MjData(self.model)
        joint_map = build_joint_map(self.model)
        policy = WalkAmpPolicy(
            self.model,
            data,
            joint_map,
            self.config["policy"],
            self.config["walkamp"],
            self.config["simulation"]["control_dt"],
        )
        setup_initial_pose(self.model, data, joint_map, policy, 1.0, 0.015)
        target, history, action = policy.step()
        self.assertEqual(history.shape, (840,))
        self.assertEqual(action.shape, (23,))
        frames = history.reshape(10, 84)
        for frame in frames[1:]:
            np.testing.assert_array_equal(frame, frames[0])
        first_history = history.copy()
        _, second_history, _ = policy.step()
        np.testing.assert_array_equal(second_history[:-84], first_history[84:])
        self.assertEqual(
            [joint_map.names[index] for index in policy.policy_indices],
            WALKAMP_JOINT_NAMES,
        )
        hold = self.config["walkamp"]["uncontrolled_joints"]
        indices = np.array([joint_map.index[name] for name in hold["joint_names"]])
        np.testing.assert_allclose(target.q[indices], hold["positions"])
        np.testing.assert_allclose(target.kp[indices], hold["kp"])
        np.testing.assert_allclose(target.kd[indices], hold["kd"])
        self.assertTrue(np.all(target.effort[indices] > 0.0))

    def test_local_full_model_is_not_official_evt2_equivalent(self) -> None:
        import mujoco

        self.assertTrue(compare_models(self.model, self.model)["equivalent"])
        candidate = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "mjcf" / "dex_evt_full.xml"))
        report = compare_models(self.model, candidate)
        self.assertFalse(report["equivalent"])
        self.assertGreater(len(report["categories"]["body_poses_and_inertias"]), 0)
        self.assertGreater(len(report["categories"]["collision_geometries"]), 0)
        self.assertEqual(len(report["categories"]["control_interfaces"]), 0)


if __name__ == "__main__":
    unittest.main()
