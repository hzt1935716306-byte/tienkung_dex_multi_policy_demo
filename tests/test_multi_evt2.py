from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tienkung_demo.motion_evt2 import MotionEvt2Policy
from tienkung_demo.command_bus import CommandReader, CommandWriter, StatusReader, StatusWriter
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
    continuous_group_transition_target,
    continuous_transition_target,
    endpoint_progress,
    joint_group_indices,
    prealign_group_target,
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
        self.assertEqual(endpoint_progress(0, 40), 0.0)
        self.assertEqual(endpoint_progress(39, 40), 1.0)

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

    def test_grouped_prealign_only_moves_configured_joint_groups(self) -> None:
        _, walkamp, _, _ = self.build_stack()
        balance = walkamp.neutral_target("balance")
        motion = walkamp.neutral_target("motion")
        motion.q += 0.25
        motion.kp += 25.0
        motion.kd += 2.5
        motion.feedforward += 3.0
        target = prealign_group_target(
            balance,
            motion,
            {"legs": 0.0, "waist": 0.0, "arms": 1.0},
            self.joint_map,
        )
        groups = joint_group_indices(self.joint_map)
        np.testing.assert_allclose(target.q[groups["legs"]], balance.q[groups["legs"]])
        np.testing.assert_allclose(target.q[groups["waist"]], balance.q[groups["waist"]])
        np.testing.assert_allclose(target.q[groups["arms"]], motion.q[groups["arms"]])
        np.testing.assert_array_equal(target.kp, balance.kp)
        np.testing.assert_array_equal(target.kd, balance.kd)
        np.testing.assert_array_equal(target.feedforward, balance.feedforward)

    def test_continuous_entry_has_exact_complete_target_endpoints(self) -> None:
        _, walkamp, _, _ = self.build_stack()
        anchor = walkamp.neutral_target("anchor")
        balance_start = walkamp.neutral_target("balance_start")
        balance = walkamp.neutral_target("balance")
        motion = walkamp.neutral_target("motion")
        for index, field in enumerate(("q", "kp", "kd", "feedforward", "effort"), 1):
            getattr(anchor, field)[:] += 0.1 * index
            getattr(balance, field)[:] -= 0.2 * index
            getattr(motion, field)[:] += 0.3 * index
        anchor.torque_scale = 0.7
        balance_start.torque_scale = 0.8
        balance.torque_scale = 0.9
        motion.torque_scale = 1.0

        start = continuous_group_transition_target(
            anchor,
            balance_start,
            balance,
            motion,
            {"legs": 0.0, "waist": 0.0, "arms": 0.0},
            self.joint_map,
            anchored_groups={"legs", "waist", "arms"},
        )
        finish = continuous_group_transition_target(
            anchor,
            balance_start,
            balance,
            motion,
            {"legs": 1.0, "waist": 1.0, "arms": 1.0},
            self.joint_map,
            anchored_groups={"legs", "waist", "arms"},
        )
        for field in ("q", "kp", "kd", "feedforward", "effort"):
            np.testing.assert_allclose(getattr(start, field), getattr(anchor, field))
            np.testing.assert_allclose(getattr(finish, field), getattr(motion, field))
        self.assertAlmostEqual(start.torque_scale, anchor.torque_scale)
        self.assertAlmostEqual(finish.torque_scale, motion.torque_scale)

    def test_continuous_exit_has_exact_complete_target_endpoints(self) -> None:
        _, walkamp, _, _ = self.build_stack()
        anchor = walkamp.neutral_target("anchor")
        source_start = walkamp.neutral_target("source_start")
        live_source = walkamp.neutral_target("live_source")
        destination = walkamp.neutral_target("destination")
        for index, field in enumerate(
            ("q", "kp", "kd", "feedforward", "effort"),
            1,
        ):
            getattr(anchor, field)[:] += 0.1 * index
            getattr(source_start, field)[:] -= 0.2 * index
            getattr(live_source, field)[:] += 0.3 * index
            getattr(destination, field)[:] -= 0.4 * index
        anchor.torque_scale = 0.7
        source_start.torque_scale = 0.8
        live_source.torque_scale = 0.9
        destination.torque_scale = 1.0

        start = continuous_transition_target(
            anchor,
            source_start,
            live_source,
            destination,
            0.0,
            "start",
        )
        finish = continuous_transition_target(
            anchor,
            source_start,
            live_source,
            destination,
            1.0,
            "finish",
        )
        midpoint = continuous_transition_target(
            anchor,
            source_start,
            live_source,
            destination,
            0.5,
            "midpoint",
        )
        frozen_midpoint = continuous_transition_target(
            anchor,
            source_start,
            source_start,
            destination,
            0.5,
            "frozen_midpoint",
        )
        for field in ("q", "kp", "kd", "feedforward", "effort"):
            np.testing.assert_allclose(getattr(start, field), getattr(anchor, field))
            np.testing.assert_allclose(
                getattr(finish, field),
                getattr(destination, field),
            )
        self.assertAlmostEqual(start.torque_scale, anchor.torque_scale)
        self.assertAlmostEqual(finish.torque_scale, destination.torque_scale)
        self.assertGreater(
            float(np.max(np.abs(midpoint.q - frozen_midpoint.q))),
            0.0,
        )

    def test_default_entry_mode_is_continuous_full_body(self) -> None:
        self.assertEqual(
            self.config["controller"]["entry"]["mode"],
            "full_body_continuous",
        )

    def test_default_exit_modes_are_action_specific(self) -> None:
        exit_config = self.config["controller"]["exit"]
        self.assertEqual(exit_config["mode"], "continuous")
        self.assertEqual(exit_config["handoff_target"], "neutral")
        self.assertEqual(
            self.config["motions"]["a"]["walkamp_reentry_history_mode"],
            "measured",
        )
        self.assertEqual(
            self.config["motions"]["b"]["walkamp_reentry_history_mode"],
            "repeated",
        )

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
        self.assertGreater(limiter.last_requested_maximum_delta, 9.0)
        self.assertAlmostEqual(limiter.last_applied_maximum_delta, 0.03)

    def test_motion_specific_exit_settings(self) -> None:
        self.assertTrue(self.config["motions"]["a"]["exit_window"]["enabled"])
        self.assertAlmostEqual(
            self.config["motions"]["a"]["exit_window"]["seconds_before_end"],
            0.8,
        )
        self.assertAlmostEqual(
            self.config["motions"]["a"]["walkamp_reentry_phase_time"],
            0.31875,
        )
        self.assertTrue(self.config["motions"]["b"]["exit_window"]["enabled"])
        self.assertAlmostEqual(
            self.config["motions"]["b"]["exit_window"]["seconds_before_end"],
            0.8,
        )
        self.assertEqual(
            self.config["motions"]["b"]["walkamp_reentry_phase_time"],
            0.31875,
        )
        self.assertEqual(self.config["motions"]["a"]["transition_out_seconds"], 0.6)
        self.assertEqual(self.config["motions"]["b"]["transition_out_seconds"], 0.6)

    def test_handoff_and_recovery_have_distinct_limits(self) -> None:
        import mujoco

        data, walkamp, _, controller = self.build_stack()
        contacts = {
            "left_contact_count": 1.0,
            "right_contact_count": 1.0,
            "left_normal_force": 100.0,
            "right_normal_force": 100.0,
        }
        neutral = walkamp.neutral_target("handoff_test")
        handoff, handoff_reasons, _ = controller.handoff_feasible(contacts, neutral)
        recovered, recovery_reasons, _ = controller.recovery_complete(contacts)
        self.assertTrue(handoff, handoff_reasons)
        self.assertTrue(recovered, recovery_reasons)

        pitch = 0.15
        data.qpos[3:7] = np.array(
            [np.cos(pitch / 2.0), 0.0, np.sin(pitch / 2.0), 0.0]
        )
        mujoco.mj_forward(self.model, data)
        handoff, handoff_reasons, _ = controller.handoff_feasible(contacts, neutral)
        recovered, recovery_reasons, _ = controller.recovery_complete(contacts)
        self.assertFalse(handoff)
        self.assertIn("pitch", handoff_reasons)
        self.assertTrue(recovered, recovery_reasons)

    def test_terminal_wait_is_bounded_and_enters_failed_state(self) -> None:
        _, _, _, controller = self.build_stack()
        record = controller.request_command("bow", "test")
        controller._begin_transition_out("test_terminal_wait")
        no_support = {
            "left_contact_count": 0.0,
            "right_contact_count": 0.0,
            "left_normal_force": 0.0,
            "right_normal_force": 0.0,
        }
        timeout_steps = controller._duration_steps(
            self.config["controller"]["handoff"]["terminal_wait_timeout_seconds"]
        )
        for _ in range(timeout_steps + 2):
            controller.step(no_support)
            if controller.state == ControllerState.FAILED:
                break
        self.assertEqual(controller.state, ControllerState.FAILED)
        self.assertEqual(record.status, CommandStatus.FAILED.value)
        self.assertIn("handoff_timeout", controller.terminal_failure_reason)
        self.assertTrue(
            any(event["event"] == "terminal_failure" for event in controller.event_history)
        )

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

    def test_walkamp_preview_does_not_mutate_policy_or_simulation(self) -> None:
        data, walkamp, _, _ = self.build_stack()
        walkamp.record_measured_state(walkamp.neutral_target("executed"))
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        command = walkamp.command.copy()
        last_actions = walkamp.last_actions.copy()
        actions = walkamp.actions.copy()
        history = walkamp.history.copy()
        timer = walkamp.timer_gait
        first_observation = walkamp.first_observation
        last_observation = walkamp.last_observation.copy()
        last_target = walkamp.last_target
        measured_frames = [
            (
                frame.angular_velocity.copy(),
                frame.projected_gravity.copy(),
                frame.joint_position.copy(),
                frame.joint_velocity.copy(),
                frame.executed_action.copy(),
            )
            for frame in walkamp.measured_frames
        ]

        preview = walkamp.preview_from_live_state(
            phase_time=0.2125,
            history_mode="measured",
            previous_target=walkamp.neutral_target("previous"),
        )

        self.assertEqual(preview.history.shape, (840,))
        self.assertEqual(preview.action.shape, (23,))
        self.assertTrue(np.all(np.isfinite(preview.history)))
        self.assertTrue(np.all(np.isfinite(preview.action)))
        np.testing.assert_array_equal(data.qpos, qpos)
        np.testing.assert_array_equal(data.qvel, qvel)
        np.testing.assert_array_equal(walkamp.command, command)
        np.testing.assert_array_equal(walkamp.last_actions, last_actions)
        np.testing.assert_array_equal(walkamp.actions, actions)
        np.testing.assert_array_equal(walkamp.history, history)
        self.assertEqual(walkamp.timer_gait, timer)
        self.assertEqual(walkamp.first_observation, first_observation)
        np.testing.assert_array_equal(walkamp.last_observation, last_observation)
        self.assertIs(walkamp.last_target, last_target)
        self.assertEqual(len(walkamp.measured_frames), len(measured_frames))
        for frame, expected in zip(walkamp.measured_frames, measured_frames):
            for actual, saved in zip(
                (
                    frame.angular_velocity,
                    frame.projected_gravity,
                    frame.joint_position,
                    frame.joint_velocity,
                    frame.executed_action,
                ),
                expected,
            ):
                np.testing.assert_array_equal(actual, saved)

    def test_measured_history_uses_real_recorded_states(self) -> None:
        import mujoco

        data, walkamp, _, _ = self.build_stack()
        first_target = walkamp.neutral_target("first_executed")
        walkamp.record_measured_state(first_target)
        joint = int(self.joint_map.qpos_adr[walkamp.policy_indices[0]])
        velocity = int(self.joint_map.qvel_adr[walkamp.policy_indices[0]])
        data.qpos[joint] += 0.03
        data.qvel[velocity] = 0.2
        mujoco.mj_forward(self.model, data)
        second_target = walkamp.neutral_target("second_executed")
        second_target.q[walkamp.policy_indices[0]] += 0.04
        walkamp.record_measured_state(second_target)
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        result = walkamp.begin_from_live_state(
            phase_time=0.2125,
            history_mode="measured",
            previous_target=second_target,
        )
        frames = walkamp.history.reshape(10, 84)

        self.assertEqual(result["history_mode"], "measured")
        self.assertEqual(result["measured_frames"], 2.0)
        self.assertTrue(np.all(np.isfinite(walkamp.history)))
        self.assertFalse(np.array_equal(frames[-2], frames[-1]))
        self.assertAlmostEqual(float(frames[-1][9]), 0.03, places=6)
        np.testing.assert_array_equal(data.qpos, qpos)
        np.testing.assert_array_equal(data.qvel, qvel)

    def test_command_lifecycle_and_busy_rejection(self) -> None:
        _, _, _, controller = self.build_stack()
        self.assertEqual(controller.state, ControllerState.WALKAMP_IDLE)

        unsupported = controller.request_command("dance", "test")
        self.assertEqual(unsupported.status, CommandStatus.REJECTED.value)
        self.assertEqual(unsupported.reason, "unsupported_action")
        self.assertEqual(controller.state, ControllerState.WALKAMP_IDLE)

        accepted = controller.request_command("bow", "test")
        self.assertEqual(accepted.status, CommandStatus.WAITING_FOR_READY.value)
        self.assertEqual(controller.state, ControllerState.READY_CHECK)
        accepted_history = [
            item["status"]
            for item in controller.command_history
            if item["command_id"] == accepted.command_id
        ]
        self.assertEqual(
            accepted_history,
            [
                CommandStatus.RECEIVED.value,
                CommandStatus.ACCEPTED.value,
                CommandStatus.WAITING_FOR_READY.value,
            ],
        )

        busy = controller.request_command("wave", "test")
        self.assertEqual(busy.status, CommandStatus.REJECTED.value)
        self.assertIn("controller_busy", busy.reason)

    def test_moving_action_brakes_with_independent_rate_limit(self) -> None:
        _, walkamp, _, controller = self.build_stack()
        controller.set_walk_command((0.2, -0.1, 0.12), "test")
        self.assertEqual(controller.state, ControllerState.WALKAMP_MOVING)

        record = controller.request_command("bow", "test", "moving-request")
        self.assertEqual(record.status, CommandStatus.BRAKING.value)
        self.assertEqual(controller.state, ControllerState.BRAKE)
        before = walkamp.command.copy()
        controller._step_brake()
        after = walkamp.command.copy()
        expected_delta = np.array([0.4, 0.35, 0.6]) * controller.control_dt
        np.testing.assert_allclose(
            np.abs(before) - np.abs(after),
            expected_delta,
            atol=1.0e-6,
        )

    def test_zero_command_with_live_motion_still_enters_brake(self) -> None:
        data, _, _, controller = self.build_stack()
        data.qvel[0] = 0.1
        record = controller.request_command("wave", "test")
        self.assertEqual(record.status, CommandStatus.BRAKING.value)
        self.assertEqual(controller.state, ControllerState.BRAKE)

    def test_brake_cancel_keeps_walkamp_feedback_and_marks_cancelled(self) -> None:
        _, walkamp, _, controller = self.build_stack()
        controller.set_walk_command((0.2, 0.0, 0.0), "test")
        action = controller.request_command("bow", "test")
        abort = controller.request_command("r", "test")

        self.assertEqual(action.status, CommandStatus.CANCELLED.value)
        self.assertEqual(abort.status, CommandStatus.COMPLETED.value)
        self.assertEqual(controller.state, ControllerState.WALKAMP_IDLE)
        np.testing.assert_array_equal(walkamp.command, np.zeros(3))

    def test_external_request_id_is_idempotent(self) -> None:
        _, _, _, controller = self.build_stack()
        first = controller.request_command("bow", "voice", "same-request")
        duplicate = controller.request_command("bow", "voice", "same-request")
        self.assertIs(first, duplicate)
        self.assertEqual(len(controller.command_records), 1)

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


class CommandBusTests(unittest.TestCase):
    def test_command_envelope_expiry_and_status_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command_path = Path(directory) / "commands.jsonl"
            status_path = Path(directory) / "status.jsonl"
            reader = CommandReader(command_path)
            writer = CommandWriter(command_path, source="test")
            request_id = writer.send("bow", "test bow", ttl_seconds=10.0)
            records = reader.read_records()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].request_id, request_id)
            self.assertEqual(records[0].command, "bow")
            self.assertFalse(records[0].expired)

            status_reader = StatusReader(status_path)
            status_writer = StatusWriter(status_path)
            status_writer.send({"request_id": request_id, "status": "COMPLETED"})
            terminal = status_reader.wait_for_terminal(request_id, timeout=0.2)
            self.assertIsNotNone(terminal)
            self.assertEqual(terminal["status"], "COMPLETED")

            expired_id = writer.send("wave", ttl_seconds=0.0)
            time.sleep(0.001)
            expired = reader.read_records()
            self.assertEqual(expired[0].request_id, expired_id)
            self.assertTrue(expired[0].expired)


if __name__ == "__main__":
    unittest.main()
