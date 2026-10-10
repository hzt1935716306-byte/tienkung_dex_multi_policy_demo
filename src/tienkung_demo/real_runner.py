from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any, Iterable

import numpy as np

from .command_bus import CommandReader, StatusWriter
from .hardware.ankle_adapter import AnkleConversionError, SptlibAnkleAdapter
from .hardware.joint_mapper import HardwareJointMapper
from .hardware.safety_supervisor import HardwareSafetySupervisor
from .hardware.state_estimator import ImuAdapter
from .hardware.tiangong_ros2 import OfficialRosStateBuffer, TiangongRos2StateSource
from .motion_evt2 import TRAINING_NOMINAL_GAINS
from .multi_evt2_runner import load_multi_evt2_config
from .runtime.control_target import copy_control_target
from .runtime.policy_runtime import MotionRuntime, RuntimeJointLayout, WalkAmpRuntime
from .runtime.robot_state import RobotStateSnapshot
from .runtime.shadow_controller import ShadowMultiPolicyController, ShadowStep
from .walkamp import load_walkamp_config


def _resolve(value: str, directory: Path) -> str:
    path = Path(value).expanduser()
    return str((directory / path).resolve() if not path.is_absolute() else path.resolve())


def load_real_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if config.get("schema") != "tienkung.real_runtime.v1":
        raise ValueError("Unsupported real runtime config schema")
    if config.get("mode") != "shadow" or not config.get("shadow_only", False):
        raise ValueError("run_real.py only accepts mode=shadow and shadow_only=true")
    if config.get("publish_motor_commands", False):
        raise ValueError("run_real.py refuses publish_motor_commands=true")
    config["_path"] = str(config_path)
    directory = config_path.parent
    for name in ("hardware_contract", "walkamp_config", "multi_policy_config"):
        config[name] = _resolve(config[name], directory)
    for name, value in config["policies"].items():
        config["policies"][name] = _resolve(value, directory)
    root = directory.parent
    for name, value in config["logging"].items():
        config["logging"][name] = _resolve(value, root)
    return config


def build_mapper(config: dict[str, Any]) -> HardwareJointMapper:
    values = config["hardware_mapping"]
    return HardwareJointMapper(
        zero_position=values["zero_position_official_order"],
        direction=values["direction_official_order"],
        semantic_offset=values["semantic_offset_official_order"],
        current_to_torque=values["current_to_torque_official_order"],
    )


def build_runtimes(config: dict[str, Any]):
    layout_values = config["joint_layout"]
    layout = RuntimeJointLayout(
        tuple(layout_values["names"]),
        np.asarray(layout_values["ranges"], dtype=np.float64),
        np.asarray(layout_values["efforts"], dtype=np.float64),
    )
    walk_config = load_walkamp_config(config["walkamp_config"])
    walkamp = WalkAmpRuntime(config["policies"]["walkamp"], walk_config["walkamp"], layout)
    multi_config = load_multi_evt2_config(config["multi_policy_config"])
    motion_common = dict(multi_config.get("motion_control", {}))
    motions = {}
    for key, policy_name in (("a", "bow"), ("b", "wave")):
        motion_config = dict(motion_common)
        motion_config.update(multi_config["motions"][key])
        motions[key] = MotionRuntime(
            config["policies"][policy_name],
            motion_config,
            layout,
            walkamp.neutral_target(),
            gain_override=TRAINING_NOMINAL_GAINS,
        )
    controller = ShadowMultiPolicyController(
        walkamp,
        motions,
        config["state_machine"],
        float(config["control_dt"]),
    )
    return layout, walkamp, motions, controller


def read_snapshots(path: str | Path) -> Iterable[RobotStateSnapshot]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                yield RobotStateSnapshot.from_json_dict(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Invalid snapshot at line {line_number}: {exc}") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only TienKung DEX policy Shadow Mode")
    parser.add_argument("--config", default="configs/real_evt2.json")
    parser.add_argument("--offline-observations", default="")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--walk-command", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--action", choices=("a", "b"), default=None)
    parser.add_argument("--command-file", default="")
    parser.add_argument("--status-file", default="")
    parser.add_argument("--record-states", default="")
    parser.add_argument("--runtime-log", default="")
    parser.add_argument("--summary", default="")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    config = load_real_config(args.config)
    layout, walkamp, motions, controller = build_runtimes(config)
    supervisor = HardwareSafetySupervisor.from_files(config, config["hardware_contract"])
    mapper = build_mapper(config)
    if tuple(layout.names) != mapper.runtime_names:
        raise ValueError("Policy runtime and hardware joint orders differ")
    if args.check_only:
        result = {
            "mode": "shadow",
            "motor_publishers_created": 0,
            "onnx": {"walkamp": [840, 23], "bow": [104, 19], "wave": [104, 19]},
            "hardware_contract_blockers": supervisor.contract_blockers(),
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    controller.set_walk_command(args.walk_command)
    command_reader = CommandReader(args.command_file or config.get("command_file"))
    status_writer = StatusWriter(args.status_file or config.get("status_file"))
    source = None
    ankle_adapter = None
    if args.offline_observations:
        snapshots = iter(read_snapshots(args.offline_observations))
    else:
        ankle_config = config["ankle"]
        try:
            ankle_adapter = SptlibAnkleAdapter(
                ankle_config["parallel_kp"], ankle_config["parallel_kd"]
            )
        except AnkleConversionError as exc:
            raise RuntimeError(f"Live Shadow Mode blocked: {exc}") from exc
        state_buffer = OfficialRosStateBuffer(
            mapper,
            ImuAdapter(config["imu"]),
            ankle_adapter,
            ankle_state_space=ankle_config["state_space"],
        )
        source = TiangongRos2StateSource(state_buffer, config["ros2"]["node_name"])
        source.start()
        snapshots = None

    record_path = Path(args.record_states or config["logging"]["state_record"])
    runtime_path = Path(args.runtime_log or config["logging"]["runtime_log"])
    summary_path = Path(args.summary or config["logging"]["summary"])
    for path in (record_path, runtime_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    inference_ms: list[float] = []
    loop_ms: list[float] = []
    state_age_ms: list[float] = []
    source_skew_ms: list[float] = []
    period_jitter_ms: list[float] = []
    stale_count = 0
    blocked_count = 0
    events: list[str] = []
    executed = 0
    action_requested = False
    active_request_id: str | None = None
    previous_loop_start: float | None = None
    period = float(config["control_dt"])
    next_deadline = time.perf_counter()

    print("[SHADOW] motor command publication is disabled; zero ROS command publishers are created")
    print(f"[SHADOW] config={config['_path']} joints=29 control_dt={period:.3f}s")
    try:
        with record_path.open("w", encoding="utf-8") as state_file, runtime_path.open("w", encoding="utf-8") as log_file:
            while executed < args.max_steps:
                loop_start = time.perf_counter()
                if previous_loop_start is not None:
                    period_jitter_ms.append(abs(loop_start - previous_loop_start - period) * 1000.0)
                previous_loop_start = loop_start
                if source is not None:
                    source.spin_once(period)
                    try:
                        state = source.buffer.snapshot()
                    except RuntimeError:
                        continue
                    safety_now = time.monotonic()
                else:
                    assert snapshots is not None
                    try:
                        state = next(snapshots)
                    except StopIteration:
                        break
                    safety_now = state.received_monotonic
                if args.action and not action_requested:
                    controller.request_motion(args.action)
                    action_requested = True
                for record in command_reader.read_records():
                    if record.expired:
                        status_writer.send({"request_id": record.request_id, "status": "REJECTED", "reason": "expired"})
                        continue
                    aliases = {"a": "a", "bow": "a", "b": "b", "wave": "b"}
                    if record.command == "r":
                        accepted = controller.request_recovery()
                        status_writer.send({"request_id": record.request_id, "status": "CANCELLED" if accepted else "REJECTED", "reason": "shadow_recovery_request"})
                        continue
                    key = aliases.get(record.command)
                    accepted = key is not None and controller.request_motion(key)
                    if accepted:
                        active_request_id = record.request_id
                    status_writer.send({
                        "request_id": record.request_id,
                        "status": "ACCEPTED" if accepted else "REJECTED",
                        "command": record.command,
                        "reason": "" if accepted else "unsupported_or_busy",
                    })

                state_decision = supervisor.evaluate(state, None, safety_now)
                if state_decision.shadow_inference_allowed:
                    inference_start = time.perf_counter()
                    result = controller.step(state)
                    inference_elapsed = (time.perf_counter() - inference_start) * 1000.0
                    decision = supervisor.evaluate(state, result.target, safety_now)
                else:
                    # Invalid input must not advance policy history, gait phase, motion
                    # time, or the command state machine, even in read-only Shadow Mode.
                    result = ShadowStep(
                        target=copy_control_target(
                            controller.last_target, "shadow_state_rejected"
                        ),
                        state=controller.state,
                        observation=None,
                        action=None,
                        event=None,
                        readiness_blockers=(),
                    )
                    inference_elapsed = 0.0
                    decision = state_decision
                age_ms = state.age(safety_now) * 1000.0
                timestamp_values = list(state.source_timestamps.values())
                skew_ms = 0.0 if not timestamp_values else (max(timestamp_values) - min(timestamp_values)) * 1000.0
                state_age_ms.append(age_ms)
                source_skew_ms.append(skew_ms)
                if "state_stale" in decision.blockers:
                    stale_count += 1
                if not decision.shadow_inference_allowed:
                    blocked_count += 1
                if result.event:
                    events.append(result.event)
                    if active_request_id is not None and result.event == "motion_started":
                        status_writer.send({"request_id": active_request_id, "status": "EXECUTING"})
                    elif active_request_id is not None and result.event == "recovery_complete":
                        status_writer.send({"request_id": active_request_id, "status": "COMPLETED"})
                        active_request_id = None
                    elif active_request_id is not None and result.event == "ready_failed":
                        status_writer.send(
                            {
                                "request_id": active_request_id,
                                "status": "FAILED",
                                "reason": "ready_timeout",
                                "readiness_blockers": list(result.readiness_blockers),
                            }
                        )
                        active_request_id = None
                state_file.write(json.dumps(state.to_json_dict(), ensure_ascii=False) + "\n")
                log_file.write(
                    json.dumps(
                        {
                            "step": executed,
                            "state": result.state.value,
                            "event": result.event,
                            "target_label": result.target.label,
                            "target_min": float(np.min(result.target.q)),
                            "target_max": float(np.max(result.target.q)),
                            "observation_shape": None if result.observation is None else list(result.observation.shape),
                            "action_shape": None if result.action is None else list(result.action.shape),
                            "inference_ms": inference_elapsed,
                            "state_age_ms": age_ms,
                            "source_timestamp_skew_ms": skew_ms,
                            "readiness_blockers": list(result.readiness_blockers),
                            "safety_blockers": list(decision.blockers),
                            "safety_warnings": list(decision.warnings),
                            "motor_output_allowed": decision.motor_output_allowed,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                inference_ms.append(inference_elapsed)
                executed += 1
                loop_ms.append((time.perf_counter() - loop_start) * 1000.0)
                if source is not None and not args.no_realtime:
                    next_deadline += period
                    delay = next_deadline - time.perf_counter()
                    if delay > 0.0:
                        time.sleep(delay)
    finally:
        if source is not None:
            source.close()

    def percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return float(ordered[min(len(ordered) - 1, int(fraction * len(ordered)))])

    summary = {
        "mode": "shadow",
        "steps": executed,
        "motor_publishers_created": 0,
        "motor_commands_published": 0,
        "final_controller_state": controller.state.value,
        "events": events,
        "command_events": controller.command_events,
        "hardware_contract_blockers": supervisor.contract_blockers(),
        "stale_state_count": stale_count,
        "stale_state_rate": 0.0 if executed == 0 else stale_count / executed,
        "blocked_state_count": blocked_count,
        "inference_ms": {
            "mean": statistics.fmean(inference_ms) if inference_ms else 0.0,
            "p95": percentile(inference_ms, 0.95),
            "maximum": max(inference_ms, default=0.0),
        },
        "loop_ms": {
            "mean": statistics.fmean(loop_ms) if loop_ms else 0.0,
            "p95": percentile(loop_ms, 0.95),
            "maximum": max(loop_ms, default=0.0),
        },
        "state_age_ms": {
            "mean": statistics.fmean(state_age_ms) if state_age_ms else 0.0,
            "p95": percentile(state_age_ms, 0.95),
            "maximum": max(state_age_ms, default=0.0),
        },
        "source_timestamp_skew_ms": {
            "p95": percentile(source_skew_ms, 0.95),
            "maximum": max(source_skew_ms, default=0.0),
        },
        "period_jitter_ms": {
            "p95": percentile(period_jitter_ms, 0.95),
            "maximum": max(period_jitter_ms, default=0.0),
        },
        "artifacts": {"states": str(record_path), "runtime": str(runtime_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
