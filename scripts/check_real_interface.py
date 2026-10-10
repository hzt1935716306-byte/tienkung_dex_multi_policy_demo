#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tienkung_demo.hardware.ankle_adapter import AnkleConversionError, SptlibAnkleAdapter
from tienkung_demo.hardware.state_estimator import ImuAdapter
from tienkung_demo.hardware.tiangong_ros2 import (
    OFFICIAL_COMMAND_TOPICS,
    OFFICIAL_MESSAGE_TYPES,
    OFFICIAL_STATE_TOPICS,
    OfficialRosStateBuffer,
    TiangongRos2StateSource,
)
from tienkung_demo.real_runner import build_mapper, build_runtimes, load_real_config
from tienkung_demo.hardware.safety_supervisor import HardwareSafetySupervisor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the read-only official TienKung ROS2 interface")
    parser.add_argument("--config", default="configs/real_evt2.json")
    parser.add_argument("--live", action="store_true", help="Subscribe until one complete state is received")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--output", default="artifacts/real_interface_check.json")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def ros_graph() -> dict[str, str]:
    try:
        result = subprocess.run(
            ["ros2", "topic", "list", "-t"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {"error": str(exc)}
    return {"output": result.stdout}


def main() -> int:
    args = parse_args()
    config = load_real_config(args.config)
    layout, _, _, _ = build_runtimes(config)
    mapper = build_mapper(config)
    supervisor = HardwareSafetySupervisor.from_files(config, config["hardware_contract"])
    report = {
        "mode": "shadow",
        "motor_publication_enabled": False,
        "motor_publishers_created": 0,
        "joint_count": len(layout.names),
        "joint_order_matches_hardware_mapper": tuple(layout.names) == mapper.runtime_names,
        "official_state_topics": OFFICIAL_STATE_TOPICS,
        "official_command_topics_reference_only": OFFICIAL_COMMAND_TOPICS,
        "official_message_types": OFFICIAL_MESSAGE_TYPES,
        "dependencies": {
            name: module_available(name)
            for name in ("onnxruntime", "rclpy", "bodyctrl_msgs", "sptlib_python")
        },
        "hardware_contract_blockers": supervisor.contract_blockers(),
        "live": None,
    }
    if args.live:
        ankle_config = config["ankle"]
        try:
            ankle = SptlibAnkleAdapter(ankle_config["parallel_kp"], ankle_config["parallel_kd"])
            buffer = OfficialRosStateBuffer(
                mapper,
                ImuAdapter(config["imu"]),
                ankle,
                ankle_state_space=ankle_config["state_space"],
            )
            source = TiangongRos2StateSource(buffer, config["ros2"]["node_name"] + "_check")
            source.start()
            deadline = time.monotonic() + args.timeout
            snapshot = None
            try:
                while time.monotonic() < deadline:
                    source.spin_once(0.02)
                    try:
                        snapshot = buffer.snapshot()
                        break
                    except RuntimeError:
                        pass
            finally:
                source.close()
            if snapshot is None:
                report["live"] = {"ok": False, "error": "timed out waiting for all four state topics"}
            else:
                report["live"] = {
                    "ok": True,
                    "source_timestamps": dict(snapshot.source_timestamps),
                    "validity": dict(snapshot.validity),
                    "maximum_temperature": float(snapshot.joint_temperature.max()),
                    "motor_errors": int((snapshot.motor_error != 0.0).sum()),
                }
        except (AnkleConversionError, RuntimeError, ValueError) as exc:
            report["live"] = {"ok": False, "error": str(exc), "ros_graph": ros_graph()}

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.strict:
        static_ok = report["joint_order_matches_hardware_mapper"] and all(report["dependencies"].values())
        live_ok = not args.live or bool(report["live"] and report["live"].get("ok"))
        return 0 if static_ok and live_ok and not report["hardware_contract_blockers"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
