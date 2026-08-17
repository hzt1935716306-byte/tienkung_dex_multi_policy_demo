#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tienkung_demo.config import load_config


def check_config(path: Path) -> None:
    import mujoco
    import onnxruntime as ort

    config = load_config(path)
    model = mujoco.MjModel.from_xml_path(config["model"])
    print(f"[OK] {path.name}: MuJoCo model nq={model.nq}, nv={model.nv}, nu={model.nu}")

    if config.get("walk_policy"):
        session = ort.InferenceSession(config["walk_policy"], providers=["CPUExecutionProvider"])
        if session.get_inputs()[0].shape != [1, 750] or session.get_outputs()[0].shape != [1, 20]:
            raise RuntimeError("walk.onnx must have interface [1, 750] -> [1, 20]")
        print(f"[OK] walk.onnx: input={session.get_inputs()[0].shape}, output={session.get_outputs()[0].shape}")
    for key, motion in config.get("motions", {}).items():
        session = ort.InferenceSession(motion["path"], providers=["CPUExecutionProvider"])
        inputs = {item.name: item.shape for item in session.get_inputs()}
        if inputs != {"obs": [1, 104], "time_step": [1, 1]}:
            raise RuntimeError(f"motion {key} has an unsupported ONNX interface: {inputs}")
        print(f"[OK] key {key}: {Path(motion['path']).name}, inputs={inputs}")

    expected_actuators = 19 if config.get("motions") else 20
    if model.nu != expected_actuators:
        raise RuntimeError(f"{path.name} has {model.nu} actuators; expected {expected_actuators}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the standalone TienKung demo installation")
    parser.add_argument("--smoke", action="store_true", help="Run short headless simulations after static checks")
    args = parser.parse_args()

    import mujoco
    import numpy
    import onnxruntime

    print(f"[OK] numpy={numpy.__version__}, mujoco={mujoco.__version__}, onnxruntime={onnxruntime.__version__}")
    check_config(ROOT / "configs" / "demo.json")
    check_config(ROOT / "configs" / "walk_only.json")

    if args.smoke:
        from tienkung_demo.simulator import run

        print("[TEST] 19-DOF combined walk/motion baseline (50 control steps)")
        run(["--config", str(ROOT / "configs" / "demo.json"), "--no-viewer", "--max-steps", "50", "--no-debug"])
        print("[TEST] 20-DOF walk baseline (50 control steps)")
        run(["--config", str(ROOT / "configs" / "walk_only.json"), "--no-viewer", "--max-steps", "50", "--no-debug"])
    print("[OK] setup check completed")


if __name__ == "__main__":
    main()
