from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class TensorContract:
    name: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class OnnxContract:
    name: str
    inputs: tuple[TensorContract, ...]
    outputs: tuple[TensorContract, ...]
    required_metadata: tuple[str, ...] = ()


WALKAMP_CONTRACT = OnnxContract(
    name="walkamp",
    inputs=(TensorContract("*", (1, 840)),),
    outputs=(TensorContract("*", (1, 23)),),
)

MOTION_CONTRACT = OnnxContract(
    name="beyond_mimic_19dof",
    inputs=(TensorContract("obs", (1, 104)), TensorContract("time_step", (1, 1))),
    outputs=(
        TensorContract("actions", (1, 19)),
        TensorContract("joint_pos", (1, 19)),
        TensorContract("joint_vel", (1, 19)),
        TensorContract("body_pos_w", (1, 24, 3)),
        TensorContract("body_quat_w", (1, 24, 4)),
        TensorContract("body_lin_vel_w", (1, 24, 3)),
        TensorContract("body_ang_vel_w", (1, 24, 3)),
    ),
    required_metadata=(
        "joint_names",
        "joint_stiffness",
        "joint_damping",
        "default_joint_pos",
        "action_scale",
        "observation_names",
        "command_names",
    ),
)


def create_cpu_session(path: str | Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_mem_pattern = False
    options.enable_mem_reuse = True
    return ort.InferenceSession(
        str(Path(path).expanduser().resolve()),
        options,
        providers=["CPUExecutionProvider"],
    )


def validate_session(session, contract: OnnxContract) -> Mapping[str, str]:
    actual_inputs = {item.name: tuple(item.shape) for item in session.get_inputs()}
    actual_outputs = {item.name: tuple(item.shape) for item in session.get_outputs()}

    def check(actual: Mapping[str, tuple[int, ...]], expected: Sequence[TensorContract], kind: str) -> None:
        if len(actual) != len(expected):
            raise ValueError(f"{contract.name} {kind} count is {len(actual)}, expected {len(expected)}")
        for item in expected:
            if item.name == "*":
                if item.shape not in actual.values():
                    raise ValueError(f"{contract.name} has no {kind} tensor with shape {item.shape}: {actual}")
            elif actual.get(item.name) != item.shape:
                raise ValueError(
                    f"{contract.name} {kind} {item.name!r} is {actual.get(item.name)}, expected {item.shape}"
                )

    check(actual_inputs, contract.inputs, "input")
    check(actual_outputs, contract.outputs, "output")
    metadata = session.get_modelmeta().custom_metadata_map
    missing = [name for name in contract.required_metadata if name not in metadata]
    if missing:
        raise ValueError(f"{contract.name} metadata is missing {missing}")
    providers = session.get_providers()
    if not providers or providers[0] != "CPUExecutionProvider":
        raise ValueError(f"{contract.name} must use CPUExecutionProvider first, got {providers}")
    return metadata
