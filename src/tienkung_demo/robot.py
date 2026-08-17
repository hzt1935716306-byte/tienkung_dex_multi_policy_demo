from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ALL_JOINT_NAMES = [
    "hip_roll_l_joint",
    "hip_pitch_l_joint",
    "hip_yaw_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
    "hip_roll_r_joint",
    "hip_pitch_r_joint",
    "hip_yaw_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "elbow_yaw_l_joint",
    "wrist_pitch_l_joint",
    "wrist_roll_l_joint",
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "elbow_yaw_r_joint",
    "wrist_pitch_r_joint",
    "wrist_roll_r_joint",
]


WALK_MUJOCO_NAMES = [
    "hip_roll_l_joint",
    "hip_pitch_l_joint",
    "hip_yaw_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
    "hip_roll_r_joint",
    "hip_pitch_r_joint",
    "hip_yaw_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
]


WALK_POLICY_NAMES = [
    "hip_roll_l_joint",
    "hip_roll_r_joint",
    "shoulder_pitch_l_joint",
    "shoulder_pitch_r_joint",
    "hip_pitch_l_joint",
    "hip_pitch_r_joint",
    "shoulder_roll_l_joint",
    "shoulder_roll_r_joint",
    "hip_yaw_l_joint",
    "hip_yaw_r_joint",
    "shoulder_yaw_l_joint",
    "shoulder_yaw_r_joint",
    "knee_pitch_l_joint",
    "knee_pitch_r_joint",
    "elbow_pitch_l_joint",
    "elbow_pitch_r_joint",
    "ankle_pitch_l_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_l_joint",
    "ankle_roll_r_joint",
]


MOTION_BODY_NAMES = [
    "pelvis",
    "hip_pitch_l_link",
    "hip_roll_l_link",
    "hip_yaw_l_link",
    "knee_pitch_l_link",
    "ankle_pitch_l_link",
    "ankle_roll_l_link",
    "hip_pitch_r_link",
    "hip_roll_r_link",
    "hip_yaw_r_link",
    "knee_pitch_r_link",
    "ankle_pitch_r_link",
    "ankle_roll_r_link",
    "waist_yaw_link",
    "waist_roll_link",
    "waist_pitch_link",
    "shoulder_pitch_l_link",
    "shoulder_roll_l_link",
    "shoulder_yaw_l_link",
    "elbow_pitch_l_link",
    "shoulder_pitch_r_link",
    "shoulder_roll_r_link",
    "shoulder_yaw_r_link",
    "elbow_pitch_r_link",
]


WALK_DEFAULTS = {
    "hip_roll_l_joint": 0.0,
    "hip_pitch_l_joint": -0.5,
    "hip_yaw_l_joint": 0.0,
    "knee_pitch_l_joint": 1.0,
    "ankle_pitch_l_joint": -0.5,
    "ankle_roll_l_joint": 0.0,
    "hip_roll_r_joint": 0.0,
    "hip_pitch_r_joint": -0.5,
    "hip_yaw_r_joint": 0.0,
    "knee_pitch_r_joint": 1.0,
    "ankle_pitch_r_joint": -0.5,
    "ankle_roll_r_joint": 0.0,
    "shoulder_pitch_l_joint": 0.0,
    "shoulder_roll_l_joint": 0.1,
    "shoulder_yaw_l_joint": 0.0,
    "elbow_pitch_l_joint": -0.3,
    "shoulder_pitch_r_joint": 0.0,
    "shoulder_roll_r_joint": -0.1,
    "shoulder_yaw_r_joint": 0.0,
    "elbow_pitch_r_joint": -0.3,
}


HOLD_KP = {
    "hip_pitch_l_joint": 300.0,
    "hip_roll_l_joint": 300.0,
    "hip_yaw_l_joint": 150.0,
    "knee_pitch_l_joint": 350.0,
    "ankle_pitch_l_joint": 30.0,
    "ankle_roll_l_joint": 16.8,
    "hip_pitch_r_joint": 300.0,
    "hip_roll_r_joint": 300.0,
    "hip_yaw_r_joint": 150.0,
    "knee_pitch_r_joint": 350.0,
    "ankle_pitch_r_joint": 30.0,
    "ankle_roll_r_joint": 16.8,
    "waist_yaw_joint": 400.0,
    "waist_roll_joint": 400.0,
    "waist_pitch_joint": 400.0,
    "shoulder_pitch_l_joint": 150.0,
    "shoulder_roll_l_joint": 80.0,
    "shoulder_yaw_l_joint": 60.0,
    "elbow_pitch_l_joint": 150.0,
    "elbow_yaw_l_joint": 30.0,
    "wrist_pitch_l_joint": 20.0,
    "wrist_roll_l_joint": 20.0,
    "shoulder_pitch_r_joint": 150.0,
    "shoulder_roll_r_joint": 80.0,
    "shoulder_yaw_r_joint": 60.0,
    "elbow_pitch_r_joint": 150.0,
    "elbow_yaw_r_joint": 30.0,
    "wrist_pitch_r_joint": 20.0,
    "wrist_roll_r_joint": 20.0,
}

HOLD_KD = {
    name: (15.0 if "waist" in name else 5.0 if "hip" in name or "knee" in name else 2.0)
    for name in ALL_JOINT_NAMES
}


WALK_KP = {
    "hip_roll_l_joint": 700.0,
    "hip_pitch_l_joint": 700.0,
    "hip_yaw_l_joint": 500.0,
    "knee_pitch_l_joint": 700.0,
    "ankle_pitch_l_joint": 30.0,
    "ankle_roll_l_joint": 16.8,
    "hip_roll_r_joint": 700.0,
    "hip_pitch_r_joint": 700.0,
    "hip_yaw_r_joint": 500.0,
    "knee_pitch_r_joint": 700.0,
    "ankle_pitch_r_joint": 30.0,
    "ankle_roll_r_joint": 16.8,
    "shoulder_pitch_l_joint": 60.0,
    "shoulder_roll_l_joint": 20.0,
    "shoulder_yaw_l_joint": 10.0,
    "elbow_pitch_l_joint": 10.0,
    "shoulder_pitch_r_joint": 60.0,
    "shoulder_roll_r_joint": 20.0,
    "shoulder_yaw_r_joint": 10.0,
    "elbow_pitch_r_joint": 10.0,
}

WALK_KD = {
    name: (10.0 if "hip" in name or "knee" in name else 2.5 if "ankle_pitch" in name else 1.4 if "ankle_roll" in name else 3.0 if "shoulder_pitch" in name else 1.5 if "shoulder_roll" in name else 1.0)
    for name in WALK_MUJOCO_NAMES
}

WALK_EFFORT = {
    "hip_roll_l_joint": 180.0,
    "hip_pitch_l_joint": 300.0,
    "hip_yaw_l_joint": 180.0,
    "knee_pitch_l_joint": 300.0,
    "ankle_pitch_l_joint": 60.0,
    "ankle_roll_l_joint": 30.0,
    "hip_roll_r_joint": 180.0,
    "hip_pitch_r_joint": 300.0,
    "hip_yaw_r_joint": 180.0,
    "knee_pitch_r_joint": 300.0,
    "ankle_pitch_r_joint": 60.0,
    "ankle_roll_r_joint": 30.0,
    "shoulder_pitch_l_joint": 52.5,
    "shoulder_roll_l_joint": 52.5,
    "shoulder_yaw_l_joint": 52.5,
    "elbow_pitch_l_joint": 52.5,
    "shoulder_pitch_r_joint": 52.5,
    "shoulder_roll_r_joint": 52.5,
    "shoulder_yaw_r_joint": 52.5,
    "elbow_pitch_r_joint": 52.5,
}


def values_from_dict(
    values: dict[str, float], names: list[str] | None = None, default: float = 0.0
) -> np.ndarray:
    selected_names = ALL_JOINT_NAMES if names is None else names
    return np.array([values.get(name, default) for name in selected_names], dtype=np.float64)


@dataclass
class JointMap:
    names: list[str]
    qpos_adr: np.ndarray
    qvel_adr: np.ndarray
    actuator_ids: np.ndarray
    ranges: np.ndarray
    efforts: np.ndarray
    index: dict[str, int]


@dataclass
class ControlTarget:
    q: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    effort: np.ndarray
    torque_scale: float
    label: str


def build_joint_map(model) -> JointMap:
    import mujoco

    qpos_adr = []
    qvel_adr = []
    actuator_ids = []
    ranges = []
    efforts = []
    names = []
    for name in ALL_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            continue
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{name}")
        if actuator_id < 0:
            raise ValueError(f"MuJoCo model is missing motor actuator for {name!r}.")
        names.append(name)
        qpos_adr.append(model.jnt_qposadr[joint_id])
        qvel_adr.append(model.jnt_dofadr[joint_id])
        actuator_ids.append(actuator_id)
        ranges.append(model.jnt_range[joint_id].copy())
        force_range = model.actuator_forcerange[actuator_id]
        effort = max(abs(force_range[0]), abs(force_range[1]))
        if effort <= 0.0:
            control_range = model.actuator_ctrlrange[actuator_id]
            effort = max(abs(control_range[0]), abs(control_range[1]))
        if effort <= 0.0:
            joint_force_range = model.jnt_actfrcrange[joint_id]
            effort = max(abs(joint_force_range[0]), abs(joint_force_range[1]))
        if effort <= 0.0:
            raise ValueError(f"No usable effort limit for motor_{name}.")
        efforts.append(effort)
    return JointMap(
        names=names,
        qpos_adr=np.asarray(qpos_adr, dtype=np.int32),
        qvel_adr=np.asarray(qvel_adr, dtype=np.int32),
        actuator_ids=np.asarray(actuator_ids, dtype=np.int32),
        ranges=np.asarray(ranges, dtype=np.float64),
        efforts=np.asarray(efforts, dtype=np.float64),
        index={name: index for index, name in enumerate(names)},
    )


def base_target(joint_map: JointMap, label: str) -> ControlTarget:
    q = values_from_dict(WALK_DEFAULTS, joint_map.names)
    kp = values_from_dict(HOLD_KP, joint_map.names)
    kd = values_from_dict(HOLD_KD, joint_map.names)
    effort = np.minimum(joint_map.efforts, 30.0)
    return ControlTarget(q=q, kp=kp, kd=kd, effort=effort, torque_scale=1.0, label=label)
