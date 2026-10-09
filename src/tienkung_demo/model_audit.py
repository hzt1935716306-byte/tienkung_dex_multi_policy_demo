from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _object_names(model, object_type, count: int) -> list[str]:
    import mujoco

    return [mujoco.mj_id2name(model, object_type, index) or f"#{index}" for index in range(count)]


def _difference(field: str, name: str, reference: Any, candidate: Any) -> dict[str, Any]:
    def serializable(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    return {
        "field": field,
        "name": name,
        "reference": serializable(reference),
        "candidate": serializable(candidate),
    }


def _arrays_equal(first: np.ndarray, second: np.ndarray, tolerance: float) -> bool:
    return first.shape == second.shape and bool(np.allclose(first, second, atol=tolerance, rtol=0.0))


def compare_models(reference, candidate, tolerance: float = 1.0e-9) -> dict[str, Any]:
    import mujoco

    report: dict[str, Any] = {
        "equivalent": True,
        "tolerance": tolerance,
        "counts": {
            "reference": {
                "nq": reference.nq,
                "nv": reference.nv,
                "nu": reference.nu,
                "nbody": reference.nbody,
                "njnt": reference.njnt,
                "ngeom": reference.ngeom,
                "nmesh": reference.nmesh,
                "nsensor": reference.nsensor,
            },
            "candidate": {
                "nq": candidate.nq,
                "nv": candidate.nv,
                "nu": candidate.nu,
                "nbody": candidate.nbody,
                "njnt": candidate.njnt,
                "ngeom": candidate.ngeom,
                "nmesh": candidate.nmesh,
                "nsensor": candidate.nsensor,
            },
        },
        "categories": {},
    }

    count_differences = []
    for name, value in report["counts"]["reference"].items():
        candidate_value = report["counts"]["candidate"][name]
        if value != candidate_value:
            count_differences.append(_difference(name, "model", value, candidate_value))
    report["categories"]["counts"] = count_differences

    reference_joint_names = _object_names(reference, mujoco.mjtObj.mjOBJ_JOINT, reference.njnt)
    candidate_joint_names = _object_names(candidate, mujoco.mjtObj.mjOBJ_JOINT, candidate.njnt)
    reference_joint_ids = {name: index for index, name in enumerate(reference_joint_names)}
    candidate_joint_ids = {name: index for index, name in enumerate(candidate_joint_names)}
    joint_differences = []
    for name in sorted(reference_joint_ids.keys() | candidate_joint_ids.keys()):
        if name not in reference_joint_ids:
            joint_differences.append(_difference("joint_presence", name, False, True))
            continue
        if name not in candidate_joint_ids:
            joint_differences.append(_difference("joint_presence", name, True, False))
            continue
        reference_id = reference_joint_ids[name]
        candidate_id = candidate_joint_ids[name]
        scalar_fields = {
            "type": (reference.jnt_type[reference_id], candidate.jnt_type[candidate_id]),
            "qpos_order": (reference.jnt_qposadr[reference_id], candidate.jnt_qposadr[candidate_id]),
            "dof_order": (reference.jnt_dofadr[reference_id], candidate.jnt_dofadr[candidate_id]),
        }
        reference_body = mujoco.mj_id2name(
            reference, mujoco.mjtObj.mjOBJ_BODY, reference.jnt_bodyid[reference_id]
        )
        candidate_body = mujoco.mj_id2name(
            candidate, mujoco.mjtObj.mjOBJ_BODY, candidate.jnt_bodyid[candidate_id]
        )
        scalar_fields["body"] = (reference_body, candidate_body)
        for field, (reference_value, candidate_value) in scalar_fields.items():
            if reference_value != candidate_value:
                joint_differences.append(
                    _difference(field, name, reference_value, candidate_value)
                )
        for field, reference_value, candidate_value in (
            ("axis", reference.jnt_axis[reference_id], candidate.jnt_axis[candidate_id]),
            ("range", reference.jnt_range[reference_id], candidate.jnt_range[candidate_id]),
            (
                "actuator_force_range",
                reference.jnt_actfrcrange[reference_id],
                candidate.jnt_actfrcrange[candidate_id],
            ),
        ):
            if not _arrays_equal(reference_value, candidate_value, tolerance):
                joint_differences.append(_difference(field, name, reference_value, candidate_value))
    report["categories"]["joint_axes_ranges_and_order"] = joint_differences

    reference_body_names = _object_names(reference, mujoco.mjtObj.mjOBJ_BODY, reference.nbody)
    candidate_body_names = _object_names(candidate, mujoco.mjtObj.mjOBJ_BODY, candidate.nbody)
    reference_body_ids = {name: index for index, name in enumerate(reference_body_names)}
    candidate_body_ids = {name: index for index, name in enumerate(candidate_body_names)}
    inertia_differences = []
    for name in sorted(reference_body_ids.keys() | candidate_body_ids.keys()):
        if name not in reference_body_ids:
            inertia_differences.append(_difference("body_presence", name, False, True))
            continue
        if name not in candidate_body_ids:
            inertia_differences.append(_difference("body_presence", name, True, False))
            continue
        reference_id = reference_body_ids[name]
        candidate_id = candidate_body_ids[name]
        for field, reference_value, candidate_value in (
            ("mass", reference.body_mass[reference_id], candidate.body_mass[candidate_id]),
            ("inertia", reference.body_inertia[reference_id], candidate.body_inertia[candidate_id]),
            ("inertial_position", reference.body_ipos[reference_id], candidate.body_ipos[candidate_id]),
            ("inertial_quaternion", reference.body_iquat[reference_id], candidate.body_iquat[candidate_id]),
            ("body_position", reference.body_pos[reference_id], candidate.body_pos[candidate_id]),
            ("body_quaternion", reference.body_quat[reference_id], candidate.body_quat[candidate_id]),
        ):
            reference_array = np.atleast_1d(reference_value)
            candidate_array = np.atleast_1d(candidate_value)
            if not _arrays_equal(reference_array, candidate_array, tolerance):
                inertia_differences.append(_difference(field, name, reference_value, candidate_value))
    report["categories"]["body_poses_and_inertias"] = inertia_differences

    def collision_signatures(model) -> dict[str, list[dict[str, Any]]]:
        signatures: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for geom_id in range(model.ngeom):
            if model.geom_bodyid[geom_id] == 0:
                continue
            if model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0:
                continue
            body_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id]
            ) or f"body#{model.geom_bodyid[geom_id]}"
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "<unnamed>"
            signatures[body_name].append(
                {
                    "name": geom_name,
                    "type": int(model.geom_type[geom_id]),
                    "size": model.geom_size[geom_id].round(12).tolist(),
                    "position": model.geom_pos[geom_id].round(12).tolist(),
                    "quaternion": model.geom_quat[geom_id].round(12).tolist(),
                    "friction": model.geom_friction[geom_id].round(12).tolist(),
                    "contype": int(model.geom_contype[geom_id]),
                    "conaffinity": int(model.geom_conaffinity[geom_id]),
                    "condim": int(model.geom_condim[geom_id]),
                }
            )
        return dict(signatures)

    reference_collisions = collision_signatures(reference)
    candidate_collisions = collision_signatures(candidate)
    collision_differences = []
    for body_name in sorted(reference_collisions.keys() | candidate_collisions.keys()):
        reference_value = reference_collisions.get(body_name, [])
        candidate_value = candidate_collisions.get(body_name, [])
        if reference_value != candidate_value:
            collision_differences.append(
                _difference("collision_geometries", body_name, reference_value, candidate_value)
            )
    report["categories"]["collision_geometries"] = collision_differences

    reference_actuator_names = _object_names(
        reference, mujoco.mjtObj.mjOBJ_ACTUATOR, reference.nu
    )
    candidate_actuator_names = _object_names(
        candidate, mujoco.mjtObj.mjOBJ_ACTUATOR, candidate.nu
    )
    reference_actuator_ids = {name: index for index, name in enumerate(reference_actuator_names)}
    candidate_actuator_ids = {name: index for index, name in enumerate(candidate_actuator_names)}
    actuator_differences = []
    for name in sorted(reference_actuator_ids.keys() | candidate_actuator_ids.keys()):
        if name not in reference_actuator_ids:
            actuator_differences.append(_difference("actuator_presence", name, False, True))
            continue
        if name not in candidate_actuator_ids:
            actuator_differences.append(_difference("actuator_presence", name, True, False))
            continue
        reference_id = reference_actuator_ids[name]
        candidate_id = candidate_actuator_ids[name]
        reference_joint = mujoco.mj_id2name(
            reference, mujoco.mjtObj.mjOBJ_JOINT, reference.actuator_trnid[reference_id, 0]
        )
        candidate_joint = mujoco.mj_id2name(
            candidate, mujoco.mjtObj.mjOBJ_JOINT, candidate.actuator_trnid[candidate_id, 0]
        )
        if reference_joint != candidate_joint:
            actuator_differences.append(
                _difference("controlled_joint", name, reference_joint, candidate_joint)
            )
        for field, reference_value, candidate_value in (
            ("control_range", reference.actuator_ctrlrange[reference_id], candidate.actuator_ctrlrange[candidate_id]),
            ("force_range", reference.actuator_forcerange[reference_id], candidate.actuator_forcerange[candidate_id]),
            ("gear", reference.actuator_gear[reference_id], candidate.actuator_gear[candidate_id]),
            ("gain_parameters", reference.actuator_gainprm[reference_id], candidate.actuator_gainprm[candidate_id]),
            ("bias_parameters", reference.actuator_biasprm[reference_id], candidate.actuator_biasprm[candidate_id]),
        ):
            if not _arrays_equal(reference_value, candidate_value, tolerance):
                actuator_differences.append(_difference(field, name, reference_value, candidate_value))
    report["categories"]["control_interfaces"] = actuator_differences

    report["mismatch_counts"] = {
        category: len(differences) for category, differences in report["categories"].items()
    }
    report["equivalent"] = not any(report["mismatch_counts"].values())
    return report
