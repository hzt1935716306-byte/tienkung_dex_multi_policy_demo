# Third-Party Notices

This standalone project packages trained policy exports and robot assets that originated in the following local projects.

## TienKung-Lab / Legged Lab

- Used for the TienKung2 Lite walking policy, MuJoCo model, and sim-to-sim observation/control conventions.
- Upstream license: BSD 3-Clause.
- Copyright notices are preserved in this project's `LICENSE`.

## BeyondMimic / Isaac Lab / RSL-RL

- Used to train and export the TienKung Dex bow and right-hand motion policies.
- Relevant framework components are distributed under BSD 3-Clause licenses.
- The ONNX files in `policies/` are exported model artifacts, not source checkpoints.

## xgmr robot assets

- Some compact STL meshes and the diagnostic 29-DOF MJCF were adapted from the local xgmr robot assets.
- Upstream license: MIT.
- Original copyright: Copyright 2025 Yanjieze.

The exact 19-DOF action model in `assets/mjcf/dex_evt_liondance.xml` was generated from the training URDF and then augmented with MuJoCo motors, a ground plane, and lighting. Check the upstream projects before redistributing policy weights or robot CAD files outside your organization.

## Open X-Humanoid Deploy_Tienkung 3.0

- `policies/walkamp_official.onnx` is copied from `policy/walk_amp/model/policy.onnx` at commit `9786cf1a7ed9e3bead1a8de47ed6a5c251cb1869`.
- Unmodified source references are stored in `third_party/deploy_tienkung_walkamp/`.
- The WALKAMP adapter follows the joint order, gains, observation history, gait phase, and timing in the upstream `fsm_walkamp.py` and `walk_amp.yaml`.
- Upstream license: BSD 3-Clause. A copy is stored at `assets/official_evt2/DEPLOY_TIENKUNG_LICENSE.txt`.

## Open X-Humanoid xSIM_MUJOCO

- `assets/official_evt2/` contains the upstream EVT2 MuJoCo XML, URDF files, and meshes from commit `942469c660ec93e6d7ae9e78026db81cfa623591`.
- Upstream license: Apache License 2.0. A copy is stored at `assets/official_evt2/XSIM_MUJOCO_LICENSE`.
- Exact source paths and SHA-256 hashes are recorded in `assets/official_evt2/UPSTREAM_MANIFEST.json`.
