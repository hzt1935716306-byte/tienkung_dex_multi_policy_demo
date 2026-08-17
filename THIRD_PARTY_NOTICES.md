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
