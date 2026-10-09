# BeyondMimic Motion on Official EVT2: Phase 2

This phase runs the existing 19-output BeyondMimic bow and right-hand-open policies independently on the official 29-joint EVT2 MuJoCo model. It preserves the phase-one WALKAMP entry point and the original `configs/demo.json`; it does not implement online switching, voice/LLM changes, retraining, or real-robot deployment.

## Run the policies

Use the viewer:

```bash
python run_motion_evt2.py \
  --config configs/motion_evt2.json \
  --motion a \
  --mode policy

python run_motion_evt2.py \
  --config configs/motion_evt2.json \
  --motion b \
  --mode policy
```

`a` selects `policies/bow.onnx`; `b` selects `policies/right_hand_open.onnx`. Press `Esc` to close the viewer. `--mode policy` performs real 104-dimensional observation feedback inference. `--mode reference` only sends the ONNX reference joint positions through PD and is a diagnostic, not a policy-success test.

Run all reproducible headless diagnostics in the required order, reference first and policy second:

```bash
python scripts/run_motion_evt2_headless_suite.py \
  --config configs/motion_evt2.json
```

Run one case directly:

```bash
python run_motion_evt2.py \
  --config configs/motion_evt2.json \
  --motion a \
  --mode policy \
  --headless \
  --no-realtime
```

CSV logs and JSON summaries are written under `artifacts/`. The logs contain root quaternion/RPY, world and body velocity, all 29 measured joint positions and velocities, targets, applied torques and torque limits, foot contacts and normal forces, tracking error, torque saturation, and target clipping.

## Inference and control contract

Both ONNX files are checked at startup against the following exact contract:

- inputs: `obs [1,104]`, `time_step [1,1]`
- primary output: `actions [1,19]`
- reference outputs: 19 joint positions/velocities and 24-body position, quaternion, linear-velocity and angular-velocity arrays
- metadata observation order: `command,motion_anchor_ori_b,base_ang_vel,joint_pos,joint_vel,actions`
- metadata command: `motion`
- metadata joint order must match the expected 19 names exactly

The 19 policy joints are mapped by name, never by the MuJoCo index alone. The other ten joints are held at the phase-one `WalkAmpPolicy.neutral_target()` positions, gains, damping and effort limits:

```text
shoulder_roll_l_joint, shoulder_yaw_l_joint, elbow_yaw_l_joint,
wrist_pitch_l_joint, wrist_roll_l_joint,
shoulder_roll_r_joint, shoulder_yaw_r_joint, elbow_yaw_r_joint,
wrist_pitch_r_joint, wrist_roll_r_joint
```

The official XML remains unchanged with actuator groups 0, 1 and 3 disabled (`disableactuator=11`). Control is applied only through the existing `motor_*` torque actuators. Joint targets and torques are clipped to model limits.

The reference first frame initializes a standalone simulation episode once. A one-time root-height correction resolves initial collision penetration. Neither operation is called inside the control loop and neither is an acceptable online WALKAMP-to-motion transition mechanism.

## Validation results

Results recorded on 2026-10-09 using the packaged official EVT2 model and the real ONNX files:

| Motion and mode | Result | Steps | Min root z | Max abs roll/pitch | Mean tracking error | Torque saturation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Bow reference | Diagnostic failed: fell | 136/556 | 0.339 m | 1.352 rad | 0.045 rad | 0.068% |
| Hand reference | Diagnostic failed: fell | 156/685 | 0.324 m | 1.356 rad | 0.034 rad | 0.000% |
| Bow policy | Pass | 556/556 | 0.927 m | 0.720 rad | 0.090 rad | 0.000% |
| Hand policy | Pass | 685/685 | 0.977 m | 0.105 rad | 0.068 rad | 0.003% |

The reference failures are genuine: their first-frame poses and open-loop joint trajectories are not dynamically balanced under official EVT2 inertias and contacts. They fell despite low tracking error and no meaningful torque saturation. Holding the first reference frame before playback was also tested and did not resolve the fall. The feedback policy results are therefore reported separately and are the only policy-success results.

The policy runs used no gravity changes, leg locking, periodic state reset, or gain inflation. Bow target clipping occurred for 3.45% of all joint-step targets and hand target clipping for 2.51%. Only the left/right ankle-pitch targets were clipped to official XML limits; the exact per-joint rates are retained in each JSON summary.

## Tests and WALKAMP regression

Run the unit tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Run the unchanged WALKAMP four-scenario suite:

```bash
python scripts/run_walkamp_headless_suite.py \
  --config configs/walkamp_official.json
```

`tests/test_motion_evt2.py` checks the 29-joint map, exact 19+10 partition, both ONNX contracts, WALKAMP-neutral hold values for all uncontrolled joints, target limits, finite observations/actions/physics state, and the episode-start-only reset guard.

## Phase 3 switching interface

Online switching should use three explicit states: `WALKAMP`, `READY`, and `MOTION`. Every state should emit one complete 29-joint `ControlTarget`; a partial 19-joint command should never reach the motor interface.

1. `WALKAMP -> READY`: command zero velocity and wait for bounded base speed, roll/pitch, torque saturation and valid foot support.
2. `READY -> MOTION`: initialize only policy memory, time step and yaw alignment, then blend complete 29-joint targets over a bounded transition. Never rewrite `qpos` or replay the first reference state.
3. `MOTION -> READY`: finish by duration or abort on height, attitude, contact, finite-value or saturation guards.
4. `READY -> WALKAMP`: reset WALKAMP history from live state, ramp commands from zero, and retain the same motor limits.

The current standalone episode initialization is intentionally not exposed as a switching API. Phase 3 should first validate these transitions in headless MuJoCo before any real-robot integration.
