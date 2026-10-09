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

CSV logs and JSON summaries are written under `artifacts/`. The logs contain root quaternion/RPY, world and body velocity, all 29 measured joint positions and velocities, targets, applied torques and torque limits, foot contacts and normal forces, tracking error, torque saturation, target clipping, and equivalent feed-forward statistics.

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

The official XML remains unchanged with actuator groups 0, 1 and 3 disabled (`disableactuator=11`). Control is applied only through the existing `motor_*` torque actuators. Joint targets and torques remain clipped to model limits. When a policy position target is outside a joint range, the clipped position error is converted to equivalent feed-forward torque before effort limiting. This preserves IsaacLab's implicit-PD behavior without putting an out-of-range position in the MuJoCo control target.

`configs/motion_evt2.json` selects the deterministic `training_nominal` gain profile. It does not use the randomized environment-0 gains embedded by the IsaacLab ONNX exporter. The nominal values are the gains from the training robot configuration; changing or re-exporting ONNX metadata therefore no longer changes MuJoCo control behavior.

The reference first frame initializes a standalone simulation episode once. A one-time root-height correction resolves initial collision penetration. Neither operation is called inside the control loop and neither is an acceptable online WALKAMP-to-motion transition mechanism.

## Validation results

Results recorded on 2026-10-09 using the packaged official EVT2 model and the real ONNX files:

| Motion and mode | Result | Steps | Min root z | Max abs roll/pitch | Mean tracking error | Torque saturation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Bow reference | Diagnostic failed: fell | 133/556 | 0.337 m | 1.341 rad | 0.045 rad | 0.047% |
| Hand reference | Diagnostic failed: fell | 151/685 | 0.323 m | 1.353 rad | 0.036 rad | 0.000% |
| Bow policy | Pass | 556/556 | 0.944 m | 0.735 rad | 0.069 rad | 0.000% |
| Hand policy | Pass | 685/685 | 0.992 m | 0.071 rad | 0.041 rad | 0.001% |

The reference failures are genuine: their first-frame poses and open-loop joint trajectories are not dynamically balanced under official EVT2 inertias and contacts. They fell despite low tracking error and no meaningful torque saturation. Holding the first reference frame before playback was also tested and did not resolve the fall. The feedback policy results are therefore reported separately and are the only policy-success results.

The policy runs used no gravity changes, leg locking, periodic state reset, or gain inflation. Only the left/right ankle-pitch targets exceed the official XML range. Directly discarding the out-of-range position error caused the policy to lose the PD effort it learned to use in IsaacLab. In a deterministic bow comparison, preserving that effort and selecting nominal training gains reduced left/right foot contact transitions from `11/9` to `1/1`, increased double support from `87.8%` to `98.0%`, and reduced lateral root displacement from `22.1 cm` to `6.2 cm`. Torque saturation remained zero.

The target clipping rate in the JSON summaries now records range crossings, while applied torque includes the bounded equivalent feed-forward term. Position targets remain within model limits and total torque remains subject to the existing effort limit.

The initial ground-clearance calculation projects each oriented collision shape onto the world vertical axis. The earlier orientation-independent bounding-radius calculation overestimated the vertical size of the sideways foot cylinders by about 0.104 m. Correcting it and using a 2 mm motion-episode clearance moved first foot contact from step 13-14 to step 1-2. Bow peak normal force fell from about 4420 N to 1395 N; hand peak normal force fell from about 2817 N to 1513 N. This removes most of the artificial startup rocking.

Some bow movement remains intrinsic to this compatibility baseline. Its reference pelvis trajectory contains about 0.76 rad of pitch and 0.26 rad of yaw variation, and the policy was trained on the 19-joint Liondance model rather than the dynamically different official EVT2 model. Removing that residual motion safely requires reference cleanup and policy fine-tuning or retraining; root locking, gravity changes and arbitrary gain increases are deliberately not used here.

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

`tests/test_motion_evt2.py` checks the 29-joint map, exact 19+10 partition, both ONNX contracts, WALKAMP-neutral hold values for all uncontrolled joints, deterministic training gains, equivalent PD effort at clipped targets, target limits, finite observations/actions/physics state, and the episode-start-only reset guard.

## Phase 3 switching interface

Online switching should use three explicit states: `WALKAMP`, `READY`, and `MOTION`. Every state should emit one complete 29-joint `ControlTarget`; a partial 19-joint command should never reach the motor interface.

1. `WALKAMP -> READY`: command zero velocity and wait for bounded base speed, roll/pitch, torque saturation and valid foot support.
2. `READY -> MOTION`: initialize only policy memory, time step and yaw alignment, then blend complete 29-joint targets over a bounded transition. Never rewrite `qpos` or replay the first reference state.
3. `MOTION -> READY`: finish by duration or abort on height, attitude, contact, finite-value or saturation guards.
4. `READY -> WALKAMP`: reset WALKAMP history from live state, ramp commands from zero, and retain the same motor limits.

The current standalone episode initialization is intentionally not exposed as a switching API. Phase 3 should first validate these transitions in headless MuJoCo before any real-robot integration.
