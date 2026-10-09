# Official WALKAMP Phase 1

This phase adds an independent, reproducible baseline for the official 23-output WALKAMP policy on the complete 29-joint EVT2 robot. It does not participate in the existing multi-policy controller and does not change the BeyondMimic, voice, LLM, or real-robot interfaces.

## Upstream baseline

| Component | Upstream revision | Packaged file |
| --- | --- | --- |
| WALKAMP policy and inference reference | `Deploy_Tienkung` branch `3.0`, commit `9786cf1a7ed9e3bead1a8de47ed6a5c251cb1869` | `policies/walkamp_official.onnx` |
| EVT2 model and meshes | `xSIM_MUJOCO` commit `942469c660ec93e6d7ae9e78026db81cfa623591` | `assets/official_evt2/` |

The source URLs, SHA-256 hashes, and licenses are recorded in `assets/official_evt2/UPSTREAM_MANIFEST.json` and `third_party/OPEN_X_SHA256SUMS`. Unmodified `fsm_walkamp.py` and `walk_amp.yaml` snapshots are kept in `third_party/deploy_tienkung_walkamp/` for direct comparison. Startup checks require an ONNX input of `[1, 840]`, output of `[1, 23]`, and the official configuration constants.

## Independent viewer

Install the normal project dependencies, then run:

```bash
python run_walkamp.py --config configs/walkamp_official.json
```

Keyboard controls:

| Key | Command change |
| --- | --- |
| `W` / `S` | Increase/decrease forward velocity |
| `A` / `D` | Increase/decrease lateral velocity |
| `Q` / `E` | Increase/decrease yaw rate |
| `Space` | Set all three commands to zero |
| `Esc` | Quit |

This is a separate entry point. The existing command remains unchanged:

```bash
python run_sim.py --config configs/demo.json
```

## Official inference contract

Each policy observation frame contains 84 floats in this exact order:

```text
body angular velocity (3)
projected gravity (3)
vx, vy, yaw-rate command (3)
23 joint positions relative to defaults
23 joint velocities
23 previous actions
left/right gait sin, sin, cos, cos, support ratios (6)
```

The first observation is copied into all 10 history slots. Later steps discard the oldest frame and append the newest frame, producing the 840-float ONNX input. The policy runs at 100 Hz with `action_scale=0.25`; MuJoCo runs at 1 kHz with ten physics steps per policy step. The gait cycle is 0.85 s, with phase offsets `[0.38, 0.88]` and support ratios `[0.38, 0.38]`.

The 23 policy outputs are mapped by joint name in the official `walk_amp.yaml` order. The remaining six joints are held explicitly:

```text
elbow_yaw_l_joint, wrist_pitch_l_joint, wrist_roll_l_joint,
elbow_yaw_r_joint, wrist_pitch_r_joint, wrist_roll_r_joint
```

Their positions, gains, and effort limits are visible in `configs/walkamp_official.json`. All 29 targets are clipped to MuJoCo joint limits, and all torques are clipped to actuator limits; the six hold joints also use the lower configured limits.

## Reproducible headless suite

Run all four scenarios:

```bash
python scripts/run_walkamp_headless_suite.py \
  --config configs/walkamp_official.json
```

Run one scenario directly:

```bash
python run_walkamp.py \
  --config configs/walkamp_official.json \
  --scenario low_forward \
  --headless \
  --no-realtime
```

Available scenarios are `stand`, `low_forward`, `stop`, and `turn`. The suite writes a per-step CSV and JSON result under `artifacts/walkamp_headless_suite/`. CSV fields include root quaternion/RPY, world and body velocity, command, observation/action ranges, and every joint's measured position, velocity, target, torque, and torque limit.

Results from the implementation check on 2026-10-09:

| Scenario | Result | Minimum root z | Maximum abs roll/pitch | Relevant result |
| --- | --- | ---: | ---: | --- |
| Stand, 6 s | Pass | 0.981 m | 0.014 rad | XY drift 0.012 m |
| Forward, `vx=0.25` | Pass | 0.965 m | 0.111 rad | X +0.910 m; Y -0.208 m |
| Forward then stop | Pass | 0.969 m | 0.111 rad | Final XY speed 0.004 m/s |
| Turn, `yaw=0.25` | Pass | 0.975 m | 0.079 rad | Yaw +1.069 rad |

The forward test has measurable lateral and heading drift. The phase-one result establishes a functioning official baseline, not perfect trajectory tracking.

## Model compatibility audit

Generate a machine-readable comparison:

```bash
python scripts/compare_walkamp_models.py \
  --reference assets/official_evt2/urdf/evt2.xml \
  --candidate assets/mjcf/dex_evt_full.xml \
  --allow-differences \
  --output artifacts/model_compatibility_walkamp.json
```

The current `dex_evt_full.xml` is not dynamically equivalent to the official EVT2 model:

| Area | Audit result |
| --- | --- |
| Model size | Official: 31 bodies, 67 geoms, 39 meshes; local full: 33, 107, 44 |
| Joint axes/order | Matching |
| Joint constraints | 4 range differences and 16 joint actuator-force-range differences |
| Body properties | 143 differences across mass, inertia, inertial pose, and body pose |
| Collision geometry | 26 body-level differences |
| MuJoCo actuator interface | All 58 actuator bindings/ranges/gears/gain/bias entries match |

Because inertias and collision geometry differ, this phase deliberately runs WALKAMP on the packaged official EVT2 model. Running it on `dex_evt_full.xml` must be treated as a separate compatibility experiment and must not be described as an equivalent port.
