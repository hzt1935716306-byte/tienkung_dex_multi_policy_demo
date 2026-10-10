# TienKung DEX Phase 4 Read-only Deployment

This branch is a software-adaptation and Shadow Mode branch. It cannot enable
motor output. Do not interpret these commands as permission to control a robot.

## Architecture

```text
Official ROS2 status topics
        |
OfficialRosStateBuffer -- sptlib parallel->serial ankle conversion
        |
RobotStateSnapshot (29 joints + explicit validity)
        |
WalkAmpRuntime / MotionRuntime
        |
ShadowMultiPolicyController
        |
HardwareSafetySupervisor
        |
29-joint candidate ControlTarget -> log only

There is no ROS motor-command publisher.
```

MuJoCo entry points (`run_walkamp.py`, `run_motion_evt2.py`,
`run_multi_evt2.py`) and the Phase 3B voice/command bus remain unchanged.

## 1. Development-host checks

Use an environment containing the project requirements:

```bash
cd /home/zt/project/tienkung/tienkung_dex_multi_policy_demo

/home/zt/miniconda3/envs/beyondmimic/bin/python -m pytest -q

/home/zt/miniconda3/envs/beyondmimic/bin/python \
  scripts/check_real_interface.py \
  --config configs/real_evt2.json

/home/zt/miniconda3/envs/beyondmimic/bin/python \
  run_real.py --config configs/real_evt2.json --check-only
```

`--check-only` loads all three ONNX files and validates configuration without
starting ROS. It does not publish commands.

## 2. Offline observation replay

Use a JSONL file produced by the read-only recorder:

```bash
/home/zt/miniconda3/envs/beyondmimic/bin/python \
  scripts/replay_real_observations.py \
  --config configs/real_evt2.json \
  --offline-observations /path/to/robot_states.jsonl \
  --max-steps 1000 \
  --no-realtime
```

Add `--action a` or `--action b` only to exercise the candidate Shadow state
flow. The replayed physical state is immutable, so this is not a dynamics test.

## 3. Target-computer prerequisites

Before a live check, the on-site operator must verify:

1. The robot is the expected EVT2 29-motor revision.
2. OS, Python ABI and ROS2 distribution match the official binary packages.
3. `bodyctrl_msgs` and `sptlib_python` come from the approved official/vendor
   release and load in the same Python process as this project.
4. The vendor controller remains running and no custom motor publisher is
   active.
5. Reading the four status topics is permitted.

This project intentionally does not install the official `.deb` or binary wheel
automatically.

## 4. Live read-only interface check

In a shell on the robot computer, source the actual vendor ROS2 environment.
The exact workspace path must come from the operator; do not guess it.

```bash
source /opt/ros/humble/setup.bash
source /path/from/vendor/install/setup.bash

cd /path/to/tienkung_dex_multi_policy_demo

python3 scripts/check_real_interface.py \
  --config configs/real_evt2.json \
  --live \
  --timeout 10
```

This creates four subscriptions and zero publishers. Inspect the generated
`artifacts/real_interface_check.json`. Do not continue if IDs, types, rates,
timestamps, IMU values, temperatures or errors are wrong.

## 5. Live Shadow Mode

After the read-only check passes:

```bash
python3 run_real.py \
  --config configs/real_evt2.json \
  --max-steps 3000
```

Expected startup text includes:

```text
[SHADOW] motor command publication is disabled; zero ROS command publishers are created
```

Outputs are written under `artifacts/`:

- `real_shadow_states.jsonl`: timestamped input snapshots;
- `real_shadow_runtime.jsonl`: policy targets, latency and safety decisions;
- `real_shadow_summary.json`: aggregate performance and blockers.

Do not request an action in live Shadow Mode until a validated base-velocity and
foot-contact/force source has been connected. Without those estimators the
controller remains at `READY_CHECK`, reports the unavailable fields, and returns
a `FAILED/ready_timeout` status after the configured bounded wait.

After those estimators are validated, the existing text/voice client can target
the Shadow Command Bus without gaining any motor authority:

```bash
python3 scripts/voice_control.py --config configs/real_evt2.json --text
```

The LLM remains limited to the `a`/`b` whitelist. It cannot alter gains, limits,
joint targets or control authority.

## 6. Hardware-contract completion

For every entry in `configs/hardware_contract_evt2.json`, attach real evidence
and have the vendor or responsible site engineer approve it. In particular:

- capture all 29 CAN IDs and map each to physical positive motion;
- load actual zero offsets, signs and limits rather than identity placeholders;
- calibrate IMU orientation and angular-velocity frames;
- validate sptlib ordering, units, calibration and failure handling;
- identify reliable base velocity and bilateral contact/force estimates;
- confirm lower-level `pos/spd/kp/kd/tor` units and watchdog behavior;
- prove exclusive controller ownership and a tested recovery procedure;
- test physical E-stop and vendor protective stop under supervision.

Editing all statuses to `confirmed` does not enable commands: the implementation
still has no ROS command publisher and `HardwareSafetySupervisor` always returns
`motor_output_allowed=false`.

## 7. Protected motion plan for a future reviewed branch

Only after all contract evidence is accepted should a separate branch add a
hard-gated publisher. The on-site sequence is:

1. read-only state and policy inference;
2. vendor-approved supported single-joint/limited posture interface test;
3. protected WALKAMP standing test in a support rig;
4. low-speed walk and brake tests;
5. separate bow and wave tests;
6. complete WALKAMP/BRAKE/READY/MOTION/RECOVER flow;
7. voice trigger only after the control chain is already accepted.

Failure at any step stops progression. A Python zero-torque command is not an
E-stop, a fixed PD pose is not a universal recovery, and the original controller
must never be disabled automatically.
