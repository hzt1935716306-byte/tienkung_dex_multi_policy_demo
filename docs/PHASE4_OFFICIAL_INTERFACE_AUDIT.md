# Phase 4 Official Interface Audit

Date: 2026-10-10

This audit is a software reference audit. No real TienKung robot was connected, no
vendor controller was stopped, and no motor command was published.

## Fixed upstream references

| Project | Branch/revision used | Purpose |
|---|---|---|
| Open-X-Humanoid/Deploy_Tienkung | branch `3.0`, commit `9786cf1a7ed9e3bead1a8de47ed6a5c251cb1869` | Primary ROS2 and policy deployment reference |
| Open-X-Humanoid/xSIM_MUJOCO | commit `942469c660ec93e6d7ae9e78026db81cfa623591` | EVT2 MuJoCo model reference |
| Open-X-Humanoid/xMimic | commit `4a9f8bac4aaec07cd01b0ec9a4b1188735174983` | BeyondMimic observation and robot configuration reference |

The audited Deploy_Tienkung files are `rl_control_node.py`,
`common/robot_interface.py`, `common/robot_data.py`,
`common/body_id_map.py`, `config/dex_config.yaml`,
`policy/walk_amp/fsm_walkamp.py`, and
`policy/beyond_mimic/fsm_beyond_mimic.py`.

## ROS2 contract found in official source

State subscriptions use QoS depth 10:

| Topic | Type | Fields consumed |
|---|---|---|
| `/leg/status` | `bodyctrl_msgs/msg/MotorStatusMsg` | `header.stamp`, `status[]` |
| `/arm/status` | `bodyctrl_msgs/msg/MotorStatusMsg` | `header.stamp`, `status[]` |
| `/waist/status` | `bodyctrl_msgs/msg/MotorStatusMsg` | `header.stamp`, `status[]` |
| `/imu/status` | `bodyctrl_msgs/msg/Imu` | `euler`, `orientation`, angular velocity, acceleration, `error` |

`MotorStatus` contains `name:uint16`, `pos`, `speed`, `current`,
`temperature`, and `error:uint32`. The official implementation converts current
to torque using `ct_scale` and applies zero/direction calibration.

The official command topics are `/leg/cmd_ctrl`, `/arm/cmd_ctrl`, and
`/waist/cmd_ctrl`, all using `bodyctrl_msgs/msg/CmdMotorCtrl`. Each `MotorCtrl`
contains `name`, `kp`, `kd`, `pos`, `spd`, and `tor`. Phase 4 only encodes this
candidate structure for tests; it does not create publishers.

## Official 29-motor map

The runtime name is this repository's canonical EVT2 name. `l_wrist_yaw` and
`r_wrist_yaw` correspond to XML joints historically named
`elbow_yaw_l_joint` and `elbow_yaw_r_joint`.

| Index | CAN ID | Official name | Runtime name | Group |
|---:|---:|---|---|---|
| 0 | 51 | `l_hip_pitch` | `hip_pitch_l_joint` | leg |
| 1 | 52 | `l_hip_roll` | `hip_roll_l_joint` | leg |
| 2 | 53 | `l_hip_yaw` | `hip_yaw_l_joint` | leg |
| 3 | 54 | `l_knee` | `knee_pitch_l_joint` | leg |
| 4 | 55 | `l_ankle_pitch` | `ankle_pitch_l_joint` | leg |
| 5 | 56 | `l_ankle_roll` | `ankle_roll_l_joint` | leg |
| 6 | 61 | `r_hip_pitch` | `hip_pitch_r_joint` | leg |
| 7 | 62 | `r_hip_roll` | `hip_roll_r_joint` | leg |
| 8 | 63 | `r_hip_yaw` | `hip_yaw_r_joint` | leg |
| 9 | 64 | `r_knee` | `knee_pitch_r_joint` | leg |
| 10 | 65 | `r_ankle_pitch` | `ankle_pitch_r_joint` | leg |
| 11 | 66 | `r_ankle_roll` | `ankle_roll_r_joint` | leg |
| 12 | 33 | `waist_yaw` | `waist_yaw_joint` | waist |
| 13 | 32 | `waist_roll` | `waist_roll_joint` | waist |
| 14 | 31 | `waist_pitch` | `waist_pitch_joint` | waist |
| 15 | 11 | `l_shoulder_pitch` | `shoulder_pitch_l_joint` | arm |
| 16 | 12 | `l_shoulder_roll` | `shoulder_roll_l_joint` | arm |
| 17 | 13 | `l_shoulder_yaw` | `shoulder_yaw_l_joint` | arm |
| 18 | 14 | `l_elbow` | `elbow_pitch_l_joint` | arm |
| 19 | 15 | `l_wrist_yaw` | `elbow_yaw_l_joint` | arm |
| 20 | 16 | `l_wrist_pitch` | `wrist_pitch_l_joint` | arm |
| 21 | 17 | `l_wrist_roll` | `wrist_roll_l_joint` | arm |
| 22 | 21 | `r_shoulder_pitch` | `shoulder_pitch_r_joint` | arm |
| 23 | 22 | `r_shoulder_roll` | `shoulder_roll_r_joint` | arm |
| 24 | 23 | `r_shoulder_yaw` | `shoulder_yaw_r_joint` | arm |
| 25 | 24 | `r_elbow` | `elbow_pitch_r_joint` | arm |
| 26 | 25 | `r_wrist_yaw` | `elbow_yaw_r_joint` | arm |
| 27 | 26 | `r_wrist_pitch` | `wrist_pitch_r_joint` | arm |
| 28 | 27 | `r_wrist_roll` | `wrist_roll_r_joint` | arm |

This mapping is confirmed against official source, not against the user's
physical robot. Physical correspondence remains an open hardware-contract item.

## Policy contracts

| Policy | Input | Output | Period | Joint partition |
|---|---|---|---:|---|
| WALKAMP | one `[1,840]` tensor: 10 x 84 history | `[1,23]` | 0.01 s | 23 policy + 6 explicit hold |
| Bow | `obs [1,104]`, `time_step [1,1]` | `actions [1,19]` plus reference outputs | 0.01 s | 19 policy + 10 explicit hold |
| Wave | `obs [1,104]`, `time_step [1,1]` | `actions [1,19]` plus reference outputs | 0.01 s | 19 policy + 10 explicit hold |

`WalkAmpRuntime` preserves the official 10-frame history, gait phase, previous
action, 23-joint order, default angles, action scale and gains. `MotionRuntime`
validates all ONNX inputs, outputs and metadata before inference and preserves
reference time, previous action and live yaw alignment.

## Ankle audit

The official hardware path treats the four ankle channels as parallel actuator
measurements, converts them to serial policy coordinates with
`sptlib_python.funcSPTrans`, and converts serial targets back before command
encoding. This repository calls the same API and rejects a false conversion
return. It contains no replacement kinematic formula and no identity fallback.

Official reference gains for the four parallel ankle channels are Kp 15.0 and
Kd 1.25. Their suitability for the actual hardware revision is unconfirmed.

## State availability gap

The four official state topics provide joint state, current, temperature, motor
error and IMU data. They do not provide the following values used by the
Phase 3B READY/handoff safety checks:

- base linear velocity;
- left/right foot contact;
- left/right normal support force;
- world CoM position and velocity.

These fields remain invalid in `RobotStateSnapshot`. The Shadow state machine
therefore refuses to progress from `READY_CHECK` on live official-topic data
until a vendor or validated estimator supplies them. No zero velocity or fake
double support is inserted.

## Important unresolved contract items

The complete fail-closed list is in
[`configs/hardware_contract_evt2.json`](../configs/hardware_contract_evt2.json).
It includes robot/firmware revision, zero offsets and directions, real limits,
IMU mount transform, ankle calibration, command units/watchdog, estimator
sources, exclusive control ownership, E-stop and mechanical protection.

The official reference's identity offsets and broad YAML limits are not treated
as physical calibration. The existing local `dex_evt_full.xml` was previously
shown not equivalent to the official EVT2 model; all current simulation entries
continue to use `assets/official_evt2/urdf/evt2.xml`.
