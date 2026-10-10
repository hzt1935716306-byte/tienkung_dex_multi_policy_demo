# Phase 4 Software Validation

Date: 2026-10-10

## Scope

This report covers ONNX contracts, numerical runtime decoupling, official-format
message parsing, 29-joint candidate command encoding, safety gates and offline
Shadow Mode. It contains no result from a physical robot.

## Automated tests

Command:

```bash
/home/zt/miniconda3/envs/beyondmimic/bin/python -m pytest -q
```

Result at implementation time: all 60 tests passed, including all Phase 1-3B
regressions. The Phase 4 tests additionally verify:

- official CAN IDs and 12+3+14 partition;
- complete 29-joint state and command mapping;
- duplicate, missing and unknown motor rejection;
- IMU and ankle conversion failure rejection;
- disabled command sink and absence of ROS publisher creation;
- stale state, motor error, temperature, joint and target safety checks;
- WALKAMP and both BeyondMimic ONNX contracts;
- all three policies accept a snapshot parsed from official ROS message shapes;
- missing real-world READY estimators produce a bounded `ready_timeout` failure;
- old MuJoCo inference versus decoupled Runtime on identical state.

Numerical comparison tolerances are `1e-6` for WALKAMP observation/action,
`1e-5` for BeyondMimic observation/action, and `1e-7` for motion joint target.

## Offline Shadow replay

A 10-second, 1000-sample state recording was generated from the official EVT2
MuJoCo model at 100 Hz. Its contact flags and forces came from MuJoCo contacts;
they were not manually asserted. This recording is explicitly marked
`mujoco_recording_not_hardware`.

| Replay | Result | Mean inference | p95 inference | Max inference | p95 complete loop |
|---|---|---:|---:|---:|---:|
| WALKAMP idle | 1000/1000, remained idle | 0.088 ms | 0.091 ms | 0.331 ms | 0.233 ms |
| Bow request | full candidate state flow completed | 0.194 ms | 0.262 ms | 0.426 ms | 0.402 ms |
| Wave request | full candidate state flow completed | 0.216 ms | 0.264 ms | 0.429 ms | 0.404 ms |

All three runs reported:

```text
motor_publishers_created: 0
motor_commands_published: 0
stale_state_count: 0
```

Bow and wave each produced the ordered events `ready_confirmed`,
`pre_align_complete`, `motion_started`, `motion_complete`,
`walkamp_acquired`, and `recovery_complete`.

These timing values are CPU ONNX and Python-loop measurements on the development
computer. They exclude real ROS middleware, sensor transport and target-computer
scheduling. Replay completion proves software flow only; it does not prove
physical stability because candidate targets were not fed back into the recorded
trajectory.

## Static interface check

Command:

```bash
python scripts/check_real_interface.py --config configs/real_evt2.json
```

Observed on the development host:

- `onnxruntime`: available;
- `rclpy`: unavailable in the active conda shell;
- `bodyctrl_msgs`: unavailable;
- `sptlib_python`: unavailable;
- all hardware-contract confirmations: pending.

Therefore live read-only Shadow Mode has not run on this host. The official
Deploy_Tienkung repository includes a Humble `bodyctrl_msgs` package and a CPython
3.10 `sptlib_python` wheel, but compatibility and installation on the target must
be verified with the vendor/robot operator. This project does not copy or
silently install those binary packages.

## Acceptance status

| Item | Status |
|---|---|
| Three ONNX contracts load on CPUExecutionProvider | Pass |
| Decoupled runtime matches existing simulation inference | Pass |
| Official-format state parser and 29-joint map | Pass in unit tests |
| Candidate `pos/spd/kp/kd/tor` encoding | Pass with fake checked sptlib adapter |
| Conversion with real official sptlib binary | Not run on target |
| Actual ROS2 topic types, rates and latency | Not run on target |
| Actual robot calibration and model match | Blocked by hardware access |
| READY/handoff estimators on robot | Blocked; official topics do not supply them |
| Motor publication | Intentionally unavailable |
| Physical motion and multi-policy stability | Not tested, no claim |

## Existing simulation regressions

The original headless suites were rerun after the Phase 4 changes:

- WALKAMP `stand`, `low_forward`, `stop`, and `turn`: all passed;
- BeyondMimic real policy inference: bow completed 556/556 and wave completed
  685/685, both passed;
- reference-only motion playback remained diagnostic and fell, as already
  classified by the suite; it was not reported as policy success;
- Phase 3B quick walking-command suite: all nine brake/action/cancel scenarios
  passed and returned to `WALKAMP_IDLE`.
