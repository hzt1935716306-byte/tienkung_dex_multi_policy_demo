# EVT2 Multi-Policy Phase 3A

## Scope

Phase 3A puts the official WALKAMP policy and the two existing BeyondMimic policies in one official EVT2 MuJoCo simulation. It does not modify the three neural-network files, the legacy `configs/demo.json` path, voice/LLM code, or real-robot interfaces.

The default controller is always WALKAMP. A zero velocity command remains a WALKAMP command and never starts a motion automatically.

## Run

Interactive viewer:

```bash
python run_multi_evt2.py --config configs/multi_evt2.json
```

Controls:

| Input | Result |
| --- | --- |
| `W/S` | increase/decrease forward velocity |
| `A/D` | increase/decrease lateral velocity |
| `Q/E` | increase/decrease yaw rate |
| `Space` | set WALKAMP command to zero |
| `J`, `a`, `bow` | request bow |
| `K`, `b`, `wave` | request wave |
| `R`, `r` | request safe abort and recovery |
| `Esc` | close the simulation |

External command bus:

```bash
python scripts/send_command.py bow --config configs/multi_evt2.json
python scripts/send_command.py wave --config configs/multi_evt2.json
python scripts/send_command.py r --config configs/multi_evt2.json
```

The command lifecycle is logged as `RECEIVED`, `ACCEPTED`, `EXECUTING`, then `COMPLETED`, `REJECTED`, or `FAILED`. Unsupported commands and new actions received while busy are rejected rather than preempting the active controller.

## State Machine

```text
WALKAMP_IDLE
  -> READY_CHECK
  -> PRE_ALIGN
  -> TRANSITION_IN
  -> MOTION
  -> TRANSITION_OUT
  -> RECOVER
  -> WALKAMP_IDLE
```

`READY_CHECK` requires acceptable root height, roll/pitch, base velocity, joint velocity, bilateral foot contact, and support force for a configured stable interval. `PRE_ALIGN` moves the physical 29-joint target toward the motion start while retaining WALKAMP feedback gains. Leg, waist, and arm deltas and tracking errors have separate limits.

The motion time index remains at frame zero during pre-alignment and transition-in. It starts advancing only after `MOTION` acquires control.

At normal completion, motion feedback remains active until the measured robot state is suitable for takeover. Phase 3A.1 adds a verified early tail window for bow and a bounded terminal wait; wave retains the terminal-frame path. WALKAMP then rebuilds its 840-value history, previous action, and gait phase from the live state. Recovery is declared complete only after another measured stable interval. See [Phase 3A.1 bow recovery](BOW_RECOVERY_PHASE3A1.md) for the newer exit behavior and tests.

An `r` command waits for a verified support and posture window. If no safe early window is available, the motion controller remains active until a safe terminal region; it is never replaced by a zero-torque or direct state reset.

## Target Handoff

Both policies first produce a `ControlTarget` in the same 29-joint model order. Network actions are never blended directly. The physical targets include:

- joint position target;
- Kp and Kd;
- feedforward effort;
- effort limit;
- torque scale.

The handoff uses:

```text
alpha(s) = 10*s^3 - 15*s^4 + 6*s^5
q_des = (1-alpha)*q_old + alpha*q_new
```

The same alpha is applied to the other `ControlTarget` fields. Both policies are evaluated against current simulator state during a handoff. A per-joint target-rate limiter and a per-joint torque-rate limiter run after blending. Exactly one complete 29-joint target is sent to MuJoCo each control cycle.

`MotionEvt2Policy.begin_from_live_state()` aligns reference yaw and seeds previous actions without modifying `qpos`, `qvel`, or the floating base. The original standalone `reset_episode_to_reference()` remains available only for the Phase 2 episode-start diagnostic.

## Configuration

All Phase 3A parameters are in `configs/multi_evt2.json`:

- policy and official EVT2 model paths;
- action mapping and duration;
- readiness, pre-alignment, takeover, and recovery thresholds;
- 0.4 s transition-in and 0.6 s transition-out defaults;
- leg, waist, and arm target-rate and torque-rate limits;
- deterministic idle, bow, wave, and abort scenarios.

The listed transition durations are validated defaults for this model and these policies, not a general real-robot safety guarantee. The configuration retains the official EVT2 actuator disable mask and uses only the `motor_*` torque actuator group.

## Logging

CSV logs contain root pose and velocity, commanded and actual joints, target Kp/Kd/feedforward/effort, torque, policy observation/action ranges, foot contacts and normal force, transition alpha, maximum target rate, maximum torque rate, saturation fraction, and slew-limited fraction.

State and command lifecycle events are emitted as structured JSON lines. Each headless run also writes a JSON summary under `artifacts/`.

## Reproducible Tests

Unit and contract tests:

```bash
python -m unittest discover -s tests -v
```

Phase 3A four-scenario suite:

```bash
python scripts/run_multi_evt2_headless_suite.py \
  --config configs/multi_evt2.json
```

Regression suites:

```bash
python scripts/run_walkamp_headless_suite.py \
  --config configs/walkamp_official.json

python scripts/run_motion_evt2_headless_suite.py \
  --config configs/motion_evt2.json
```

Validated on 2026-10-09:

| Scenario | Result | Minimum root z | Maximum abs roll/pitch | Torque saturation |
| --- | --- | ---: | ---: | ---: |
| idle | PASS | 0.988 m | 0.006 rad | 0.0% |
| bow | PASS | 0.935 m | 0.728 rad | 0.0% |
| wave | PASS | 0.988 m | 0.140 rad | 0.0% |
| abort | PASS | 0.960 m | 0.140 rad | 0.0% |

All 25 unit tests passed. The four official WALKAMP scenarios passed. Both Phase 2 policy runs passed their complete trajectories; the open-loop reference runs remain diagnostic failures as already documented and are not counted as policy success.

## Real-Robot Boundary

This phase validates one MuJoCo process only. It does not prove real-robot safety. Before deployment, the same `ControlTarget` and state-machine interfaces should be connected to a hardware safety layer with measured actuator limits, watchdogs, emergency stop, communication timeout handling, calibration checks, and staged low-gain/robot-supported tests.
