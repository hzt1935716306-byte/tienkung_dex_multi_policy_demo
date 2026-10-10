# Phase 3B: Walking Commands and Voice Integration

Branch: `feature/walk-command-voice-phase3b`

Base: `feature/stand-ready-phase3a4`

## Scope

This phase keeps the official EVT2 model and the existing WALKAMP, bow, and
wave ONNX files unchanged. It adds command-triggered braking from a moving
WALKAMP state and reconnects the existing DashScope voice front end to the
EVT2 multi-policy controller.

The active path is:

```text
WALKAMP_MOVING -> BRAKE -> READY_CHECK -> PRE_ALIGN -> TRANSITION_IN
               -> MOTION -> TRANSITION_OUT -> RECOVER -> WALKAMP_IDLE
```

WALKAMP remains in control throughout BRAKE and READY_CHECK. No state in this
path writes MuJoCo `qpos` or `qvel`. The existing Phase 3A entry, exit, early
handoff, measured-history, phase-preview, READY interface, target-rate, and
torque-rate logic remains in place.

## Braking

The selected simulation defaults are:

| Parameter | Value |
| --- | ---: |
| `vx` deceleration | `0.40 m/s^2` |
| `vy` deceleration | `0.35 m/s^2` |
| yaw deceleration | `0.60 rad/s^2` |
| command-zero epsilon | `0.005` |
| continuous ready time | `0.30 s` |
| ready timeout | `6.0 s` |
| maximum ready XY speed | `0.05 m/s` |
| maximum ready angular speed | `0.18 rad/s` |
| maximum ready joint speed | `0.35 rad/s` |

Reaching a zero velocity command does not mean the robot is stopped. The
controller continues to evaluate measured base velocity, angular velocity,
joint speed, attitude, bilateral support, and normal force. A timeout marks the
action failed and leaves the verified WALKAMP feedback controller active.

The previous velocity is recorded for diagnostics. Automatic velocity resume
is disabled by default (`resume_previous_velocity=false`), so a completed
action returns to zero-speed WALKAMP.

## Command Lifecycle

Each external request has a UUID and follows:

```text
RECEIVED -> ACCEPTED -> BRAKING -> WAITING_FOR_READY -> EXECUTING -> COMPLETED
```

Terminal alternatives are `REJECTED`, `CANCELLED`, and `FAILED`. JSONL command
and status channels are configured separately:

```text
/tmp/tienkung_dex_commands.jsonl
/tmp/tienkung_dex_command_status.jsonl
```

Retries with the same request ID are idempotent. Expired requests are rejected.
A second action can wait in the voice-side one-item queue; requests that reach a
busy controller are rejected and never preempt the active policy. `R` during
BRAKE or READY_CHECK cancels the pending action while retaining WALKAMP
feedback. `R` after BeyondMimic takes control uses the existing safe recovery
path.

## Simulation Commands

Interactive keyboard demo:

```bash
cd /home/zt/project/tienkung/tienkung_dex_multi_policy_demo
/home/zt/miniconda3/envs/beyondmimic/bin/python run_multi_evt2.py \
  --config configs/multi_evt2.json
```

Controls: `W/S` forward/backward, `A/D` lateral, `Q/E` yaw, `Space` stop,
`J` bow, `K` wave, `R` safe cancel/recovery, and `Esc` quit.

External commands from another terminal:

```bash
python scripts/send_command.py bow --config configs/multi_evt2.json
python scripts/send_command.py wave --config configs/multi_evt2.json
python scripts/send_command.py r --config configs/multi_evt2.json
```

Text command front end:

```bash
python scripts/voice_control.py --config configs/multi_evt2.json --text
```

Real DashScope voice front end:

```bash
export DASHSCOPE_API_KEY="YOUR_REAL_KEY"
python scripts/voice_control.py --config configs/multi_evt2.json --mic 8
```

The LLM is only permitted to emit whitelisted action IDs: bow maps to `a` and
wave maps to `b`. Walking velocity remains under keyboard or remote-control
ownership. Voice/network work runs outside the MuJoCo control loop.

For a microphone on a client computer and MuJoCo on a server, configure
passwordless SSH first and run:

```bash
python scripts/voice_control.py \
  --config configs/multi_evt2.json \
  --ssh-target USER@SERVER_IP \
  --remote-project /absolute/path/to/tienkung_dex_multi_policy_demo \
  --remote-python /absolute/path/to/python
```

Remote status is obtained by `scripts/wait_status.py`; completion is no longer
estimated from `duration_steps`.

## Reproducible Tests

Phase 3B.1 full suite, including 20 alternating no-reset cycles:

```bash
python scripts/run_walk_command_suite.py \
  --config configs/multi_evt2.json \
  --output-dir artifacts/walk_command_phase3b
```

Text-to-controller end-to-end tests:

```bash
python scripts/run_voice_text_integration.py \
  --config configs/multi_evt2.json \
  --output-dir artifacts/voice_text_phase3b
```

Unit and regression tests:

```bash
python -m pytest -q
python scripts/run_multi_evt2_headless_suite.py --config configs/multi_evt2.json
python scripts/run_walkamp_headless_suite.py --config configs/walkamp_official.json
python scripts/run_motion_evt2_headless_suite.py --config configs/motion_evt2.json
```

## Results

All nine Phase 3B.1 directional and edge-case scenarios passed. The 20-cycle
test used one continuous MuJoCo state and completed 20/20 actions with no fall,
timeout, failed command, or torque saturation. Mean request-to-ready time was
`1.068 s`; maximum stopping distance was `0.036 m`; minimum root height was
`0.946 m`; maximum applied per-cycle joint target change was `0.08 rad`.

The isolated `0.20 m/s` forward tests required `2.78 s` and `0.461 m` from
request to a continuously ready state. This is the conservative measured-stop
result, not merely the command ramp time. It should be recalibrated before any
hardware use.

Both text flows passed:

```text
请鞠个躬 -> a -> BRAKE -> READY -> bow -> RECOVER -> COMPLETED
向我挥挥手 -> b -> BRAKE -> READY -> wave -> RECOVER -> COMPLETED
```

Their command transport delays were below `4 ms` in the local test. Full
status sequences were `RECEIVED, ACCEPTED, BRAKING, WAITING_FOR_READY,
EXECUTING, COMPLETED`.

The real microphone/cloud test is not accepted as passed. PyAudio and an input
device were available, but the DashScope WebSocket closed before the connected
callback. The current shell also had no explicit `DASHSCOPE_API_KEY`. Text
integration therefore proves the control and status path, but cloud credentials,
network access, and a spoken microphone trial remain an external prerequisite.

Compact checked-in results are in `docs/data/phase3b_results.json`; complete
runtime logs and summaries are generated under `artifacts/`.

## Safety Boundary

These are MuJoCo results, not hardware certification. The braking rates,
stopping distance, contact thresholds, and controller timeouts require separate
hardware validation with the vendor safety stack active. This phase does not
modify or disable any real-robot service.
