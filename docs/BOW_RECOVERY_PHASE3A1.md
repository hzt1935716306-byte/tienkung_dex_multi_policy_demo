# Phase 3A.1 Bow Recovery

## Scope

This change is limited to the bow exit and WALKAMP recovery path on branch
`feature/bow-recovery-phase3a1`. It does not change any ONNX policy, the EVT2
XML, gravity, joint limits, motion gains, WALKAMP gains, the legacy 19-DOF
entry point, or a real-robot interface. Wave retains its original terminal
handoff behavior as a regression control.

## Reproduced Failure

The Phase 3A baseline is commit `6f98d7e`. Its 20-command scenario runs all
commands in one continuous MuJoCo state. No reset occurs between bows.

```bash
python scripts/reproduce_bow_recovery_baseline.py
```

The failure reproduced deterministically on the second bow at step 2031:

| point | step | pitch (rad) | horizontal speed (m/s) | max joint speed (rad/s) | support |
| --- | ---: | ---: | ---: | ---: | --- |
| motion terminal frame | 1874 | 0.081 | 0.158 | 0.434 | both feet |
| transition-out in progress | 1934 | 0.272 | 0.941 | 7.048 | asymmetric |
| WALKAMP acquired | 1955 | 0.390 | 1.391 | 8.623 | right foot lost |
| fall threshold | 2030 | 1.397 | 2.802 | 8.099 | no support |

The bow body motion reaches its end with a small forward trend but is still
supported. The old terminal-frame wait and subsequent policy interpolation
amplify that trend. The actual fall occurs after WALKAMP acquisition in
`RECOVER`. Therefore this was not classified as a failure of the bow body
motion alone or as a pure terminal-wait failure.

The first baseline bow happened to recover, but moved 0.229 m in its heading
direction between transition-out start and recovery completion. At takeover it
had 0.110 rad pitch, 0.679 m/s forward world velocity, 9.704 rad/s maximum
joint speed, and one foot without contact.

## Implementation

Bow now has a configurable 0.8 s tail candidate window. The policy timeline
continues normally while the controller checks the measured state. A handoff
requires 0.15 s of consecutive valid samples; one good frame is insufficient.
The deterministic nominal run opens the window at reference frame 475 and
accepts it at frame 489. Reference frames around 490 are already settled: the
measured reference maximum joint speed is about 0.161 rad/s and the maximum
joint-position difference from the final frame is about 0.026 rad.

`handoff_feasible()` checks:

- root height, roll, pitch, pitch angular velocity, planar and angular speed;
- bilateral contact and normal support force;
- maximum measured joint speed;
- measured leg, waist, and arm posture error from WALKAMP neutral;
- motion-to-WALKAMP target difference for each joint group.

`recovery_complete()` is a separate, post-acquisition stability check. It does
not require the stricter pre-handoff target agreement, but still requires
valid posture, velocity, contacts, support force, and joint speed for 0.3 s.

Once the tail is prequalified, WALKAMP history and previous actions are built
from live state immediately. The controller does not repeat the same wait in
`TRANSITION_OUT`. Motion and WALKAMP inference continue from current MuJoCo
state during the 29-joint quintic target blend. No `qpos` or `qvel` reset is
used.

If no safe terminal window appears within 0.8 s, the controller enters the
explicit `FAILED` state, logs the rejected metrics and reason, and the runner
stops. It does not hold the last motion frame indefinitely, switch to zero
torque, or force WALKAMP to take over an invalid state.

Structured events distinguish:

- `motion_exit_window_opened`;
- `handoff_condition_met`;
- `motion_execution_ended` and `motion_timeline_ended`;
- `transition_out_started` and `walkamp_interpolation_started`;
- `walkamp_control_acquired`;
- `recovery_completed` or `terminal_failure`.

Each event includes pelvis pose, pitch angular velocity, horizontal velocity,
foot contacts and forces, joint speed, action id, and timing. CSV/JSON summaries
also report target jump, target rate, torque, torque rate, and saturation,
including a separate aggregate for `TRANSITION_OUT`.

## Selected Parameters

The four requested transition durations all avoided a fall in the nominal
test, but their takeover dynamics were different:

| duration | pitch rate at takeover | horizontal speed | max joint speed | conclusion |
| ---: | ---: | ---: | ---: | --- |
| 0.2 s | 1.709 rad/s | 0.097 m/s | 6.810 rad/s | too abrupt |
| 0.3 s | 0.336 rad/s | 0.279 m/s | 2.299 rad/s | excess translation |
| 0.4 s | 0.458 rad/s | 0.179 m/s | 2.572 rad/s | larger attitude transient |
| 0.6 s | 0.181 rad/s | 0.028 m/s | 1.532 rad/s | selected |

WALKAMP live-history initialization was tested at gait phase times 0, 0.2125,
0.425, and 0.6375 s. The selected 0.2125 s phase gave the best overall nominal
handoff: 0.028 m/s takeover speed and 0.0086 m forward recovery displacement.
In the 20-bow test it reduced mean forward recovery displacement from 0.0524 m
at phase 0 to 0.0332 m and reduced single-foot takeover events from 19/20 to
1/20. It does have a higher worst-case takeover joint-speed transient
(9.347 rad/s versus 6.932 rad/s at phase 0), so this is a simulation result,
not a claim of hardware readiness.

The selected values in `configs/multi_evt2.json` are:

```text
bow exit window       0.8 s
handoff stable time   0.15 s
terminal wait bound   0.8 s
transition out        0.6 s
bow WALKAMP phase     0.2125 s
wave WALKAMP phase    0.0 s (unchanged)
recovery stable time  0.3 s
```

## Test Results

Run the complete Phase 3A.1 suite:

```bash
python scripts/run_bow_recovery_suite.py \
  --config configs/multi_evt2.json \
  --baseline-summary artifacts/bow_recovery_baseline_reproduced/repeat20_summary.json
```

The suite covers the four transition durations, four WALKAMP phase candidates,
nominal bow, small forward/backward initial-state offsets, a tail velocity and
pitch-rate perturbation, and 20 consecutive bows in one physical state.

| scenario | result | completed | mean forward recovery displacement | takeover contact loss |
| --- | --- | ---: | ---: | ---: |
| nominal | PASS | 1/1 | 0.0086 m | 0 |
| small forward initial offset | PASS | 1/1 | 0.0328 m | 0 |
| small backward initial offset | PASS | 1/1 | 0.0189 m | 0 |
| tail perturbation | PASS | 1/1 | 0.0188 m | 0 |
| continuous repeated bow | PASS | 20/20 | 0.0332 m | 1/20 |

The 20-bow test executes 20400 control cycles without resetting MuJoCo. Its
minimum root height is 0.945 m, maximum torque is 139.4 Nm, maximum torque-rate
sample is 3327.9 Nm/s, and torque saturation rate is 0%. During transition-out,
the worst requested/applied one-cycle target deltas are 0.708/0.076 rad; the
target limiter therefore remains active instead of passing the raw jump to the
PD controller. The baseline completed only one bow and fell on the second at
root height 0.348 m.

Regression commands:

```bash
python -m pytest -q
python scripts/run_multi_evt2_headless_suite.py --config configs/multi_evt2.json
python scripts/run_walkamp_headless_suite.py --config configs/walkamp_official.json
python scripts/run_motion_evt2_headless_suite.py --config configs/motion_evt2.json
```

The Phase 2 reference modes remain expected diagnostic failures; both real
BeyondMimic policy modes pass their complete trajectories.

## Conclusions

1. The unstable trend begins near the bow tail, is amplified during the old
   wait/interpolation, and the reproduced fall occurs after WALKAMP takeover.
2. A repeatable safe window exists around bow reference frames 475-489 under
   nominal and tested small-perturbation conditions.
3. A 0.6 s exit blend with 0.15 s prequalification and WALKAMP phase 0.2125 s
   gives the best tested balance of pose, velocity, support, and displacement.
4. The continuous 20-bow test passes 20/20 without any physical-state reset.
5. The standard forward recovery displacement falls from 0.229 m in the first
   successful baseline bow to 0.0086 m; repeated-test mean is 0.0332 m. Extra
   recovery stepping is substantially reduced but not proven eliminated for
   all states.
6. The current ONNX is sufficient for this tested MuJoCo envelope. A dedicated
   learned recovery or a retrained standing tail is still required before
   claiming recovery from larger disturbances or hardware-safe operation. If
   no feasible tail window is found, the controller now reports that limitation
   instead of hiding it with a forced takeover.
