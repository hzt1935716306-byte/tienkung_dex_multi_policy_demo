# Architecture

## Runtime Data Flow

```text
keyboard ---------------------------> MultiPolicyController
voice/text -> JSONL command file ----> MultiPolicyController
                                             |
                                             v
                                WalkPolicy or MotionPolicy
                                             |
                                             v
                               joint target + PD gains
                                             |
                                             v
                                       MuJoCo motors
```

`run_sim.py` adds `src/` to `sys.path` and starts `tienkung_demo.simulator`. Configuration paths are resolved relative to the selected JSON file, so the repository can be moved without editing absolute paths.

## Modules

- `config.py`: loads JSON and resolves project-relative model/policy paths.
- `robot.py`: joint maps, default poses, PD gains, effort limits, and policy order maps.
- `policies.py`: ONNX inference and exact observation construction for walking and motion policies.
- `simulator.py`: MuJoCo loop, PD control, keyboard callback, transitions, and fall checks.
- `command_bus.py`: append-only JSONL communication between voice and simulation processes.
- `voice.py`: DashScope realtime audio, LLM action labels, text fallback, and command dispatch.

## Motion Startup

The combined demo treats walk and idle as two states of the locomotion baseline. With a zero velocity command, the configured `idle_motion_key` policy runs continuously at frame 0 as a learned balance controller. A nonzero velocity command blends into the walk policy; clearing the command blends back into idle.

Motion entry uses one control tick (`0.01 s`). On completion, the action blends directly into the learned idle controller. The old motion-frame-0 return and neutral hold remain available behind `return_via_motion_start`, but are disabled by default because they can leave the robot without adequate balance authority during recovery.

## Security

DashScope credentials are read only from `DASHSCOPE_API_KEY`. No API key is stored in source code or JSON configuration.
