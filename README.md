# TienKung Dex Multi-Policy MuJoCo Demo

这是一个可独立运行、可直接上传 GitHub 的 TienKung 机器人演示工程。它已经把模型、ONNX 策略、MuJoCo 控制器、JSON 配置、键盘命令和 DashScope 语音控制整理到同一个目录，不依赖原始 IsaacLab 训练仓库运行。

## 当前能力

| 入口 | 机器人模型 | 功能 | 验证状态 |
| --- | --- | --- | --- |
| `configs/demo.json` | Dex EVT Liondance，19 DOF | 单窗口静止稳定、WASD 行走、键盘动作、外部命令 | 已验证 |
| `configs/walk_only.json` | TienKung2 Lite，20 DOF | 原生行走策略诊断 | 已验证 |
| `configs/unified_experimental.json` | Dex EVT，29 DOF | 单窗口尝试同时加载行走和动作 | 不稳定，仅供诊断 |
| `configs/walkamp_official.json` | 官方 xSIM EVT2，完整 29 关节 | 独立运行官方 WALKAMP 840→23 行走策略 | 四场景无头测试通过 |
| `configs/motion_evt2.json` | 官方 xSIM EVT2，完整 29 关节 | 独立运行 BeyondMimic 104→19 鞠躬/摊手策略 | 两个真实策略完整轨迹通过 |

重要：三个 checkpoint 并不是在同一个 action space 上训练的。本项目在 19-DOF 动作模型上运行全部策略，行走策略多出的 4 个肩 roll/yaw 输出保留在观测历史中但不施加到固定关节，腰部在行走期间由 PD 保持。该适配已通过 MuJoCo 连续切换测试；实机部署前仍应按 [策略兼容说明](docs/POLICY_COMPATIBILITY.md) 做限幅和低增益测试。

## 官方 WALKAMP 第一阶段基线

独立运行官方 23 维 WALKAMP 策略：

```bash
python run_walkamp.py --config configs/walkamp_official.json
```

按键仍采用游戏式控制：`W/S` 前后、`A/D` 横移、`Q/E` 转向、`Space` 停止、`Esc` 退出。这个入口使用原样打包的官方 xSIM EVT2 模型，不参与原有 19 维动作切换，因此不会影响 `configs/demo.json`。

四场景无界面回归：

```bash
python scripts/run_walkamp_headless_suite.py \
  --config configs/walkamp_official.json
```

完整观测定义、23/29 关节映射、日志字段、模型审计结论与测试结果见 [WALKAMP 第一阶段文档](docs/WALKAMP_PHASE1.md)。现有 `assets/mjcf/dex_evt_full.xml` 的惯量和碰撞体与官方 evt2 不等效，因此本阶段没有宣称它可以替代官方模型。

## BeyondMimic 动作第二阶段基线

在官方 29 关节 EVT2 模型中分别运行两个 19 维动作策略：

```bash
python run_motion_evt2.py --config configs/motion_evt2.json --motion a --mode policy
python run_motion_evt2.py --config configs/motion_evt2.json --motion b --mode policy
```

`a` 为鞠躬，`b` 为右手摊开。该入口只验证单个动作，不会与 WALKAMP 在线切换，也不会修改旧的 `configs/demo.json`。无头回归会先运行开环参考轨迹诊断，再运行真实 ONNX 策略：

```bash
python scripts/run_motion_evt2_headless_suite.py --config configs/motion_evt2.json
```

参考轨迹在官方动力学模型中会跌倒，但两个反馈策略均可完成完整轨迹；因此参考模式的失败不会被误报成策略失败或策略成功。映射、日志、测试结果及限制见 [BeyondMimic EVT2 第二阶段文档](docs/MOTION_EVT2_PHASE2.md)。

动作入口固定使用训练名义 PD，不依赖 ONNX 导出时随机写入的环境增益。超出关节范围的策略期望角仍会被裁剪，但被裁掉的位置误差会转换为受力矩限幅约束的等效前馈，从而与 IsaacLab 隐式 PD 的控制语义一致。

## 目录结构

```text
.
├── assets/                 # 19-DOF 动作模型、20-DOF 行走模型及网格
├── configs/                # 所有命令行参数、策略和动作标签
├── docs/                   # 架构与兼容性说明
├── policies/               # 已导出的 ONNX 策略
├── scripts/                # 环境检查、语音控制、手动发命令
├── src/tienkung_demo/      # 控制器源码
├── run_sim.py              # MuJoCo 主入口
├── run_walkamp.py          # 官方 WALKAMP 独立入口
├── run_motion_evt2.py      # 官方 EVT2 上的独立 BeyondMimic 动作入口
└── setup.sh                # 安装脚本
```

## 1. 从零安装

推荐 Ubuntu 22.04、Python 3.10，并使用独立 Conda 环境：

```bash
cd /path/to/tienkung_dex_multi_policy_demo

conda create -n tienkung_demo python=3.10 -y
conda activate tienkung_demo

chmod +x setup.sh
./setup.sh
python scripts/check_setup.py --smoke
```

安装语音功能：

```bash
sudo apt update
sudo apt install -y portaudio19-dev
./setup.sh --voice
```

已有环境也可指定 Python：

```bash
PYTHON_BIN=/path/to/conda/envs/tienkung_demo/bin/python ./setup.sh
```

## 2. 运行单窗口完整 Demo

一条命令启动静止稳定、键盘行走、动作和外部命令接收：

```bash
python run_sim.py --config configs/demo.json
```

仿真窗口按键：

| 按键 | 动作 |
| --- | --- |
| `W` / `S` | 前进 / 后退 |
| `A` / `D` | 左移 / 右移 |
| `Q` / `E` | 左转 / 右转 |
| `Space` | 行走速度清零并进入静止稳定 |
| `J` | 鞠躬 |
| `K` | 右手摊开/摆手 |
| `R` | 中止当前动作并平滑恢复 |
| `Esc` | 退出 |

零速度时使用动作策略第 0 帧作为学习型稳定器，因此不会让周期行走策略持续原地踏步。有速度命令时自动平滑切入行走策略；按 `Space`、触发动作或动作结束时自动平滑切回静止稳定器。动作返回不再先回放动作首帧，避免无平衡阶段导致倒地。整个过程保持在同一个 MuJoCo 窗口中。

`configs/demo.json` 中的 `simulation.idle_motion_key` 指定静止稳定器使用哪个动作，当前为 `b`。如果清空该值，零速时会退回行走策略加 PI 定点保持，此时仍可能有轻微踏步。

不连接网络和麦克风时，可先在终端 2 测试文字触发：

```bash
python scripts/voice_control.py --config configs/demo.json --text
```

可识别的示例语义：

- 鞠躬：`请鞠个躬`、`谢谢您`、`向大家问好`、`对不起`
- 摆手：`请挥挥手`、`欢迎大家`、`再见`、`打个招呼`、`你好`

启动 DashScope 实时语音：

```bash
export DASHSCOPE_API_KEY="你的真实 DashScope API Key"
python scripts/voice_control.py --config configs/demo.json
```

指定麦克风索引：

```bash
python scripts/voice_control.py --config configs/demo.json --mic 2
```

### 本地麦克风控制远程服务器

普通 SSH 不会把本地麦克风设备映射到服务器。推荐在身边的电脑运行语音模块，通过 SSH 只发送识别后的动作命令；服务器继续运行 MuJoCo。

服务器终端：

```bash
cd /absolute/path/to/tienkung_dex_multi_policy_demo
python run_sim.py --config configs/demo.json
```

本地电脑先验证免密 SSH，避免动作线程等待密码：

```bash
ssh-copy-id USER@SERVER_IP
ssh USER@SERVER_IP true
```

本地电脑克隆本仓库并安装语音依赖后，先用文字测试远程链路：

```bash
python scripts/voice_control.py \
  --config configs/demo.json \
  --text \
  --ssh-target USER@SERVER_IP \
  --remote-project /absolute/path/to/tienkung_dex_multi_policy_demo
```

确认输入“鞠躬”或“挥手”可以控制服务器后，再启动本地麦克风：

```bash
export DASHSCOPE_API_KEY="你的真实 DashScope API Key"
python scripts/voice_control.py \
  --config configs/demo.json \
  --ssh-target USER@SERVER_IP \
  --remote-project /absolute/path/to/tienkung_dex_multi_policy_demo
```

`--remote-project` 必须是服务器上的绝对路径。远程发送脚本只依赖 Python 标准库，默认使用服务器的 `python3`；需要指定解释器时增加 `--remote-python /path/to/python`。

语音进程与 MuJoCo 通过 `/tmp/tienkung_dex_commands.jsonl` 通信。也可手动从另一个终端发命令：

```bash
python scripts/send_command.py a
python scripts/send_command.py b
python scripts/send_command.py r
```

## 3. 单独诊断原生行走策略

```bash
python run_sim.py --config configs/walk_only.json
```

| 按键 | 行走命令 |
| --- | --- |
| `W` / `S` | 增加前进 / 后退速度 |
| `A` / `D` | 增加左移 / 右移速度 |
| `Q` / `E` | 增加左转 / 右转角速度 |
| `Space` | 速度清零 |
| `Esc` | 退出 |

每次按键按固定步长累加速度，控制台会打印当前 `vx`、`vy` 和 `yaw`。原数字键 `8/2/4/6/7/9/5` 和数字小键盘仍然兼容。正常演示请优先使用 `configs/demo.json`；`walk_only.json` 只用于对照原始 20-DOF 行走模型。

## 4. 无界面测试

```bash
python run_sim.py --config configs/demo.json \
  --no-viewer --auto-motion a --auto-motion-after 100 --max-steps 800

python run_sim.py --config configs/demo.json \
  --no-viewer --auto-motion b --auto-motion-after 100 --max-steps 950

python run_sim.py --config configs/walk_only.json \
  --no-viewer --max-steps 1000
```

## 5. 修改动作和配置

所有演示参数都在 `configs/demo.json` 中。新增动作时，把 ONNX 放入 `policies/`，然后增加一个单字符键：

```json
"c": {
  "name": "new_motion",
  "display_name": "新动作",
  "path": "../policies/new_motion.onnx",
  "duration_steps": 500,
  "start_step": 0,
  "control_mode": "policy",
  "torque_scale": 1.0,
  "effort_scale": 1.0,
  "voice_keywords": ["新动作", "演示一下"]
}
```

ONNX 必须带有本项目使用的 BeyondMimic 元数据：`joint_names`、`joint_stiffness`、`joint_damping`、`default_joint_pos` 和 `action_scale`。

## 6. 策略来源

- `policies/bow.onnx`：由 `2026-08-16_00-02-03_bow_skeleton1/model_9999.pt` 导出。
- `policies/right_hand_open.onnx`：由 `2026-08-16_00-04-49_右手摊开1_Skeleton2/model_9999.pt` 导出。
- `policies/walk.onnx`：由 `2026-08-16_17-28-17/model_49999.pt` 对应的 TienKung-Lab 导出策略提供。

运行时只需要 ONNX，不需要把 `.pt` checkpoint 或原训练仓库一起复制。

## 7. 常见问题

`ModuleNotFoundError`：确认已在项目根目录执行 `./setup.sh`，并使用同一个 Conda 环境运行。

MuJoCo 窗口打不开：确认当前终端具有桌面会话和 `DISPLAY`；服务器上改用 `--no-viewer`。默认 MJCF 已包含天空、棋盘地面、环境光和双向补光；若窗口仍全黑，检查显卡驱动及 `MUJOCO_GL` 设置。

`No module named pyaudio`：安装 `portaudio19-dev` 后重新执行 `./setup.sh --voice`。

DashScope websocket 超时：检查 API Key、网络、代理和防火墙；先用 `--text` 验证动作链路。

实验性 29-DOF 联合配置中机器人倒地：请使用默认的 19-DOF `configs/demo.json` 适配方案；长期实机部署仍建议统一机器人定义后重新训练或微调。

## License

源码按 BSD 3-Clause 许可发布。机器人资产和训练权重的来源与再分发注意事项见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
