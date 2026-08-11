# Q1 ASAP Sim2Real 部署包

这是 Q1 motion tracking 的独立部署包，用于拷贝到 Q1 机载电脑上运行。它不依赖完整的 ASAP 训练仓库。

## 目录结构

```text
q1_asap_2real/
├── scripts/
│   └── sim2real_q1_motion_tracking_ros2.py
├── config/
│   └── q1_sim2real_base.yaml
├── robots/q1/
│   ├── q1_22dof_box.urdf
│   └── q1_22dof.yaml
├── policies/
│   └── lateral_raise/
│       └── model_6000.onnx
└── motions/
    ├── lateral_raise/
    │   └── lq1_ateral_raise.pkl
    ├── bolt/
    │   └── q1_bolt.pkl
    ├── football/
    │   └── q1_shoot.pkl
    └── stand/
        └── q1_stand_still.pkl
```

## 关键逻辑

脚本会从 `--motion-file` 自动推断动作时长：

```text
cycle_time = pose_aa.shape[0] / fps
```

当前已打包的 motion：

```text
lateral_raise: 157 / 50 = 3.14s
bolt:           99 / 50 = 1.98s
football:      154 / 50 = 3.08s
stand:         125 / 50 = 2.50s
```

因此一般不需要为每个动作单独写 YAML。新增动作时，把 ONNX 放到 `policies/<动作名>/`，把 motion pkl 放到 `motions/<动作名>/`，启动时替换 `--policy-path` 和 `--motion-file` 即可。


## 双策略状态机与离线 MuJoCo 验证

当前 `scripts/sim2real_q1_motion_tracking_ros2.py` 使用 stand + mimic 双策略状态机：

```text
HOLD_CURRENT -> GET_READY -> STAND_POLICY
-> PRE_MIMIC_INTERP -> PRE_MIMIC_HOLD -> MIMIC_POLICY
-> POST_MIMIC_HOLD -> POST_MIMIC_INTERP -> STAND_POLICY
```

- stand policy 在 mimic 期间继续 shadow inference，phase/history/counter 不重置。
- mimic 在 `MIMIC_POLICY` 期间控制全部 22 DoF。
- mimic 结束时，post handoff 以实际发送的 `command_target` 为起点；下半身在 `--post-handoff-s`（默认 0.5 s）内使用 minimum-jerk 向最新 stand target 过渡，上半身保持该实际命令的尾姿态。
- `--post-handoff-s` 必须满足 `0 < post_handoff_s <= post_hold_s`。

Linux headless 服务器可运行 ROS-free 回归脚本：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/env_genesis

python scripts/test_q1_dual_policy_mujoco_headless.py \
  --stand-policy-path policies/stand/model_8000.onnx \
  --stand-motion-file motions/stand/standv2_raw.pkl \
  --stand-config config/q1_sim2real_base.yaml \
  --mimic-policy-path policies/lateral_raise/model_6000.onnx \
  --mimic-motion-file motions/lateral_raise/lateral_raise_ref_motion_raw.pkl \
  --mimic-config config/q1_sim2real_base.yaml \
  --urdf robots/q1/q1_22dof_box.urdf \
  --robot-yaml robots/q1/q1_22dof.yaml \
  --output logs/q1_dual_policy_headless.npz \
  --summary-output logs/q1_dual_policy_headless_summary.yaml \
  --control-mode kinematic
```

`fixed-root + kinematic` 只验证观测、history、ONNX、状态机和 target 流水线；**不能**验证落地接触、站立稳定性，也不能使用其视频判断机器人是否落地。

需要验证 MuJoCo 地面接触时，必须使用带 floor/collision/freejoint 的物理 MJCF，并启用自由根 PD：

```bash
python scripts/test_q1_dual_policy_mujoco_headless.py \
  ...相同的 stand/mimic 参数... \
  --physics-mjcf /path/to/q1.xml \
  --control-mode pd \
  --free-root \
  --video-output logs/q1_dual_policy_contact.mp4
```

该模式在首次 stand ONNX inference 前提供临时外力辅助；首次 stand target 产出后撤销辅助，之后由 MuJoCo `mj_step` 计算接触。它仍只是 sim2sim，不代表真机验证。

### 当前 handoff 验证状态

minimum-jerk handoff 已消除 mimic 结束首帧的多腿同步跳变，并消除了 measured-q 上身尾姿态端点跳变；但当前自由根 PD 模型中，latest stand lower target 在 handoff 尾端仍快速变化，左右 ankle pitch 仍可能触发 `max_target_step_rad=0.08`。因此：

- 不要把当前 physical handoff 视频视为“已完成真机 handoff 验证”。
- 不要擅自全局降低 `max_target_step_rad`；需要先决定延长 handoff/post hold，或增加 handoff 专用的连续 stand-reference filter。
- 所有真机流程必须在人扶/保护、可立即急停的条件下进行。

## 机载环境准备

先进入部署包目录，并 source Q1 ROS2/SDK 工作空间：

```bash
cd ~/q1_asap_2real
source /path/to/q1_ros2_ws/install/setup.bash
```

如果实际 topic 名和默认值不同，启动时改下面这些参数：

```text
--joint-state-topic /aima/hal/joint/state
--joint-command-topic /aima/hal/joint/command
--imu-topic /aima/hal/imu/state
```

## Dry Run

先 dry-run。它会加载 policy、订阅机器人状态、计算 obs 和 action，但不会发布真实电机命令。

```bash
python scripts/sim2real_q1_motion_tracking_ros2.py \
  --config config/q1_sim2real_base.yaml \
  --policy-path policies/lateral_raise/model_6000.onnx \
  --motion-file motions/lateral_raise/lq1_ateral_raise.pkl \
  --urdf robots/q1/q1_22dof_box.urdf \
  --robot-yaml robots/q1/q1_22dof.yaml \
  --enable-policy \
  --dry-run
```

启动日志里应看到：

```text
cycle_time=3.140000s
policy_dt=0.0200s (50.0Hz)
```

同时重点检查：

```text
joint_age 和 base_age 正常刷新
静止站立时 projected_gravity 接近 [0, 0, -1]
ONNX input 是 [1, 365]，output 是 [1, 22]
```

如果 `projected_gravity` 明显不对，优先检查 IMU 坐标系、四元数顺序和 topic 类型，不要直接开电机。

## 开电机运行

只有在 dry-run 的 topic、IMU、关节顺序和安全检查都确认正常后，再开电机：

```bash
python scripts/sim2real_q1_motion_tracking_ros2.py \
  --config config/q1_sim2real_base.yaml \
  --policy-path policies/lateral_raise/model_6000.onnx \
  --motion-file motions/lateral_raise/lq1_ateral_raise.pkl \
  --urdf robots/q1/q1_22dof_box.urdf \
  --robot-yaml robots/q1/q1_22dof.yaml \
  --enable-policy \
  --enable-motors \
  --risk-confirm I_UNDERSTAND_REAL_ROBOT_RISK
```

## 按键（双策略）

```text
i = 插值到 default_dof_pos（GET_READY）
] = 进入 STAND_POLICY
[ = 从 STAND_POLICY 开始 mimic 过渡
o = 从 PRE_MIMIC_INTERP / PRE_MIMIC_HOLD / MIMIC_POLICY 安全返回 post handoff
q = 紧急阻尼
```

推荐流程：

```text
1. 启动脚本，等待 HOLD_CURRENT
2. 按 i，等待 GET_READY 完成并回到 HOLD_CURRENT
3. 按 ]，先观察 stand policy
4. 只在 stand 稳定、有人保护时按 [ 开始 mimic
5. 需要提前停止 mimic 时按 o；异常时按 q
```

## 切换动作

例如使用 bolt motion，需要有对应的 bolt ONNX，然后改成：

```bash
--policy-path policies/bolt/<bolt_model>.onnx \
--motion-file motions/bolt/q1_bolt.pkl
```

例如使用 football motion：

```bash
--policy-path policies/football/<football_model>.onnx \
--motion-file motions/football/q1_shoot.pkl
```

例如使用 stand motion：

```bash
--policy-path policies/stand/<stand_model>.onnx \
--motion-file motions/stand/q1_stand_still.pkl
```

注意：当前包里只有 lateral_raise 的 ONNX 已放入 `policies/`。`bolt`、`football` 和 `stand` 目前只放了 motion，使用前还需要放入对应训练导出的 ONNX。

## 注意事项

- 当前脚本的 obs 是 `[1, 365]`。
- `--fake-base-state` 只能用于吊起或台架验证，不能用于站立 motion tracking。
- `config/q1_sim2real_base.yaml` 是通用机器人和控制参数，正常新增动作不需要改它。
- `cycle_time` 默认从 motion pkl 推断，不要手动写死 bolt 或其他动作时长。
