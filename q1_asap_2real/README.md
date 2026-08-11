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

## 按键

```text
i = 插值到默认站立位姿
 a = policy armed
p = 开始执行 policy
o = 停止/保持当前位置
q = 紧急阻尼
```

推荐流程：

```text
1. 启动脚本
2. 等待进入 HOLD_CURRENT
3. 按 i，插值到默认姿态
4. 按 a，进入 POLICY_ARMED
5. 按 p，开始动作
6. 异常时按 q
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
