# G1 Motion Tracking + Locomotion 双策略架构

## 总览

G1 真机部署采用**双策略并联**架构：一个 locomotion 策略持续运行（控制下肢行走/站立），多个 mimic 策略按需接管（控制全身做特定动作）。两个策略都是独立训练的 ONNX 模型，由 `deepmimic_dec_loco.py` 统一调度。

```
         locomotion policy (ONNX)               mimic policy (ONNX)
               ↓                                       ↓
         下肢 12 DOF                              全 身 23/29 DOF
               ↓                                       ↓
         ┌─────────────────────────────────────────────────┐
         │          deepmimic_dec_loco.py                   │
         │  · 策略切换 / 插值调度                            │
         │  · 观测拼接 / history 管理                        │
         │  · 键盘 / 手柄输入                                │
         └─────────────────────────────────────────────────┘
                              ↓
                     Unitree SDK2 DDS
                              ↓
                         G1 真机
```

---

## 1. 训练阶段

### 1.1 Locomotion 策略

**训练任务**：velocity tracking（速度跟踪）

- 输入：本体感知 (base_ang_vel, projected_gravity, dof_pos/vel) + 指令 (lin_vel, ang_vel, stand flag, phase) + history (4 帧 × 若干项)
- 输出：**12 DOF 下肢动作**（hip × 6 + knee × 2 + ankle × 4）
- 上肢：训练时不参与，固定在默认姿态

### 1.2 Mimic 策略

**训练任务**：motion tracking（运动模仿）

每个 mimic 策略对应一个特定动作（CR7 跳、侧跳、踢球、扣篮等），使用 DeepMimic 架构：

- 输入：本体感知 + `ref_motion_phase` (参考运动进度 0→1) + history
- 输出：**全套 23/29 DOF**

分别训练，各自导出为独立的 ONNX 文件。

---

## 2. 真机部署架构

核心代码文件：

| 文件 | 职责 |
|------|------|
| `sim2real/rl_policy/base_policy.py` | `BasePolicy` 基类：ONNX 加载、关节状态接收、PD 指令发送、键盘/手柄交互、GET_READY |
| `sim2real/rl_policy/deepmimic_dec_loco.py` | `MotionTrackingDecLocoPolicy`：双策略调度、观测拼接、插值逻辑 |
| `sim2real/config/g1_29dof_hist.yaml` | 机器人参数、PD 增益、观测维度、mimic 策略映射表 |
| `sim2real/utils/state_processor.py` | 读取 Unitree 底层状态，打包为 robot_state_data |
| `sim2real/utils/command_sender.py` | 通过 SDK2 发送关节指令 |
| `sim2real/utils/history_handler.py` | 环形 history buffer |

### 2.1 状态机

```
         上电
          ↓
    原地 PD 保持 (use_policy_action=False)
          ↓  按 i
     GET_READY: 当前姿态 → default_dof_angles 线性插值 (500 步 ≈ 10s)
          ↓  按 ]
    ┌─ LOCOMOTION 模式 (默认) ───────────────────────────┐
    │  · locomotion 策略控制下肢 12 DOF                     │
    │  · 上身固定在 loco_upper_body_dof_pos                 │
    │  · 持续循环，无终止                                    │
    │  · 键盘 WASD/QE 调整速度指令，z 归零                   │
    │  · 手柄：摇杆控制速度，十字键选 mimic                   │
    └──────────────────────────────────────────────────┘
          ↓  按 [  或  手柄 select
    ┌─ MIMIC 模式 ──────────────────────────────────────┐
    │  Phase 1: 上身插值 (1.5s)                           │
    │    ref_upper_dof_pos 从当前值 → mimic 初始上身姿态     │
    │  Phase 2: mimic 策略接管全身                          │
    │    motion 播完 (phase ≥ 1.0) 自动触发 Phase 4         │
    │  Phase 3: (跳过，保留)                                │
    │  Phase 4: 上身回插 (2.0s = 0.5s gap + 1.5s)          │
    │    ref_upper_dof_pos 从 mimic 结束姿态 → loco 默认姿态 │
    └──────────────────────────────────────────────────┘
          ↓  自动
    回到 LOCOMOTION 模式
```

### 2.2 观测构造

两种模式构造不同的观测向量，见 `prepare_obs_for_rl()`。

**LOCOMOTION 模式**（无 history 版本）：

```
[last_action(12), base_ang_vel×0.25(3), ang_vel_cmd(1), lin_vel_cmd(2),
 stand_cmd(1), cos_phase(1), dof_pos_minus_default(29), dof_vel×0.05(29),
 projected_gravity(3), ref_upper_dof_pos(17), sin_phase(1)]
```

- `last_action` 只取前 12 维（下肢），因为 locomotion 策略只输出下肢
- `ang_vel_cmd` / `lin_vel_cmd` 从键盘 WASD 或手柄摇杆实时更新
- `stand_cmd`: 0=站立模式 1=行走模式
- `ref_upper_dof_pos`: mimic 结束后的目标上肢姿态或 loco 默认上肢姿态

**MIMIC 模式**（无 history 版本）：

```
[last_action(23), base_ang_vel×0.25(3),
 dof_pos_minus_default(23), dof_vel×0.05(23),
 projected_gravity(3), phase(1)]
```

- 使用 `policy_mimic_robot_dofs` 掩码，只取 mimic 策略需要的 DOF（23 vs 29）
- `phase`: 当前 motion 播放进度 [0, 1]，由 `(current_time - frame_start_time) / motion_length_s` 计算

**History**：两种模式都支持独立 history（`history_loco_config` 和 `history_mimic_config`），通过 `USE_HISTORY_LOCO` 和 `USE_HISTORY_MIMIC` 开关控制。

### 2.3 Action 构造

两种模式的 action 拼接策略不同，见 `get_policy_action()`。

**LOCOMOTION 模式**：

```python
# 策略输出 (12,) — 只控制下肢
policy_action = locomotion_policy(obs)  # shape (12,)

# 拼接: 下肢用策略输出, 上肢用 ref_upper_dof_pos
scaled = policy_action * 0.25                          # 下肢 action
scaled = concat([scaled, ref_upper_dof_pos], axis=1)   # 拼接上肢 → (29,)

# 最终指令
q_target = scaled + default_dof_angles
```

**MIMIC 模式**：

```python
# 策略输出 (23,) — 全身 (g1_29dof_anneal_23dof)
policy_action = mimic_policy(obs)  # shape (23,)

# 扩展为 29 DOF，非活动关节填 0
full = zeros(29)
full[active_dofs] = policy_action
scaled = full * 0.25

q_target = scaled + default_dof_angles
```

### 2.4 动作空间映射

config 中 `mimic_robot_types` 定义了每个 mimic 策略使用的机器人类型：

```yaml
robot_dofs:
  "g1_29dof":              [1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,1, 1,1,1,1,1,1,1, 1,1,1,1,1,1,1]  # 29 DOF
  "g1_29dof_anneal_23dof": [1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,1, 1,1,1,1,0,0,0, 1,1,1,1,0,0,0]  # 23 DOF (无手腕)
```

训练时 G1 用 `g1_29dof_anneal_23dof`（23 RL 关节），6 个手腕关节不参与 RL。部署时 `full_policy_action` 根据掩码填充，手腕关节始终填 0（保持 `default_dof_angles` 位置）。

---

## 3. 插值机制详解

### 3.1 GET_READY → LOCOMOTION 启动

不涉及策略切换，只是 PD 目标值的线性插值：

```python
q_target = dof_pos + (default_dof_angles - dof_pos) * (init_count / 500)
```

500 步 × 20ms = 10 秒。插值完成后策略输出的 `q_target` 相对于 `default_dof_angles`，目标连续。

### 3.2 LOCOMOTION → MIMIC

**上身插值**（Phase 1, 1.5s 线性）：

```python
alpha = elapsed / 1.5
ref_upper_dof_pos = (1 - alpha) * current_upper_pose + alpha * start_upper_dof_pos[mimic_idx]
```

下身始终由 locomotion 策略控制（此时通常 `stand_cmd=1` 即站立状态，locomotion 策略会让机器人站稳）。

插值完成后：
- `history_handler.reset([0])` — 清空 history，避免 loco 模式的命令/phase history 污染 mimic 观测
- 切换到 mimic 策略
- `frame_start_time` 重置为当前时间，开始 motion 播放

### 3.3 MIMIC → LOCOMOTION

**自动触发**（`phase ≥ 1.0`）：

1. 记录当前上身姿态作为 `end_upper_dof_pos`，腰 roll/pitch 归零
2. `ref_upper_dof_pos = end_upper_dof_pos`
3. 切换到 locomotion 策略

**上身回插**（Phase 4, 0.5s 静置 + 1.5s 插值）：

```python
alpha = max((elapsed - 0.5) / 1.5, 0.0)   # 前 0.5s 保持 mimic 结束姿态
ref_upper_dof_pos = (1 - alpha) * end_upper_dof_pos + alpha * loco_upper_body_dof_pos
```

`loco_upper_body_dof_pos` 定义在 config：
```yaml
# 腰: 0 0 0
# 左肩: 0, 0.3, 0, 1.0 (shoulder_pitch, roll, yaw, elbow)
# 左手腕: 0, 0, 0
# 右肩: 0, -0.3, 0, 1.0
# 右手腕: 0, 0, 0
```

这样 mimic 结束后机器人不会突然"散架"——上身缓慢回到对称站姿，下身持续由 locomotion 策略保持平衡。

**紧急退出**（按键 `[` 在 mimic 进行中）：
和自动结束一样走 Phase 4 回插，但不经过 0.5s 静置（`interpolation_emergency = True`）。

---

## 4. 多 Mimic 策略管理

config 中预定义了多个 mimic 策略：

```yaml
mimic_models:
  "CR7_level1": "model_191500.onnx"
  "kick_level3": "model_240000.onnx"
  "jump_forward_level1": "model_191500.onnx"
  "lebron_level1": "model_233500.onnx"
  "side_jump_level3": "model_245000.onnx"
  ...
```

每个策略有对应的：
- `start_upper_body_dof_pos`: mimic 起始时上身应该摆什么姿态（从参考 motion 第一帧提取）
- `motion_length_s`: motion 总时长
- `mimic_robot_types`: 使用的 DOF 掩码

切换方式：
- 键盘：`;` 下一个，`'` 上一个（仅在 LOCOMOTION 模式下有效）
- 手柄：R1 → 下一个，L1 → 上一个

---

## 5. 安全机制

### 5.1 关节限位

```python
q_target = np.clip(q_target, motor_pos_lower_limit_list, motor_pos_upper_limit_list)
```

### 5.2 Action clip

```python
policy_action = np.clip(policy_action, -100, 100)
```

### 5.3 PD 增益可调

运行时通过键盘/手柄调节 KP：
- `5/6`: KP ± 0.01
- `4/7`: KP ± 0.1
- `0`: 重置 KP = 1.0（即原始值）

### 5.4 软停止

按键 `o` → `use_policy_action = False`，策略不再更新目标，机器人 PD 保持在当前 `q_target`。

`ElasticBand` 机制（config `ENABLE_ELASTIC_BAND: True`）预留了阻尼模式接口，但 G1 主控制循环中未显式调用。

---

## 6. 数据流总结

```
Unitree LowState (500Hz)
        │
        ▼
  state_processor._prepare_low_state()
        │
        ▼
  robot_state_data: [timestamp×2, pos×3, quat×4, joint_pos×29,
                     base_lin_vel×3, base_ang_vel×3, joint_vel×29]
        │
        ▼
  prepare_obs_for_rl()  ← 根据 policy_locomotion_mimic_flag 选择观测格式
        │
        ▼
  policy(obs)  ← ONNX 推理
        │
        ▼
  get_policy_action()  ← 插值 + 拼接 + action_scale
        │
        ▼
  q_target = scaled_action + default_dof_angles
        │
        ▼
  command_sender.send_command(q_target, dq=0, tau=0)
        │
        ▼
  Unitree SDK2 → 电机 PD 控制器 (500Hz)
```
