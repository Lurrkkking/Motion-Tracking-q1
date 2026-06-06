# Q1 Motion Tracking 搭建记录

> 从官方 [LeCAR-Lab/ASAP](https://github.com/LeCAR-Lab/ASAP.git) 出发，在 branch `q1_tracking_test` 上完成 Q1 机器人的 motion_tracking 训练与 eval 视频导出。

## 环境准备

### Conda 环境

从已有的 `q1_humanoidverse`（已跑通 Q1 locomotion）克隆新环境 `q1_motion`，放到数据盘：

```bash
conda create -y --prefix /root/autodl-tmp/conda_envs/q1_motion --clone q1_humanoidverse
```

清理环境中的路径冲突（克隆会带入 HumanoidVerse 的 `.pth` 文件）：

```bash
rm /root/autodl-tmp/conda_envs/q1_motion/lib/python3.8/site-packages/_editable_impl_humanoidverse.pth
```

安装 ASAP_official 的包（覆盖 HumanoidVerse 的 egg-link）：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/q1_motion
pip install -e /root/autodl-tmp/ASAP_official/.
pip install -e /root/autodl-tmp/ASAP_official/isaac_utils
```

### 分支

```bash
cd /root/autodl-tmp/ASAP_official
git checkout -b q1_tracking_test
```

---

## Phase 1：验证官方 G1 Motion Tracking

在改 Q1 之前，先确认官方代码本身没问题。

### 修复官方 obs 配置缺失字段

官方 `config/obs/motion_tracking/motion_tracking.yaml` 缺少 `add_noise_currculum` 等 7 个必填字段（`legged_robot_base.py` 会直接访问这些 key），补上：

```yaml
add_noise_currculum: False
noise_initial_value: 0.05
noise_value_max: 1.0
noise_value_min: 0.00001
soft_dof_pos_curriculum_degree: 0.00001
soft_dof_pos_curriculum_level_down_threshold: 40
soft_dof_pos_curriculum_level_up_threshold: 42
```

### 运行命令

```bash
python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  +robot=g1/g1_29dof_anneal_23dof \
  +obs=motion_tracking/motion_tracking \
  +rewards=motion_tracking/reward_motion_tracking_basic \
  +domain_rand=NO_domain_rand \
  +terrain=terrain_locomotion_plane \
  headless=True num_envs=1 \
  'robot.motion.motion_file=humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles'
```

结果：env instantiate 成功，simulation 创建成功，PPO iteration 正常。

---

## Phase 2：复制 Q1 Asset 和 Robot Config

从已验证的 HumanoidVerse 复制 Q1 资源：

```bash
# Robot 数据（URDF + meshes）
cp -r /root/autodl-tmp/HumanoidVerse/humanoidverse/data/robots/q1 \
      /root/autodl-tmp/ASAP_official/humanoidverse/data/robots/q1

# Robot 配置
cp -r /root/autodl-tmp/HumanoidVerse/humanoidverse/config/robot/q1 \
      /root/autodl-tmp/ASAP_official/humanoidverse/config/robot/q1
```

Q1 核心参数（来自 HumanoidVerse 已验证配置）：

| 参数 | 值 |
|---|---|
| DOF 数量 | 22 |
| Body 数量 | 23（head_link 被 collapse_fixed_joints 合并到 torso） |
| 出生高度 | `pos.z = 0.41m` |
| PD stiffness | hip/knee 30, ankle 20, waist 80, shoulder/elbow 30 |
| PD damping | hip/knee/shoulder/elbow 1.5, ankle 1.0, waist 2.0 |
| Asset armature | 0.004 |
| Per-DOF armature | 0.01 × 22 |
| 动作缩放 | `action_scale = 0.25` |

---

## Phase 3：给 Q1 补全 Motion Tracking 字段

### 3.1 参考 G1 的 motion block 结构

官方 G1 的 `config/robot/g1/g1_29dof_anneal_23dof.yaml` 有一个完整的 `motion:` block，包含：

- `motion_file` — motion data 路径
- `asset` — MJCF XML 路径（用于 motion lib 加载骨架）
- `humanoid_type` / 各种 flag
- `body_names` / `dof_names`（link 名，用于 motion fitting）
- `limb_weight_group` — 肢体分组
- `nums_extend_bodies` + `extend_config` — 扩展 body（手、头）
- `motion_tracking_link` — VR 3-point 跟踪的 body
- `lower_body_link` / `upper_body_link` — 上下半身分组
- `pelvis_link` / `base_link` / `hips_link`
- `joint_matches` — 机器人 body → SMPL joint 映射
- `smpl_pose_modifier` — SMPL 坐标系修正
- `visualization` — marker 颜色

### 3.2 关键差异：Q1 vs G1

| | G1 | Q1 |
|---|---|---|
| DOF | 23（多 waist_pitch） | 22 |
| Config body_names | 24 | 23 |
| Motion data joints | 27 (= 24 + 3 extend) | 24（bolt）/ 23（cr7） |
| Waist 结构 | waist_yaw → waist_roll → torso | waist_roll → torso（waist_yaw 是两者间的 joint） |
| 扩展 body | hands×2 + head = 3 | 仅 head = 1 |
| Ref motion phase obs | 不需要（G1 有 `dof` 字段） | 需要（Q1 motion 无 `dof` 字段） |

### 3.3 Body 数量对齐（核心难点）

Motion tracking env 的数据流：

```
Motion data (pose_aa)  →  FK (body_names_augment = XML + extend)
                       →  global_translation_extend (ref body positions)
Simulator (URDF)       →  _rigid_body_pos_extend (robot body positions)
                       →  二者做 diff → reward
```

**G1 的对齐方式**：
- XML: 24 bodies
- extend: 3（hands + head）
- `body_names_augment` = 27
- Motion `pose_aa` = 27 joints ← 完美匹配
- `dof_pos` 计算：`pose[..., 1:24]` = 23 ← 匹配 sim 的 23 DOF

**Q1 需要满足**：
- `XML bodies + extend = pose_aa joints`
- `dof_pos 维度 = sim DOF 数量 (22)`

**最终方案**（匹配 bolt.pkl 的 24 joints）：
- XML: 23 bodies（不含 head_link）
- extend: 1（head_link，offset [0,0,0]）
- `body_names_augment` = 24 ← 匹配 motion 24 joints
- `dof_pos` = `pose[..., 1:23]` = 22 ← 匹配 sim 22 DOF

### 3.4 Motion Tracking Link

VR 3-point tracking 硬编码了 `heading_inv_rot.repeat(3, 1)`（3 个跟踪点）。Q1 只用 feet×2 + torso：

```yaml
motion_tracking_link:
  - "left_ankle_roll_link"
  - "right_ankle_roll_link"
  - "torso_link"
```

### 3.5 创建 MJCF XML

Motion lib 用 MJCF XML 加载骨架树做 FK。为 Q1 创建 `data/robots/q1/q1_22dof_box.xml`：

- 23 bodies（pelvis + 12 leg + 2 waist/torso + 8 arm，无 head_link）
- 22 actuators（对应 22 DOF）
- 1 floating_base_joint（free type，不计入 DOF）

关键：head_link 不在 XML 中（否则 `num_joints - 1 ≠ num_dof`），而是通过 `extend_config` 在运行时加到 sim body list。

### 3.6 复制 Q1 Motion 和专用 Obs/Reward 配置

```bash
# Motion files
mkdir -p humanoidverse/data/motions/q1
cp /root/autodl-tmp/ASAP/humanoidverse/data/motions/q1/*.pkl \
   humanoidverse/data/motions/q1/

# Q1 专用 obs + reward（使用 ref_motion_phase 而非 body tracking obs）
cp .../obs/motion_tracking/q1_deepmimic_a2c_nolinvel_LARGEnoise_history.yaml \
   humanoidverse/config/obs/motion_tracking/
cp .../rewards/motion_tracking/reward_motion_tracking_q1_cr7.yaml \
   humanoidverse/config/rewards/motion_tracking/
```

修复 obs 维度计算：将硬编码 `+9`（G1 的 3 个扩展 body）改为动态：

```yaml
dif_local_rigid_body_pos: ${eval:'3 * ${robot.num_bodies} + 3 * ${robot.motion.nums_extend_bodies}'}
```

---

## Phase 4 & 5：训练

### 启动脚本

`.sh/run_q1_motion_tracking.sh`：

```bash
#!/bin/bash
set -euo pipefail

MOTION_FILE="${MOTION_FILE:-humanoidverse/data/motions/q1/q1_bolt.pkl}"
EXP_NAME="${EXP_NAME:-MotionTracking_Q1_Basic}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/q1_motion
cd /root/autodl-tmp/ASAP_official

exec python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  +robot=q1/q1_22dof \
  +obs=motion_tracking/q1_deepmimic_a2c_nolinvel_LARGEnoise_history \
  +rewards=motion_tracking/reward_motion_tracking_q1_cr7 \
  +domain_rand=NO_domain_rand \
  +terrain=terrain_locomotion_plane \
  "robot.motion.motion_file=${MOTION_FILE}" \
  project_name=TEST_Q1 \
  experiment_name="${EXP_NAME}" \
  num_envs=256 \
  headless=True
```

### 验证结果

| 阶段 | num_envs | 结果 |
|---|---|---|
| Phase 4 | 1 | env instantiate/reset/step 正常，reward 计算正常 |
| Phase 5 | 64 | PPO iteration 正常，~1.37s/iter |
| Phase 5 | 128 | PPO iteration 正常，~1.22s/iter |
| 完整训练 | 256 | 正常运行，可保存 checkpoint |

---

## Phase 6：Eval 视频导出

官方 `eval_agent.py` 和 `isaacgym.py` 缺少 offscreen 录屏功能，需要从旧版移植。

### 6.1 eval_agent.py

添加录制参数解析 + `disable_keyboard_listener` 控制：

```python
auto_record = bool(config.get("auto_record", False))
auto_record_num_frames = int(config.get("auto_record_num_frames", 600))
disable_keyboard_listener = bool(config.get("disable_keyboard_listener", True))
offscreen_record = bool(config.get("offscreen_record", False))
# ... 注入到 config.env.config
```

### 6.2 simulator/isaacgym/isaacgym.py

**`__init__`**：添加录制状态变量，保留 `save_rendering_dir`。

**`setup/setup_terrain` → `prepare_sim`**：
- `headless and not offscreen_record` 时才禁用 graphics device（录屏需要 GPU 渲染）
- `prepare_sim` 末尾调用 `_setup_offscreen_recording()`

**新增三个方法**：
- `_setup_offscreen_recording()` — 创建 camera sensor + cv2 VideoWriter
- `_write_offscreen_frame()` — 每帧捕获 camera image → RGBA → BGR → write
- `finalize_recording()` — release VideoWriter

**`render()` 开头**：优先写 offscreen frame，若无 viewer 则 early return。

### 6.3 envs/base_task/base_task.py

```python
# 修改前（offscreen 模式下 render 根本不会被调用）
def render(self, sync_frame_time=True):
    if self.viewer:
        self.simulator.render(sync_frame_time)

# 修改后
def render(self, sync_frame_time=True):
    if self.viewer or getattr(self.simulator, "offscreen_record", False):
        self.simulator.render(sync_frame_time)
```

### 6.4 agents/ppo/ppo.py

```python
# 修改前：死循环
while True:
    ...

# 修改后：有 eval_steps 限制，fallback 到 auto_record_num_frames
eval_steps = int(self.config.get("eval_steps", -1))
if eval_steps <= 0:
    fallback_steps = int(self.env.config.get("auto_record_num_frames", -1))
    if fallback_steps > 0:
        eval_steps = fallback_steps
while step < eval_steps:
    ...
```

### 6.5 最终 eval 命令

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/q1_motion
cd /root/autodl-tmp/ASAP_official

python humanoidverse/eval_agent.py \
    '+checkpoint=/root/autodl-tmp/ASAP_official/logs/TEST_Q1/<run_dir>/model_1000.pt' \
    ++auto_record=true \
    ++auto_record_num_frames=150 \
    ++offscreen_record=true \
    ++offscreen_record_fps=50 \
    ++disable_keyboard_listener=true
```

视频输出到 checkpoint 目录下的 `renderings/ckpt_<N>/` 中。

---

## 改动的文件清单

### 新增

| 文件 | 说明 |
|---|---|
| `config/robot/q1/q1_22dof.yaml` | Q1 完整 robot config（从 HumanoidVerse 复制 + 补齐 motion block） |
| `data/robots/q1/` | Q1 URDF + meshes（从 HumanoidVerse 复制） |
| `data/robots/q1/q1_22dof_box.xml` | MJCF 骨架 XML（新建，23 bodies + 22 actuators） |
| `data/motions/q1/*.pkl` | Q1 motion files（bolt, cr7, jump, stand） |
| `config/obs/motion_tracking/q1_deepmimic_a2c_nolinvel_LARGEnoise_history.yaml` | Q1 专用 obs 配置 |
| `config/rewards/motion_tracking/reward_motion_tracking_q1_cr7.yaml` | Q1 专用 reward 配置 |
| `.sh/run_q1_motion_tracking.sh` | 训练启动脚本 |
| `docs/q1_motion_tracking_setup.md` | 本文档 |

### 修改

| 文件 | 改动 |
|---|---|
| `config/obs/motion_tracking/motion_tracking.yaml` | 补 `add_noise_currculum` 等 7 个必填字段（官方原版缺失） |
| `eval_agent.py` | 加 `auto_record`/`offscreen_record` 参数解析，`disable_keyboard_listener` |
| `simulator/isaacgym/isaacgym.py` | 加 offscreen 录屏全套逻辑 |
| `envs/base_task/base_task.py` | `render()` 增加 offscreen_record 检查 |
| `agents/ppo/ppo.py` | `evaluate_policy` 从死循环改为 `eval_steps` 控制 |

---

## 已知局限 & 后续工作

1. **Motion file**：bolt.pkl 的 24 joints 匹配当前配置。cr7_motion.pkl（23 joints）也和当前 XML 的 23 body 骨架兼容。如需新 motion，确保 `pose_aa.shape[1] == XML_bodies + nums_extend_bodies`。

2. **VR tracking 3 点**：当前用双脚 + torso 做 tracking。如果后续 motion data 包含手部轨迹，需要改 XML 和 extend_config 来支持。

3. **Domain randomization**：当前训练使用 `NO_domain_rand`。后续可以加上 randomization 提升 sim2real 效果。

4. **eval 脚本**：`evaluate_policy` 的 `while step < eval_steps` 限制的是 step 数而非 episode 数。如果 episode 提前终止，env 会自动 reset 继续下一步。
