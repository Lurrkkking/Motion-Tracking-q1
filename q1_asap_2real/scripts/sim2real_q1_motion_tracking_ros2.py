#!/usr/bin/env python3
"""
Q1 motion-tracking sim2real ROS2 bridge.

This script is intentionally conservative. With --fake-base-state it is only a
bench/suspended validation tool: fake base angular velocity and gravity are not
valid for standing motion tracking.

Policy inference runs at the training control rate, while joint commands are
republished by a separate high-rate timer, matching the Q1 SDK joint control
example's 0.002 s command period.
"""

import argparse
import os
import pickle
import signal
import sys
import termios
import threading
import time
import tty
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import yaml

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
except ImportError:
    rclpy = None
    Node = object
    DurabilityPolicy = None
    HistoryPolicy = None
    QoSProfile = None
    ReliabilityPolicy = None

try:
    from aimdk_msgs.msg import JointCommand, JointCommandArray, JointStateArray
except ImportError:
    JointCommand = None
    JointCommandArray = None
    JointStateArray = None
try:
    from sensor_msgs.msg import Imu
except ImportError:
    Imu = None
try:
    from rosidl_runtime_py.utilities import get_message
except ImportError:
    get_message = None
try:
    from nav_msgs.msg import Odometry
except ImportError:
    Odometry = None


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NUM_ACTIONS = 22
HISTORY_KEYS = [
    "actions", "base_ang_vel", "dof_pos", "dof_vel",
    "projected_gravity", "ref_motion_phase",
]
SINGLE_SLICES = {
    "actions": slice(0, 22),
    "base_ang_vel": slice(22, 25),
    "dof_pos": slice(25, 47),
    "dof_vel": slice(47, 69),
    "projected_gravity": slice(69, 72),
    "ref_motion_phase": slice(72, 73),
}

FALLBACK_LOWER_LIMITS = np.array([
    -3.0543, -0.69813, -1.5708, 0.0, -0.7854, -0.34907,
    -3.0543, -1.5708, -1.5708, 0.0, -0.7854, -0.34907,
    -0.2618, -1.5708,
    -3.1416, -0.087266, -1.5708, -0.87266,
    -3.1416, -2.7925, -1.5708, -0.87266,
], dtype=np.float32)
FALLBACK_UPPER_LIMITS = np.array([
    1.5708, 1.5708, 1.5708, 2.4435, 0.43633, 0.34907,
    1.5708, 0.69813, 1.5708, 2.4435, 0.43633, 0.34907,
    0.2618, 1.5708,
    1.5708, 2.7925, 1.5708, 1.6581,
    1.5708, 0.087266, 1.5708, 1.6581,
], dtype=np.float32)


class ControlState(Enum):
    WAIT_STATE = "WAIT_STATE"                           # 等待传感器数据就绪
    HOLD_CURRENT = "HOLD_CURRENT"                       # 保持当前位姿不动
    GET_READY = "GET_READY"                             # 插值到默认站立位姿
    POLICY_ARMED = "POLICY_ARMED"                       # 策略已加载，等待启动信号
    POLICY = "POLICY"                                   # 策略运行中
    STOP = "STOP"                                       # 停在当前目标
    EMERGENCY_DAMPING = "EMERGENCY_DAMPING"             # 紧急阻尼（零刚度+高阻尼）
    ERROR = "ERROR"                                     # 错误状态


def read_conf(config_file):
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cfg = {}
    cfg["num_single_obs"] = int(config["num_single_obs"])           # 单帧观测维度 = 73
    cfg["simulation_dt"] = float(config["simulation_dt"])
    cfg["cycle_time"] = float(config.get("cycle_time", 0.0))
    cfg["frame_stack"] = int(config["frame_stack"])
    cfg["num_actions"] = int(config["num_actions"])
    cfg["control_decimation"] = int(config["control_decimation"])   # 控制降频比，比如 10
    cfg["simulation_duration"] = float(config.get("simulation_duration", config["cycle_time"]))
    cfg["default_dof_pos"] = np.array(config["default_dof_pos"], dtype=np.float32)              # 默认站立位姿的 22 个关节角，GET_READY 阶段的目标就是插值到这个位置。
    cfg["obs_scale_base_ang_vel"] = float(config["obs_scale_base_ang_vel"])
    cfg["obs_scale_dof_pos"] = float(config["obs_scale_dof_pos"])
    cfg["obs_scale_dof_vel"] = float(config["obs_scale_dof_vel"])
    cfg["obs_scale_gvec"] = float(config["obs_scale_gvec"])
    cfg["obs_scale_refmotion"] = float(config["obs_scale_refmotion"])
    cfg["obs_scale_hist"] = float(config["obs_scale_hist"])
    cfg["clip_observations"] = float(config["clip_observations"])
    cfg["clip_actions"] = float(config["clip_actions"])
    cfg["action_scale"] = float(config["action_scale"])
    cfg["kps"] = np.array(config["kps"], dtype=np.float32) * float(config.get("kp_scale", 1.0))
    cfg["kds"] = np.array(config["kds"], dtype=np.float32) * float(config.get("kd_scale", 1.0))
    cfg["time_offset"] = float(config.get("time_offset", 0.0))                                      # 相位初始偏移
    cfg["phase_wrap"] = bool(config.get("phase_wrap", False))                                       # 是否循环
    cfg["stop_at_motion_end"] = bool(config.get("stop_at_motion_end", True))                        # 运动结束自动停
    cfg["action_filter_alpha"] = float(config.get("action_filter_alpha", 1.0))                      # EMA 平滑系数
    cfg["target_pos_rate_limit"] = float(config.get("target_pos_rate_limit", 0.0))                  # 目标位置速率限制
    cfg["dof_names"] = list(config.get("joint_names", config.get("dof_names", [])))
    return cfg


def infer_cycle_time_from_motion(motion_file):
    if not motion_file:
        return None, ""
    try:
        with open(motion_file, "rb") as f:
            data = pickle.load(f)
    except Exception:
        try:
            import joblib
            data = joblib.load(motion_file)
        except Exception as exc:
            raise RuntimeError(f"failed to read motion file {motion_file}: {exc}") from exc
    if isinstance(data, dict):
        first_key = sorted(data.keys())[0]
        motion = data[first_key]
    else:
        first_key = "motion"
        motion = data
    if "pose_aa" not in motion or "fps" not in motion:
        raise RuntimeError(f"motion file {motion_file} must contain pose_aa and fps")
    num_frames = int(np.asarray(motion["pose_aa"]).shape[0])
    fps = float(motion["fps"])
    if num_frames <= 0 or fps <= 0.0:
        raise RuntimeError(f"invalid motion length in {motion_file}: frames={num_frames}, fps={fps}")
    return num_frames / fps, f"{motion_file}::{first_key} ({num_frames} frames / {fps:g} fps)"


def load_q1_joint_limits(dof_names, urdf_path, robot_yaml_path):                                    # 限位读取顺序 urdf-yaml-硬编码
    urdf_path = Path(urdf_path)
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        limits_by_name = {}
        for joint in root.findall("joint"):
            name = joint.get("name")
            limit = joint.find("limit")
            if not name or limit is None:
                continue
            if "lower" not in limit.attrib or "upper" not in limit.attrib:
                continue
            limits_by_name[name] = (float(limit.get("lower")), float(limit.get("upper")))
        missing = [name for name in dof_names if name not in limits_by_name]
        if missing:
            raise ValueError(f"missing URDF limits for joints: {missing}")
        lower = np.array([limits_by_name[name][0] for name in dof_names], dtype=np.float32)
        upper = np.array([limits_by_name[name][1] for name in dof_names], dtype=np.float32)
        if lower.shape[0] != NUM_ACTIONS or upper.shape[0] != NUM_ACTIONS:
            raise ValueError("Q1 URDF limit vector length mismatch")
        return lower, upper, str(urdf_path)
    except Exception as exc:
        print(f"[WARN] Failed to load Q1 joint limits from {urdf_path}: {exc}")

    q1_yaml = Path(robot_yaml_path)       # yaml兜底
    try:
        with open(q1_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        robot = data["robot"]
        lower = np.array(robot["dof_pos_lower_limit_list"], dtype=np.float32)
        upper = np.array(robot["dof_pos_upper_limit_list"], dtype=np.float32)
        if lower.shape[0] != NUM_ACTIONS or upper.shape[0] != NUM_ACTIONS:
            raise ValueError("Q1 yaml limit vector length mismatch")
        return lower, upper, str(q1_yaml)
    except Exception as exc:                                                
        print(f"[WARN] Failed to load Q1 joint limits from {q1_yaml}: {exc}")
        return FALLBACK_LOWER_LIMITS.copy(), FALLBACK_UPPER_LIMITS.copy(), "fallback_constants"     # 前文硬编码兜底


def create_history(cfg):
    frame_stack = cfg["frame_stack"]
    num_actions = cfg["num_actions"]
    hist_dict = {
        "actions": np.zeros((frame_stack, num_actions), dtype=np.float32),
        "base_ang_vel": np.zeros((frame_stack, 3), dtype=np.float32),
        "dof_pos": np.zeros((frame_stack, num_actions), dtype=np.float32),
        "dof_vel": np.zeros((frame_stack, num_actions), dtype=np.float32),
        "projected_gravity": np.zeros((frame_stack, 3), dtype=np.float32),
        "ref_motion_phase": np.zeros((frame_stack, 1), dtype=np.float32),
    }
    hist_obs = [hist_dict[key].reshape(1, -1) for key in HISTORY_KEYS]
    return hist_dict, np.concatenate(hist_obs, axis=1).astype(np.float32)


def update_hist_obs(hist_dict, obs_single):                 # 每一帧策略推理后更新历史
    for key in HISTORY_KEYS:
        slc = SINGLE_SLICES[key]
        arr = np.delete(hist_dict[key], -1, axis=0)         # 删掉最旧的一行（第 5 帧）
        arr = np.vstack((obs_single[0, slc], arr))          # 新帧放最前面（第 1 帧）
        hist_dict[key] = arr.astype(np.float32)
    return np.concatenate(
        [hist_dict[key].reshape(1, -1) for key in HISTORY_KEYS], axis=1
    ).astype(np.float32)


def compute_ref_motion_phase(counter, cfg):                         # 用时间反推相位，策略跟上
    sim_time = cfg["time_offset"] + (counter + 1) * cfg["simulation_dt"]
    if cfg["phase_wrap"]:
        return float((sim_time % cfg["cycle_time"]) / cfg["cycle_time"])
    phase_time = min(sim_time, cfg["cycle_time"] - 1e-6) if cfg["stop_at_motion_end"] else sim_time
    return float(np.clip(phase_time / cfg["cycle_time"], 0.0, 1.0))


def get_obs(hist_obs_c, hist_dict, state, action, counter, cfg):
    hist_len = cfg["frame_stack"] * cfg["num_single_obs"]
    ref_motion_phase = compute_ref_motion_phase(counter, cfg)       # 参考相位输入到obs
    # 构建单帧观测 (73 维)
    obs_single = np.zeros([1, cfg["num_single_obs"]], dtype=np.float32)
    obs_single[0, 0:22] = action
    obs_single[0, 22:25] = state["base_ang_vel"] * cfg["obs_scale_base_ang_vel"]
    obs_single[0, 25:47] = (state["dof_pos"] - cfg["default_dof_pos"]) * cfg["obs_scale_dof_pos"]
    obs_single[0, 47:69] = state["dof_vel"] * cfg["obs_scale_dof_vel"]
    obs_single[0, 69:72] = state["projected_gravity"] * cfg["obs_scale_gvec"]
    obs_single[0, 72] = ref_motion_phase * cfg["obs_scale_refmotion"]
    #拼接"当前帧部分历史 + 当前帧"
    obs_all = np.zeros([1, (cfg["frame_stack"] + 1) * cfg["num_single_obs"]], dtype=np.float32)     # 4+1 × 73 = 365 维
    obs_all[0, 0:22] = obs_single[0, 0:22]
    obs_all[0, 22:25] = obs_single[0, 22:25]
    obs_all[0, 25:47] = obs_single[0, 25:47]
    obs_all[0, 47:69] = obs_single[0, 47:69]
    obs_all[0, 69:69 + hist_len] = hist_obs_c[0] * cfg["obs_scale_hist"]        # 历史帧 × 放大
    obs_all[0, 69 + hist_len:69 + hist_len + 3] = obs_single[0, 69:72]
    obs_all[0, 69 + hist_len + 3] = obs_single[0, 72]
    hist_obs_new = update_hist_obs(hist_dict, obs_single)
    obs_all = np.clip(obs_all, -cfg["clip_observations"], cfg["clip_observations"])
    return obs_all.astype(np.float32), hist_obs_new


def load_onnx_policy(policy_path, expected_obs_dim, expected_act_dim):
    if ort is None:
        raise RuntimeError("onnxruntime is not available; install it or run without --enable-policy")
    if not os.path.isfile(policy_path):
        raise FileNotFoundError(f"ONNX policy not found: {policy_path}")
    session = ort.InferenceSession(policy_path)
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    if inp.shape[-1] != expected_obs_dim:
        raise ValueError(f"ONNX input dim {inp.shape[-1]} != expected obs dim {expected_obs_dim}")
    if out.shape[-1] != expected_act_dim:
        raise ValueError(f"ONNX output dim {out.shape[-1]} != expected action dim {expected_act_dim}")
    dummy = np.zeros((1, expected_obs_dim), dtype=np.float32)
    result = session.run([out.name], {inp.name: dummy})[0]
    if result.shape != (1, expected_act_dim) or not np.isfinite(result).all():
        raise RuntimeError(f"ONNX dummy inference failed: shape={result.shape}")
    return {"session": session, "input_name": inp.name, "output_name": out.name}


def infer_policy_action(policy, obs_buff, clip_actions):
    raw_action = policy["session"].run(
        [policy["output_name"]], {policy["input_name"]: obs_buff}
    )[0]
    raw_action = np.asarray(raw_action).reshape(-1).astype(np.float32)
    if not np.isfinite(raw_action).all():
        raise RuntimeError("Policy returned nonfinite action")
    return np.clip(raw_action, -clip_actions, clip_actions)


def apply_action_postprocess(raw_action, prev_action, prev_target_dof_pos, cfg, policy_dt):
    alpha = cfg["action_filter_alpha"]
    action = alpha * raw_action + (1.0 - alpha) * prev_action
    target_from_action = action * cfg["action_scale"] + cfg["default_dof_pos"]
    if cfg["target_pos_rate_limit"] > 0:
        max_delta = cfg["target_pos_rate_limit"] * policy_dt
        target_dof_pos = np.clip(target_from_action, prev_target_dof_pos - max_delta, prev_target_dof_pos + max_delta)
    else:
        target_dof_pos = target_from_action
    return action.astype(np.float32), target_dof_pos.astype(np.float32)


def quat_rotate_inverse_wxyz(q, v):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    v = np.asarray(v, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return v.copy()
    q = q / norm
    qw = q[0]
    qvec = q[1:4]
    return (
        v * (2.0 * qw * qw - 1.0)
        - np.cross(qvec, v) * qw * 2.0
        + qvec * np.dot(qvec, v) * 2.0
    )


def copy_if_present(dst, src, name):
    if src is not None and hasattr(dst, name) and hasattr(src, name):
        setattr(dst, name, getattr(src, name))


def optional_float_attr(obj, names):
    for name in names:
        if hasattr(obj, name):
            return float(getattr(obj, name))
    return None


class KeyboardReader:
    def __init__(self, callback):
        self.callback = callback
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.old_settings = None

    def start(self):
        if not sys.stdin.isatty():
            print("[WARN] stdin is not a TTY; keyboard controls disabled")
            return
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except termios.error:
                pass

    def _run(self):
        import select

        self.old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not self.stop_event.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    ch = sys.stdin.read(1)
                    if ch:
                        self.callback(ch)
        finally:
            self.stop()


class Q1MotionTrackingNode(Node):
    def __init__(self, args, cfg, limits):
        super().__init__("q1_motion_tracking_sim2real")
        self.args = args
        self.cfg = cfg
        self.lower_limits, self.upper_limits, self.limit_source = limits
        self.dof_names = cfg["dof_names"]
        self.policy_dt = cfg["simulation_dt"] * cfg["control_decimation"]

        self.state = ControlState.WAIT_STATE
        self.latest_joint_msg_time = 0.0
        self.latest_base_msg_time = 0.0
        self.latest_joint_state_msg = None
        self.latest_state_by_name: Dict[str, object] = {}
        self.policy_joint_index = {name: i for i, name in enumerate(self.dof_names)}
        self.latest_dof_pos = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.latest_dof_vel = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.latest_target_dof_pos = cfg["default_dof_pos"].copy()
        self.command_target = self.latest_target_dof_pos.copy()
        self.last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.policy = None
        self.main_policy = None
        self.stand_policy = None
        self.main_cfg = cfg.copy()
        self.stand_cfg = cfg.copy()
        if "stand_cycle_time" in cfg:
            self.stand_cfg["cycle_time"] = float(cfg["stand_cycle_time"])
            self.stand_cfg["phase_wrap"] = True
            self.stand_cfg["stop_at_motion_end"] = False
            self.stand_cfg["simulation_duration"] = max(float(self.stand_cfg.get("simulation_duration", 0.0)), self.stand_cfg["cycle_time"])
        self.active_policy_name = "main"
        self.hist_dict, self.hist_obs_c = create_history(cfg)
        self.policy_counter = 0
        self.get_ready_start_time = 0.0
        self.get_ready_start_pos = None
        self.clamp_count_limits = 0
        self.clamp_count_step = 0
        self.command_publish_count = 0
        self.simulated_command_count = 0
        self.policy_count = 0
        self.last_action_max = 0.0
        self.last_target_delta_max = 0.0
        self.last_command_step_max = 0.0
        self.max_abs_dof_vel_seen = 0.0
        self.max_abs_base_ang_vel_seen = 0.0
        self.imu_adapter_mode = "fake_base_state" if args.fake_base_state else "none"
        self.motors_enabled = self.compute_motor_enable_gate(log=False)
        self.last_error = ""

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.command_pub = self.create_publisher(JointCommandArray, args.joint_command_topic, qos)
        self.state_sub = self.create_subscription(JointStateArray, args.joint_state_topic, self.on_joint_state, qos)
        self.imu_sub = None
        self.odom_sub = None
        if args.imu_topic:
            imu_msg_type = args.imu_msg_type or "sensor_msgs/msg/Imu"
            if imu_msg_type == "sensor_msgs/msg/Imu":
                if Imu is None:
                    raise RuntimeError("sensor_msgs/Imu is unavailable, cannot use --imu-topic")
                self.imu_adapter_mode = "sensor_msgs/msg/Imu"
                self.imu_sub = self.create_subscription(Imu, args.imu_topic, self.on_imu, qos)
            else:
                if get_message is None:
                    raise RuntimeError("rosidl_runtime_py is unavailable, cannot load --imu-msg-type")
                imu_cls = get_message(imu_msg_type)
                self.imu_adapter_mode = f"unsupported_custom:{imu_msg_type}"
                self.imu_sub = self.create_subscription(imu_cls, args.imu_topic, self.on_custom_imu, qos)
        if args.odom_topic:
            if Odometry is None:
                raise RuntimeError("nav_msgs/Odometry is unavailable, cannot use --odom-topic")
            self.odom_sub = self.create_subscription(Odometry, args.odom_topic, self.on_odom, qos)

        if args.enable_policy:
            expected_obs_dim = (cfg["frame_stack"] + 1) * cfg["num_single_obs"]
            self.main_policy = load_onnx_policy(args.policy_path, expected_obs_dim, cfg["num_actions"])
            self.policy = self.main_policy
            if args.stand_policy_path:
                self.stand_policy = load_onnx_policy(args.stand_policy_path, expected_obs_dim, cfg["num_actions"])
        self.motors_enabled = self.compute_motor_enable_gate(log=False)

        self.create_timer(args.command_period_s, self.on_command_timer)
        self.create_timer(self.policy_dt, self.on_policy_timer)
        self.create_timer(0.05, self.on_safety_timer)
        self.create_timer(args.print_rate_s, self.on_print_timer)

        self.keyboard = KeyboardReader(self.on_key)
        if not args.no_keyboard:
            self.keyboard.start()
        self.log_startup()

    def compute_motor_enable_gate(self, log=True):
        reasons = []
        if not self.args.enable_motors:
            reasons.append("--enable-motors not set")
        if self.args.risk_confirm != "I_UNDERSTAND_REAL_ROBOT_RISK":
            reasons.append("risk token missing")
        if self.args.fake_base_state:
            reasons.append("fake base state")
        if not self.base_state_ready():
            reasons.append("base state not ready")
        if self.latest_joint_msg_time <= 0.0:
            reasons.append("joint state not ready")
        if self.args.enable_policy and self.policy is None:
            reasons.append("policy not loaded")
        enabled = len(reasons) == 0
        if log and reasons:
            self.get_logger().info("real joint command disabled: " + ", ".join(reasons))
        return enabled

    def dry_run_active(self):
        return self.args.dry_run or not self.motors_enabled

    def log_startup(self):
        self.get_logger().warn(
            "This script commands Q1 joints only when --enable-motors and the risk token are set. "
            "Ensure low-level developer mode is active and high-level motion control is not conflicting."
        )
        if self.args.fake_base_state:
            self.get_logger().error(
                "FAKE BASE STATE ENABLED: base_ang_vel=0 and projected_gravity=[0,0,-1]. "
                "This is ONLY for suspended/bench validation and is forbidden for standing motion tracking."
            )
        self.get_logger().info(
            f"policy_dt={self.policy_dt:.4f}s ({1.0 / self.policy_dt:.1f}Hz), "
            f"command_period_s={self.args.command_period_s:.4f}s "
            f"({1.0 / self.args.command_period_s:.1f}Hz), dry_run_active={self.dry_run_active()}"
        )
        self.get_logger().info(
            f"joint_state_topic={self.args.joint_state_topic}, "
            f"joint_command_topic={self.args.joint_command_topic}, imu_topic={self.args.imu_topic}, "
            f"imu_adapter={self.imu_adapter_mode}, limits={self.limit_source}"
        )
        self.get_logger().info("Keys: i=GET_READY, a=ARM, p=POLICY, o=STOP/HOLD, q=EMERGENCY_DAMPING")

    def on_joint_state(self, msg):
        self.latest_joint_msg_time = time.monotonic()
        self.latest_joint_state_msg = msg
        self.latest_state_by_name = {joint.name: joint for joint in msg.joints}
        ok, dof_pos, dof_vel = self.extract_joint_state()
        if ok:
            self.latest_dof_pos = dof_pos
            self.latest_dof_vel = dof_vel
            self.max_abs_dof_vel_seen = max(self.max_abs_dof_vel_seen, float(np.max(np.abs(dof_vel))))
            if not self.check_joint_state_safety(msg, dof_pos, dof_vel):
                return
            if self.state == ControlState.WAIT_STATE and self.base_state_ready():
                self.latest_target_dof_pos = self.latest_dof_pos.copy()
                self.command_target = self.latest_dof_pos.copy()
                self.transition(ControlState.HOLD_CURRENT, "joint/base state ready")
            self.motors_enabled = self.compute_motor_enable_gate(log=False)

    def on_imu(self, msg):
        q = msg.orientation
        quat_wxyz = np.array([q.w, q.x, q.y, q.z], dtype=np.float64)
        base_ang_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], dtype=np.float32)
        projected_gravity = quat_rotate_inverse_wxyz(quat_wxyz, [0.0, 0.0, -1.0]).astype(np.float32)
        self.accept_base_state(base_ang_vel, projected_gravity, "imu")

    def on_custom_imu(self, _msg):
        # Custom IMU adapters must explicitly map the SDK message into base_ang_vel,
        # projected_gravity, and latest_base_msg_time after inspecting the real .msg fields.
        self.enter_error(
            "unsupported custom IMU message type. Run: ros2 topic type "
            f"{self.args.imu_topic}; then: ros2 interface show <returned_type>; "
            "add an explicit custom IMU adapter that sets base_ang_vel, projected_gravity, "
            "and latest_base_msg_time. This script does not guess custom IMU fields."
        )

    def on_odom(self, msg):
        q = msg.pose.pose.orientation
        quat_wxyz = np.array([q.w, q.x, q.y, q.z], dtype=np.float64)
        projected_gravity = quat_rotate_inverse_wxyz(quat_wxyz, [0.0, 0.0, -1.0]).astype(np.float32)
        base_ang_vel = np.array([
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ], dtype=np.float32)
        self.accept_base_state(base_ang_vel, projected_gravity, "odom")

    def accept_base_state(self, base_ang_vel, projected_gravity, source):
        base_ang_vel = np.asarray(base_ang_vel, dtype=np.float32).reshape(3)
        projected_gravity = np.asarray(projected_gravity, dtype=np.float32).reshape(3)
        if not np.isfinite(base_ang_vel).all() or not np.isfinite(projected_gravity).all():
            self.enter_error(f"nonfinite {source} base state")
            return
        max_abs = float(np.max(np.abs(base_ang_vel)))
        self.max_abs_base_ang_vel_seen = max(self.max_abs_base_ang_vel_seen, max_abs)
        if max_abs > self.args.max_base_ang_vel:
            self.enter_error(
                f"base angular velocity {max_abs:.3f} exceeds --max-base-ang-vel {self.args.max_base_ang_vel:.3f}"
            )
            return
        self.base_ang_vel = np.clip(base_ang_vel, -self.args.max_base_ang_vel, self.args.max_base_ang_vel).astype(np.float32)
        self.projected_gravity = projected_gravity.astype(np.float32)
        self.latest_base_msg_time = time.monotonic()
        self.motors_enabled = self.compute_motor_enable_gate(log=False)

    def extract_joint_state(self) -> Tuple[bool, np.ndarray, np.ndarray]:
        missing = [name for name in self.dof_names if name not in self.latest_state_by_name]
        if missing:
            if self.state != ControlState.WAIT_STATE:
                self.enter_error(f"missing joints in state: {missing}")
            return False, self.latest_dof_pos, self.latest_dof_vel
        dof_pos = np.array([self.latest_state_by_name[name].position for name in self.dof_names], dtype=np.float32)
        dof_vel = np.array([self.latest_state_by_name[name].velocity for name in self.dof_names], dtype=np.float32)
        if not np.isfinite(dof_pos).all() or not np.isfinite(dof_vel).all():
            self.enter_error("nonfinite joint state")
            return False, self.latest_dof_pos, self.latest_dof_vel
        return True, dof_pos, dof_vel

    def check_joint_state_safety(self, msg, dof_pos, dof_vel):
        del dof_pos
        max_vel = float(np.max(np.abs(dof_vel)))
        if max_vel > self.args.max_dof_vel:
            self.enter_error(f"dof velocity {max_vel:.3f} exceeds --max-dof-vel {self.args.max_dof_vel:.3f}")
            return False
        for joint in msg.joints:
            motor_temp = optional_float_attr(joint, ["motor_temp", "motor_temperature", "temperature"])
            coil_temp = optional_float_attr(joint, ["coil_temp", "coil_temperature"])
            if motor_temp is not None and motor_temp > self.args.max_motor_temp_c:
                self.enter_error(
                    f"joint {joint.name} motor temp {motor_temp:.1f}C exceeds {self.args.max_motor_temp_c:.1f}C"
                )
                return False
            if coil_temp is not None and coil_temp > self.args.max_coil_temp_c:
                self.enter_error(
                    f"joint {joint.name} coil temp {coil_temp:.1f}C exceeds {self.args.max_coil_temp_c:.1f}C"
                )
                return False
        return True

    def base_state_ready(self):
        if self.args.fake_base_state:
            return True
        return self.latest_base_msg_time > 0.0

    def current_state_ready(self):
        return self.latest_joint_msg_time > 0.0 and self.base_state_ready()

    def state_age_ok(self):
        now = time.monotonic()
        if self.latest_joint_msg_time <= 0.0 or now - self.latest_joint_msg_time > self.args.state_timeout_s:
            return False
        if not self.args.fake_base_state:
            if self.latest_base_msg_time <= 0.0 or now - self.latest_base_msg_time > self.args.state_timeout_s:
                return False
        return True

    def transition(self, new_state, reason):
        if self.state == new_state:
            return
        self.get_logger().info(f"STATE {self.state.value} -> {new_state.value}: {reason}")
        self.state = new_state
        if new_state == ControlState.GET_READY:
            self.get_ready_start_time = time.monotonic()
            self.get_ready_start_pos = self.latest_dof_pos.copy()
        elif new_state == ControlState.POLICY:
            self.reset_policy_runtime()
        elif new_state == ControlState.STOP:
            self.command_target = self.latest_target_dof_pos.copy()
        elif new_state in (ControlState.EMERGENCY_DAMPING, ControlState.ERROR):
            self.command_target = self.latest_dof_pos.copy()

    def reset_policy_runtime(self):
        self.policy_counter = 0
        self.last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.hist_dict, self.hist_obs_c = create_history(self.cfg)
        self.latest_target_dof_pos = self.latest_dof_pos.copy()
        self.command_target = self.latest_target_dof_pos.copy()
        self.last_action_max = 0.0
        self.get_logger().info(f"Policy runtime reset for {self.active_policy_name}")

    def switch_to_stand_policy(self):
        if self.stand_policy is None:
            self.transition(ControlState.STOP, "motion phase reached end")
            return
        self.policy = self.stand_policy
        self.cfg = self.stand_cfg
        self.active_policy_name = "stand"
        self.reset_policy_runtime()
        self.get_logger().warn("Main motion ended; switched to looping stand policy")

    def enter_emergency(self, reason):
        self.last_error = reason
        self.get_logger().error(f"EMERGENCY_DAMPING: {reason}")
        self.last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.latest_target_dof_pos = self.latest_dof_pos.copy()
        self.command_target = self.latest_dof_pos.copy()
        self.transition(ControlState.EMERGENCY_DAMPING, reason)

    def enter_error(self, reason):
        self.enter_emergency(reason)
        self.transition(ControlState.ERROR, reason)

    def on_key(self, ch):
        if ch == "i":
            if self.current_state_ready():
                self.transition(ControlState.GET_READY, "keyboard i")
            else:
                self.get_logger().warn("Cannot GET_READY before joint/base state is ready")
        elif ch == "a":
            if not self.args.enable_policy:
                self.get_logger().warn("Policy is disabled; restart with --enable-policy to arm")
            elif self.current_state_ready():
                self.transition(ControlState.POLICY_ARMED, "keyboard a")
            else:
                self.get_logger().warn("Cannot arm before joint/base state is ready")
        elif ch == "p":
            if not self.args.enable_policy:
                self.get_logger().warn("Policy is disabled; restart with --enable-policy")
            elif self.state == ControlState.POLICY_ARMED:
                self.transition(ControlState.POLICY, "keyboard p")
            else:
                self.get_logger().warn("Press 'a' to enter POLICY_ARMED before 'p'")
        elif ch == "o":
            self.transition(ControlState.STOP, "keyboard o")
        elif ch == "q" or ch == "\x03":
            self.enter_emergency("keyboard emergency damping")

    def on_safety_timer(self):
        if self.state == ControlState.WAIT_STATE:
            return
        if not self.state_age_ok():
            self.enter_error("state timeout")
            return
        if not np.isfinite(self.command_target).all():
            self.enter_error("nonfinite command target")
            return
        if not np.isfinite(self.base_ang_vel).all() or not np.isfinite(self.projected_gravity).all():
            self.enter_error("nonfinite base state")
            return
        max_base = float(np.max(np.abs(self.base_ang_vel)))
        if max_base > self.args.max_base_ang_vel:
            self.enter_error(f"base angular velocity {max_base:.3f} exceeds limit")
            return
        self.motors_enabled = self.compute_motor_enable_gate(log=False)

    def on_policy_timer(self):
        if self.state == ControlState.GET_READY:
            self.update_get_ready_target()
            return
        if self.state != ControlState.POLICY:
            return
        if not self.state_age_ok():
            self.enter_error("state timeout before policy inference")
            return
        try:
            state = {
                "dof_pos": self.latest_dof_pos.copy(),
                "dof_vel": self.latest_dof_vel.copy(),
                "base_ang_vel": self.base_ang_vel.copy(),
                "projected_gravity": self.projected_gravity.copy(),
            }
            if not all(np.isfinite(v).all() for v in state.values()):
                raise RuntimeError("nonfinite input state")
            obs, self.hist_obs_c = get_obs(self.hist_obs_c, self.hist_dict, state, self.last_action, self.policy_counter, self.cfg)
            if not np.isfinite(obs).all():
                raise RuntimeError("nonfinite observation")
            raw_action = infer_policy_action(self.policy, obs, self.cfg["clip_actions"])
            action, target = apply_action_postprocess(raw_action, self.last_action, self.latest_target_dof_pos, self.cfg, self.policy_dt)
            target = self.apply_target_safety(target)
            self.last_action = action
            self.last_target_delta_max = float(np.max(np.abs(target - self.latest_target_dof_pos)))
            self.latest_target_dof_pos = target
            self.last_action_max = float(np.max(np.abs(raw_action)))
            self.policy_counter += self.cfg["control_decimation"]
            self.policy_count += 1
            if self.cfg["stop_at_motion_end"] and not self.cfg["phase_wrap"]:
                elapsed = self.policy_counter * self.cfg["simulation_dt"]
                if elapsed >= self.cfg["cycle_time"]:
                    if self.active_policy_name == "stand":
                        self.policy_counter = 0
                    else:
                        self.switch_to_stand_policy()
        except Exception as exc:
            self.enter_error(f"policy update failed: {exc}")

    def update_get_ready_target(self):
        if self.get_ready_start_pos is None:
            self.get_ready_start_pos = self.latest_dof_pos.copy()
        elapsed = time.monotonic() - self.get_ready_start_time
        alpha = float(np.clip(elapsed / max(self.args.get_ready_duration_s, 1e-6), 0.0, 1.0))
        target = (1.0 - alpha) * self.get_ready_start_pos + alpha * self.cfg["default_dof_pos"]
        target = self.apply_target_safety(target.astype(np.float32), step_reference=self.command_target)
        self.last_target_delta_max = float(np.max(np.abs(target - self.latest_target_dof_pos)))
        self.latest_target_dof_pos = target
        if alpha >= 1.0:
            self.transition(ControlState.HOLD_CURRENT, "get-ready complete")

    def apply_target_safety(self, target, step_reference=None):
        target = np.asarray(target, dtype=np.float32).copy()
        if not np.isfinite(target).all():
            raise RuntimeError("nonfinite target")
        if self.args.max_target_step_rad > 0:
            ref = self.latest_target_dof_pos if step_reference is None else step_reference
            before = target.copy()
            target = np.clip(target, ref - self.args.max_target_step_rad, ref + self.args.max_target_step_rad)
            if not np.allclose(before, target):
                self.clamp_count_step += 1
        before_limits = target.copy()
        margin = float(self.args.joint_limit_margin_rad)
        target = np.clip(target, self.lower_limits + margin, self.upper_limits - margin)
        if not np.allclose(before_limits, target):
            self.clamp_count_limits += 1
        return target.astype(np.float32)

    def clamp_command_target_to_limits(self, target):
        margin = float(self.args.joint_limit_margin_rad)
        before = np.asarray(target, dtype=np.float32)
        clamped = np.clip(before, self.lower_limits + margin, self.upper_limits - margin)
        if not np.allclose(before, clamped):
            self.clamp_count_limits += 1
        return clamped.astype(np.float32)

    def update_command_ramp(self):
        desired = self.clamp_command_target_to_limits(self.latest_target_dof_pos)
        before = self.command_target.copy()
        max_step = float(self.args.max_command_step_rad)
        if max_step > 0.0:
            self.command_target = np.clip(desired, before - max_step, before + max_step).astype(np.float32)
        else:
            self.command_target = desired.copy()
        self.command_target = self.clamp_command_target_to_limits(self.command_target)
        self.last_command_step_max = float(np.max(np.abs(self.command_target - before)))

    def make_command_msg(self, damping_frame=False):
        if self.latest_joint_state_msg is None:
            return None
        cmd = JointCommandArray()
        latest_header = getattr(self.latest_joint_state_msg, "header", None)
        if hasattr(cmd, "header"):
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.header.sequence = self.command_publish_count + self.simulated_command_count
            copy_if_present(cmd.header, latest_header, "frame_id")
        copy_if_present(cmd, self.latest_joint_state_msg, "meas_stamp")
        emergency = damping_frame or self.state in (ControlState.EMERGENCY_DAMPING, ControlState.ERROR)
        safe_command_target = self.clamp_command_target_to_limits(self.command_target)
        for state_joint in self.latest_joint_state_msg.joints:
            joint = JointCommand()
            joint.name = state_joint.name
            idx = self.policy_joint_index.get(state_joint.name)
            if emergency:
                joint.position = float(state_joint.position)
                joint.stiffness = 0.0
                joint.damping = float(self.args.emergency_damping)
            elif idx is not None:
                joint.position = float(safe_command_target[idx])
                joint.stiffness = float(self.cfg["kps"][idx])
                joint.damping = float(self.cfg["kds"][idx])
            else:
                joint.position = float(state_joint.position)
                joint.stiffness = float(self.args.hold_uncontrolled_stiffness)
                joint.damping = float(self.args.hold_uncontrolled_damping)
            joint.velocity = 0.0
            joint.effort = 0.0
            cmd.joints.append(joint)
        return cmd

    def publish_command_once(self, damping_frame=False):
        cmd = self.make_command_msg(damping_frame=damping_frame)
        if cmd is None:
            return
        if self.dry_run_active():
            self.simulated_command_count += 1
            return
        self.command_pub.publish(cmd)
        self.command_publish_count += 1

    def on_command_timer(self):
        if self.state == ControlState.WAIT_STATE:
            return
        if self.state not in (ControlState.EMERGENCY_DAMPING, ControlState.ERROR):
            self.update_command_ramp()
        self.publish_command_once()

    def on_print_timer(self):
        now = time.monotonic()
        joint_age = now - self.latest_joint_msg_time if self.latest_joint_msg_time > 0 else float("inf")
        base_age = 0.0 if self.args.fake_base_state else (
            now - self.latest_base_msg_time if self.latest_base_msg_time > 0 else float("inf")
        )
        phase = compute_ref_motion_phase(self.policy_counter, self.cfg)
        max_abs_dof_vel = float(np.max(np.abs(self.latest_dof_vel)))
        max_abs_base_ang_vel = float(np.max(np.abs(self.base_ang_vel)))
        self.get_logger().info(
            f"state={self.state.value} active_policy={self.active_policy_name} phase={phase:.3f} dry_run_active={self.dry_run_active()} "
            f"joint_age={joint_age:.3f}s base_age={base_age:.3f}s "
            f"policy_count={self.policy_count} command_count={self.command_publish_count} "
            f"simulated_command_count={self.simulated_command_count} "
            f"action_max={self.last_action_max:.3f} target_delta_max={self.last_target_delta_max:.3f} "
            f"command_step_max={self.last_command_step_max:.3f} "
            f"clamp_step={self.clamp_count_step} clamp_limit={self.clamp_count_limits} "
            f"max_abs_dof_vel={max_abs_dof_vel:.3f} max_abs_base_ang_vel={max_abs_base_ang_vel:.3f} "
            f"projected_gravity={np.array2string(self.projected_gravity, precision=3)} "
            f"last_error={self.last_error}"
        )
        if max_abs_base_ang_vel < 0.3 and np.linalg.norm(self.projected_gravity - np.array([0.0, 0.0, -1.0])) > 0.35:
            self.get_logger().warn(
                "projected_gravity is not close to [0, 0, -1] while base angular velocity is small; "
                "check IMU coordinate frame or quaternion order"
            )

    def publish_emergency_frames(self, frames=10):
        if self.state != ControlState.EMERGENCY_DAMPING:
            self.enter_emergency("shutdown")
        for _ in range(frames):
            self.publish_command_once(damping_frame=True)
            time.sleep(max(float(self.args.command_period_s), 0.001))

    def shutdown(self):
        self.publish_emergency_frames(10)
        self.keyboard.stop()


def resolve_path(path):
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(_REPO_ROOT, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Q1 ROS2 sim2real motion-tracking bridge")
    parser.add_argument("--config", type=str, default="config/q1_sim2real_base.yaml")
    parser.add_argument("--policy-path", type=str, default="policies/lateral_raise/model_6000.onnx")
    parser.add_argument("--motion-file", type=str, default="motions/lateral_raise/lq1_ateral_raise.pkl")
    parser.add_argument("--stand-policy-path", type=str, default="policies/stand/model_5000.onnx")
    parser.add_argument("--stand-motion-file", type=str, default="motions/stand/q1_stand_still.pkl")
    parser.add_argument("--cycle-time", type=float, default=0.0)
    parser.add_argument("--urdf", type=str, default="robots/q1/q1_22dof_box.urdf")
    parser.add_argument("--robot-yaml", type=str, default="robots/q1/q1_22dof.yaml")
    parser.add_argument("--enable-policy", action="store_true")
    parser.add_argument("--joint-state-topic", type=str, default="/aima/hal/joint/state")
    parser.add_argument("--joint-command-topic", type=str, default="/aima/hal/joint/command")
    parser.add_argument("--imu-topic", type=str, default="/aima/hal/imu/state")
    parser.add_argument("--imu-msg-type", type=str, default="sensor_msgs/msg/Imu")
    parser.add_argument("--odom-topic", type=str, default="")
    parser.add_argument("--fake-base-state", action="store_true")
    parser.add_argument("--ack-fake-base-bench-only", action="store_true")
    parser.add_argument("--command-period-s", type=float, default=0.002)
    parser.add_argument("--max-command-step-rad", type=float, default=0.01)
    parser.add_argument("--emergency-damping", type=float, default=5.0)
    parser.add_argument("--hold-uncontrolled-stiffness", type=float, default=0.0)
    parser.add_argument("--hold-uncontrolled-damping", type=float, default=5.0)
    parser.add_argument("--state-timeout-s", type=float, default=0.2)
    parser.add_argument("--get-ready-duration-s", type=float, default=3.0)
    parser.add_argument("--max-target-step-rad", type=float, default=0.08)
    parser.add_argument("--joint-limit-margin-rad", type=float, default=0.02)
    parser.add_argument("--max-base-ang-vel", type=float, default=10.0)
    parser.add_argument("--max-dof-vel", type=float, default=30.0)
    parser.add_argument("--max-motor-temp-c", type=float, default=80.0)
    parser.add_argument("--max-coil-temp-c", type=float, default=80.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enable-motors", action="store_true")
    parser.add_argument("--risk-confirm", type=str, default="")
    parser.add_argument("--print-rate-s", type=float, default=1.0)
    parser.add_argument("--no-keyboard", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.command_period_s <= 0.0:
        raise ValueError("--command-period-s must be > 0")
    if args.command_period_s > 0.01:
        raise ValueError("--command-period-s must be <= 0.01 (>=100Hz); default 0.002 is 500Hz")
    if args.max_command_step_rad <= 0.0:
        raise ValueError("--max-command-step-rad must be > 0")
    if args.emergency_damping < 0.0:
        raise ValueError("--emergency-damping must be >= 0")
    if args.state_timeout_s <= 0.0:
        raise ValueError("--state-timeout-s must be > 0")
    if args.max_base_ang_vel <= 0.0 or args.max_dof_vel <= 0.0:
        raise ValueError("--max-base-ang-vel and --max-dof-vel must be > 0")
    if args.enable_policy and not args.policy_path:
        raise ValueError("--policy-path is required with --enable-policy")
    if bool(args.stand_policy_path) != bool(args.stand_motion_file):
        raise ValueError("--stand-policy-path and --stand-motion-file must be provided together")
    if args.fake_base_state and args.enable_policy and not args.ack_fake_base_bench_only:
        raise ValueError(
            "--fake-base-state with --enable-policy requires --ack-fake-base-bench-only. "
            "This is only allowed for suspended/bench tests, not standing motion tracking."
        )
    if args.enable_motors:
        if args.risk_confirm != "I_UNDERSTAND_REAL_ROBOT_RISK":
            raise ValueError("--enable-motors requires --risk-confirm I_UNDERSTAND_REAL_ROBOT_RISK")
        if args.fake_base_state:
            raise ValueError("--enable-motors is forbidden with --fake-base-state")
        if not args.enable_policy:
            raise ValueError("--enable-motors requires --enable-policy and a checked ONNX policy")
    if args.imu_msg_type != "sensor_msgs/msg/Imu":
        print(
            "WARNING: custom IMU message type requested. This script will not guess fields; "
            f"run 'ros2 topic type {args.imu_topic}' and 'ros2 interface show <returned_type>', "
            "then implement an explicit adapter callback."
        )
    if not args.fake_base_state and not args.imu_topic and not args.odom_topic:
        raise ValueError(
            "No base-state source configured. Provide --imu-topic/--odom-topic, or use "
            "--fake-base-state for suspended/bench validation only."
        )


def main():
    args = parse_args()
    args.config = resolve_path(args.config)
    args.policy_path = resolve_path(args.policy_path)
    args.motion_file = resolve_path(args.motion_file)
    args.stand_policy_path = resolve_path(args.stand_policy_path)
    args.stand_motion_file = resolve_path(args.stand_motion_file)
    args.urdf = resolve_path(args.urdf)
    args.robot_yaml = resolve_path(args.robot_yaml)
    validate_args(args)
    if rclpy is None or JointCommandArray is None:
        raise RuntimeError(
            "ROS2/Q1 SDK Python packages are unavailable. Source the SDK workspace first, "
            "for example: source /root/autodl-tmp/q1_sim2real文档/202606091310/install/setup.bash"
        )

    cfg = read_conf(args.config)
    if args.cycle_time > 0.0:
        cfg["cycle_time"] = float(args.cycle_time)
        cycle_time_source = "--cycle-time"
    else:
        inferred_cycle_time, motion_desc = infer_cycle_time_from_motion(args.motion_file)
        if inferred_cycle_time is not None:
            cfg["cycle_time"] = float(inferred_cycle_time)
            cycle_time_source = motion_desc
        else:
            cycle_time_source = "config"
    if cfg["cycle_time"] <= 0.0:
        raise ValueError("cycle_time must be provided by --cycle-time, --motion-file, or config")
    cfg["simulation_duration"] = float(max(cfg.get("simulation_duration", 0.0), cfg["cycle_time"]))
    stand_cycle_source = "disabled"
    if args.stand_motion_file:
        stand_cycle_time, stand_cycle_source = infer_cycle_time_from_motion(args.stand_motion_file)
        cfg["stand_cycle_time"] = float(stand_cycle_time)
    if cfg["num_actions"] != NUM_ACTIONS or len(cfg["dof_names"]) != NUM_ACTIONS:
        raise ValueError("Q1 config must contain 22 actions and 22 joint names")
    limits = load_q1_joint_limits(cfg["dof_names"], args.urdf, args.robot_yaml)

    print("=" * 72)
    print("Q1 Motion Tracking Sim2Real ROS2")
    print("=" * 72)
    print(f"config={args.config}")
    print(f"policy_path={args.policy_path}")
    print(f"motion_file={args.motion_file}")
    print(f"cycle_time={cfg['cycle_time']:.6f}s source={cycle_time_source}")
    print(f"stand_policy_path={args.stand_policy_path}")
    print(f"stand_motion_file={args.stand_motion_file}")
    if args.stand_motion_file:
        print(f"stand_cycle_time={cfg['stand_cycle_time']:.6f}s source={stand_cycle_source}")
    print(f"urdf={args.urdf}")
    print(f"robot_yaml={args.robot_yaml}")
    print(f"joint_state_topic={args.joint_state_topic}")
    print(f"joint_command_topic={args.joint_command_topic}")
    print(f"imu_topic={args.imu_topic}")
    print(f"imu_msg_type={args.imu_msg_type} adapter_mode={'sensor_msgs/msg/Imu' if args.imu_msg_type == 'sensor_msgs/msg/Imu' else 'unsupported_custom'}")
    print(f"enable_policy={args.enable_policy} enable_motors={args.enable_motors} dry_run_active={not args.enable_motors or args.dry_run}")
    print(f"fake_base_state={args.fake_base_state}")
    if args.fake_base_state:
        print("WARNING: fake-base-state is bench/suspended only; do not use for standing motion tracking.")
    print()

    rclpy.init()
    node = Q1MotionTrackingNode(args, cfg, limits)

    def handle_sigint(signum, _frame):
        node.get_logger().warn(f"Received signal {signum}; entering EMERGENCY_DAMPING")
        node.enter_emergency("SIGINT/SIGTERM")
        node.publish_emergency_frames(10)
        rclpy.shutdown()

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    try:
        rclpy.spin(node)
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
