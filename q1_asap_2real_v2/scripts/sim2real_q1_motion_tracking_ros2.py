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
import importlib.util
import os
import pickle
import signal
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import yaml

try:
    import termios
    import tty
except ImportError:
    termios = None
    tty = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None
try:
    import mujoco
except ImportError:
    mujoco = None
try:
    import mujoco.viewer
except ImportError:
    pass

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
LOWER_BODY_INDICES = np.arange(0, 12, dtype=np.int64)
UPPER_BODY_INDICES = np.arange(12, 22, dtype=np.int64)
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
    WAIT_STATE = "WAIT_STATE"
    HOLD_CURRENT = "HOLD_CURRENT"
    GET_READY = "GET_READY"
    STAND_POLICY = "STAND_POLICY"
    PRE_MIMIC_INTERP = "PRE_MIMIC_INTERP"
    PRE_MIMIC_HOLD = "PRE_MIMIC_HOLD"
    MIMIC_POLICY = "MIMIC_POLICY"
    POST_MIMIC_HOLD = "POST_MIMIC_HOLD"
    POST_MIMIC_INTERP = "POST_MIMIC_INTERP"
    EMERGENCY_DAMPING = "EMERGENCY_DAMPING"
    ERROR = "ERROR"


@dataclass
class PolicyRuntime:
    """All phase-conditioned state belonging to exactly one ONNX policy."""
    name: str
    policy: dict
    cfg: dict
    motion_file: str
    cycle_time: float
    phase_wrap: bool
    policy_path: str
    counter: int = 0
    last_action: np.ndarray = None
    hist_dict: dict = None
    hist_obs: np.ndarray = None
    latest_raw_action: np.ndarray = None
    latest_target: np.ndarray = None


def _load_motion_entry(motion_file):
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
        print(f"motion pkl top-level keys ({motion_file}): {list(data.keys())}")
        nested = [(key, value) for key, value in data.items() if isinstance(value, dict)]
        entry_name, motion = nested[0] if nested else ("<top-level>", data)
    else:
        entry_name, motion = "<root>", data
    if not isinstance(motion, dict):
        raise RuntimeError(f"motion file {motion_file} entry {entry_name} is not a dict")
    print(f"motion pkl selected entry={entry_name}; keys={list(motion.keys())}")
    return motion, entry_name


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
    cfg["motion_joint_order"] = config.get("motion_joint_order")
    cfg["stand_upper_body_dof_pos"] = config.get("stand_upper_body_dof_pos")
    cfg["mimic_models"] = config.get("mimic_models", {})
    return cfg


def load_motion_joint_trajectory(motion_file, expected_joint_names, declared_joint_order=None):
    """Return a Q1 joint trajectory only when its order is provable.

    pose_aa is deliberately not converted here: this deployment bridge has no
    SMPL-to-Q1 retargeter and must never guess one on real hardware.
    """
    motion, entry_name = _load_motion_entry(motion_file)
    trajectory_key = next((key for key in ("dof_pos", "joint_pos", "joint_positions", "qpos", "robot_dof_pos") if key in motion), None)
    if trajectory_key is None:
        if "pose_aa" in motion:
            print("motion pkl contains pose_aa but no retargeted Q1 joint trajectory")
        return None, f"{motion_file}::{entry_name}: no Q1 joint trajectory"
    q = np.asarray(motion[trajectory_key], dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != len(expected_joint_names):
        raise ValueError(f"{motion_file}::{trajectory_key} must have shape [frames,{len(expected_joint_names)}], got {q.shape}")
    names = motion.get("joint_names")
    name_field = "joint_names"
    if names is None:
        names = motion.get("dof_names")
        name_field = "dof_names"
    if names is not None:
        names = list(names)
        if len(names) != len(expected_joint_names) or set(names) != set(expected_joint_names):
            raise ValueError(f"{motion_file} {name_field} do not exactly match policy joint names")
        mapping = [names.index(name) for name in expected_joint_names]
        return q[:, mapping], f"{motion_file}::{entry_name}.{trajectory_key} mapped by {name_field}"
    if declared_joint_order != "policy":
        raise ValueError(
            f"{motion_file} has a {q.shape[1]}-DoF trajectory but no joint_names; "
            "set motion_joint_order: policy explicitly to use positional order"
        )
    return q, f"{motion_file}::{entry_name}.{trajectory_key} declared policy order"


def infer_cycle_time_from_motion(motion_file):
    if not motion_file:
        return None, ""
    motion, entry_name = _load_motion_entry(motion_file)
    if "fps" not in motion:
        return None, f"{motion_file}::{entry_name}: no fps"
    frame_source = next((key for key in ("dof_pos", "joint_pos", "joint_positions", "qpos", "robot_dof_pos", "pose_aa") if key in motion), None)
    if frame_source is None:
        return None, f"{motion_file}::{entry_name}: no frame trajectory"
    num_frames = int(np.asarray(motion[frame_source]).shape[0])
    fps = float(motion["fps"])
    if num_frames <= 0 or fps <= 0.0:
        raise RuntimeError(f"invalid motion length in {motion_file}: frames={num_frames}, fps={fps}")
    return num_frames / fps, f"{motion_file}::{entry_name} ({num_frames} frames / {fps:g} fps)"


def resolve_start_upper_pose(motion_file, policy_joint_names, upper_body_indices, yaml_fallback, declared_joint_order=None):
    q, source = load_motion_joint_trajectory(motion_file, policy_joint_names, declared_joint_order)
    if q is not None:
        return q[0, upper_body_indices].astype(np.float32), source
    if yaml_fallback is None:
        raise ValueError(
            f"Cannot resolve start upper pose for {motion_file}: {source}; "
            "provide a retargeted Q1 trajectory or YAML start_upper_body_dof_pos"
        )
    fallback = np.asarray(yaml_fallback, dtype=np.float32).reshape(-1)
    if fallback.shape != (len(upper_body_indices),) or not np.isfinite(fallback).all():
        raise ValueError("YAML start_upper_body_dof_pos must contain 10 finite values")
    return fallback, "YAML start_upper_body_dof_pos fallback (no retargeted Q1 trajectory)"


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


def compute_runtime_phase(runtime):
    sim_time = runtime.cfg["time_offset"] + runtime.counter * runtime.cfg["simulation_dt"]
    if runtime.phase_wrap:
        return float((sim_time % runtime.cycle_time) / runtime.cycle_time)
    return float(min(sim_time, runtime.cycle_time - 1e-6) / runtime.cycle_time)


def get_obs(hist_obs_c, hist_dict, state, action, counter, cfg, ref_motion_phase=None):
    hist_len = cfg["frame_stack"] * cfg["num_single_obs"]
    if ref_motion_phase is None:
        ref_motion_phase = compute_ref_motion_phase(counter, cfg)
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


class MujocoSimBackend:
    """Small in-process MuJoCo robot used as a safe stand-in for real Q1 topics."""

    def __init__(self, args, cfg):
        if mujoco is None:
            raise RuntimeError("mujoco is not installed. Run: python3 -m pip install mujoco")
        try:
            builder_path = Path(__file__).resolve().with_name("visualize_q1_live_mujoco_ros2.py")
            spec = importlib.util.spec_from_file_location("q1_live_mujoco_builder", builder_path)
            builder = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(builder)
            build_model_xml = builder.build_model_xml
        except Exception as exc:
            raise RuntimeError(
                "failed to import the URDF->MJCF builder from visualize_q1_live_mujoco_ros2.py"
            ) from exc

        mjcf_xml, actual_geometry = build_model_xml(
            args.urdf,
            args.mujoco_root_height,
            fixed_base=False,
            visual_geometry=args.mujoco_visual_geometry,
        )
        self.lock = threading.RLock()
        self.model = mujoco.MjModel.from_xml_string(mjcf_xml)
        self.model.opt.timestep = float(args.command_period_s)
        self.data = mujoco.MjData(self.model)
        self.dof_names = list(cfg["dof_names"])
        self.default_dof_pos = cfg["default_dof_pos"].astype(np.float64)
        self.root_height = float(args.mujoco_root_height)
        self.lock_root = not bool(args.mujoco_free_root)
        self.control_mode = args.mujoco_control_mode
        self.kinematic_tau = float(args.mujoco_kinematic_tau)
        self.kp_scale = float(args.mujoco_kp_scale)
        self.kd_scale = float(args.mujoco_kd_scale)
        self.joint_armature = float(args.mujoco_joint_armature)
        self.joint_damping = float(args.mujoco_joint_damping)
        self.actual_geometry = actual_geometry
        self.joint_qposadr = {}
        self.joint_dofadr = {}
        for name in self.dof_names:
            try:
                joint = self.model.joint(name)
                self.joint_qposadr[name] = int(joint.qposadr[0])
                self.joint_dofadr[name] = int(joint.dofadr[0])
            except KeyError as exc:
                raise RuntimeError(f"MuJoCo model is missing policy joint {name}") from exc
        for dofadr in self.joint_dofadr.values():
            self.model.dof_armature[dofadr] = max(float(self.model.dof_armature[dofadr]), self.joint_armature)
            self.model.dof_damping[dofadr] = max(float(self.model.dof_damping[dofadr]), self.joint_damping)

        self.root_qposadr = None
        self.root_dofadr = None
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                self.root_qposadr = int(self.model.jnt_qposadr[joint_id])
                self.root_dofadr = int(self.model.jnt_dofadr[joint_id])
                break
        self.reset()

    def reset(self):
        with self.lock:
            self.data.qpos[:] = self.model.qpos0
            self.data.qvel[:] = 0.0
            if self.root_qposadr is not None:
                self.data.qpos[self.root_qposadr:self.root_qposadr + 3] = [0.0, 0.0, self.root_height]
                self.data.qpos[self.root_qposadr + 3:self.root_qposadr + 7] = [1.0, 0.0, 0.0, 0.0]
            for idx, name in enumerate(self.dof_names):
                self.data.qpos[self.joint_qposadr[name]] = self.default_dof_pos[idx]
            mujoco.mj_forward(self.model, self.data)

    def _lock_root_pose(self):
        if not self.lock_root or self.root_qposadr is None:
            return
        self.data.qpos[self.root_qposadr:self.root_qposadr + 3] = [0.0, 0.0, self.root_height]
        self.data.qpos[self.root_qposadr + 3:self.root_qposadr + 7] = [1.0, 0.0, 0.0, 0.0]
        if self.root_dofadr is not None:
            self.data.qvel[self.root_dofadr:self.root_dofadr + 6] = 0.0

    def step(self, target_dof_pos, kps, kds):
        with self.lock:
            target_dof_pos = np.asarray(target_dof_pos, dtype=np.float64).reshape(NUM_ACTIONS)
            self._lock_root_pose()
            if self.control_mode == "kinematic":
                dt = float(self.model.opt.timestep)
                alpha = float(np.clip(dt / max(self.kinematic_tau, dt), 0.0, 1.0))
                for idx, name in enumerate(self.dof_names):
                    qposadr = self.joint_qposadr[name]
                    dofadr = self.joint_dofadr[name]
                    before = float(self.data.qpos[qposadr])
                    after = before + alpha * (float(target_dof_pos[idx]) - before)
                    self.data.qpos[qposadr] = after
                    self.data.qvel[dofadr] = (after - before) / dt
                self.data.qfrc_applied[:] = 0.0
                self._lock_root_pose()
                mujoco.mj_forward(self.model, self.data)
                return

            kps = np.asarray(kps, dtype=np.float64).reshape(NUM_ACTIONS) * self.kp_scale
            kds = np.asarray(kds, dtype=np.float64).reshape(NUM_ACTIONS) * self.kd_scale
            self.data.qfrc_applied[:] = 0.0
            for idx, name in enumerate(self.dof_names):
                qposadr = self.joint_qposadr[name]
                dofadr = self.joint_dofadr[name]
                pos = float(self.data.qpos[qposadr])
                vel = float(self.data.qvel[dofadr])
                self.data.qfrc_applied[dofadr] = kps[idx] * (target_dof_pos[idx] - pos) - kds[idx] * vel
            mujoco.mj_step(self.model, self.data)
            self._lock_root_pose()
            mujoco.mj_forward(self.model, self.data)

    def snapshot(self):
        with self.lock:
            dof_pos = np.array(
                [self.data.qpos[self.joint_qposadr[name]] for name in self.dof_names],
                dtype=np.float32,
            )
            dof_vel = np.array(
                [self.data.qvel[self.joint_dofadr[name]] for name in self.dof_names],
                dtype=np.float32,
            )
            if self.root_qposadr is not None:
                quat_wxyz = np.array(
                    self.data.qpos[self.root_qposadr + 3:self.root_qposadr + 7],
                    dtype=np.float64,
                )
            else:
                quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            if self.root_dofadr is not None:
                base_ang_vel = np.array(
                    self.data.qvel[self.root_dofadr + 3:self.root_dofadr + 6],
                    dtype=np.float32,
                )
            else:
                base_ang_vel = np.zeros(3, dtype=np.float32)
            if self.lock_root:
                base_ang_vel[:] = 0.0
            projected_gravity = quat_rotate_inverse_wxyz(quat_wxyz, [0.0, 0.0, -1.0]).astype(np.float32)
            return dof_pos, dof_vel, base_ang_vel, projected_gravity


class MujocoPhysicsBackend:
    """Collision-enabled free-root PD backend shared with the headless regression."""

    is_physics_backend = True

    def __init__(self, args, cfg):
        if not Path(args.physics_mjcf).is_file():
            raise FileNotFoundError(f"physics MJCF not found: {args.physics_mjcf}")
        physics_path = Path(__file__).resolve().with_name("test_q1_dual_policy_mujoco_headless.py")
        spec = importlib.util.spec_from_file_location("q1_physics_backend", physics_path)
        physics_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(physics_module)
        physics_args = SimpleNamespace(
            physics_mjcf=args.physics_mjcf,
            timestep=args.command_period_s,
            solver_iterations=args.physics_solver_iterations,
            solver_ls_iterations=args.physics_solver_ls_iterations,
            root_height=args.mujoco_root_height,
            assist_kp_pos=args.physics_assist_kp_pos,
            assist_kd_pos=args.physics_assist_kd_pos,
            assist_kp_rot=args.physics_assist_kp_rot,
            assist_kd_rot=args.physics_assist_kd_rot,
        )
        physics_cfg = dict(cfg)
        physics_cfg["root_height"] = float(args.mujoco_root_height)
        helper = SimpleNamespace(quat_rotate_inverse_wxyz=quat_rotate_inverse_wxyz)
        self._impl = physics_module.MujocoPhysicsBackend(mujoco, helper, physics_args, physics_cfg)
        self.lock = threading.RLock()
        self.model, self.data = self._impl.model, self._impl.data
        self.actual_geometry, self.control_mode = "physics_mjcf", "pd"

    def release_assist(self, sim_time):
        with self.lock:
            self._impl.release_assist(sim_time)

    def step(self, target_dof_pos, kps, kds):
        with self.lock:
            self._impl.step(target_dof_pos, kps, kds)

    def snapshot(self):
        with self.lock:
            return self._impl.snapshot()

    def physics_status(self):
        with self.lock:
            return {
                "contacts": int(self._impl.data.ncon),
                "max_contacts": int(self._impl.max_contacts),
                "root_z": float(self._impl.data.qpos[self._impl.root_qposadr + 2]),
                "min_root_z": float(self._impl.min_root_z),
                "assist_enabled": bool(self._impl.assist_enabled),
                "assist_release_s": self._impl.assist_release_s,
            }


class KeyboardReader:
    def __init__(self, callback):
        self.callback = callback
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.old_settings = None

    def start(self):
        if termios is None or tty is None:
            print("[WARN] termios/tty unavailable on this platform; keyboard controls disabled. Use --no-keyboard to silence this warning.")
            return
        if not sys.stdin.isatty():
            print("[WARN] stdin is not a TTY; keyboard controls disabled")
            return
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.old_settings is not None and termios is not None:
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
    def __init__(self, args, stand_cfg, mimic_cfg, limits, mimic_start_upper, mimic_start_source, stand_start_source):
        super().__init__("q1_motion_tracking_sim2real")
        self.args = args
        self.cfg = stand_cfg  # Existing command/MuJoCo paths use the stand hardware config.
        self.stand_cfg = stand_cfg
        self.mimic_cfg = mimic_cfg
        self.lower_limits, self.upper_limits, self.limit_source = limits
        self.dof_names = stand_cfg["dof_names"]
        self.policy_dt = stand_cfg["simulation_dt"] * stand_cfg["control_decimation"]
        self.control_lock = threading.RLock()

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
        self.latest_target_dof_pos = stand_cfg["default_dof_pos"].copy()
        self.command_target = self.latest_target_dof_pos.copy()
        self.stand_runtime = None
        self.mimic_runtimes = []
        self.selected_mimic_idx = 0
        self.mimic_start_upper = mimic_start_upper
        self.mimic_start_source = mimic_start_source
        self.stand_start_source = stand_start_source
        self.pre_interp_start_upper = None
        self.post_hold_upper = None
        self.post_handoff_start_target = None
        self.post_handoff_last_log_time = 0.0
        self.post_handoff_limit_clamp_count = 0
        self.state_enter_time = time.monotonic()
        self.get_ready_completed = False
        self.mujoco_auto_start_requested = False
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
        self.command_loop_count = 0
        self.command_overrun_count = 0
        self.command_max_lateness_ms = 0.0
        self.command_last_interval_ms = 0.0
        self.command_start_ns = time.monotonic_ns()
        self.command_last_ns = 0
        self.policy_stand_inference_ms = 0.0
        self.policy_mimic_inference_ms = 0.0
        self.policy_callback_ms = 0.0
        self.policy_overrun_count = 0
        self.policy_overrun_last_log_time = 0.0
        self.command_thread_error = ""
        self.command_thread_last_error_log_time = 0.0
        self.max_abs_dof_vel_seen = 0.0
        self.max_abs_base_ang_vel_seen = 0.0
        self.imu_adapter_mode = "mujoco_sim" if args.mujoco_sim else ("fake_base_state" if args.fake_base_state else "none")
        self.motors_enabled = self.compute_motor_enable_gate(log=False)
        self.last_error = ""
        self.mujoco_ready_time = 0.0
        self.mujoco_auto_start_done = False
        self.first_stand_onnx_done = False
        self.mujoco_backend = None
        if args.mujoco_sim:
            backend_cls = MujocoPhysicsBackend if args.physics_mjcf else MujocoSimBackend
            self.mujoco_backend = backend_cls(args, stand_cfg)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.command_pub = None
        self.state_sub = None
        self.imu_sub = None
        self.odom_sub = None
        if not args.mujoco_sim:
            self.command_pub = self.create_publisher(JointCommandArray, args.joint_command_topic, qos)
            self.state_sub = self.create_subscription(JointStateArray, args.joint_state_topic, self.on_joint_state, qos)
        if not args.mujoco_sim and args.imu_topic:
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
        if not args.mujoco_sim and args.odom_topic:
            if Odometry is None:
                raise RuntimeError("nav_msgs/Odometry is unavailable, cannot use --odom-topic")
            self.odom_sub = self.create_subscription(Odometry, args.odom_topic, self.on_odom, qos)

        if args.enable_policy:
            self.stand_runtime = self.create_policy_runtime(
                "stand", args.stand_policy_path, args.stand_motion_file, stand_cfg, phase_wrap=True
            )
            self.mimic_runtimes = [self.create_policy_runtime(
                args.mimic_name, args.mimic_policy_path, args.mimic_motion_file, mimic_cfg, phase_wrap=False
            )]
        self.motors_enabled = self.compute_motor_enable_gate(log=False)
        if self.args.mujoco_sim:
            self.refresh_mujoco_state()

        self.create_timer(self.policy_dt, self.on_policy_timer)
        self.create_timer(0.05, self.on_safety_timer)
        self.create_timer(args.print_rate_s, self.on_print_timer)

        self.command_stop_event = threading.Event()
        self.command_thread = threading.Thread(
            target=self.command_loop, daemon=True, name="q1_command_loop"
        )
        self.command_thread.start()

        self.keyboard = KeyboardReader(self.on_key)
        if not args.no_keyboard:
            self.keyboard.start()
        self.log_startup()

    def create_policy_runtime(self, name, policy_path, motion_file, cfg, phase_wrap):
        expected_obs_dim = (cfg["frame_stack"] + 1) * cfg["num_single_obs"]
        policy = load_onnx_policy(policy_path, expected_obs_dim, cfg["num_actions"])
        cycle_time = float(cfg["cycle_time"])
        if cycle_time <= 0.0:
            raise ValueError(f"{name} cycle_time must be positive")
        runtime = PolicyRuntime(
            name=name, policy=policy, cfg=cfg, motion_file=motion_file,
            cycle_time=cycle_time, phase_wrap=phase_wrap, policy_path=policy_path,
        )
        self.reset_policy_runtime(runtime, keep_global_target=True)
        return runtime

    def reset_policy_runtime(self, runtime, keep_global_target=False):
        runtime.counter = 0
        runtime.last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        runtime.latest_raw_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        runtime.hist_dict, runtime.hist_obs = create_history(runtime.cfg)
        runtime.latest_target = self.latest_dof_pos.copy()
        if not keep_global_target:
            self.latest_target_dof_pos = self.latest_dof_pos.copy()
            self.command_target = self.latest_target_dof_pos.copy()
        self.get_logger().info(f"{runtime.name} runtime reset: phase/history/action cleared")

    def build_runtime_obs(self, runtime, state):
        obs, runtime.hist_obs = get_obs(
            runtime.hist_obs, runtime.hist_dict, state, runtime.last_action, runtime.counter, runtime.cfg,
            ref_motion_phase=compute_runtime_phase(runtime),
        )
        if not np.isfinite(obs).all():
            raise RuntimeError(f"{runtime.name} nonfinite observation")
        return obs

    def infer_runtime_target(self, runtime, state):
        obs = self.build_runtime_obs(runtime, state)
        raw_action = infer_policy_action(runtime.policy, obs, runtime.cfg["clip_actions"])
        action, target = apply_action_postprocess(
            raw_action, runtime.last_action, runtime.latest_target, runtime.cfg, self.policy_dt
        )
        runtime.last_action = action
        runtime.latest_raw_action = raw_action
        runtime.latest_target = target
        runtime.counter += runtime.cfg["control_decimation"]
        return target.astype(np.float32)

    def selected_mimic(self):
        return self.mimic_runtimes[self.selected_mimic_idx]

    def compute_motor_enable_gate(self, log=True):
        reasons = []
        if self.args.mujoco_sim:
            reasons.append("mujoco sim mode")
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
        if self.args.enable_policy and (self.stand_runtime is None or not self.mimic_runtimes):
            reasons.append("stand/mimic policy not loaded")
        enabled = len(reasons) == 0
        if log and reasons:
            self.get_logger().info("real joint command disabled: " + ", ".join(reasons))
        return enabled

    def dry_run_active(self):
        return self.args.dry_run or not self.motors_enabled or self.args.mujoco_sim

    def log_startup(self):
        if self.args.mujoco_sim:
            self.get_logger().warn(
                "MUJOCO SIM MODE ENABLED: no real Q1 state topics are subscribed and no real joint "
                "commands are published. Policy targets are applied only inside the MuJoCo backend."
            )
            self.get_logger().info(
                f"mujoco_visual_geometry={self.args.mujoco_visual_geometry}, "
                f"actual_geometry={self.mujoco_backend.actual_geometry}, "
                f"root_height={self.args.mujoco_root_height:.3f}, "
                f"root_lock={not self.args.mujoco_free_root}, "
                f"control_mode={self.args.mujoco_control_mode}, "
                f"auto_start_policy={self.args.mujoco_auto_start_policy}"
            )
            if self.args.physics_mjcf:
                self.get_logger().warn(
                    f"PHYSICS MJCF MODE: xml={self.args.physics_mjcf}; free-root PD contacts are active. "
                    "Temporary pelvis assist will be released after the first stand ONNX target."
                )
        else:
            self.get_logger().warn(
                "This script commands Q1 joints only when --enable-motors and the risk token are set. "
                "Ensure low-level developer mode is active and high-level motion control is not conflicting."
            )
        if self.args.fake_base_state:
            self.get_logger().error(
                "FAKE BASE STATE ENABLED: base_ang_vel=0 and projected_gravity=[0,0,-1]. "
                "This is ONLY for suspended/bench validation and is forbidden for standing motion tracking."
            )
        self.get_logger().warn(
            "Q1 must already be in low-level developer mode. High-level "
            "STAND_UP / BIPED_STAND_DEFAULT must not be active. This process is the only low-level joint command source."
        )
        self.get_logger().info(
            f"stand upper source={self.stand_start_source}; mimic upper source={self.mimic_start_source}; "
            f"selected_mimic={self.selected_mimic().name if self.mimic_runtimes else 'none'}"
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
        self.get_logger().info("Keys: i=GET_READY, ]=STAND_POLICY, ;/'=select mimic, [=start mimic, o=return/hold stand, q=EMERGENCY_DAMPING")

    def refresh_mujoco_state(self):
        dof_pos, dof_vel, base_ang_vel, projected_gravity = self.mujoco_backend.snapshot()
        now = time.monotonic()
        joints = [
            SimpleNamespace(name=name, position=float(dof_pos[idx]), velocity=float(dof_vel[idx]))
            for idx, name in enumerate(self.dof_names)
        ]
        with self.control_lock:
            self.latest_joint_state_msg = SimpleNamespace(joints=joints, header=SimpleNamespace(frame_id="mujoco"))
            self.latest_state_by_name = {joint.name: joint for joint in joints}
            self.latest_dof_pos = dof_pos
            self.latest_dof_vel = dof_vel
            self.base_ang_vel = base_ang_vel
            self.projected_gravity = projected_gravity
            self.latest_joint_msg_time = now
            self.latest_base_msg_time = now
            self.max_abs_dof_vel_seen = max(self.max_abs_dof_vel_seen, float(np.max(np.abs(dof_vel))))
            self.max_abs_base_ang_vel_seen = max(self.max_abs_base_ang_vel_seen, float(np.max(np.abs(base_ang_vel))))
            if self.state == ControlState.WAIT_STATE:
                self.latest_target_dof_pos = self.latest_dof_pos.copy()
                self.command_target = self.latest_dof_pos.copy()
                self.mujoco_ready_time = now
                self.transition(ControlState.HOLD_CURRENT, "mujoco sim state ready")
        self.maybe_auto_start_policy(now, "mujoco state refresh")

    def maybe_auto_start_policy(self, now=None, source="timer"):
        with self.control_lock:
            if not self.args.mujoco_auto_start_policy or self.mujoco_auto_start_done or not self.args.mujoco_sim:
                return
            if not self.args.enable_policy or self.stand_runtime is None or self.state != ControlState.HOLD_CURRENT:
                return
            if not self.current_state_ready():
                return
            now = time.monotonic() if now is None else now
            if self.mujoco_ready_time <= 0.0:
                self.mujoco_ready_time = now
            if now - self.mujoco_ready_time < self.args.mujoco_start_delay_s:
                return
            self.mujoco_auto_start_done = True
            self.mujoco_auto_start_requested = True
            self.transition(ControlState.GET_READY, f"mujoco auto get-ready ({source})")

    def on_joint_state(self, msg):
        with self.control_lock:
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
        with self.control_lock:
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
        if self.state == ControlState.ERROR:
            return False
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
        if self.args.mujoco_sim:
            return True
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
        with self.control_lock:
            old_state = self.state
            if old_state == new_state:
                return
            self.get_logger().info(f"STATE {old_state.value} -> {new_state.value}: {reason}")
            self.state = new_state
            self.state_enter_time = time.monotonic()
            if new_state == ControlState.GET_READY:
                self.get_ready_start_time = time.monotonic()
                self.get_ready_start_pos = self.latest_dof_pos.copy()
                self.get_ready_completed = False
            elif new_state == ControlState.STAND_POLICY:
                # Reset once only on the deliberate HOLD_CURRENT -> STAND_POLICY handoff.
                self.enter_stand_policy(reason, reset_runtime=(old_state == ControlState.HOLD_CURRENT))
            elif new_state in (ControlState.EMERGENCY_DAMPING, ControlState.ERROR):
                self.command_target = self.latest_dof_pos.copy()

    def enter_stand_policy(self, reason, reset_runtime=False):
        if reset_runtime:
            self.reset_policy_runtime(self.stand_runtime)
        self.last_action_max = 0.0
        self.get_logger().info(f"stand policy takes all 22 DoF: {reason}; reset_runtime={reset_runtime}")

    def start_mimic_transition(self):
        with self.control_lock:
            self.pre_interp_start_upper = (
                self.command_target[UPPER_BODY_INDICES].copy()
                if np.isfinite(self.command_target).all() else self.latest_dof_pos[UPPER_BODY_INDICES].copy()
            )
            mimic = self.selected_mimic()
            self.get_logger().info(
                f"mimic start: selected={mimic.name} phase={compute_runtime_phase(mimic):.3f} "
                f"start_upper={np.array2string(self.mimic_start_upper, precision=3)} "
                f"measured_upper={np.array2string(self.latest_dof_pos[UPPER_BODY_INDICES], precision=3)}"
            )
            self.transition(ControlState.PRE_MIMIC_INTERP, "keyboard [")

    def finish_mimic_and_start_post_hold(self, reason):
        with self.control_lock:
            # This is the 500 Hz-ramped target actually sent to the PD controller,
            # not measured q and not the un-ramped policy output.
            if not np.isfinite(self.command_target).all():
                raise RuntimeError("cannot start post handoff from nonfinite command_target")
            self.post_handoff_start_target = self.command_target.copy()
            self.post_hold_upper = self.post_handoff_start_target[UPPER_BODY_INDICES].copy()
            # A stand shadow rollout during mimic accumulates states outside
            # its training distribution.  Reset phase/history/action from the
            # measured post-mimic state before stand regains control.
            self.reset_policy_runtime(self.stand_runtime, keep_global_target=True)
            stand_lower = self.stand_runtime.latest_target[LOWER_BODY_INDICES].copy()
            lower_gap = float(np.max(np.abs(
                self.post_handoff_start_target[LOWER_BODY_INDICES] - stand_lower
            )))
            self.post_handoff_last_log_time = 0.0
            self.get_logger().info(
                f"mimic end handoff (stand runtime re-primed): selected={self.selected_mimic().name} "
                f"phase={compute_runtime_phase(self.selected_mimic()):.3f} "
                f"start_lower={np.array2string(self.post_handoff_start_target[LOWER_BODY_INDICES], precision=3)} "
                f"stand_lower={np.array2string(stand_lower, precision=3)} "
                f"gap_max={lower_gap:.4f} post_handoff_s={self.args.post_handoff_s:.3f}"
            )
            self.transition(ControlState.POST_MIMIC_HOLD, reason)

    def enter_emergency(self, reason):
        with self.control_lock:
            if self.state == ControlState.EMERGENCY_DAMPING and self.last_error == reason:
                return
            self.last_error = reason
            self.get_logger().error(f"EMERGENCY_DAMPING: {reason}")
            if self.stand_runtime is not None:
                self.stand_runtime.last_action.fill(0.0)
            for runtime in self.mimic_runtimes:
                runtime.last_action.fill(0.0)
            self.latest_target_dof_pos = self.latest_dof_pos.copy()
            self.command_target = self.latest_dof_pos.copy()
            self.transition(ControlState.EMERGENCY_DAMPING, reason)

    def enter_error(self, reason):
        with self.control_lock:
            if self.state == ControlState.ERROR and self.last_error == reason:
                return
            self.last_error = reason
            self.get_logger().error(f"ERROR: {reason}; continuing damping command output")
            if self.stand_runtime is not None:
                self.stand_runtime.last_action.fill(0.0)
            for runtime in self.mimic_runtimes:
                runtime.last_action.fill(0.0)
            self.latest_target_dof_pos = self.latest_dof_pos.copy()
            self.command_target = self.latest_dof_pos.copy()
            self.transition(ControlState.ERROR, reason)

    def on_key(self, ch):
        if ch == "i":
            if self.current_state_ready() and self.state in (ControlState.HOLD_CURRENT, ControlState.STAND_POLICY):
                self.transition(ControlState.GET_READY, "keyboard i")
            else:
                self.get_logger().warn("GET_READY requires ready state and HOLD_CURRENT/STAND_POLICY")
        elif ch == "]":
            if not self.args.enable_policy:
                self.get_logger().warn("Policy is disabled; restart with --enable-policy")
            elif self.state == ControlState.HOLD_CURRENT and self.get_ready_completed:
                self.transition(ControlState.STAND_POLICY, "keyboard ]")
            else:
                self.get_logger().warn("Press i and wait for GET_READY to complete before ]")
        elif ch == ";":
            if self.mimic_runtimes:
                self.selected_mimic_idx = (self.selected_mimic_idx + 1) % len(self.mimic_runtimes)
                self.get_logger().info(f"selected mimic={self.selected_mimic().name}")
        elif ch == "'":
            if self.mimic_runtimes:
                self.selected_mimic_idx = (self.selected_mimic_idx - 1) % len(self.mimic_runtimes)
                self.get_logger().info(f"selected mimic={self.selected_mimic().name}")
        elif ch == "[":
            if self.state == ControlState.STAND_POLICY:
                self.start_mimic_transition()
            else:
                self.get_logger().warn("[ is only allowed from STAND_POLICY")
        elif ch == "o":
            if self.state in (ControlState.PRE_MIMIC_INTERP, ControlState.PRE_MIMIC_HOLD, ControlState.MIMIC_POLICY):
                self.finish_mimic_and_start_post_hold("keyboard o safe return")
            elif self.state == ControlState.STAND_POLICY:
                self.get_logger().info("already in STAND_POLICY; continuing stand control")
        elif ch == "q" or ch == "\x03":
            self.enter_emergency("keyboard emergency damping")

    def on_safety_timer(self):
        if not self.command_thread.is_alive():
            self.enter_error("command thread exited unexpectedly")
            return
        if self.state == ControlState.WAIT_STATE:
            return
        if self.state == ControlState.ERROR:
            return
        now = time.monotonic()
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
        self.maybe_auto_start_policy(now, "safety timer")
        self.motors_enabled = self.compute_motor_enable_gate(log=False)

    def on_policy_timer(self):
        with self.control_lock:
            current_state = self.state
        if current_state == ControlState.GET_READY:
            self.update_get_ready_target()
            return
        if current_state not in (
            ControlState.STAND_POLICY, ControlState.PRE_MIMIC_INTERP, ControlState.PRE_MIMIC_HOLD,
            ControlState.MIMIC_POLICY, ControlState.POST_MIMIC_HOLD, ControlState.POST_MIMIC_INTERP,
        ):
            return
        if not self.state_age_ok():
            self.enter_error("state timeout before policy inference")
            return
        try:
            callback_start_ns = time.monotonic_ns()
            overrun_log = None
            with self.control_lock:
                state = {
                    "dof_pos": self.latest_dof_pos.copy(),
                    "dof_vel": self.latest_dof_vel.copy(),
                    "base_ang_vel": self.base_ang_vel.copy(),
                    "projected_gravity": self.projected_gravity.copy(),
                }
                state_enter_time = self.state_enter_time
                pre_interp_start_upper = None if self.pre_interp_start_upper is None else self.pre_interp_start_upper.copy()
                post_hold_upper = None if self.post_hold_upper is None else self.post_hold_upper.copy()
                post_handoff_start_target = None if self.post_handoff_start_target is None else self.post_handoff_start_target.copy()
                mimic_start_upper = self.mimic_start_upper.copy()
                mimic = self.selected_mimic()
            if not all(np.isfinite(v).all() for v in state.values()):
                raise RuntimeError("nonfinite input state")
            if current_state == ControlState.MIMIC_POLICY:
                # Pause stand shadow rollout while mimic owns all 22 DoF.
                # finish_mimic_and_start_post_hold re-primes it from measured q.
                stand_target = self.stand_runtime.latest_target.copy()
                stand_ms = 0.0
            else:
                stand_start_ns = time.monotonic_ns()
                stand_target = self.infer_runtime_target(self.stand_runtime, state)
                stand_ms = (time.monotonic_ns() - stand_start_ns) / 1e6
            if current_state == ControlState.STAND_POLICY and not self.first_stand_onnx_done:
                self.first_stand_onnx_done = True
                if getattr(self.mujoco_backend, "is_physics_backend", False):
                    self.mujoco_backend.release_assist(time.monotonic())
                    self.get_logger().info("PHYSICS ASSIST released after first stand ONNX target")
            mimic_ms = 0.0
            active_runtime = self.stand_runtime
            transition_after_commit = None
            if current_state == ControlState.STAND_POLICY:
                final_target = stand_target
            elif current_state == ControlState.PRE_MIMIC_INTERP:
                alpha = float(np.clip((time.monotonic() - state_enter_time) / self.args.pre_interp_s, 0.0, 1.0))
                final_target = stand_target.copy()
                final_target[UPPER_BODY_INDICES] = (1.0 - alpha) * pre_interp_start_upper + alpha * mimic_start_upper
                if alpha >= 1.0:
                    transition_after_commit = (ControlState.PRE_MIMIC_HOLD, "pre mimic interpolation complete")
            elif current_state == ControlState.PRE_MIMIC_HOLD:
                final_target = stand_target.copy()
                final_target[UPPER_BODY_INDICES] = mimic_start_upper
                if time.monotonic() - state_enter_time >= self.args.pre_hold_s:
                    transition_after_commit = (ControlState.MIMIC_POLICY, "pre mimic hold complete")
            elif current_state == ControlState.MIMIC_POLICY:
                mimic_start_ns = time.monotonic_ns()
                final_target = self.infer_runtime_target(mimic, state)
                mimic_ms = (time.monotonic_ns() - mimic_start_ns) / 1e6
                active_runtime = mimic
                if mimic.counter * mimic.cfg["simulation_dt"] >= mimic.cycle_time:
                    transition_after_commit = (ControlState.POST_MIMIC_HOLD, "mimic phase reached end")
            elif current_state == ControlState.POST_MIMIC_HOLD:
                if post_handoff_start_target is None or post_hold_upper is None:
                    raise RuntimeError("POST_MIMIC_HOLD is missing handoff start target")
                elapsed = time.monotonic() - state_enter_time
                u = float(np.clip(elapsed / self.args.post_handoff_s, 0.0, 1.0))
                blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
                final_target = stand_target.copy()
                final_target[LOWER_BODY_INDICES] = (
                    (1.0 - blend) * post_handoff_start_target[LOWER_BODY_INDICES]
                    + blend * stand_target[LOWER_BODY_INDICES]
                )
                final_target[UPPER_BODY_INDICES] = post_hold_upper
                if elapsed >= self.args.post_hold_s:
                    transition_after_commit = (ControlState.POST_MIMIC_INTERP, "post mimic hold complete")
            else:  # POST_MIMIC_INTERP
                alpha = float(np.clip((time.monotonic() - state_enter_time) / self.args.post_interp_s, 0.0, 1.0))
                final_target = stand_target.copy()
                final_target[UPPER_BODY_INDICES] = (1.0 - alpha) * post_hold_upper + alpha * stand_target[UPPER_BODY_INDICES]
                if alpha >= 1.0:
                    transition_after_commit = (ControlState.STAND_POLICY, "post mimic interpolation complete")
            with self.control_lock:
                # A keyboard/safety transition during inference wins; discard stale output.
                if self.state != current_state:
                    return
                previous_target = self.latest_target_dof_pos.copy()
                raw_lower_delta = float(np.max(np.abs(final_target[LOWER_BODY_INDICES] - previous_target[LOWER_BODY_INDICES])))
                target, step_limited_names = self.apply_target_safety(final_target, return_step_info=True)
                self.last_target_delta_max = float(np.max(np.abs(target - previous_target)))
                self.latest_target_dof_pos = target
                callback_ms = (time.monotonic_ns() - callback_start_ns) / 1e6
                self.last_action_max = float(np.max(np.abs(active_runtime.latest_raw_action)))
                self.policy_count += 1
                self.policy_stand_inference_ms = stand_ms
                self.policy_mimic_inference_ms = mimic_ms
                self.policy_callback_ms = callback_ms
                if current_state == ControlState.POST_MIMIC_HOLD:
                    elapsed = time.monotonic() - state_enter_time
                    final_lower_delta = float(np.max(np.abs(
                        target[LOWER_BODY_INDICES] - previous_target[LOWER_BODY_INDICES]
                    )))
                    lower_limited = [name for name in step_limited_names if name in set(self.dof_names[:12])]
                    if lower_limited:
                        self.post_handoff_limit_clamp_count += 1
                    now = time.monotonic()
                    if now - self.post_handoff_last_log_time >= 0.25 or elapsed >= self.args.post_handoff_s:
                        self.post_handoff_last_log_time = now
                        u = float(np.clip(elapsed / self.args.post_handoff_s, 0.0, 1.0))
                        blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
                        self.get_logger().info(
                            f"post handoff blend={blend:.3f} raw_lower_delta={raw_lower_delta:.4f} "
                            f"final_lower_delta={final_lower_delta:.4f} step_limited={bool(lower_limited)} "
                            f"joints={lower_limited}"
                        )
                if callback_ms > self.policy_dt * 1000.0:
                    self.policy_overrun_count += 1
                    now = time.monotonic()
                    if now - self.policy_overrun_last_log_time >= 1.0:
                        self.policy_overrun_last_log_time = now
                        overrun_log = (
                            f"policy callback overrun {callback_ms:.2f}ms > {self.policy_dt * 1000.0:.2f}ms "
                            f"(count={self.policy_overrun_count})"
                        )
                if transition_after_commit is not None:
                    next_state, reason = transition_after_commit
                    if next_state == ControlState.MIMIC_POLICY:
                        self.reset_policy_runtime(mimic, keep_global_target=True)
                    elif next_state == ControlState.POST_MIMIC_HOLD:
                        self.finish_mimic_and_start_post_hold(reason)
                    else:
                        self.transition(next_state, reason)
            if overrun_log is not None:
                self.get_logger().warn(overrun_log)
        except Exception as exc:
            self.enter_error(f"policy update failed: {exc}")

    def update_get_ready_target(self):
        with self.control_lock:
            if self.get_ready_start_pos is None:
                self.get_ready_start_pos = self.latest_dof_pos.copy()
            elapsed = time.monotonic() - self.get_ready_start_time
            alpha = float(np.clip(elapsed / max(self.args.get_ready_duration_s, 1e-6), 0.0, 1.0))
            target = (1.0 - alpha) * self.get_ready_start_pos + alpha * self.cfg["default_dof_pos"]
            target = self.apply_target_safety(target.astype(np.float32), step_reference=self.command_target)
            self.last_target_delta_max = float(np.max(np.abs(target - self.latest_target_dof_pos)))
            self.latest_target_dof_pos = target
            if alpha >= 1.0:
                self.get_ready_completed = True
                self.transition(ControlState.HOLD_CURRENT, "get-ready complete")
                if self.mujoco_auto_start_requested:
                    self.mujoco_auto_start_requested = False
                    self.transition(ControlState.STAND_POLICY, "mujoco auto start after get-ready")

    def apply_target_safety(self, target, step_reference=None, return_step_info=False):
        target = np.asarray(target, dtype=np.float32).copy()
        if not np.isfinite(target).all():
            raise RuntimeError("nonfinite target")
        step_limited_names = []
        if self.args.max_target_step_rad > 0:
            ref = self.latest_target_dof_pos if step_reference is None else step_reference
            before = target.copy()
            target = np.clip(target, ref - self.args.max_target_step_rad, ref + self.args.max_target_step_rad)
            step_limited_names = [
                self.dof_names[idx] for idx in np.flatnonzero(np.abs(before - target) > 1e-7)
            ]
            if step_limited_names:
                self.clamp_count_step += 1
        before_limits = target.copy()
        margin = float(self.args.joint_limit_margin_rad)
        target = np.clip(target, self.lower_limits + margin, self.upper_limits - margin)
        if not np.allclose(before_limits, target):
            self.clamp_count_limits += 1
        target = target.astype(np.float32)
        return (target, step_limited_names) if return_step_info else target

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

    def make_command_msg(self, joints_snapshot, command_target, control_state, frame_id="", meas_stamp=None):
        if not joints_snapshot:
            return None
        cmd = JointCommandArray()
        if hasattr(cmd, "header"):
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.header.sequence = self.command_publish_count + self.simulated_command_count
            if hasattr(cmd.header, "frame_id"):
                cmd.header.frame_id = frame_id
        if meas_stamp is not None and hasattr(cmd, "meas_stamp"):
            cmd.meas_stamp = meas_stamp
        emergency = control_state in (ControlState.EMERGENCY_DAMPING, ControlState.ERROR)
        safe_command_target = np.asarray(command_target, dtype=np.float32)
        for name, position in joints_snapshot:
            joint = JointCommand()
            joint.name = name
            idx = self.policy_joint_index.get(name)
            if emergency:
                joint.position = float(position)
                joint.stiffness = 0.0
                joint.damping = float(self.args.emergency_damping)
            elif idx is not None:
                joint.position = float(safe_command_target[idx])
                joint.stiffness = float(self.cfg["kps"][idx])
                joint.damping = float(self.cfg["kds"][idx])
            else:
                joint.position = float(position)
                joint.stiffness = float(self.args.hold_uncontrolled_stiffness)
                joint.damping = float(self.args.hold_uncontrolled_damping)
            joint.velocity = 0.0
            joint.effort = 0.0
            cmd.joints.append(joint)
        return cmd

    def command_loop(self):
        period_ns = int(self.args.command_period_s * 1e9)
        next_deadline = time.monotonic_ns()
        while not self.command_stop_event.is_set():
            next_deadline += period_ns
            thread_error_log = None
            try:
                self.command_loop_step()
            except Exception as exc:
                error_text = f"command thread failed: {type(exc).__name__}: {exc}"
                now = time.monotonic()
                with self.control_lock:
                    self.command_thread_error = error_text
                    if now - self.command_thread_last_error_log_time >= 1.0:
                        self.command_thread_last_error_log_time = now
                        thread_error_log = error_text
                try:
                    self.enter_error(error_text)
                except Exception:
                    # The loop must remain alive even if error handling itself is impaired.
                    pass
            if thread_error_log is not None:
                self.get_logger().error(thread_error_log)
            now_ns = time.monotonic_ns()
            remain_ns = next_deadline - now_ns
            if remain_ns > 0:
                self.command_stop_event.wait(remain_ns / 1e9)
            else:
                with self.control_lock:
                    self.command_overrun_count += 1
                    self.command_max_lateness_ms = max(self.command_max_lateness_ms, -remain_ns / 1e6)
                if now_ns - next_deadline >= period_ns:
                    next_deadline = now_ns

    def command_loop_step(self):
        now_ns = time.monotonic_ns()
        with self.control_lock:
            control_state = self.state
            if control_state == ControlState.WAIT_STATE:
                return
            if control_state not in (ControlState.EMERGENCY_DAMPING, ControlState.ERROR):
                self.update_command_ramp()
            command_target = self.command_target.copy()
            msg = self.latest_joint_state_msg
            joints_snapshot = [] if msg is None else [
                (joint.name, float(joint.position)) for joint in msg.joints
            ]
            header = getattr(msg, "header", None)
            frame_id = getattr(header, "frame_id", "")
            meas_stamp = getattr(msg, "meas_stamp", None)
            dry_run = self.args.dry_run or not self.motors_enabled or self.args.mujoco_sim
            self.command_loop_count += 1
            if self.command_last_ns:
                self.command_last_interval_ms = (now_ns - self.command_last_ns) / 1e6
            self.command_last_ns = now_ns
        if self.args.mujoco_sim:
            if control_state not in (ControlState.EMERGENCY_DAMPING, ControlState.ERROR):
                self.mujoco_backend.step(command_target, self.cfg["kps"], self.cfg["kds"])
            self.refresh_mujoco_state()
            with self.control_lock:
                self.simulated_command_count += 1
            return
        if self.command_pub is None or (rclpy is not None and not rclpy.ok()) or not joints_snapshot:
            return
        cmd = self.make_command_msg(joints_snapshot, command_target, control_state, frame_id, meas_stamp)
        if cmd is None:
            return
        if dry_run:
            with self.control_lock:
                self.simulated_command_count += 1
            return
        self.command_pub.publish(cmd)
        with self.control_lock:
            self.command_publish_count += 1

    def on_print_timer(self):
        now = time.monotonic()
        with self.control_lock:
            joint_age = now - self.latest_joint_msg_time if self.latest_joint_msg_time > 0 else float("inf")
            base_age = 0.0 if self.args.fake_base_state else (
                now - self.latest_base_msg_time if self.latest_base_msg_time > 0 else float("inf")
            )
            state = self.state
            active_runtime = self.selected_mimic() if state == ControlState.MIMIC_POLICY and self.mimic_runtimes else self.stand_runtime
            phase = compute_runtime_phase(active_runtime) if active_runtime is not None else 0.0
            stand_counter = self.stand_runtime.counter if self.stand_runtime is not None else 0
            mimic_counter = self.selected_mimic().counter if self.mimic_runtimes else 0
            max_abs_dof_vel = float(np.max(np.abs(self.latest_dof_vel)))
            max_abs_base_ang_vel = float(np.max(np.abs(self.base_ang_vel)))
            gravity = self.projected_gravity.copy()
            command_hz = self.command_loop_count / max((time.monotonic_ns() - self.command_start_ns) / 1e9, 1e-6)
            dry_run = self.dry_run_active()
        physics_status = self.mujoco_backend.physics_status() if getattr(self.mujoco_backend, "is_physics_backend", False) else None
        self.get_logger().info(
            f"state={state.value} active_policy={active_runtime.name if active_runtime else 'none'} phase={phase:.3f} dry_run_active={dry_run} "
            f"stand_counter={stand_counter} mimic_counter={mimic_counter} "
            f"joint_age={joint_age:.3f}s base_age={base_age:.3f}s "
            f"policy_count={self.policy_count} command_count={self.command_publish_count} "
            f"simulated_command_count={self.simulated_command_count} "
            f"action_max={self.last_action_max:.3f} target_delta_max={self.last_target_delta_max:.3f} "
            f"command_step_max={self.last_command_step_max:.3f} "
            f"clamp_step={self.clamp_count_step} clamp_limit={self.clamp_count_limits} "
            f"max_abs_dof_vel={max_abs_dof_vel:.3f} max_abs_base_ang_vel={max_abs_base_ang_vel:.3f} "
            f"projected_gravity={np.array2string(gravity, precision=3)} "
            f"last_error={self.last_error} "
            f"stand_ms={self.policy_stand_inference_ms:.2f} mimic_ms={self.policy_mimic_inference_ms:.2f} "
            f"policy_cb_ms={self.policy_callback_ms:.2f}/{self.policy_dt * 1000.0:.2f} policy_overrun={self.policy_overrun_count} "
            f"command_loop_count={self.command_loop_count} command_hz={command_hz:.1f} "
            f"command_interval_ms={self.command_last_interval_ms:.3f} command_overrun={self.command_overrun_count} "
            f"command_max_late_ms={self.command_max_lateness_ms:.3f} "
            f"physics={physics_status if physics_status is not None else 'off'}"
        )
        if max_abs_base_ang_vel < 0.3 and np.linalg.norm(gravity - np.array([0.0, 0.0, -1.0])) > 0.35:
            self.get_logger().warn(
                "projected_gravity is not close to [0, 0, -1] while base angular velocity is small; "
                "check IMU coordinate frame or quaternion order"
            )

    def shutdown(self):
        if not self.args.mujoco_sim and rclpy is not None and rclpy.ok():
            self.enter_emergency("shutdown")
            # Keep the sole command thread alive for at least ten damping periods.
            self.command_stop_event.wait(max(10.0 * self.args.command_period_s, 0.02))
        self.command_stop_event.set()
        if self.command_thread.is_alive():
            self.command_thread.join(timeout=1.0)
        if self.command_thread.is_alive():
            self.get_logger().error("command thread did not exit within 1.0s")
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
    parser.add_argument("--motion-file", type=str, default="motions/lateral_raise/lateral_raise_ref_motion_asap.pkl")
    parser.add_argument("--stand-policy-path", type=str, default="")
    parser.add_argument("--stand-motion-file", type=str, default="motions/stand/q1_stand_still.pkl")
    parser.add_argument("--stand-config", type=str, default="")
    parser.add_argument("--mimic-policy-path", type=str, default="")
    parser.add_argument("--mimic-motion-file", type=str, default="")
    parser.add_argument("--mimic-config", type=str, default="")
    parser.add_argument("--mimic-name", type=str, default="lateral_raise")
    parser.add_argument("--cycle-time", type=float, default=0.0)
    parser.add_argument("--urdf", type=str, default="robots/q1/q1_22dof_box.urdf")
    parser.add_argument("--robot-yaml", type=str, default="robots/q1/q1_22dof.yaml")
    parser.add_argument("--enable-policy", action="store_true")
    parser.add_argument(
        "--upper-body-policy-only",
        action="store_true",
        help="deprecated and forbidden: dual-policy mimic always controls all 22 DoF",
    )
    parser.add_argument("--joint-state-topic", type=str, default="/aima/hal/joint/state")
    parser.add_argument("--joint-command-topic", type=str, default="/aima/hal/joint/command")
    parser.add_argument("--imu-topic", type=str, default="/aima/hal/imu/state")
    parser.add_argument("--imu-msg-type", type=str, default="sensor_msgs/msg/Imu")
    parser.add_argument("--odom-topic", type=str, default="")
    parser.add_argument("--mujoco-sim", action="store_true", help="use an in-process MuJoCo Q1 instead of real robot topics")
    parser.add_argument("--mujoco-viewer", action="store_true", help="open a MuJoCo viewer for --mujoco-sim")
    parser.add_argument("--mujoco-viewer-hz", type=float, default=60.0)
    parser.add_argument("--mujoco-auto-start-policy", action="store_true", help="automatically run GET_READY then enter STAND_POLICY in --mujoco-sim")
    parser.add_argument("--mujoco-start-delay-s", type=float, default=1.0)
    parser.add_argument("--mujoco-root-height", type=float, default=0.42)
    parser.add_argument(
        "--mujoco-control-mode",
        choices=("kinematic", "pd"),
        default="kinematic",
        help="kinematic is stable for policy visualization; pd is an experimental dynamics mode",
    )
    parser.add_argument("--mujoco-kinematic-tau", type=float, default=0.04)
    parser.add_argument("--mujoco-kp-scale", type=float, default=0.05)
    parser.add_argument("--mujoco-kd-scale", type=float, default=1.0)
    parser.add_argument("--mujoco-joint-armature", type=float, default=0.01)
    parser.add_argument("--mujoco-joint-damping", type=float, default=0.2)
    parser.add_argument(
        "--mujoco-visual-geometry",
        choices=("auto", "mesh", "simplified"),
        default="auto",
        help="geometry used when generating the MuJoCo sim model from --urdf",
    )
    parser.add_argument("--mujoco-free-root", action="store_true", help="let MuJoCo simulate the free root instead of locking it upright")
    parser.add_argument("--physics-mjcf", type=str, default="", help="collision-enabled MJCF for free-root PD ground-contact simulation")
    parser.add_argument("--physics-solver-iterations", type=int, default=100)
    parser.add_argument("--physics-solver-ls-iterations", type=int, default=50)
    parser.add_argument("--physics-assist-kp-pos", type=float, default=300.0)
    parser.add_argument("--physics-assist-kd-pos", type=float, default=100.0)
    parser.add_argument("--physics-assist-kp-rot", type=float, default=50.0)
    parser.add_argument("--physics-assist-kd-rot", type=float, default=10.0)
    parser.add_argument("--fake-base-state", action="store_true")
    parser.add_argument("--ack-fake-base-bench-only", action="store_true")
    parser.add_argument("--command-period-s", type=float, default=0.002)
    parser.add_argument("--max-command-step-rad", type=float, default=0.01)
    parser.add_argument("--emergency-damping", type=float, default=5.0)
    parser.add_argument("--hold-uncontrolled-stiffness", type=float, default=0.0)
    parser.add_argument("--hold-uncontrolled-damping", type=float, default=5.0)
    parser.add_argument("--state-timeout-s", type=float, default=0.2)
    parser.add_argument("--get-ready-duration-s", type=float, default=10.0)
    parser.add_argument("--pre-interp-s", type=float, default=None)
    parser.add_argument("--pre-hold-s", type=float, default=None)
    parser.add_argument("--post-hold-s", type=float, default=None)
    parser.add_argument("--post-handoff-s", type=float, default=0.5)
    parser.add_argument("--post-interp-s", type=float, default=None)
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
    if args.upper_body_policy_only:
        raise ValueError("--upper-body-policy-only is forbidden: formal mimic playback must control all 22 DoF")
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
    if args.mujoco_sim and args.enable_motors:
        raise ValueError("--mujoco-sim is a simulator-only mode and cannot be combined with --enable-motors")
    if args.mujoco_sim and args.fake_base_state:
        raise ValueError("--mujoco-sim already provides base state; do not combine it with --fake-base-state")
    if args.physics_mjcf:
        if not args.mujoco_sim:
            raise ValueError("--physics-mjcf requires --mujoco-sim")
        if args.mujoco_control_mode != "pd" or not args.mujoco_free_root:
            raise ValueError("--physics-mjcf requires --mujoco-control-mode pd --mujoco-free-root")
        if not Path(args.physics_mjcf).is_file():
            raise FileNotFoundError(f"physics MJCF not found: {args.physics_mjcf}")
        if args.physics_solver_iterations <= 0 or args.physics_solver_ls_iterations <= 0:
            raise ValueError("physics solver iterations must be positive")
    if args.mujoco_viewer and not args.mujoco_sim:
        raise ValueError("--mujoco-viewer requires --mujoco-sim")
    if args.mujoco_viewer_hz <= 0.0:
        raise ValueError("--mujoco-viewer-hz must be > 0")
    if args.mujoco_start_delay_s < 0.0:
        raise ValueError("--mujoco-start-delay-s must be >= 0")
    if args.mujoco_auto_start_policy and not args.mujoco_sim:
        raise ValueError("--mujoco-auto-start-policy requires --mujoco-sim")
    if args.mujoco_auto_start_policy and not args.enable_policy:
        raise ValueError("--mujoco-auto-start-policy requires --enable-policy")
    if args.mujoco_root_height <= 0.0:
        raise ValueError("--mujoco-root-height must be > 0")
    if args.mujoco_kinematic_tau <= 0.0:
        raise ValueError("--mujoco-kinematic-tau must be > 0")
    if args.mujoco_kp_scale <= 0.0 or args.mujoco_kd_scale <= 0.0:
        raise ValueError("--mujoco-kp-scale and --mujoco-kd-scale must be > 0")
    if args.mujoco_joint_armature < 0.0 or args.mujoco_joint_damping < 0.0:
        raise ValueError("--mujoco-joint-armature and --mujoco-joint-damping must be >= 0")
    if args.enable_policy and not args.stand_policy_path:
        raise ValueError("--stand-policy-path is required with --enable-policy")
    if args.enable_policy and not args.mimic_policy_path:
        raise ValueError("--mimic-policy-path (or legacy --policy-path) is required with --enable-policy")
    for name in ("pre_interp_s", "pre_hold_s", "post_hold_s", "post_interp_s", "get_ready_duration_s"):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
    if args.post_handoff_s <= 0.0 or args.post_handoff_s > args.post_hold_s:
        raise ValueError("--post-handoff-s must satisfy 0 < post-handoff-s <= post-hold-s")
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
    if not args.mujoco_sim and not args.fake_base_state and not args.imu_topic and not args.odom_topic:
        raise ValueError(
            "No base-state source configured. Provide --imu-topic/--odom-topic, or use "
            "--fake-base-state for suspended/bench validation only."
        )


def run_mujoco_viewer(node, args):
    if not args.mujoco_sim or node.mujoco_backend is None:
        raise RuntimeError("MuJoCo viewer requires --mujoco-sim")
    if mujoco is None or not hasattr(mujoco, "viewer"):
        raise RuntimeError("mujoco.viewer is unavailable. Reinstall MuJoCo with: python3 -m pip install mujoco")

    backend = node.mujoco_backend
    sleep_s = 1.0 / max(float(args.mujoco_viewer_hz), 1.0)
    node.get_logger().info(
        f"Opening MuJoCo viewer at {args.mujoco_viewer_hz:.1f}Hz. "
        "Use terminal keys i/]/;/\'/[/o/q for the dual-policy state machine."
    )
    with mujoco.viewer.launch_passive(backend.model, backend.data) as viewer:
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 140
        viewer.cam.elevation = -15
        while viewer.is_running() and rclpy.ok():
            with backend.lock:
                viewer.sync()
            time.sleep(sleep_s)


def main():
    args = parse_args()
    args.config = resolve_path(args.config)
    # Legacy single-policy arguments remain the mimic fallback only.
    args.mimic_policy_path = args.mimic_policy_path or args.policy_path
    args.mimic_motion_file = args.mimic_motion_file or args.motion_file
    args.stand_config = args.stand_config or args.config
    args.mimic_config = args.mimic_config or args.config
    args.stand_policy_path = resolve_path(args.stand_policy_path) if args.stand_policy_path else ""
    args.stand_motion_file = resolve_path(args.stand_motion_file)
    args.mimic_policy_path = resolve_path(args.mimic_policy_path)
    args.mimic_motion_file = resolve_path(args.mimic_motion_file)
    args.stand_config = resolve_path(args.stand_config)
    args.mimic_config = resolve_path(args.mimic_config)
    args.urdf = resolve_path(args.urdf)
    args.robot_yaml = resolve_path(args.robot_yaml)
    args.physics_mjcf = resolve_path(args.physics_mjcf) if args.physics_mjcf else ""
    if rclpy is None:
        raise RuntimeError(
            "ROS2 Python package rclpy is unavailable. Source the SDK workspace first, "
            "for example: source /home/hoho/q1_wordandexample/202606091310/install/setup.bash"
        )
    if not args.mujoco_sim and (JointCommandArray is None or JointStateArray is None):
        raise RuntimeError(
            "ROS2/Q1 SDK Python packages are unavailable. Source the SDK workspace first, "
            "for example: source /home/hoho/q1_wordandexample/202606091310/install/setup.bash"
        )
    if args.mujoco_sim and mujoco is None:
        raise RuntimeError("MuJoCo is unavailable. Install it with: python3 -m pip install mujoco")

    stand_cfg = read_conf(args.stand_config)
    mimic_cfg = read_conf(args.mimic_config)
    stand_cycle, stand_cycle_source = infer_cycle_time_from_motion(args.stand_motion_file)
    mimic_cycle, mimic_cycle_source = infer_cycle_time_from_motion(args.mimic_motion_file)
    if stand_cycle is not None:
        stand_cfg["cycle_time"] = float(stand_cycle)
    if args.cycle_time > 0.0:
        mimic_cfg["cycle_time"] = float(args.cycle_time)
        mimic_cycle_source = "--cycle-time (legacy mimic override)"
    elif mimic_cycle is not None:
        mimic_cfg["cycle_time"] = float(mimic_cycle)
    stand_cfg["phase_wrap"] = True
    stand_cfg["stop_at_motion_end"] = False
    mimic_cfg["phase_wrap"] = False
    mimic_cfg["stop_at_motion_end"] = True
    for role, cfg in (("stand", stand_cfg), ("mimic", mimic_cfg)):
        if cfg["cycle_time"] <= 0.0:
            raise ValueError(f"{role} cycle_time must come from its motion pkl or config")
        cfg["simulation_duration"] = float(max(cfg.get("simulation_duration", 0.0), cfg["cycle_time"]))
        if cfg["num_actions"] != NUM_ACTIONS or len(cfg["dof_names"]) != NUM_ACTIONS:
            raise ValueError(f"{role} config must contain 22 actions and 22 joint names")
    stand_policy_dt = stand_cfg["simulation_dt"] * stand_cfg["control_decimation"]
    mimic_policy_dt = mimic_cfg["simulation_dt"] * mimic_cfg["control_decimation"]
    if not np.isclose(stand_policy_dt, mimic_policy_dt):
        raise ValueError(f"stand/mimic policy_dt must match; got {stand_policy_dt} and {mimic_policy_dt}")
    if stand_cfg["dof_names"] != mimic_cfg["dof_names"]:
        raise ValueError("stand and mimic policy joint order must be identical")
    model_cfg = mimic_cfg["mimic_models"].get(args.mimic_name, {})
    for arg_name, yaml_key, default in (
        ("pre_interp_s", "pre_interp_s", 1.5),
        ("pre_hold_s", "pre_hold_s", 1.0),
        ("post_hold_s", "post_hold_s", 0.5),
        ("post_interp_s", "post_interp_s", 1.5),
    ):
        if getattr(args, arg_name) is None:
            setattr(args, arg_name, float(model_cfg.get(yaml_key, default)))
    mimic_fallback = model_cfg.get("start_upper_body_dof_pos")
    mimic_order = model_cfg.get("motion_joint_order", mimic_cfg.get("motion_joint_order"))
    mimic_start_upper, mimic_start_source = resolve_start_upper_pose(
        args.mimic_motion_file, mimic_cfg["dof_names"], UPPER_BODY_INDICES, mimic_fallback, mimic_order
    )
    try:
        _, stand_start_source = resolve_start_upper_pose(
            args.stand_motion_file, stand_cfg["dof_names"], UPPER_BODY_INDICES,
            stand_cfg.get("stand_upper_body_dof_pos"), stand_cfg.get("motion_joint_order")
        )
    except ValueError:
        stand_start_source = "stand default_dof_pos upper body fallback (no retargeted Q1 trajectory/YAML override)"
        print("WARNING:", stand_start_source)
    limits = load_q1_joint_limits(stand_cfg["dof_names"], args.urdf, args.robot_yaml)
    validate_args(args)

    print("=" * 72)
    print("Q1 Motion Tracking Sim2Real ROS2")
    print("=" * 72)
    print(f"stand_config={args.stand_config}")
    print(f"stand_policy_path={args.stand_policy_path}")
    print(f"stand_motion_file={args.stand_motion_file}")
    print(f"stand_cycle_time={stand_cfg['cycle_time']:.6f}s source={stand_cycle_source}")
    print(f"mimic_config={args.mimic_config}")
    print(f"mimic_policy_path={args.mimic_policy_path}")
    print(f"mimic_motion_file={args.mimic_motion_file}")
    print(f"mimic_cycle_time={mimic_cfg['cycle_time']:.6f}s source={mimic_cycle_source}")
    print(f"policy_dt stand={stand_policy_dt:.6f}s mimic={mimic_policy_dt:.6f}s")
    print(f"stand_upper_pose_source={stand_start_source}")
    print(f"mimic_start_upper_pose_source={mimic_start_source}")
    print(f"urdf={args.urdf}")
    print(f"robot_yaml={args.robot_yaml}")
    print(f"joint_state_topic={args.joint_state_topic}")
    print(f"joint_command_topic={args.joint_command_topic}")
    print(f"imu_topic={args.imu_topic}")
    print(f"imu_msg_type={args.imu_msg_type} adapter_mode={'sensor_msgs/msg/Imu' if args.imu_msg_type == 'sensor_msgs/msg/Imu' else 'unsupported_custom'}")
    print(
        f"mujoco_sim={args.mujoco_sim} mujoco_root_height={args.mujoco_root_height:.3f} "
        f"mujoco_visual_geometry={args.mujoco_visual_geometry} mujoco_free_root={args.mujoco_free_root} "
        f"mujoco_viewer={args.mujoco_viewer} mujoco_control_mode={args.mujoco_control_mode} "
        f"mujoco_auto_start_policy={args.mujoco_auto_start_policy}"
    )
    print(f"enable_policy={args.enable_policy} enable_motors={args.enable_motors} dry_run_active={not args.enable_motors or args.dry_run}")
    print("WARNING: Q1 must already be in low-level developer mode. High-level STAND_UP / BIPED_STAND_DEFAULT must not be active.")
    print(f"fake_base_state={args.fake_base_state}")
    if args.fake_base_state:
        print("WARNING: fake-base-state is bench/suspended only; do not use for standing motion tracking.")
    print()

    rclpy.init()
    node = Q1MotionTrackingNode(
        args, stand_cfg, mimic_cfg, limits, mimic_start_upper, mimic_start_source, stand_start_source
    )

    def handle_sigint(signum, _frame):
        if args.mujoco_sim:
            node.get_logger().warn(f"Received signal {signum}; shutting down MuJoCo sim mode")
        else:
            node.get_logger().warn(f"Received signal {signum}; entering EMERGENCY_DAMPING")
            node.enter_emergency("SIGINT/SIGTERM")
            node.command_stop_event.wait(max(10.0 * args.command_period_s, 0.02))
        rclpy.shutdown()

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    spin_thread = None
    try:
        if args.mujoco_viewer:
            spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
            spin_thread.start()
            run_mujoco_viewer(node, args)
            if rclpy.ok():
                rclpy.shutdown()
        else:
            rclpy.spin(node)
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if spin_thread is not None:
            spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
