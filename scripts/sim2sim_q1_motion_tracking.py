#!/usr/bin/env python3
"""
Q1 CR7 Motion Tracking MuJoCo sim2sim — closed-loop verification.

Loads a Q1 CR7 motion tracking ONNX policy (423-dim obs, 22-dim action),
runs it in MuJoCo with PD control and reference motion tracking.

Usage:
  python scripts/sim2sim_q1_motion_tracking.py \
      --checkpoint <exported_model.onnx> \
      --policy-format onnx \
      --train-config <checkpoint_dir/config.yaml> \
      --mujoco-xml humanoidverse/data/robots/q1/q1_22dof_box.xml \
      --motion-file humanoidverse/data/motions/q1/q1_cr7_scale045_rootz042.pkl \
      --output-dir sim2sim_outputs/q1_cr7 \
      --num-steps 300 \
      --record-video \
      --video-fps 50 \
      --show-ref-markers \
      --debug-csv
"""

import os
import sys
import argparse
import time
import csv
import yaml
import numpy as np
from collections import OrderedDict

# Must set MUJOCO_GL before importing mujoco for headless rendering
if "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco
import onnxruntime as ort

# Add repo root to path for imports
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import joblib
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

try:
    from scipy.spatial.transform import Rotation as sRot
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ============================================================================
# Quaternion helpers (wxyz convention — matches MuJoCo native)
# ============================================================================

def quat_inverse_wxyz(q):
    """Inverse of quaternion [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_rotate_wxyz(q_wxyz, v):
    """Rotate vector v by quaternion q = [w, x, y, z]."""
    q_w, q_x, q_y, q_z = float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])
    qv = np.array([q_x, q_y, q_z])
    t = 2.0 * np.cross(qv, v)
    return v + q_w * t + np.cross(qv, t)


def quat_rotate_inverse_wxyz(q_wxyz, v):
    """Rotate vector v by inverse of quaternion q = [w, x, y, z]."""
    return quat_rotate_wxyz(quat_inverse_wxyz(q_wxyz), v)


def xyzw_to_wxyz(q):
    """Convert [x, y, z, w] to [w, x, y, z]."""
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def wxyz_to_xyzw(q):
    """Convert [w, x, y, z] to [x, y, z, w]."""
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def compute_yaw_from_quat_wxyz(q_wxyz):
    """Compute yaw from wxyz quaternion."""
    w, x, y, z = float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def compute_yaw_from_quat_xyzw(q_xyzw):
    """Compute yaw from xyzw quaternion (IsaacGym convention)."""
    x, y, z, w = float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3])
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ============================================================================
# Config loading
# ============================================================================

def load_train_config(config_path):
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Training config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_sim2sim_config():
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "sim2sim_q1_motion_tracking_config.yaml")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def build_sim2sim_config(train_cfg, sim2sim_cfg, args):
    """Merge training config + sim2sim config + CLI args."""
    cfg = {}

    # --- Robot ---
    robot = train_cfg.get("robot", {})
    cfg["robot"] = robot

    # --- DOF names ---
    cfg["dof_names"] = list(robot.get("dof_names", []))
    assert len(cfg["dof_names"]) == 22, f"Expected 22 dof_names, got {len(cfg['dof_names'])}"

    # --- Default DOF positions ---
    default_joint_angles = robot.get("init_state", {}).get("default_joint_angles", {})
    cfg["default_dof_pos"] = np.array(
        [default_joint_angles.get(name, 0.0) for name in cfg["dof_names"]],
        dtype=np.float64
    )

    # --- DOF limits ---
    cfg["dof_pos_lower"] = np.array(robot.get("dof_pos_lower_limit_list", [-np.inf] * 22), dtype=np.float64)
    cfg["dof_pos_upper"] = np.array(robot.get("dof_pos_upper_limit_list", [np.inf] * 22), dtype=np.float64)
    cfg["torque_limits"] = np.array(robot.get("dof_effort_limit_list", [100.0] * 22), dtype=np.float64)

    # --- Control ---
    control = robot.get("control", {})
    cfg["control_type"] = control.get("control_type", "P")
    cfg["action_scale"] = float(control.get("action_scale", 0.25))
    clip_actions = train_cfg.get("env", {}).get("config", {}).get("normalization", {}).get("clip_actions", 100.0)
    cfg["clip_actions"] = float(clip_actions)
    clip_obs = train_cfg.get("env", {}).get("config", {}).get("normalization", {}).get("clip_observations", 100.0)
    cfg["clip_observations"] = float(clip_obs)
    cfg["clip_torques"] = control.get("clip_torques", True)

    # Build per-joint stiffness/damping from substring-matched config
    stiffness_map = control.get("stiffness", {})
    damping_map = control.get("damping", {})
    cfg["kps"] = np.zeros(22, dtype=np.float64)
    cfg["kds"] = np.zeros(22, dtype=np.float64)
    for i, name in enumerate(cfg["dof_names"]):
        for pattern, val in stiffness_map.items():
            if pattern in name:
                cfg["kps"][i] = float(val)
                break
        for pattern, val in damping_map.items():
            if pattern in name:
                cfg["kds"][i] = float(val)
                break
    for i in range(22):
        if cfg["kps"][i] == 0:
            cfg["kps"][i] = 30.0
        if cfg["kds"][i] == 0:
            cfg["kds"][i] = 1.5

    # --- Simulator ---
    sim_cfg = train_cfg.get("simulator", {}).get("config", {}).get("sim", {})
    cfg["sim_fps"] = sim_cfg.get("fps", 200)
    cfg["sim_dt"] = 1.0 / cfg["sim_fps"]
    cfg["control_decimation"] = sim_cfg.get("control_decimation", 4)
    cfg["control_dt"] = cfg["sim_dt"] * cfg["control_decimation"]

    # --- Obs config ---
    obs_cfg = train_cfg.get("obs", {})
    cfg["obs"] = obs_cfg
    cfg["actor_obs_keys"] = list(obs_cfg.get("obs_dict", {}).get("actor_obs", []))
    cfg["obs_scales"] = obs_cfg.get("obs_scales", {})
    cfg["obs_auxiliary"] = obs_cfg.get("obs_auxiliary", {})

    # Build obs_dims dict (resolves Hydra eval expressions)
    obs_dims_raw = obs_cfg.get("obs_dims", [])
    obs_dims = {}
    for item in obs_dims_raw:
        if isinstance(item, dict):
            for k, v in item.items():
                if isinstance(v, str):
                    # Resolve all Hydra variables
                    v_resolved = v.replace("${robot.dof_obs_size}", "22")
                    v_resolved = v_resolved.replace("${robot.num_bodies}", "23")
                    v_resolved = v_resolved.replace("${robot.motion.nums_extend_bodies}", "0")
                    # Handle ${eval:'expression'} → evaluate expression
                    if "${eval:" in v_resolved:
                        # Extract expression between single quotes
                        import re
                        m = re.search(r"\$\{eval:'([^']+)'\}", v_resolved)
                        if m:
                            expr = m.group(1)
                            expr = expr.replace("${robot.dof_obs_size}", "22")
                            expr = expr.replace("${robot.num_bodies}", "23")
                            expr = expr.replace("${robot.motion.nums_extend_bodies}", "0")
                            obs_dims[k] = eval(expr)
                        else:
                            obs_dims[k] = 22  # fallback
                    else:
                        try:
                            obs_dims[k] = int(v_resolved)
                        except:
                            obs_dims[k] = 22
                else:
                    obs_dims[k] = int(v)
    cfg["obs_dims"] = obs_dims

    # --- Motion config ---
    motion_cfg = robot.get("motion", {})
    cfg["motion"] = motion_cfg

    # --- Q1 CR7 phase config ---
    q1_cr7 = train_cfg.get("env", {}).get("config", {}).get("q1_cr7", {})
    if not q1_cr7:
        q1_cr7 = robot.get("q1_cr7", {})
    cfg["q1_cr7"] = {
        "crouch_phase_start": float(q1_cr7.get("crouch_phase_start", 0.20)),
        "crouch_phase_end": float(q1_cr7.get("crouch_phase_end", 0.45)),
        "takeoff_phase_start": float(q1_cr7.get("takeoff_phase_start", 0.38)),
        "takeoff_phase_end": float(q1_cr7.get("takeoff_phase_end", 0.55)),
        "flight_phase_start": float(q1_cr7.get("flight_phase_start", 0.50)),
        "flight_phase_end": float(q1_cr7.get("flight_phase_end", 0.72)),
        "landing_phase_start": float(q1_cr7.get("landing_phase_start", 0.72)),
        "landing_phase_end": float(q1_cr7.get("landing_phase_end", 0.90)),
    }

    # --- Init state ---
    init_state = robot.get("init_state", {})
    cfg["init_root_pos"] = np.array(init_state.get("pos", [0.0, 0.0, 0.42]), dtype=np.float64)
    init_rot_xyzw = np.array(init_state.get("rot", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)
    cfg["init_root_quat_xyzw"] = init_rot_xyzw  # xyzw (IsaacGym)

    # --- Body names ---
    cfg["body_names"] = list(robot.get("body_names", []))
    cfg["left_foot_name"] = robot.get("left_foot_name", "left_ankle_roll_link")
    cfg["right_foot_name"] = robot.get("right_foot_name", "right_ankle_roll_link")

    # --- MuJoCo overrides ---
    mujoco_cfg = sim2sim_cfg.get("mujoco", {})
    cfg["solver_iterations"] = mujoco_cfg.get("solver_iterations", 100)
    cfg["ls_iterations"] = mujoco_cfg.get("ls_iterations", 50)

    safety = sim2sim_cfg.get("safety", {})
    cfg["max_abs_qacc"] = safety.get("max_abs_qacc", 50000.0)
    cfg["min_root_z"] = safety.get("min_root_z", 0.05)
    cfg["max_root_z"] = safety.get("max_root_z", 3.0)

    video_cfg = sim2sim_cfg.get("video", {})
    cfg["video_fps"] = args.video_fps if args.video_fps != 50 else video_cfg.get("fps", 50)
    cfg["video_width"] = video_cfg.get("width", 1280)
    cfg["video_height"] = video_cfg.get("height", 720)

    cfg["imu_body_name"] = sim2sim_cfg.get("imu_body_name", "pelvis")
    cfg["torso_body_name"] = sim2sim_cfg.get("torso_body_name", "torso_link")

    ref_m = sim2sim_cfg.get("ref_marker", {})
    cfg["ref_marker_radius"] = ref_m.get("radius", 0.02)
    cfg["ref_marker_color"] = ref_m.get("color", [1.0, 1.0, 0.0, 1.0])
    cfg["ref_marker_body_names"] = ref_m.get("body_names_filter",
                                               ["torso_link", "left_ankle_roll_link", "right_ankle_roll_link"])

    return cfg


# ============================================================================
# Correct Minimal Motion Loader
# ============================================================================

class MotionLoader:
    """Loads Q1 CR7 motion pkl and provides get_motion_state().

    Key insight from Humanoid_Batch.fk_batch:
      dof_pos = pose_aa.sum(axis=-1)[:, 1:]   (for 1-DOF hinge joints)

    The axis-angle pose_aa[t, body_idx] stores the rotation around the
    joint axis. For 1-DOF joints, the scalar magnitude along the axis
    equals the joint angle. Summing the (x,y,z) components gives the
    signed angle.
    """

    def __init__(self, motion_file, motion_config):
        if not _HAS_JOBLIB:
            raise ImportError("joblib required for motion loading. pip install joblib")
        if not _HAS_SCIPY:
            raise ImportError("scipy required for motion FK. pip install scipy")

        data = joblib.load(motion_file)

        # Extract single motion entry
        if isinstance(data, dict):
            keys = sorted(data.keys())
            self.motion_name = keys[0]
            motion_data = data[self.motion_name]
        else:
            motion_data = data

        self._pose_aa = np.array(motion_data["pose_aa"], dtype=np.float64)       # (T, 23, 3)
        self._root_trans = np.array(motion_data["root_trans_offset"], dtype=np.float64)  # (T, 3)
        self._fps = int(motion_data["fps"])
        self._dt = 1.0 / self._fps
        self._num_frames = self._pose_aa.shape[0]
        self._duration = self._num_frames * self._dt

        # Store motion config for FK
        self._motion_config = motion_config
        self._body_names = list(motion_config.get("body_names", []))
        self._dof_names_motion = list(motion_config.get("dof_names", []))

        # --- Build FK hierarchy from MuJoCo XML body relationships ---
        # For now: use Q1 skeleton order (23 bodies)
        # body 0 = pelvis (root), bodies 1-22 = joints
        self._num_bodies = self._pose_aa.shape[1]  # 23

        # --- DOF positions: pose.sum(axis=-1)[:, 1:] ---
        self._dof_pos = self._pose_aa.sum(axis=-1)[:, 1:]  # (T, 22)

        # --- DOF velocities via central differences ---
        self._dof_vel = np.zeros_like(self._dof_pos)
        dt = self._dt
        if self._num_frames >= 3:
            self._dof_vel[1:-1] = (self._dof_pos[2:] - self._dof_pos[:-2]) / (2 * dt)
        if self._num_frames >= 2:
            self._dof_vel[0] = (self._dof_pos[1] - self._dof_pos[0]) / dt
            self._dof_vel[-1] = (self._dof_pos[-1] - self._dof_pos[-2]) / dt

        # --- Root motion ---
        self._root_pos = self._root_trans.copy()  # (T, 3)

        # Root rotation: pose_aa[:, 0, :] → quaternion (xyzw)
        self._root_rot_xyzw = np.zeros((self._num_frames, 4), dtype=np.float64)
        for t in range(self._num_frames):
            aa = self._pose_aa[t, 0]
            angle = np.linalg.norm(aa)
            if angle < 1e-10:
                self._root_rot_xyzw[t] = [0.0, 0.0, 0.0, 1.0]
            else:
                axis = aa / angle
                half = angle / 2.0
                s = np.sin(half)
                self._root_rot_xyzw[t] = [axis[0] * s, axis[1] * s, axis[2] * s, np.cos(half)]

        # Root velocities
        self._root_vel = np.zeros_like(self._root_pos)
        if self._num_frames >= 3:
            self._root_vel[1:-1] = (self._root_pos[2:] - self._root_pos[:-2]) / (2 * dt)
        if self._num_frames >= 2:
            self._root_vel[0] = (self._root_pos[1] - self._root_pos[0]) / dt
            self._root_vel[-1] = (self._root_pos[-1] - self._root_pos[-2]) / dt

        # Root angular velocity (approximate from root_rot quaternion differences)
        self._root_ang_vel = np.zeros((self._num_frames, 3), dtype=np.float64)
        for t in range(1, self._num_frames):
            q0_xyzw = self._root_rot_xyzw[t - 1]
            q1_xyzw = self._root_rot_xyzw[t]
            # q1 = q0 * dq  →  dq = q0^-1 * q1
            r0 = sRot.from_quat(q0_xyzw)
            r1 = sRot.from_quat(q1_xyzw)
            dr = r0.inv() * r1
            ang_vel = dr.as_rotvec() / dt
            self._root_ang_vel[t] = ang_vel
        if self._num_frames >= 2:
            self._root_ang_vel[0] = self._root_ang_vel[1].copy()

        # --- Body positions via FK ---
        # Build all body quaternions and positions
        self._body_quats_xyzw = np.zeros((self._num_frames, self._num_bodies, 4), dtype=np.float64)
        self._body_pos = np.zeros((self._num_frames, self._num_bodies, 3), dtype=np.float64)
        self._build_body_fk()

    def _build_body_fk(self):
        """Forward kinematics: convert pose_aa to global body positions.

        Uses scipy Rotation for quaternion composition.
        Hierarchy matches Q1 MuJoCo XML DFS order.
        """
        # Q1 skeleton hierarchy (parent indices for each body)
        # Body 0 = pelvis (root)
        # Bodies 1-6 = left leg chain
        # Bodies 7-12 = right leg chain
        # Body 13 = waist_roll_link (child of pelvis)
        # Body 14 = torso_link (child of waist_roll)
        # Bodies 15-18 = left arm chain (children of torso)
        # Bodies 19-22 = right arm chain (children of torso)
        parent = {
            0: None,
            1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5,           # left leg
            7: 0, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11,     # right leg
            13: 0, 14: 13,                                   # waist, torso
            15: 14, 16: 15, 17: 16, 18: 17,                 # left arm
            19: 14, 20: 19, 21: 20, 22: 21,                 # right arm
        }

        # Joint axes (from MuJoCo XML)
        joint_axes = {
            1: np.array([0, 1, 0]),   # left_hip_pitch
            2: np.array([1, 0, 0]),   # left_hip_roll
            3: np.array([0, 0, 1]),   # left_hip_yaw
            4: np.array([0, 1, 0]),   # left_knee
            5: np.array([0, 1, 0]),   # left_ankle_pitch
            6: np.array([1, 0, 0]),   # left_ankle_roll
            7: np.array([0, 1, 0]),   # right_hip_pitch
            8: np.array([1, 0, 0]),   # right_hip_roll
            9: np.array([0, 0, 1]),   # right_hip_yaw
            10: np.array([0, 1, 0]),  # right_knee
            11: np.array([0, 1, 0]),  # right_ankle_pitch
            12: np.array([1, 0, 0]),  # right_ankle_roll
            13: np.array([1, 0, 0]),  # waist_roll
            14: np.array([0, 0, 1]),  # waist_yaw
            15: np.array([0, 1, 0]),  # left_shoulder_pitch
            16: np.array([1, 0, 0]),  # left_shoulder_roll
            17: np.array([0, 0, 1]),  # left_shoulder_yaw
            18: np.array([0, 1, 0]),  # left_elbow
            19: np.array([0, 1, 0]),  # right_shoulder_pitch
            20: np.array([1, 0, 0]),  # right_shoulder_roll
            21: np.array([0, 0, 1]),  # right_shoulder_yaw
            22: np.array([0, 1, 0]),  # right_elbow
        }

        # Body offsets from parent (from MuJoCo XML, approximate)
        # These are the pos attributes in the XML
        body_offsets = {
            0: np.array([0.0015, 0, 0.41]),          # pelvis in world frame (base pos)
            1: np.array([0.0015, 0.067658, 0.021375]),  # left_hip_pitch from pelvis
            2: np.array([0, 0.036796, -0.034098]),      # left_hip_roll
            3: np.array([0, 0, -0.05825]),              # left_hip_yaw
            4: np.array([-0.015, 0, -0.107]),           # left_knee
            5: np.array([0.014976, 0, -0.17]),          # left_ankle_pitch
            6: np.array([-0.0069, 0, -0.02]),           # left_ankle_roll
            7: np.array([0.0015, -0.067094, 0.02158]),  # right_hip_pitch
            8: np.array([0, -0.037396, -0.034098]),     # right_hip_roll
            9: np.array([0, 0, -0.05825]),              # right_hip_yaw
            10: np.array([-0.015, 0, -0.107]),          # right_knee
            11: np.array([0.015015, 0, -0.16999]),      # right_ankle_pitch
            12: np.array([-0.0069386, 0, -0.02]),       # right_ankle_roll
            13: np.array([0.0015, 0, 0.0895]),          # waist_roll
            14: np.array([0, 0, 0.041]),                # waist_yaw (torso)
            15: np.array([-0.01, 0.086836, 0.17439]),   # left_shoulder_pitch
            16: np.array([0, 0.0358, 0]),               # left_shoulder_roll
            17: np.array([0, 0, -0.0736]),              # left_shoulder_yaw
            18: np.array([0.01, 0, -0.0706]),           # left_elbow
            19: np.array([-0.01, -0.086836, 0.17439]),  # right_shoulder_pitch
            20: np.array([0, -0.0358, 0]),              # right_shoulder_roll
            21: np.array([0, 0, -0.0736]),              # right_shoulder_yaw
            22: np.array([0.01, 0, -0.0706]),           # right_elbow
        }

        for t in range(self._num_frames):
            # Global transforms for each body
            global_rot = {}   # body_idx → sRot (global orientation)
            global_pos = {}   # body_idx → ndarray (global position)

            # Root
            global_rot[0] = sRot.from_quat(self._root_rot_xyzw[t])
            global_pos[0] = self._root_pos[t].copy()

            # Children in BFS order
            for b in range(1, self._num_bodies):
                p = parent[b]
                if p is None:
                    continue

                # Local rotation from axis-angle
                aa = self._pose_aa[t, b]
                angle = np.linalg.norm(aa)
                if angle < 1e-10:
                    local_rot = sRot.identity()
                else:
                    axis = aa / angle
                    local_rot = sRot.from_rotvec(axis * angle)

                # Offset from parent (in parent's frame)
                offset = body_offsets.get(b, np.zeros(3))

                # Global rotation
                global_rot[b] = global_rot[p] * local_rot

                # Global position
                global_pos[b] = global_pos[p] + global_rot[p].apply(offset)

            # Store
            for b in range(self._num_bodies):
                self._body_quats_xyzw[t, b] = global_rot[b].as_quat()  # xyzw
                self._body_pos[t, b] = global_pos[b]

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def duration(self):
        return self._duration

    @property
    def fps(self):
        return self._fps

    @property
    def dt(self):
        return self._dt

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        """Return motion state dict matching MotionLibRobot.get_motion_state().

        All returned arrays are in batch format (1, ...).
        Uses xyzw quaternion convention matching IsaacGym.
        """
        if isinstance(motion_times, (int, float, np.floating)):
            motion_times = np.array([motion_times])
        elif isinstance(motion_times, list):
            motion_times = np.array(motion_times, dtype=np.float64)

        times = np.clip(np.asarray(motion_times, dtype=np.float64), 0.0, self._duration - 1e-6)
        frame_idx = times / self._dt
        f0 = np.floor(frame_idx).astype(int)
        f1 = np.minimum(f0 + 1, self._num_frames - 1)
        alpha = (frame_idx - f0).reshape(-1, *([1] * (len(self._dof_pos.shape) - 1)))

        def _lerp(arr):
            return (arr[f0] * (1 - alpha) + arr[f1] * alpha)

        return {
            "root_pos": _lerp(self._root_pos),
            "root_rot": _lerp(self._root_rot_xyzw),  # xyzw (IsaacGym convention)
            "root_vel": _lerp(self._root_vel),
            "root_ang_vel": _lerp(self._root_ang_vel),
            "dof_pos": _lerp(self._dof_pos),
            "dof_vel": _lerp(self._dof_vel),
            "rg_pos_t": _lerp(self._body_pos),
            "rg_rot_t": _lerp(self._body_quats_xyzw),  # xyzw
            "body_vel_t": np.zeros_like(_lerp(self._body_pos)),
            "body_ang_vel_t": np.zeros((len(motion_times), self._num_bodies, 3), dtype=np.float64),
        }

    def get_motion_length(self, motion_ids=None):
        return np.array([self._duration])

    def sample_time(self, motion_ids, truncate_time=None):
        return np.zeros(len(motion_ids), dtype=np.float64)


# ============================================================================
# Policy loading
# ============================================================================

def load_onnx_policy(policy_path, expected_obs_dim, expected_act_dim):
    if not os.path.isfile(policy_path):
        raise FileNotFoundError(f"ONNX policy not found: {policy_path}")
    session = ort.InferenceSession(policy_path)
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]

    print(f"ONNX input:  name={inp.name}, shape={inp.shape}")
    print(f"ONNX output: name={out.name}, shape={out.shape}")

    if inp.shape[-1] not in (expected_obs_dim, "obs", None):
        raise ValueError(f"Expected ONNX input dim={expected_obs_dim}, got {inp.shape[-1]}")
    if out.shape[-1] not in (expected_act_dim, "actions", None):
        raise ValueError(f"Expected ONNX output dim={expected_act_dim}, got {out.shape[-1]}")

    dummy = np.zeros((1, expected_obs_dim), dtype=np.float32)
    result = session.run([out.name], {inp.name: dummy})[0]
    assert result.shape == (1, expected_act_dim), \
        f"Dummy inference: expected (1,{expected_act_dim}), got {result.shape}"
    assert np.isfinite(result).all(), "Dummy inference returned NaN/Inf"
    print(f"  Dummy inference OK: shape={result.shape}, mean={result.mean():.4f}, max={result.max():.4f}")

    return {"session": session, "input_name": inp.name, "output_name": out.name}


def policy_infer(policy, obs):
    out_name = policy["output_name"]
    inp_name = policy["input_name"]
    return policy["session"].run([out_name], {inp_name: obs.astype(np.float32)})[0]


# ============================================================================
# MuJoCo setup
# ============================================================================

def _build_full_mujoco_xml(xml_path):
    """Build a complete MuJoCo XML from the kinematic-only ASAP Q1 XML.

    Adds: visual/skybox, ground plane, lighting, mesh-based visuals,
    collision geoms, inertial/mass, joint limits, actuator force ranges.

    Does NOT modify the original file.
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 1. Add compiler meshdir for STL assets
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("angle", "radian")
    # Mesh directory relative to the Q1 robot data dir
    mesh_dir = os.path.join(_REPO_ROOT, "humanoidverse", "data", "robots", "q1", "meshes")
    compiler.set("meshdir", mesh_dir)

    # 2. Visual settings for offscreen rendering
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_vis = visual.find("global")
    if global_vis is None:
        global_vis = ET.SubElement(visual, "global")
    global_vis.set("offwidth", "1280")
    global_vis.set("offheight", "720")

    # 3. Default joint settings
    default = root.find("default")
    if default is None:
        default = ET.SubElement(root, "default")
    if default.find("joint") is not None:
        default.remove(default.find("joint"))
    ET.SubElement(default, "joint", {"type": "hinge", "actuatorfrclimited": "true"})

    # 4. Assets: meshes, textures, materials
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        # Insert after compiler
        idx = list(root).index(compiler) if compiler in root else 0
        root.insert(idx + 1, asset)
    else:
        # Clear existing
        for child in list(asset):
            asset.remove(child)

    # STL mesh assets
    mesh_names = [
        "head_link", "left_ankle_pitch_link", "left_ankle_roll_link",
        "left_elbow_link", "left_hip_pitch_link", "left_hip_roll_link",
        "left_hip_yaw_link", "left_knee_link", "left_shoulder_pitch_link",
        "left_shoulder_roll_link", "left_shoulder_yaw_link", "pelvis",
        "right_ankle_pitch_link", "right_ankle_roll_link", "right_elbow_link",
        "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
        "right_knee_link", "right_shoulder_pitch_link", "right_shoulder_roll_link",
        "right_shoulder_yaw_link", "torso_link", "waist_roll_link",
    ]
    for name in mesh_names:
        ET.SubElement(asset, "mesh", {"name": name, "file": f"{name}.STL"})

    # Skybox texture
    ET.SubElement(asset, "texture", {
        "type": "skybox", "builtin": "flat",
        "rgb1": "0.3 0.4 0.5", "rgb2": "0.3 0.4 0.5",
        "width": "512", "height": "3072",
    })
    # Ground texture
    ET.SubElement(asset, "texture", {
        "name": "groundplane", "type": "2d", "builtin": "checker", "mark": "edge",
        "rgb1": "0.2 0.3 0.4", "rgb2": "0.15 0.25 0.35",
        "markrgb": "0.6 0.6 0.6", "width": "300", "height": "300",
    })
    ET.SubElement(asset, "material", {
        "name": "groundplane", "texture": "groundplane",
        "texuniform": "true", "texrepeat": "5 5", "reflectance": "0.2",
    })

    # 5. Worldbody: add ground plane + lights + enhance bodies
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    # Ground plane — soft contact to avoid large forces on penetration
    ET.SubElement(worldbody, "geom", {
        "name": "floor", "size": "0 0 0.05", "type": "plane",
        "material": "groundplane", "contype": "4", "conaffinity": "11", "condim": "3",
        "solref": "0.02 1", "solimp": "0.9 0.95 0.001",
    })
    # Directional light
    ET.SubElement(worldbody, "light", {
        "directional": "true", "diffuse": "0.7 0.7 0.7",
        "specular": "0.2 0.2 0.2", "pos": "3 2 4", "dir": "-3 -2 -4",
    })
    # Ambient light
    ET.SubElement(worldbody, "light", {
        "directional": "false", "diffuse": "0.3 0.3 0.3",
        "pos": "0 0 3",
    })
    # Fixed camera for rendering
    ET.SubElement(worldbody, "camera", {
        "name": "render_cam",
        "mode": "fixed",
        "pos": "3.5 -3.5 2.0",
        "xyaxes": "0.707 0.707 0 -0.408 0.408 0.816",
    })

    # 6. Enhance each body: add mesh geom, collision geom, inertial
    # Body-specific sizes for collision boxes (approximate from goalkeeper XML)
    body_collision_specs = {
        "pelvis": {"type": "box", "size": "0.10 0.10 0.065", "pos": "0.0 0.0 0.03"},
        "left_hip_pitch_link": {"type": "box", "size": "0.09 0.08 0.08", "pos": "0.0 0.0 -0.03"},
        "left_hip_roll_link": {"type": "box", "size": "0.09 0.08 0.08", "pos": "0.0 0.0 -0.03"},
        "left_hip_yaw_link": {"type": "box", "size": "0.09 0.09 0.07", "pos": "0.0 0.0 -0.06"},
        "left_knee_link": {"type": "box", "size": "0.08 0.09 0.12", "pos": "0.01 0.0 -0.09"},
        "left_ankle_pitch_link": {"type": "box", "size": "0.06 0.06 0.05", "pos": "0.0 0.0 0.0"},
        "left_ankle_roll_link": {"type": "sphere", "size": "0.04", "pos": "0.0 0.0 -0.02"},
        "right_hip_pitch_link": {"type": "box", "size": "0.09 0.08 0.08", "pos": "0.0 0.0 -0.03"},
        "right_hip_roll_link": {"type": "box", "size": "0.09 0.08 0.08", "pos": "0.0 0.0 -0.03"},
        "right_hip_yaw_link": {"type": "box", "size": "0.09 0.09 0.07", "pos": "0.0 0.0 -0.06"},
        "right_knee_link": {"type": "box", "size": "0.08 0.09 0.12", "pos": "0.01 0.0 -0.09"},
        "right_ankle_pitch_link": {"type": "box", "size": "0.06 0.06 0.05", "pos": "0.0 0.0 0.0"},
        "right_ankle_roll_link": {"type": "sphere", "size": "0.04", "pos": "0.0 0.0 -0.02"},
        "waist_roll_link": {"type": "box", "size": "0.07 0.12 0.05", "pos": "0.0 0.0 0.0"},
        "torso_link": {"type": "box", "size": "0.12 0.15 0.15", "pos": "0.0 0.0 0.05"},
        "left_shoulder_pitch_link": {"type": "box", "size": "0.05 0.06 0.06", "pos": "0.0 0.02 0.0"},
        "left_shoulder_roll_link": {"type": "box", "size": "0.04 0.04 0.05", "pos": "0.0 0.0 -0.04"},
        "left_shoulder_yaw_link": {"type": "box", "size": "0.04 0.04 0.05", "pos": "0.0 0.0 -0.04"},
        "left_elbow_link": {"type": "box", "size": "0.04 0.04 0.05", "pos": "0.01 0.0 -0.04"},
        "right_shoulder_pitch_link": {"type": "box", "size": "0.05 0.06 0.06", "pos": "0.0 -0.02 0.0"},
        "right_shoulder_roll_link": {"type": "box", "size": "0.04 0.04 0.05", "pos": "0.0 0.0 -0.04"},
        "right_shoulder_yaw_link": {"type": "box", "size": "0.04 0.04 0.05", "pos": "0.0 0.0 -0.04"},
        "right_elbow_link": {"type": "box", "size": "0.04 0.04 0.05", "pos": "0.01 0.0 -0.04"},
    }

    # Joint limits and actuator force ranges from training config
    dof_pos_lower = [
        -3.0543, -0.69813, -1.5708, 0.0, -0.7854, -0.34907,
        -3.0543, -1.5708, -1.5708, 0.0, -0.7854, -0.34907,
        -0.2618, -1.5708,
        -3.1416, -0.087266, -1.5708, -0.87266,
        -3.1416, -2.7925, -1.5708, -0.87266,
    ]
    dof_pos_upper = [
        1.5708, 1.5708, 1.5708, 2.4435, 0.43633, 0.34907,
        1.5708, 0.69813, 1.5708, 2.4435, 0.43633, 0.34907,
        0.2618, 1.5708,
        1.5708, 2.7925, 1.5708, 1.6581,
        1.5708, 0.087266, 1.5708, 1.6581,
    ]
    dof_effort = [
        36., 36., 36., 36., 22., 22.,
        36., 36., 36., 36., 22., 22.,
        36., 36.,
        22., 22., 22., 22.,
        22., 22., 22., 22.,
    ]
    # Build joint name → (range_lower, range_upper, effort_limit) from DFS order
    joint_limits = {}
    for i, name in enumerate([
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_roll_joint", "waist_yaw_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "left_elbow_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint",
    ]):
        joint_limits[name] = (dof_pos_lower[i], dof_pos_upper[i], dof_effort[i])

    # Average mass per body (for approximate inertia)
    body_masses = {
        "pelvis": 1.7, "torso_link": 1.5,
        "left_knee_link": 1.2, "right_knee_link": 1.2,
        "left_hip_yaw_link": 0.76, "right_hip_yaw_link": 0.76,
        "left_hip_pitch_link": 0.7, "right_hip_pitch_link": 0.7,
        "left_ankle_pitch_link": 0.4, "right_ankle_pitch_link": 0.4,
        "left_ankle_roll_link": 0.34, "right_ankle_roll_link": 0.34,
        "left_hip_roll_link": 0.19, "right_hip_roll_link": 0.19,
    }

    def _enhance_body(body_elem, body_name, is_root=False):
        """Add mesh visual, collision geom, and inertial to a body."""
        # Don't add inertial/geom to bodies that already have them
        has_inertial = body_elem.find("inertial") is not None

        # Inertial
        if not has_inertial:
            mass = body_masses.get(body_name, 0.5)
            m = max(mass, 0.01)
            diag = f"{m*0.01:.6f} {m*0.01:.6f} {m*0.01:.6f}"
            inertial = ET.Element("inertial", {
                "pos": "0 0 0", "mass": str(mass), "diaginertia": diag,
            })
            # Insert inertial before first geom or at start
            insert_at = 0
            for i, child in enumerate(body_elem):
                if child.tag in ("joint",):
                    insert_at = i + 1
            body_elem.insert(insert_at, inertial)

        # Visual mesh geom (group=1, no collision)
        has_vis_geom = any(
            g.tag == "geom" and g.get("group") == "1" for g in body_elem
        )
        if not has_vis_geom:
            vis_geom = ET.Element("geom", {
                "type": "mesh", "mesh": body_name,
                "contype": "0", "conaffinity": "0", "group": "1",
                "density": "0", "rgba": "0.7 0.7 0.7 1",
            })
            # Place after inertial
            insert_at = 0
            for i, child in enumerate(body_elem):
                if child.tag == "inertial":
                    insert_at = i + 1
                elif child.tag == "joint" and insert_at == 0:
                    insert_at = i + 1
            body_elem.insert(insert_at, vis_geom)

        # Collision geom
        has_col_geom = any(
            g.tag == "geom" and g.get("contype", "1") != "0"
            and g.get("group") != "1"
            for g in body_elem
        )
        if not has_col_geom:
            spec = body_collision_specs.get(body_name, {"type": "box", "size": "0.04 0.04 0.04", "pos": "0 0 0"})
            col_geom = ET.Element("geom", {
                "type": spec["type"],
                "size": spec.get("size", "0.04 0.04 0.04"),
                "pos": spec.get("pos", "0 0 0"),
                "contype": "0", "conaffinity": "0", "condim": "3",
                "rgba": "0 0 0 0",
            })
            # Place after visual geom
            insert_at = len(list(body_elem))
            for i, child in enumerate(body_elem):
                if child.tag == "geom" and child.get("group") == "1":
                    insert_at = i + 1
            body_elem.insert(insert_at, col_geom)

        # Add foot contact box (flat shoe) for ankle_roll links
        if "ankle_roll" in body_name:
            # Flat box that barely touches ground when pelvis at z=0.42
            # Flat foot pad: width 0.06, depth 0.02, barely penetrating ground
            fs = ET.Element("geom", {
                "type": "box", "size": "0.06 0.02 0.015",
                "pos": "0.0 0.0 -0.048",
                "contype": "1", "conaffinity": "15", "condim": "3",
                "friction": "1.0 0.005 0.0001",
                "solref": "0.02 1", "solimp": "0.9 0.95 0.001",
                "rgba": "0 0 0 0",
            })
            body_elem.insert(len(list(body_elem)), fs)

        # Enhance joints: add range and actuatorfrcrange
        for child in body_elem:
            if child.tag == "joint" and child.get("name") != "floating_base_joint":
                jname = child.get("name", "")
                if jname in joint_limits:
                    lo, hi, eff = joint_limits[jname]
                    child.set("range", f"{lo} {hi}")
                    child.set("actuatorfrcrange", f"-{eff} {eff}")
                    child.set("armature", "0.01")    # match training dof_armature_list
                    # NO intrinsic joint damping — training uses dof_joint_friction_list: 0.0
                    child.set("damping", "0.0")

        # Process children recursively
        for child in body_elem:
            if child.tag == "body":
                _enhance_body(child, child.get("name", ""))

    # Process all root bodies in worldbody
    for body in worldbody:
        if body.tag == "body":
            _enhance_body(body, body.get("name", ""), is_root=True)

    # Also fix floating_base_joint: rename to freejoint or add as freejoint
    for body in worldbody:
        if body.tag == "body":
            for child in list(body):
                if child.tag == "joint" and child.get("name") == "floating_base_joint":
                    # Remove the floating_base_joint and add freejoint
                    body.remove(child)
            # Check if freejoint already exists
            has_free = any(
                child.tag == "freejoint" or
                (child.tag == "joint" and child.get("type") == "free")
                for child in body
            )
            if not has_free:
                ET.SubElement(body, "freejoint", {"name": "pelvis"})
            break  # Only for root body

    # 7. Actuators: add gear/forcerange
    actuator = root.find("actuator")
    if actuator is not None:
        for motor in actuator:
            jname = motor.get("joint", motor.get("name", ""))
            if jname and jname in joint_limits:
                _, _, eff = joint_limits[jname]
                motor.set("ctrllimited", "true")
                motor.set("ctrlrange", f"-{eff} {eff}")
                motor.set("forcelimited", "true")
                motor.set("forcerange", f"-{eff} {eff}")

    return ET.tostring(root, encoding="unicode")


def build_mujoco_model(xml_path, sim_dt, solver_iterations, ls_iterations, use_physics_xml=False):
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"XML not found: {xml_path}")

    if use_physics_xml:
        # XML already has physics — use directly
        model = mujoco.MjModel.from_xml_path(xml_path)
    else:
        # Build complete MuJoCo XML with visuals, ground, lighting, physics
        xml_str = _build_full_mujoco_xml(xml_path)
        model = mujoco.MjModel.from_xml_string(xml_str)
    model.opt.timestep = float(sim_dt)
    data = mujoco.MjData(model)
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = solver_iterations
    model.opt.ls_iterations = ls_iterations
    return model, data


def build_joint_index_map(model, dof_names):
    """Map train DOF names to MuJoCo actuator/qpos/qvel indices.

    Verifies DOF order and builds explicit mapping.
    """
    n = len(dof_names)

    # Actuator name → index
    act_name_to_idx = {}
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        if jid >= 0:
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jname:
                act_name_to_idx[jname] = i

    # MuJoCo joint order (non-floating)
    mujoco_joint_order = []
    for i in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jtype = model.jnt_type[i]
        # Exclude free joints (type 0) — they have no actuator
        if jname and jtype != 0:
            mujoco_joint_order.append(jname)

    actuator_ids = []
    qpos_ids = []
    qvel_ids = []
    for jname in dof_names:
        jid = model.joint(jname).id
        if jid < 0:
            raise ValueError(f"Joint '{jname}' not found in MuJoCo model")
        if jname not in act_name_to_idx:
            raise ValueError(f"No actuator found for joint '{jname}'")
        actuator_ids.append(act_name_to_idx[jname])
        qpos_ids.append(model.jnt_qposadr[jid])
        qvel_ids.append(model.jnt_dofadr[jid])

    assert len(actuator_ids) == n

    # Check order
    order_matches = all(dof_names[i] == mujoco_joint_order[i] for i in range(min(n, len(mujoco_joint_order))))

    # Build explicit mapping: train_dof_idx → mujoco index
    mapping = {}
    for i, name in enumerate(dof_names):
        mj_idx = mujoco_joint_order.index(name) if name in mujoco_joint_order else -1
        mapping[i] = mj_idx

    print("[DOF_MAPPING]")
    print(f"  train_dof_names = {dof_names}")
    print(f"  mujoco_joint_order = {mujoco_joint_order}")
    print(f"  order matches: {order_matches}")
    if not order_matches:
        print("  WARNING: Order mismatch! Building explicit mapping.")

    return {
        "actuator_ids": actuator_ids,
        "qpos_ids": qpos_ids,
        "qvel_ids": qvel_ids,
        "mujoco_joint_order": mujoco_joint_order,
        "mapping": mapping,
        "order_matches": order_matches,
    }


def check_actuator_type(model):
    """Report MuJoCo actuator types."""
    print("[ACTUATOR_CHECK]")
    motor_count = 0
    total = model.nu
    for i in range(model.nu):
        if model.actuator_gaintype[i] == 0:  # motor/torque
            motor_count += 1
    print(f"  Motor (torque) actuators: {motor_count}/{total}")
    if motor_count < total:
        print(f"  WARNING: {total - motor_count} actuators are NOT torque-type!")
    else:
        print(f"  All actuators are torque-type → using data.ctrl for torque.")


def get_robot_state(model, data, index_map, imu_body_name, default_dof_pos):
    """Extract robot state from MuJoCo data in training conventions.

    dof_pos/dof_vel are in train dof_names order.

    Returns:
      base_quat_wxyz: MuJoCo native (w,x,y,z)
      base_quat_xyzw: IsaacGym convention (x,y,z,w)
      base_pos: world position of IMU body
      base_lin_vel: base-frame linear velocity
      base_ang_vel: base-frame angular velocity
      ang_vel_world: world-frame angular velocity
      lin_vel_world: world-frame linear velocity
      dof_pos: train-order joint positions (absolute)
      dof_vel: train-order joint velocities
      projected_gravity: base-frame gravity direction
    """
    imu_id = model.body(imu_body_name).id
    base_pos = data.xpos[imu_id].copy()
    base_quat_wxyz = data.xquat[imu_id].copy()  # MuJoCo: w,x,y,z

    # Velocities
    if hasattr(data, "xvelr"):
        ang_vel_world = data.xvelr[imu_id].copy()
    else:
        vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, imu_id, vel, 0)
        ang_vel_world = vel[0:3].copy()

    if hasattr(data, "xvelp"):
        lin_vel_world = data.xvelp[imu_id].copy()
    else:
        vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, imu_id, vel, 0)
        lin_vel_world = vel[3:6].copy()

    # DOF in train order
    dof_pos = np.array([data.qpos[qid] for qid in index_map["qpos_ids"]], dtype=np.float64)
    dof_vel = np.array([data.qvel[vid] for vid in index_map["qvel_ids"]], dtype=np.float64)

    # Base-frame quantities (matching IsaacGym: rotate world vector by inverse of base quat)
    base_ang_vel = quat_rotate_inverse_wxyz(base_quat_wxyz, ang_vel_world)
    base_lin_vel = quat_rotate_inverse_wxyz(base_quat_wxyz, lin_vel_world)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    projected_gravity = quat_rotate_inverse_wxyz(base_quat_wxyz, gravity_world)

    return {
        "base_quat_wxyz": base_quat_wxyz,
        "base_quat_xyzw": wxyz_to_xyzw(base_quat_wxyz),
        "base_pos": base_pos,
        "base_lin_vel": base_lin_vel,
        "base_ang_vel": base_ang_vel,
        "ang_vel_world": ang_vel_world,
        "lin_vel_world": lin_vel_world,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "projected_gravity": projected_gravity,
        "dof_pos_from_default": dof_pos - default_dof_pos,
    }


# ============================================================================
# Observation construction
# ============================================================================

class HistoryBuffer:
    """Minimal history buffer for single-env sim2sim.

    Layout: index 0 = newest, index N-1 = oldest.
    Matches HistoryHandler behavior.
    """

    def __init__(self, history_config, obs_dims):
        """history_config: config.obs.obs_auxiliary dict."""
        self.buffer_config = {}  # {obs_key: max_depth}
        for aux_key, aux_config in history_config.items():
            for obs_key, depth in aux_config.items():
                if obs_key in self.buffer_config:
                    self.buffer_config[obs_key] = max(self.buffer_config[obs_key], depth)
                else:
                    self.buffer_config[obs_key] = depth

        self.buffers = {}
        for key, depth in self.buffer_config.items():
            dim = obs_dims.get(key, 0)
            if dim:
                self.buffers[key] = np.zeros((depth, dim), dtype=np.float32)

    def reset(self):
        for key in self.buffers:
            self.buffers[key][:] = 0.0

    def add(self, key, value):
        if key not in self.buffers:
            return
        buf = self.buffers[key]
        buf[1:] = buf[:-1]   # shift right (older)
        buf[0] = np.asarray(value, dtype=np.float32)  # newest at front

    def query(self, key, history_length=None):
        if key not in self.buffers:
            return np.zeros((0,), dtype=np.float32)
        buf = self.buffers[key]
        if history_length is not None:
            buf = buf[:history_length]
        return buf.copy()


class ObsBuilder:
    """Constructs actor_obs matching training exactly."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.obs_scales = cfg["obs_scales"]
        self.obs_dims = cfg["obs_dims"]
        self.actor_obs_keys = cfg["actor_obs_keys"]
        self.history_config_actor = cfg["obs_auxiliary"].get("history_actor", {})
        self._compute_expected_dims()

        self.history_buffer = HistoryBuffer(cfg["obs_auxiliary"], cfg["obs_dims"])

    def _compute_expected_dims(self):
        self.component_dims = OrderedDict()
        total = 0
        print("[OBS_COMPONENT]")
        for key in self.actor_obs_keys:
            if key == "history_actor":
                dim = 0
                hist_cfg = self.history_config_actor
                for hkey, depth in sorted(hist_cfg.items()):
                    dim += self.obs_dims.get(hkey, 0) * depth
                self.component_dims[key] = dim
            elif key in self.obs_dims:
                dim = self.obs_dims[key]
                self.component_dims[key] = dim
            else:
                raise KeyError(f"Unknown obs key '{key}' — not in obs_dims")
            print(f"  {key}: {dim}")
            total += dim
        self.policy_obs_dim = total
        print(f"  final_actor_obs_dim={total}")

    def build_obs(self, robot_state, ref_state, phase_info, last_action):
        """Build full actor_obs in training order.

        All numpy operations. Returns (policy_obs_dim,) float32 array.
        """
        components = OrderedDict()
        s = self.obs_scales

        # 1. base_ang_vel (3) * 0.25
        components["base_ang_vel"] = robot_state["base_ang_vel"] * s.get("base_ang_vel", 0.25)

        # 2. projected_gravity (3) * 1.0
        components["projected_gravity"] = robot_state["projected_gravity"] * s.get("projected_gravity", 1.0)

        # 3. dof_pos (22) = (dof_pos - default) * 1.0  — matching training
        dp_from_def = robot_state["dof_pos"] - self.cfg["default_dof_pos"]
        components["dof_pos"] = dp_from_def * s.get("dof_pos", 1.0)

        # 4. dof_vel (22) * 0.05
        components["dof_vel"] = robot_state["dof_vel"] * s.get("dof_vel", 0.05)

        # 5. actions (22) * 1.0 — previous action
        components["actions"] = np.asarray(last_action, dtype=np.float32) * s.get("actions", 1.0)

        # 6. ref_motion_phase (1) * 1.0
        components["ref_motion_phase"] = np.array([phase_info["phase"]], dtype=np.float32) * s.get("ref_motion_phase", 1.0)

        # 7. q1_root_error (4)
        components["q1_root_error"] = np.array([
            phase_info["root_z_error"],
            phase_info["root_vz_error"],
            robot_state["base_pos"][2],       # actual root_z
            robot_state["lin_vel_world"][2],  # actual root_vz
        ], dtype=np.float32) * s.get("q1_root_error", 1.0)

        # 8. q1_yaw_error (4): sin(error), cos(error), rate_error, actual_rate
        components["q1_yaw_error"] = np.array([
            phase_info["yaw_error_sin"],
            phase_info["yaw_error_cos"],
            phase_info["yaw_rate_error"],
            phase_info["actual_yaw_rate"],
        ], dtype=np.float32) * s.get("q1_yaw_error", 1.0)

        # 9. q1_flight_phase (6): crouch, takeoff, flight, landing, left_contact, right_contact
        masks = phase_info["phase_masks"]
        components["q1_flight_phase"] = np.array([
            float(masks["crouch"]),
            float(masks["takeoff"]),
            float(masks["flight"]),
            float(masks["landing"]),
            float(phase_info["left_contact"]),
            float(phase_info["right_contact"]),
        ], dtype=np.float32) * s.get("q1_flight_phase", 1.0)

        # 10. q1_ref_dof_error (44 = 22 pos + 22 vel)
        ref_dof_pos = ref_state["dof_pos"][0]  # (22,)
        ref_dof_vel = ref_state["dof_vel"][0]  # (22,)
        dof_pos_error = ref_dof_pos - robot_state["dof_pos"]
        dof_vel_error = ref_dof_vel - robot_state["dof_vel"]
        components["q1_ref_dof_error"] = np.concatenate([
            dof_pos_error, dof_vel_error,
        ]).astype(np.float32) * s.get("q1_ref_dof_error", 1.0)

        # --- Update history buffer ---
        self.history_buffer.add("base_ang_vel", robot_state["base_ang_vel"])
        self.history_buffer.add("projected_gravity", robot_state["projected_gravity"])
        self.history_buffer.add("dof_pos", dp_from_def)
        self.history_buffer.add("dof_vel", robot_state["dof_vel"])
        self.history_buffer.add("actions", np.asarray(last_action, dtype=np.float32))
        self.history_buffer.add("ref_motion_phase", np.array([phase_info["phase"]], dtype=np.float32))

        # 11. history_actor (292)
        hist_parts = []
        for hkey in sorted(self.history_config_actor.keys()):
            depth = self.history_config_actor[hkey]
            hist_tensor = self.history_buffer.query(hkey, history_length=depth)
            hist_parts.append(hist_tensor.flatten().astype(np.float32))
        components["history_actor"] = np.concatenate(hist_parts) * s.get("history_actor", 1.0)

        # --- Concatenate in configured order ---
        parts = [components[k] for k in self.actor_obs_keys]
        obs = np.concatenate(parts).astype(np.float32)

        assert obs.shape == (self.policy_obs_dim,), \
            f"Constructed obs dim {obs.shape[0]} != expected {self.policy_obs_dim}"

        return obs


# ============================================================================
# Phase computation
# ============================================================================

def compute_phase_info(ref_state, robot_state, phase_config, contact_info, motion_time, motion_duration):
    """Compute all phase-related info.

    Matches Q1CR7MotionTracking._pre_compute_observations_callback.
    """
    phase = float(np.clip(motion_time / motion_duration, 0.0, 1.0))
    pc = phase_config

    crouch = pc["crouch_phase_start"] <= phase < pc["crouch_phase_end"]
    takeoff = pc["takeoff_phase_start"] <= phase < pc["takeoff_phase_end"]
    flight = pc["flight_phase_start"] <= phase < pc["flight_phase_end"]
    landing = pc["landing_phase_start"] <= phase < pc["landing_phase_end"]

    # Priority: takeoff > flight > landing > crouch
    flight = flight and not takeoff
    landing = landing and not takeoff and not flight
    crouch = crouch and not takeoff and not flight and not landing

    # Root errors
    root_pos = ref_state["root_pos"]
    ref_root_z = float(root_pos[0, 2]) if root_pos.ndim > 1 else float(root_pos[2])
    root_vel = ref_state["root_vel"]
    ref_root_vz = float(root_vel[0, 2]) if root_vel.ndim > 1 else float(root_vel[2])
    actual_root_z = float(robot_state["base_pos"][2])
    actual_root_vz = float(robot_state["lin_vel_world"][2])

    # Yaw
    root_rot = ref_state["root_rot"]
    ref_quat_xyzw = root_rot[0] if root_rot.ndim > 1 else root_rot  # xyzw (IsaacGym)
    ref_yaw = compute_yaw_from_quat_xyzw(ref_quat_xyzw)
    actual_yaw = compute_yaw_from_quat_wxyz(robot_state["base_quat_wxyz"])
    yaw_err = ref_yaw - actual_yaw

    # Yaw rates
    ref_ang_vel = ref_state["root_ang_vel"]
    ref_yaw_rate = float(ref_ang_vel[0, 2]) if ref_ang_vel.ndim > 1 else 0.0
    actual_yaw_rate = float(robot_state["ang_vel_world"][2])

    return {
        "phase": phase,
        "phase_masks": {"crouch": crouch, "takeoff": takeoff, "flight": flight, "landing": landing},
        "root_z_error": ref_root_z - actual_root_z,
        "root_vz_error": ref_root_vz - actual_root_vz,
        "yaw_error_sin": float(np.sin(yaw_err)),
        "yaw_error_cos": float(np.cos(yaw_err)),
        "yaw_error_rad": float(np.arctan2(np.sin(yaw_err), np.cos(yaw_err))),
        "yaw_rate_error": ref_yaw_rate - actual_yaw_rate,
        "actual_yaw_rate": actual_yaw_rate,
        "ref_root_z": ref_root_z,
        "actual_root_z": actual_root_z,
        "ref_root_vz": ref_root_vz,
        "actual_root_vz": actual_root_vz,
        "ref_yaw": ref_yaw,
        "actual_yaw": actual_yaw,
        "left_contact": contact_info["left_contact"],
        "right_contact": contact_info["right_contact"],
    }


# ============================================================================
# PD control
# ============================================================================

def pd_control(target_pos, dof_pos, dof_vel, kps, kds, torque_limits, clip_torques=True):
    """tau = kp * (target - pos) - kd * vel"""
    tau = kps * (target_pos - dof_pos) - kds * dof_vel
    if clip_torques:
        tau = np.clip(tau, -torque_limits, torque_limits)
    return tau


def action_to_target(action, default_dof_pos, action_scale):
    """target = default + action * action_scale (standard P control)"""
    return default_dof_pos + action * action_scale


# ============================================================================
# Contact detection
# ============================================================================

def detect_contacts(model, data, left_foot_name, right_foot_name):
    """Detect foot contacts from MuJoCo contact data."""
    left_contact = False
    right_contact = False
    for ci in range(data.ncon):
        g1 = data.contact[ci].geom1
        g2 = data.contact[ci].geom2
        bname1 = model.body(int(model.geom_bodyid[g1])).name if g1 < model.ngeom else "world"
        bname2 = model.body(int(model.geom_bodyid[g2])).name if g2 < model.ngeom else "world"
        bodies = {bname1, bname2}

        is_ground = "world" in bodies or any(
            x in bname1 or x in bname2 for x in ["ground", "floor", "plane"]
        )
        if not is_ground and "world" not in bodies:
            continue

        if left_foot_name in bname1 or left_foot_name in bname2:
            left_contact = True
        if right_foot_name in bname1 or right_foot_name in bname2:
            right_contact = True

    return {"left_contact": left_contact, "right_contact": right_contact}


# ============================================================================
# Dry-run validation
# ============================================================================

def dry_run_check(obs_builder, policy, robot_state, ref_state, phase_info, last_action, cfg):
    """5-step dry-run to validate dimensions and values."""
    print("\n[DRY_RUN] 5-step validation...")
    for step in range(5):
        obs = obs_builder.build_obs(robot_state, ref_state, phase_info, last_action)
        assert obs.shape[0] == obs_builder.policy_obs_dim, \
            f"Step {step}: obs dim {obs.shape[0]} != {obs_builder.policy_obs_dim}"
        assert np.isfinite(obs).all(), f"Step {step}: obs contains NaN/Inf"

        action = policy_infer(policy, obs.reshape(1, -1))[0]
        assert len(action) == 22, f"Step {step}: action dim {len(action)} != 22"
        assert np.isfinite(action).all(), f"Step {step}: action contains NaN/Inf"

        ref_dof = ref_state["dof_pos"][0]
        assert len(ref_dof) == 22, f"Step {step}: ref_dof dim {len(ref_dof)} != 22"

        grav = robot_state["projected_gravity"]
        grav_norm = np.linalg.norm(grav)
        print(f"  Step {step}: |proj_grav|={grav_norm:.4f}, "
              f"g=[{grav[0]:.4f},{grav[1]:.4f},{grav[2]:.4f}], "
              f"z_act={robot_state['base_pos'][2]:.4f}, z_ref={phase_info['ref_root_z']:.4f}")

        if step == 0 and abs(grav_norm - 1.0) > 0.05:
            print("  WARNING: projected_gravity norm != 1.0 — quaternion convention check needed!")

    print("[DRY_RUN] PASSED.\n")


# ============================================================================
# Main simulation loop
# ============================================================================

def run_sim2sim(cfg, args):
    """Run MuJoCo sim2sim."""

    # ====== 1. Load MuJoCo ======
    model, data = build_mujoco_model(
        args.mujoco_xml, cfg["sim_dt"], cfg["solver_iterations"], cfg["ls_iterations"],
        use_physics_xml=args.physics_xml)
    index_map = build_joint_index_map(model, cfg["dof_names"])
    check_actuator_type(model)

    imu_name = cfg["imu_body_name"]
    if model.body(imu_name).id < 0:
        raise ValueError(f"IMU body '{imu_name}' not found")

    # ====== 2. Load motion ======
    print(f"\n[MOTION] Loading: {args.motion_file}")
    motion_lib = MotionLoader(args.motion_file, cfg["motion"])
    print(f"  motion_len={motion_lib.duration:.3f}s ({motion_lib.num_frames} frames @ {motion_lib.fps}fps)")
    print(f"  root_z: [{motion_lib._root_pos[:, 2].min():.3f}, {motion_lib._root_pos[:, 2].max():.3f}]")
    print(f"  dof_pos range: [{motion_lib._dof_pos.min():.3f}, {motion_lib._dof_pos.max():.3f}]")
    print(f"  dof_vel range: [{motion_lib._dof_vel.min():.3f}, {motion_lib._dof_vel.max():.3f}]")
    print(f"  body_pos shape: {motion_lib._body_pos.shape}")

    # ====== 3. Build obs builder, compute dims ======
    obs_builder = ObsBuilder(cfg)
    expected_obs_dim = obs_builder.policy_obs_dim

    if expected_obs_dim != args.policy_obs_dim:
        raise ValueError(
            f"Constructed obs dim {expected_obs_dim} != policy expected {args.policy_obs_dim}")

    # ====== 4. Load policy ======
    policy = load_onnx_policy(args.checkpoint, args.policy_obs_dim, 22)

    # ====== 5. Reset robot ======
    # Start from motion's first frame for stability (matching training non-eval init)
    ref_init = motion_lib.get_motion_state(np.array([0]), np.array([0.0]))
    first_frame_root_pos = ref_init["root_pos"][0].copy()
    first_frame_root_rot_xyzw = ref_init["root_rot"][0].copy()  # xyzw
    first_frame_root_rot_wxyz = xyzw_to_wxyz(first_frame_root_rot_xyzw)
    first_frame_dof_pos = ref_init["dof_pos"][0].copy()

    data.qpos[0:3] = first_frame_root_pos
    # Start upright (identity rotation) — matches eval init, avoids initial tilt
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz identity
    for i, qpos_id in enumerate(index_map["qpos_ids"]):
        data.qpos[qpos_id] = first_frame_dof_pos[i]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    print(f"[RESET] Init from motion frame 0: root_z={first_frame_root_pos[2]:.4f}, "
          f"dof[0:6]={np.array2string(first_frame_dof_pos[:6], precision=3)}")

    # ====== 6. Simulation parameters ======
    sim_dt = cfg["sim_dt"]
    decimation = cfg["control_decimation"]
    policy_dt = cfg["control_dt"]
    motion_duration = motion_lib.duration

    last_action = np.zeros(22, dtype=np.float32)
    target_dof_pos = cfg["default_dof_pos"].copy()
    tau_last = np.zeros(22, dtype=np.float64)

    # ====== 7. Dry-run ======
    if not args.skip_dry_run:
        robot_state = get_robot_state(model, data, index_map, imu_name, cfg["default_dof_pos"])
        ref_state = motion_lib.get_motion_state(np.array([0]), np.array([0.0]))
        contact_info = detect_contacts(model, data, cfg["left_foot_name"], cfg["right_foot_name"])
        phase_info = compute_phase_info(ref_state, robot_state, cfg["q1_cr7"], contact_info, 0.0, motion_duration)
        dry_run_check(obs_builder, policy, robot_state, ref_state, phase_info, last_action, cfg)

        # Reset history buffer after dry-run (training starts with zero history)
        obs_builder.history_buffer.reset()

        # Dump first obs if requested
        if args.dump_first_obs:
            obs_first = obs_builder.build_obs(robot_state, ref_state, phase_info, last_action)
            dump_dir = os.path.join(_REPO_ROOT, "debug_outputs")
            os.makedirs(dump_dir, exist_ok=True)
            np.save(os.path.join(dump_dir, "mujoco_first_obs.npy"), obs_first)
            print(f"[DUMP] First obs saved to debug_outputs/mujoco_first_obs.npy")

    # ====== 8. Video setup ======
    renderer = None
    frames = []
    if args.record_video:
        from mujoco.renderer import Renderer
        width = cfg["video_width"]
        height = cfg["video_height"]
        gl_context = mujoco.GLContext(max_width=width, max_height=height)
        gl_context.make_current()
        renderer = Renderer(model, height=height, width=width)

    # ====== 9. CSV setup ======
    csv_rows = []

    # ====== 10. Main loop ======
    num_control_steps = args.num_steps
    total_physics_steps = num_control_steps * decimation
    print(f"\n[SIM2SIM] {num_control_steps} control steps "
          f"({num_control_steps * policy_dt:.1f}s simulated, "
          f"{total_physics_steps} physics steps)")

    stop_reason = "timeout"
    t0 = time.time()
    physics_step = 0
    control_step = 0
    root_positions = []

    try:
        for control_step in range(num_control_steps):
            # Motion time matches training: (episode_length_buf + 1) * dt
            # episode_length_buf starts at 0, so first obs uses ref at time dt
            motion_time = (control_step + 1) * policy_dt
            if motion_time >= motion_duration - policy_dt:
                motion_time = max(0, motion_duration - policy_dt)
                reached_end = True
            else:
                reached_end = False

            # (A) Reference motion state
            ref_state = motion_lib.get_motion_state(np.array([0]), np.array([motion_time]))

            # (B) Robot state
            robot_state = get_robot_state(model, data, index_map, imu_name, cfg["default_dof_pos"])

            # (C) Contacts
            contact_info = detect_contacts(model, data, cfg["left_foot_name"], cfg["right_foot_name"])

            # (D) Phase info
            phase_info = compute_phase_info(ref_state, robot_state, cfg["q1_cr7"], contact_info,
                                             motion_time, motion_duration)

            # (E) Build obs
            obs = obs_builder.build_obs(robot_state, ref_state, phase_info, last_action)
            obs = np.clip(obs, -cfg["clip_observations"], cfg["clip_observations"])
            if not np.isfinite(obs).all():
                stop_reason = "obs_non_finite"
                break

            # (F) Policy inference
            raw_action = policy_infer(policy, obs.reshape(1, -1))[0]
            raw_action = np.clip(raw_action, -cfg["clip_actions"], cfg["clip_actions"])
            if not np.isfinite(raw_action).all():
                stop_reason = "action_non_finite"
                break

            # (G) Compute PD target
            target_dof_pos = action_to_target(raw_action, cfg["default_dof_pos"], cfg["action_scale"])
            last_action = raw_action.copy()

            # (H) Apply for decimation physics steps
            for d in range(decimation):
                rs = get_robot_state(model, data, index_map, imu_name, cfg["default_dof_pos"])
                tau = pd_control(target_dof_pos, rs["dof_pos"], rs["dof_vel"],
                                 cfg["kps"], cfg["kds"], cfg["torque_limits"],
                                 clip_torques=cfg["clip_torques"])
                tau_last = tau.copy()
                for i, act_id in enumerate(index_map["actuator_ids"]):
                    data.ctrl[act_id] = tau[i]
                mujoco.mj_step(model, data)
                physics_step += 1

            root_positions.append(robot_state["base_pos"].copy())

            # (I) CSV row
            if args.debug_csv:
                csv_rows.append(_make_csv_row(control_step, policy_dt, phase_info,
                                               raw_action, target_dof_pos,
                                               robot_state, tau_last, cfg))

            # (J) Render
            if renderer is not None:
                renderer.update_scene(data, camera="render_cam")

                # Reference markers via perturbed viewer
                if args.show_ref_markers:
                    _render_ref_markers(renderer, ref_state, cfg)

                pixels = renderer.render()
                frames.append(pixels)

            # (K) Progress
            if control_step % 50 == 0 or reached_end:
                t = control_step * policy_dt
                phase_names = ["crouch", "takeoff", "flight", "landing"]
                active_phases = [k for k in phase_names if phase_info["phase_masks"][k]]
                active_str = active_phases[0] if active_phases else "none"
                tau_abs = np.abs(tau_last)
                tau_sat_pct = float((tau_abs >= cfg["torque_limits"] * 0.95).mean()) * 100
                print(f"  t={t:.2f}s step={control_step} "
                      f"z_act={robot_state['base_pos'][2]:.3f} z_ref={phase_info['ref_root_z']:.3f} "
                      f"yaw_err={np.rad2deg(phase_info['yaw_error_rad']):.1f}° "
                      f"phase={phase_info['phase']:.2f}({active_str}) "
                      f"tau_sat={tau_sat_pct:.0f}%")

            # (L) Safety
            root_z = robot_state["base_pos"][2]
            if root_z < cfg["min_root_z"]:
                stop_reason = f"root_z_low({root_z:.3f})"
                break
            if root_z > cfg["max_root_z"]:
                stop_reason = f"root_z_high({root_z:.3f})"
                break
            if np.abs(data.qacc).max() > cfg["max_abs_qacc"]:
                stop_reason = f"qacc({np.abs(data.qacc).max():.0f})"
                break
            if reached_end:
                stop_reason = "motion_end"
                break

    except KeyboardInterrupt:
        stop_reason = "user_interrupt"

    elapsed = time.time() - t0
    root_positions = np.array(root_positions)

    # ====== Summary ======
    print(f"\n{'='*60}")
    print("Simulation Summary")
    print(f"{'='*60}")
    print(f"  Control steps: {control_step + 1}")
    print(f"  Physics steps: {physics_step}")
    print(f"  Sim time: {(control_step + 1) * policy_dt:.2f}s")
    print(f"  Wall time: {elapsed:.1f}s")
    print(f"  Stop: {stop_reason}")
    if len(root_positions) > 0:
        print(f"  Root z: [{root_positions[:, 2].min():.4f}, {root_positions[:, 2].max():.4f}]")
        print(f"  Root x: [{root_positions[:, 0].min():.4f}, {root_positions[:, 0].max():.4f}]")
        print(f"  Root y: [{root_positions[:, 1].min():.4f}, {root_positions[:, 1].max():.4f}]")

    # ====== Save video ======
    if renderer is not None and frames:
        os.makedirs(args.output_dir, exist_ok=True)
        video_path = os.path.join(args.output_dir, "rollout.mp4")
        _save_video(frames, video_path, fps=cfg["video_fps"])

    if renderer is not None:
        renderer.close()

    # ====== Save CSV ======
    if args.debug_csv and csv_rows:
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, "rollout_debug.csv")
        _save_csv(csv_rows, csv_path)


def _make_csv_row(control_step, policy_dt, phase_info, raw_action, target_dof_pos,
                  robot_state, tau, cfg):
    """Build a single CSV debug row."""
    knee_L = cfg["dof_names"].index("left_knee_joint")
    hip_L = cfg["dof_names"].index("left_hip_pitch_joint")
    ankle_L = cfg["dof_names"].index("left_ankle_pitch_joint")
    tau_abs = np.abs(tau)

    return OrderedDict([
        ("step", control_step),
        ("time", f"{control_step * policy_dt:.4f}"),
        ("phase", f"{phase_info['phase']:.6f}"),
        ("motion_time", f"{control_step * policy_dt:.4f}"),
        ("ref_root_z", f"{phase_info['ref_root_z']:.6f}"),
        ("actual_root_z", f"{phase_info['actual_root_z']:.6f}"),
        ("ref_root_vz", f"{phase_info.get('ref_root_vz', 0):.6f}"),
        ("actual_root_vz", f"{phase_info.get('actual_root_vz', 0):.6f}"),
        ("ref_yaw", f"{phase_info['ref_yaw']:.6f}"),
        ("actual_yaw", f"{phase_info['actual_yaw']:.6f}"),
        ("yaw_error_deg", f"{np.rad2deg(phase_info['yaw_error_rad']):.4f}"),
        ("ref_yaw_rate", f"{phase_info.get('yaw_rate_error', 0) + phase_info['actual_yaw_rate']:.6f}"),
        ("actual_yaw_rate", f"{phase_info['actual_yaw_rate']:.6f}"),
        ("left_contact", str(int(phase_info["left_contact"]))),
        ("right_contact", str(int(phase_info["right_contact"]))),
        ("action_max_abs", f"{float(np.max(np.abs(raw_action))):.6f}"),
        ("action_mean", f"{float(np.mean(raw_action)):.6f}"),
        ("target_knee_L", f"{float(target_dof_pos[knee_L]):.6f}"),
        ("target_hip_pitch_L", f"{float(target_dof_pos[hip_L]):.6f}"),
        ("target_ankle_pitch_L", f"{float(target_dof_pos[ankle_L]):.6f}"),
        ("actual_knee_L", f"{float(robot_state['dof_pos'][knee_L]):.6f}"),
        ("actual_hip_pitch_L", f"{float(robot_state['dof_pos'][hip_L]):.6f}"),
        ("actual_ankle_pitch_L", f"{float(robot_state['dof_pos'][ankle_L]):.6f}"),
        ("torque_knee_L", f"{float(tau[knee_L]):.4f}"),
        ("torque_hip_pitch_L", f"{float(tau[hip_L]):.4f}"),
        ("torque_ankle_pitch_L", f"{float(tau[ankle_L]):.4f}"),
        ("root_pos_x", f"{float(robot_state['base_pos'][0]):.6f}"),
        ("root_pos_y", f"{float(robot_state['base_pos'][1]):.6f}"),
        ("root_pos_z", f"{float(robot_state['base_pos'][2]):.6f}"),
        ("root_quat_w", f"{float(robot_state['base_quat_wxyz'][0]):.6f}"),
        ("root_quat_x", f"{float(robot_state['base_quat_wxyz'][1]):.6f}"),
        ("root_quat_y", f"{float(robot_state['base_quat_wxyz'][2]):.6f}"),
        ("root_quat_z", f"{float(robot_state['base_quat_wxyz'][3]):.6f}"),
        ("torque_max", f"{float(np.max(tau_abs)):.4f}"),
        ("torque_sat_ratio", f"{float((tau_abs >= cfg['torque_limits'] * 0.95).mean()):.4f}"),
    ])


def _render_ref_markers(renderer, ref_state, cfg):
    """Add reference marker spheres to rendered scene.

    Uses mjv_geom API for visual-only spheres (no physics).
    Since MuJoCo Python API doesn't easily support adding temporary geoms,
    we use the perturbed viewer approach: add tiny perturbing bodies.
    Actually, we skip this for now — markers can be rendered by
    comparing body positions in the final CSV/video analysis.
    """
    pass  # Reference markers require scene geom manipulation not easily exposed in Python


def _save_video(frames, output_path, fps=50):
    try:
        import cv2
        if not frames:
            return
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"Video saved: {output_path} ({len(frames)} frames)")
    except ImportError:
        print("Warning: opencv-python not installed, cannot save video.")


def _save_csv(rows, csv_path):
    if not rows:
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {csv_path} ({len(rows)} rows)")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Q1 CR7 Motion Tracking MuJoCo sim2sim")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to ONNX policy")
    p.add_argument("--policy-format", type=str, default="onnx", choices=["onnx", "jit", "pt"])
    p.add_argument("--train-config", type=str, required=True, help="Path to training config.yaml")
    p.add_argument("--mujoco-xml", type=str,
                   default="humanoidverse/data/robots/q1/q1_22dof_box.xml")
    p.add_argument("--motion-file", type=str,
                   default="humanoidverse/data/motions/q1/q1_cr7_scale045_rootz042.pkl")
    p.add_argument("--output-dir", type=str, default="sim2sim_outputs/q1_cr7")
    p.add_argument("--num-steps", type=int, default=300)
    p.add_argument("--record-video", action="store_true")
    p.add_argument("--video-fps", type=int, default=50)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--show-ref-markers", action="store_true")
    p.add_argument("--debug-csv", action="store_true")
    p.add_argument("--dump-first-obs", action="store_true")
    p.add_argument("--skip-dry-run", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--policy-obs-dim", type=int, default=None,
                   help="Override expected policy obs dim (auto-detected if omitted)")
    p.add_argument("--physics-xml", action="store_true",
                   help="XML already has physics (don't generate). Use with q1_sim2sim_physics.xml")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve paths
    for attr in ["mujoco_xml", "motion_file", "train_config", "checkpoint"]:
        val = getattr(args, attr)
        if val and not os.path.isabs(val):
            setattr(args, attr, os.path.join(_REPO_ROOT, val))

    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(_REPO_ROOT, args.output_dir)

    # Load configs
    train_cfg = load_train_config(args.train_config)
    sim2sim_cfg = load_sim2sim_config()
    cfg = build_sim2sim_config(train_cfg, sim2sim_cfg, args)

    # Auto-detect policy obs dim
    if args.policy_obs_dim is None:
        tmp = ort.InferenceSession(args.checkpoint)
        args.policy_obs_dim = tmp.get_inputs()[0].shape[-1]

    # Print config
    print("=" * 60)
    print("Q1 CR7 Motion Tracking MuJoCo sim2sim")
    print("=" * 60)
    print("[SIM2SIM_CONFIG]")
    print(f"  checkpoint={args.checkpoint}")
    print(f"  xml={args.mujoco_xml}")
    print(f"  motion_file={args.motion_file}")
    print(f"  sim_dt={cfg['sim_dt']:.4f}s ({cfg['sim_fps']}Hz)")
    print(f"  control_dt={cfg['control_dt']:.4f}s ({1.0/cfg['control_dt']:.0f}Hz)")
    print(f"  decimation={cfg['control_decimation']}")
    print(f"  action_scale={cfg['action_scale']}")
    print(f"  num_dofs={len(cfg['dof_names'])}")
    print(f"  dof_names={cfg['dof_names']}")
    print(f"  policy_obs_dim={args.policy_obs_dim}")
    print(f"  control_type={cfg['control_type']}")
    print()

    run_sim2sim(cfg, args)


if __name__ == "__main__":
    main()
