#!/usr/bin/env python3
"""
Q1 CR7 Motion Tracking — simplified MuJoCo sim2sim.
Modelled on fdd_asap_sim2sim.py.

Only actor_obs (365-dim). No critic_obs, no motion_lib, no reward.
ONNX policy → PD torque control → MuJoCo physics.

Obs layout (sorted alphabetical, matching training _post_config_observation_callback):
  actions(22) + base_ang_vel(3) + dof_pos(22) + dof_vel(22) +
  history_actor(292) + projected_gravity(3) + ref_motion_phase(1) = 365

History timing (matches fdd/training: previous history then update):
  obs = current_single + previous_history + current_gvec/phase
  history.update(current_single)   # after obs built

Usage:
  python scripts/sim2sim_q1_motion_tracking_simple.py \
      --config scripts/q1_sim2sim_config.yaml \
      --policy-path logs/Q1_GK/.../exported/model_1100.onnx \
      --xml-path humanoidverse/data/robots/q1/q1_22dof_box.xml \
      --allow-xml-augmentation \
      --headless \
      --video-out sim2sim_outputs/q1_cr7/q1_cr7_sim2sim.mp4 \
      --duration 4.0 \
      --video-fps 50
"""

import os
import sys
import argparse
import csv
import time
import yaml
import pickle
import joblib
import numpy as np
from collections import OrderedDict
import xml.etree.ElementTree as ET
from pathlib import Path

if "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_GMR_ROOT = "/root/autodl-tmp/GMR"

# ============================================================================
# Constants — Q1 22 DoF
# ============================================================================

NUM_ACTIONS = 22
NUM_SINGLE_OBS = 73           # 22 + 3 + 22 + 22 + 3 + 1
FRAME_STACK = 4
# final obs dim = 73 * (1 + 4) = 365

HISTORY_KEYS = [
    "actions", "base_ang_vel", "dof_pos", "dof_vel",
    "projected_gravity", "ref_motion_phase",
]

# Single-obs slice indices  (73 dim)
#   actions: 22   base_ang_vel: 3   dof_pos: 22   dof_vel: 22   gvec: 3   phase: 1
SINGLE_SLICES = {
    "actions":           slice(0, 22),
    "base_ang_vel":      slice(22, 25),
    "dof_pos":           slice(25, 47),
    "dof_vel":           slice(47, 69),
    "projected_gravity": slice(69, 72),
    "ref_motion_phase":  slice(72, 73),
}

# ============================================================================
# Config loading
# ============================================================================

def read_conf(config_file):
    """Load sim2sim config — mirrors fdd_asap_sim2sim.read_conf()."""
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cfg = {}
    cfg["num_single_obs"] = config["num_single_obs"]
    cfg["simulation_dt"] = config["simulation_dt"]
    cfg["cycle_time"] = config["cycle_time"]
    cfg["frame_stack"] = config["frame_stack"]
    cfg["num_actions"] = config["num_actions"]
    cfg["simulation_duration"] = config.get("simulation_duration", config["cycle_time"])
    cfg["control_decimation"] = config["control_decimation"]

    cfg["default_dof_pos"] = np.array(config["default_dof_pos"], dtype=np.float32)

    cfg["obs_scale_base_ang_vel"] = config["obs_scale_base_ang_vel"]
    cfg["obs_scale_dof_pos"] = config["obs_scale_dof_pos"]
    cfg["obs_scale_dof_vel"] = config["obs_scale_dof_vel"]
    cfg["obs_scale_gvec"] = config["obs_scale_gvec"]
    cfg["obs_scale_refmotion"] = config["obs_scale_refmotion"]
    cfg["obs_scale_hist"] = config["obs_scale_hist"]

    cfg["clip_observations"] = config["clip_observations"]
    cfg["clip_actions"] = config["clip_actions"]
    cfg["action_scale"] = config["action_scale"]

    cfg["kps"] = np.array(config["kps"], dtype=np.float32)
    cfg["kds"] = np.array(config["kds"], dtype=np.float32)
    cfg["kp_scale"] = float(config.get("kp_scale", 1.0))
    cfg["kd_scale"] = float(config.get("kd_scale", 1.0))
    cfg["kps"] = cfg["kps"] * cfg["kp_scale"]
    cfg["kds"] = cfg["kds"] * cfg["kd_scale"]

    cfg["tau_limit"] = np.array(config["tau_limit"], dtype=np.float32)

    cfg["xml_path"] = config["xml_path"]
    cfg["time_offset"] = float(config.get("time_offset", 0.0))
    cfg["phase_wrap"] = bool(config.get("phase_wrap", False))
    cfg["stop_at_motion_end"] = bool(config.get("stop_at_motion_end", True))
    cfg["action_filter_alpha"] = float(config.get("action_filter_alpha", 1.0))
    cfg["target_pos_rate_limit"] = float(config.get("target_pos_rate_limit", 0.0))
    cfg["safety_check_nonfinite"] = bool(config.get("safety_check_nonfinite", True))
    cfg["max_abs_qacc"] = float(config.get("max_abs_qacc", 5.0e4))
    cfg["solver_iterations"] = int(config.get("solver_iterations", 100))
    cfg["solver_ls_iterations"] = int(config.get("solver_ls_iterations", 50))

    # Paths — resolve relative paths against: cfg_dir → script_dir → repo_root
    cfg_dir = Path(config_file).resolve().parent
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent  # scripts/ → repo root
    for key in ["xml_path"]:
        if key in cfg and not os.path.isabs(cfg[key]):
            val = cfg[key]
            resolved = None
            for base in [cfg_dir, script_dir, repo_root]:
                candidate = (base / val).resolve()
                if candidate.is_file():
                    resolved = str(candidate)
                    break
            if resolved is None:
                # Default: resolve relative to repo_root (configs use repo-relative paths)
                resolved = str((repo_root / val).resolve())
            cfg[key] = resolved

    cfg["dof_names"] = list(config.get("joint_names", config.get("dof_names", [])))
    cfg["imu_body_name"] = config.get("imu_body_name", "pelvis")

    return cfg


# ============================================================================
# MuJoCo model loading + diagnostics
# ============================================================================

def check_model_diagnostics(model, xml_path, loaded_directly, augmentation_enabled,
                            ball_stripped=False):
    """Print XML/MODEL/MASS diagnostics on startup."""
    total_mass = float(model.body_mass.sum())
    print(f"[XML] loaded_directly={loaded_directly}")
    print(f"[XML] augmentation_enabled={augmentation_enabled}")
    print(f"[XML] ball_stripped={ball_stripped}")
    print(f"[MODEL] nq={model.nq} nv={model.nv} nu={model.nu} "
          f"nbody={model.nbody} njnt={model.njnt}")
    print(f"[MASS] total_mass={total_mass:.1f} kg")

    # Actuator type check
    n_motor = sum(1 for i in range(model.nu) if model.actuator_gaintype[i] == 0)
    print(f"[ACTUATOR] torque_motor: {n_motor}/{model.nu}")
    if n_motor < model.nu:
        raise ValueError("ERROR: Not all actuators are torque motors! "
                         "PD control requires torque actuators (gaintype=0).")
    # Print first few actuator ranges
    for i in range(min(3, model.nu)):
        ctrl_range = model.actuator_ctrlrange[i]
        jid = model.actuator_trnid[i, 0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) if jid >= 0 else "?"
        print(f"[ACTUATOR]   [{i}] joint={jname} ctrlrange=[{ctrl_range[0]:.1f}, {ctrl_range[1]:.1f}]")

    # Check joint damping/armature vs training config
    sample_damping = []
    sample_arm = []
    for i in range(model.njnt):
        if model.jnt_type[i] != 0:  # skip freejoints
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            dof_addr = model.jnt_dofadr[i]
            if dof_addr >= 0 and dof_addr < model.nv:
                d = model.dof_damping[dof_addr]
                a = model.dof_armature[dof_addr]
                if d > 0:
                    sample_damping.append(f"{jname}={d:.2f}")
                if len(sample_arm) < 5 and a > 0:
                    sample_arm.append(f"{jname}={a:.4f}")
    if sample_damping:
        print(f"[JOINT] damping detected: {', '.join(sample_damping[:5])} ... "
              f"(training has dof_joint_friction_list=0.0)")
    if sample_arm:
        print(f"[JOINT] armature detected: {', '.join(sample_arm)} "
              f"(training armature=0.01)")

    # Mass warning
    if total_mass < 15.0 or total_mass > 80.0:
        print(f"[MASS] WARNING: total_mass={total_mass:.1f} kg seems unreasonable "
              f"(expected ~25-50 kg for Q1)")


def _strip_ball_from_xml(xml_str, xml_dir=None):
    """Remove <body name='ball'> and fix relative meshdir for from_xml_string.

    The football body has its own freejoint — it adds 7 qpos + 6 qvel entries
    that we don't control and that mess up qpos[7:] indexing.

    Also makes meshdir absolute if xml_dir is provided, because from_xml_string
    can't resolve relative paths.
    """
    tree = ET.fromstring(xml_str) if isinstance(xml_str, str) else ET.fromstring(xml_str.decode())

    # Remove ball body
    worldbody = tree.find("worldbody")
    if worldbody is not None:
        for body in list(worldbody):
            if body.tag == "body" and body.get("name") == "ball":
                worldbody.remove(body)
                print("[XML] Removed <body name='ball'> from XML")
                break

    # Fix relative meshdir → absolute (needed for from_xml_string)
    if xml_dir:
        compiler = tree.find("compiler")
        if compiler is not None:
            meshdir = compiler.get("meshdir", "")
            if meshdir and not os.path.isabs(meshdir):
                abs_meshdir = os.path.normpath(os.path.join(xml_dir, meshdir))
                compiler.set("meshdir", abs_meshdir)
                print(f"[XML] Fixed meshdir: {meshdir} → {abs_meshdir}")

    return ET.tostring(tree, encoding="unicode")


def load_mujoco_model(xml_path, sim_dt, solver_iterations, ls_iterations,
                      allow_augmentation=False):
    """Load MuJoCo model. Default: require physics-capable XML.

    Only with allow_augmentation=True, fall back to _ensure_physics_xml()
    for kinematic-only XML files.
    """
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"XML not found: {xml_path}")

    loaded_directly = True
    augmentation_enabled = False
    ball_stripped = False

    try:
        # Strip ball before loading (ball adds unwanted freejoint + 7 qpos/6 qvel)
        with open(xml_path, "r") as f:
            raw_xml = f.read()
        if "name=\"ball\"" in raw_xml:
            print("[XML] Detected football body — stripping from in-memory XML...")
            raw_xml = _strip_ball_from_xml(raw_xml, xml_dir=os.path.dirname(xml_path))
            ball_stripped = True
            model = mujoco.MjModel.from_xml_string(raw_xml)
        else:
            model = mujoco.MjModel.from_xml_path(xml_path)
    except Exception as e_direct:
        if not allow_augmentation:
            raise RuntimeError(
                f"Failed to load XML as physics model:\n  {e_direct}\n\n"
                f"XML path: {xml_path}\n"
                f"The XML likely lacks mass/inertial/collision geoms (kinematic-only).\n"
                f"Options:\n"
                f"  1. Provide a complete MuJoCo physics XML with inertial, geoms, etc.\n"
                f"  2. Pass --allow-xml-augmentation to auto-add minimal physics at runtime.\n"
                f"     (This uses approximate masses/boxes/STL meshes from Q1 data.)"
            )
        print("[XML] Direct load failed (kinematic) — augmenting with minimal physics...")
        xml_str = _ensure_physics_xml(xml_path)
        model = mujoco.MjModel.from_xml_string(xml_str)
        loaded_directly = False
        augmentation_enabled = True
        print("[XML] Physics-augmented XML loaded")

    model.opt.timestep = float(sim_dt)
    if solver_iterations > 0:
        model.opt.iterations = solver_iterations
    if hasattr(model.opt, "ls_iterations") and ls_iterations > 0:
        model.opt.ls_iterations = ls_iterations

    data = mujoco.MjData(model)
    check_model_diagnostics(model, xml_path, loaded_directly, augmentation_enabled,
                            ball_stripped=ball_stripped)
    return model, data


# ============================================================================
# DOF mapping (Q1-specific, kept from current implementation)
# ============================================================================

def build_joint_index_map(model, dof_names):
    """Map training DOF names to MuJoCo qpos/qvel/actuator indices."""
    assert len(dof_names) == NUM_ACTIONS, \
        f"Expected {NUM_ACTIONS} dof_names, got {len(dof_names)}"

    mujoco_joint_order = []
    for i in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jtype = model.jnt_type[i]
        if jname and jtype != 0:  # exclude free joints
            mujoco_joint_order.append(jname)

    act_name_to_idx = {}
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        if jid >= 0:
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jname:
                act_name_to_idx[jname] = i

    actuator_ids, qpos_ids, qvel_ids = [], [], []
    for jname in dof_names:
        jid = model.joint(jname).id
        if jid < 0:
            raise ValueError(f"Joint '{jname}' not found in MuJoCo model")
        if jname not in act_name_to_idx:
            raise ValueError(f"No actuator for joint '{jname}'")
        actuator_ids.append(act_name_to_idx[jname])
        qpos_ids.append(model.jnt_qposadr[jid])
        qvel_ids.append(model.jnt_dofadr[jid])

    order_matches = all(
        dof_names[i] == mujoco_joint_order[i]
        for i in range(min(len(dof_names), len(mujoco_joint_order)))
    )

    print("[DOF_NAMES]")
    print(f"  train_dof_names = {dof_names}")
    print(f"  mujoco_joint_names = {mujoco_joint_order}")
    print(f"  order_matches = {order_matches}")
    if not order_matches:
        print("  WARNING: Order mismatch — using explicit mapping")

    return {
        "actuator_ids": actuator_ids,
        "qpos_ids": qpos_ids,
        "qvel_ids": qvel_ids,
        "mujoco_joint_order": mujoco_joint_order,
        "order_matches": order_matches,
    }


# ============================================================================
# Reference motion playback (left split-screen)
# ============================================================================

def _load_raw_motion_dict(motion_file):
    with open(motion_file, "rb") as f:
        try:
            motion_data = joblib.load(f)
        except Exception:
            f.seek(0)
            motion_data = pickle.load(f)
    if isinstance(motion_data, dict) and \
       not {"fps", "root_pos", "root_rot", "dof_pos", "pose_aa", "root_trans_offset"}.intersection(motion_data):
        motion_data = motion_data[next(iter(motion_data.keys()))]
    return motion_data


def load_reference_motion(motion_file, cfg):
    """Load reference pkl for split-screen playback."""
    if not motion_file:
        return None
    if not os.path.isfile(motion_file):
        raise FileNotFoundError(f"Reference motion file not found: {motion_file}")

    try:
        import importlib.util

        data_loader_path = os.path.join(
            _GMR_ROOT, "general_motion_retargeting", "data_loader.py")
        spec = importlib.util.spec_from_file_location("gmr_data_loader", data_loader_path)
        gmr_data_loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gmr_data_loader)

        _, motion_fps, root_pos, root_rot, dof_pos, _, _ = gmr_data_loader.load_robot_motion(motion_file)
        quat_note = "GMR data_loader converted stored xyzw root_rot to wxyz"
    except (KeyError, FileNotFoundError):
        motion_data = _load_raw_motion_dict(motion_file)
        motion_fps = motion_data["fps"]
        root_pos = motion_data["root_trans_offset"]
        pose_aa = np.asarray(motion_data["pose_aa"], dtype=np.float64)
        root_rot = R.from_rotvec(pose_aa[:, 0, :]).as_quat(scalar_first=True)
        dof_pos = pose_aa[:, 1:, :].sum(axis=-1)
        quat_note = "fallback converted pose_aa root rotvec to MuJoCo wxyz quaternion"

    root_pos = np.asarray(root_pos, dtype=np.float64)
    root_rot = np.asarray(root_rot, dtype=np.float64)
    dof_pos = np.asarray(dof_pos, dtype=np.float64)

    if dof_pos.ndim != 2:
        raise ValueError(f"Reference dof_pos must be 2D, got shape={dof_pos.shape}")
    if dof_pos.shape[1] < cfg["num_actions"]:
        raise ValueError(
            f"Reference dof_pos dim {dof_pos.shape[1]} < num_actions {cfg['num_actions']}")
    if dof_pos.shape[1] > cfg["num_actions"]:
        dof_pos = dof_pos[:, :cfg["num_actions"]]

    print(f"[REF_MOTION] Loaded {motion_file}")
    print(f"[REF_MOTION]   frames={len(root_pos)} fps={motion_fps} dof_dim={dof_pos.shape[1]}")
    print(f"[REF_MOTION]   root_rot is wxyz: {quat_note}")
    return {
        "fps": float(motion_fps),
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
    }


def yaw_from_quat_wxyz(quat_wxyz):
    """Return world yaw from a MuJoCo/scalar-first quaternion."""
    mat = R.from_quat(quat_wxyz, scalar_first=True).as_matrix()
    return float(np.arctan2(mat[1, 0], mat[0, 0]))


def align_quat_yaw_wxyz(quat_wxyz, target_quat_wxyz):
    """Rotate quat_wxyz so its yaw matches target_quat_wxyz, preserving tilt."""
    yaw = yaw_from_quat_wxyz(quat_wxyz)
    target_yaw = yaw_from_quat_wxyz(target_quat_wxyz)
    delta = R.from_euler("z", target_yaw - yaw)
    aligned = delta * R.from_quat(quat_wxyz, scalar_first=True)
    return aligned.as_quat(scalar_first=True)


def write_reference_qpos(ref_model, ref_data, ref_index_map, ref_motion, motion_idx,
                         root_pos_override=None, root_rot_override=None):
    """Write one decoded motion frame into ref_data.qpos and forward kinematics."""
    root_pos = ref_motion["root_pos"][motion_idx]
    if root_pos_override is not None:
        root_pos = root_pos_override
    root_rot_wxyz = ref_motion["root_rot"][motion_idx]
    if root_rot_override is not None:
        root_rot_wxyz = root_rot_override
    dof_pos = ref_motion["dof_pos"][motion_idx]

    ref_data.qpos[0:3] = root_pos
    # MuJoCo free-joint qpos expects [x,y,z,qw,qx,qy,qz]. GMR's
    # load_robot_motion() converts stored xyzw root_rot to wxyz already.
    ref_data.qpos[3:7] = root_rot_wxyz
    for i, qpos_id in enumerate(ref_index_map["qpos_ids"]):
        ref_data.qpos[qpos_id] = dof_pos[i]
    ref_data.qvel[:] = 0.0
    mujoco.mj_forward(ref_model, ref_data)


def draw_panel_label(frame, text, x=24, y=42):
    """Draw a small top-left label on an RGB frame."""
    import cv2

    img = np.ascontiguousarray(frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad_x = 12
    pad_y = 8
    cv2.rectangle(
        img,
        (x - pad_x, y - th - pad_y),
        (x + tw + pad_x, y + baseline + pad_y),
        (0, 0, 0),
        -1,
    )
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


# ============================================================================
# MuJoCo XML physics augmentation (only with --allow-xml-augmentation)
# ============================================================================

_BODY_MASS = {
    "pelvis": 1.7, "torso_link": 1.5,
    "left_knee_link": 1.2, "right_knee_link": 1.2,
    "left_hip_yaw_link": 0.76, "right_hip_yaw_link": 0.76,
    "left_hip_pitch_link": 0.7, "right_hip_pitch_link": 0.7,
    "left_ankle_pitch_link": 0.4, "right_ankle_pitch_link": 0.4,
    "left_ankle_roll_link": 0.34, "right_ankle_roll_link": 0.34,
    "left_hip_roll_link": 0.19, "right_hip_roll_link": 0.19,
}

_BODY_BOX = {
    "pelvis": "0.10 0.10 0.065 0.0 0.0 0.03",
    "left_hip_pitch_link": "0.09 0.08 0.08", "left_hip_roll_link": "0.09 0.08 0.08",
    "left_hip_yaw_link": "0.09 0.09 0.07 0.0 0.0 -0.06",
    "left_knee_link": "0.08 0.09 0.12 0.01 0.0 -0.09",
    "left_ankle_pitch_link": "0.06 0.06 0.05",
    "left_ankle_roll_link": "0.06 0.02 0.015 0.0 0.0 -0.048",
    "right_hip_pitch_link": "0.09 0.08 0.08", "right_hip_roll_link": "0.09 0.08 0.08",
    "right_hip_yaw_link": "0.09 0.09 0.07 0.0 0.0 -0.06",
    "right_knee_link": "0.08 0.09 0.12 0.01 0.0 -0.09",
    "right_ankle_pitch_link": "0.06 0.06 0.05",
    "right_ankle_roll_link": "0.06 0.02 0.015 0.0 0.0 -0.048",
    "waist_roll_link": "0.07 0.12 0.05",
    "torso_link": "0.12 0.15 0.15 0.0 0.0 0.05",
    "left_shoulder_pitch_link": "0.05 0.06 0.06 0.0 0.02 0.0",
    "left_shoulder_roll_link": "0.04 0.04 0.05 0.0 0.0 -0.04",
    "left_shoulder_yaw_link": "0.04 0.04 0.05 0.0 0.0 -0.04",
    "left_elbow_link": "0.04 0.04 0.05 0.01 0.0 -0.04",
    "right_shoulder_pitch_link": "0.05 0.06 0.06 0.0 -0.02 0.0",
    "right_shoulder_roll_link": "0.04 0.04 0.05 0.0 0.0 -0.04",
    "right_shoulder_yaw_link": "0.04 0.04 0.05 0.0 0.0 -0.04",
    "right_elbow_link": "0.04 0.04 0.05 0.01 0.0 -0.04",
}

_JOINT_LIMITS = [
    ("left_hip_pitch_joint", -3.0543, 1.5708, 36.0, 0.01),
    ("left_hip_roll_joint", -0.69813, 1.5708, 36.0, 0.01),
    ("left_hip_yaw_joint", -1.5708, 1.5708, 36.0, 0.01),
    ("left_knee_joint", 0.0, 2.4435, 36.0, 0.01),
    ("left_ankle_pitch_joint", -0.7854, 0.43633, 22.0, 0.01),
    ("left_ankle_roll_joint", -0.34907, 0.34907, 22.0, 0.01),
    ("right_hip_pitch_joint", -3.0543, 1.5708, 36.0, 0.01),
    ("right_hip_roll_joint", -1.5708, 0.69813, 36.0, 0.01),
    ("right_hip_yaw_joint", -1.5708, 1.5708, 36.0, 0.01),
    ("right_knee_joint", 0.0, 2.4435, 36.0, 0.01),
    ("right_ankle_pitch_joint", -0.7854, 0.43633, 22.0, 0.01),
    ("right_ankle_roll_joint", -0.34907, 0.34907, 22.0, 0.01),
    ("waist_roll_joint", -0.2618, 0.2618, 36.0, 0.01),
    ("waist_yaw_joint", -1.5708, 1.5708, 36.0, 0.01),
    ("left_shoulder_pitch_joint", -3.1416, 1.5708, 22.0, 0.01),
    ("left_shoulder_roll_joint", -0.087266, 2.7925, 22.0, 0.01),
    ("left_shoulder_yaw_joint", -1.5708, 1.5708, 22.0, 0.01),
    ("left_elbow_joint", -0.87266, 1.6581, 22.0, 0.01),
    ("right_shoulder_pitch_joint", -3.1416, 1.5708, 22.0, 0.01),
    ("right_shoulder_roll_joint", -2.7925, 0.087266, 22.0, 0.01),
    ("right_shoulder_yaw_joint", -1.5708, 1.5708, 22.0, 0.01),
    ("right_elbow_joint", -0.87266, 1.6581, 22.0, 0.01),
]
_JOINT_LIMITS_MAP = {j[0]: j[1:] for j in _JOINT_LIMITS}

_MESH_NAMES = [
    "head_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "left_elbow_link", "left_hip_pitch_link", "left_hip_roll_link",
    "left_hip_yaw_link", "left_knee_link", "left_shoulder_pitch_link",
    "left_shoulder_roll_link", "left_shoulder_yaw_link", "pelvis",
    "right_ankle_pitch_link", "right_ankle_roll_link", "right_elbow_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_shoulder_pitch_link", "right_shoulder_roll_link",
    "right_shoulder_yaw_link", "torso_link", "waist_roll_link",
]


def _ensure_physics_xml(xml_path):
    """Add minimal physics (inertial, collision, STL visual, ground, limits).

    Only called when --allow-xml-augmentation is passed.
    Does NOT modify the original file.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Compiler
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("angle", "radian")
    mesh_dir = os.path.join(_REPO_ROOT, "humanoidverse", "data", "robots", "q1", "meshes")
    compiler.set("meshdir", mesh_dir)
    compiler.set("autolimits", "true")

    # Visual
    if root.find("visual") is None:
        ET.SubElement(root, "visual")
    if root.find("visual/global") is None:
        ET.SubElement(root.find("visual"), "global", {"offwidth": "1280", "offheight": "720"})

    # Default
    default = root.find("default")
    if default is None:
        default = ET.SubElement(root, "default")
    for old_j in default.findall("joint"):
        default.remove(old_j)
    ET.SubElement(default, "joint", {"type": "hinge"})

    # Assets
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(list(root).index(root.find("worldbody")), asset)
    for child in list(asset):
        asset.remove(child)
    for name in _MESH_NAMES:
        ET.SubElement(asset, "mesh", {"name": name, "file": f"{name}.STL"})
    ET.SubElement(asset, "texture", {
        "type": "skybox", "builtin": "flat",
        "rgb1": "0.3 0.4 0.5", "rgb2": "0.3 0.4 0.5",
        "width": "512", "height": "512",
    })
    ET.SubElement(asset, "texture", {
        "name": "groundplane", "type": "2d", "builtin": "checker", "mark": "edge",
        "rgb1": "0.2 0.3 0.4", "rgb2": "0.15 0.25 0.35",
        "markrgb": "0.6 0.6 0.6", "width": "300", "height": "300",
    })
    ET.SubElement(asset, "material", {
        "name": "groundplane", "texture": "groundplane",
        "texuniform": "true", "texrepeat": "5 5",
    })

    # Worldbody: ground, lights (NO fixed camera — using MjvCamera free camera)
    worldbody = root.find("worldbody")
    ET.SubElement(worldbody, "geom", {
        "name": "floor", "type": "plane", "size": "0 0 0.05",
        "material": "groundplane", "contype": "4", "conaffinity": "11", "condim": "3",
        "solref": "0.02 1", "solimp": "0.9 0.95 0.001",
    })
    ET.SubElement(worldbody, "light", {
        "directional": "true", "diffuse": "0.7 0.7 0.7",
        "specular": "0.2 0.2 0.2", "pos": "3 2 4", "dir": "-3 -2 -4",
    })
    ET.SubElement(worldbody, "light", {
        "directional": "false", "diffuse": "0.3 0.3 0.3", "pos": "0 0 3",
    })

    # Enhance bodies
    def _enhance_body(body_elem):
        bname = body_elem.get("name", "")

        # Inertial
        if not any(c.tag == "inertial" for c in body_elem):
            mass = _BODY_MASS.get(bname, 0.5)
            m = max(mass, 0.01)
            diag = f"{m * 0.01:.6f} {m * 0.01:.6f} {m * 0.01:.6f}"
            inert_el = ET.Element("inertial", {
                "pos": "0 0 0", "mass": str(mass), "diaginertia": diag,
            })
            insert_at = 0
            for i, c in enumerate(body_elem):
                if c.tag == "joint":
                    insert_at = i + 1
            body_elem.insert(insert_at, inert_el)

        # Collision geom (invisible box)
        if not any(c.tag == "geom" and c.get("group", "") != "1" for c in body_elem):
            box_str = _BODY_BOX.get(bname, "0.04 0.04 0.04")
            parts = box_str.split()
            size = " ".join(parts[:3])
            pos = " ".join(parts[3:6]) if len(parts) >= 6 else "0 0 0"
            ET.SubElement(body_elem, "geom", {
                "type": "box", "size": size, "pos": pos,
                "contype": "1", "conaffinity": "15", "condim": "3",
                "rgba": "0 0 0 0",
            })

        # Visual geom (STL mesh, group=1)
        if not any(c.tag == "geom" and c.get("group") == "1" for c in body_elem):
            rgba = "0.7 0.7 0.7 1"
            ET.SubElement(body_elem, "geom", {
                "type": "mesh", "mesh": bname,
                "contype": "0", "conaffinity": "0", "group": "1",
                "density": "0", "rgba": rgba,
            })

        # Joint limits
        for c in body_elem:
            if c.tag == "joint" and c.get("name") != "floating_base_joint":
                jname = c.get("name", "")
                if jname in _JOINT_LIMITS_MAP:
                    lo, hi, eff, arm = _JOINT_LIMITS_MAP[jname]
                    c.set("range", f"{lo} {hi}")
                    c.set("armature", str(arm))
                    c.set("damping", "0.0")

        for c in body_elem:
            if c.tag == "body":
                _enhance_body(c)

    for body in list(worldbody):
        if body.tag == "body":
            for child in list(body):
                if child.tag == "joint" and child.get("name") == "floating_base_joint":
                    body.remove(child)
            has_free = any(
                child.tag == "freejoint" or
                (child.tag == "joint" and child.get("type") == "free")
                for child in body
            )
            if not has_free:
                ET.SubElement(body, "freejoint", {"name": "pelvis"})
            _enhance_body(body)
            break

    # Actuators
    actuator = root.find("actuator")
    if actuator is not None:
        for motor in actuator:
            jname = motor.get("joint", motor.get("name", ""))
            if jname and jname in _JOINT_LIMITS_MAP:
                _, _, eff, _ = _JOINT_LIMITS_MAP[jname]
                motor.set("ctrllimited", "true")
                motor.set("ctrlrange", f"-{eff} {eff}")
                motor.set("forcelimited", "true")
                motor.set("forcerange", f"-{eff} {eff}")

    return ET.tostring(root, encoding="unicode")


# ============================================================================
# State extraction (fdd style — scipy Rotation for gravity)
# ============================================================================

def get_mujoco_data(data, index_map, default_dof_pos):
    """Extract robot state. Matches fdd_asap_sim2sim.get_mujoco_data() pattern."""
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)

    # scipy Rotation uses xyzw
    quat = np.array([q[4], q[5], q[6], q[3]])
    r = R.from_quat(quat)
    base_angvel = dq[3:6]
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)

    # DOF in training order (explicit mapping — safety)
    dof_pos = np.array([data.qpos[qid] for qid in index_map["qpos_ids"]], dtype=np.double)
    dof_vel = np.array([data.qvel[vid] for vid in index_map["qvel_ids"]], dtype=np.double)

    return {
        "base_angvel": base_angvel,
        "gvec": gvec,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
    }


# ============================================================================
# History (fdd style — previous history then update)
# ============================================================================

def create_history(cfg):
    """Create zero-initialized history dict + flattened array."""
    frame_stack = cfg["frame_stack"]
    num_actions = cfg["num_actions"]
    hist_dict = {
        "actions": np.zeros((frame_stack, num_actions), dtype=np.double),
        "base_ang_vel": np.zeros((frame_stack, 3), dtype=np.double),
        "dof_pos": np.zeros((frame_stack, num_actions), dtype=np.double),
        "dof_vel": np.zeros((frame_stack, num_actions), dtype=np.double),
        "projected_gravity": np.zeros((frame_stack, 3), dtype=np.double),
        "ref_motion_phase": np.zeros((frame_stack, 1), dtype=np.double),
    }
    hist_obs = [hist_dict[key].reshape(1, -1) for key in HISTORY_KEYS]
    hist_obs_c = np.concatenate(hist_obs, axis=1)
    return hist_dict, hist_obs_c


def update_hist_obs(hist_dict, obs_single):
    """Update history: drop oldest, prepend newest single-frame obs. fdd style."""
    for key in HISTORY_KEYS:
        slc = SINGLE_SLICES[key]
        arr = np.delete(hist_dict[key], -1, axis=0)       # remove oldest (last row)
        arr = np.vstack((obs_single[0, slc], arr))          # prepend newest
        hist_dict[key] = arr
    hist_obs = np.concatenate(
        [hist_dict[key].reshape(1, -1) for key in HISTORY_KEYS], axis=1
    ).astype(np.float32)
    return hist_obs


# ============================================================================
# Phase (fdd style)
# ============================================================================

def compute_ref_motion_phase(counter, cfg):
    """Phase from physics-step counter. fdd style."""
    sim_time = cfg["time_offset"] + (counter + 1) * cfg["simulation_dt"]
    if cfg["phase_wrap"]:
        return float((sim_time % cfg["cycle_time"]) / cfg["cycle_time"])
    phase_time = min(sim_time, cfg["cycle_time"] - 1e-6) if cfg["stop_at_motion_end"] else sim_time
    return float(np.clip(phase_time / cfg["cycle_time"], 0, 1))


# ============================================================================
# Observation construction (fdd style — previous history, then update)
# ============================================================================

def get_obs(hist_obs_c, hist_dict, mj_data, action, counter, cfg):
    """Build actor_obs with PREVIOUS history, then update history.

    This matches fdd_asap_sim2sim.get_obs() exactly.
    Training flow: build obs from previous history → update history with current.

    Obs layout (Q1 22DoF, sorted keys):
      actions(22) + base_ang_vel(3) + dof_pos(22) + dof_vel(22) +
      history_actor(292) + projected_gravity(3) + ref_motion_phase(1) = 365
    """
    default_dof_pos = cfg["default_dof_pos"]
    num_obs_input = (cfg["frame_stack"] + 1) * cfg["num_single_obs"]
    num_actions = cfg["num_actions"]

    ref_motion_phase = compute_ref_motion_phase(counter, cfg)

    # Single-frame obs (73 dim, unscaled raw values in fixed order)
    obs_single = np.zeros([1, cfg["num_single_obs"]], dtype=np.float32)
    obs_single[0, 0:22] = action                                                   # actions
    obs_single[0, 22:25] = mj_data["base_angvel"] * cfg["obs_scale_base_ang_vel"]   # base_ang_vel
    obs_single[0, 25:47] = (mj_data["dof_pos"] - default_dof_pos) * cfg["obs_scale_dof_pos"]  # dof_pos
    obs_single[0, 47:69] = mj_data["dof_vel"] * cfg["obs_scale_dof_vel"]            # dof_vel
    obs_single[0, 69:72] = mj_data["gvec"] * cfg["obs_scale_gvec"]                  # projected_gravity
    obs_single[0, 72] = ref_motion_phase * cfg["obs_scale_refmotion"]               # ref_motion_phase

    # Full obs: current single terms + PREVIOUS history + current gravity/phase
    obs_all = np.zeros([1, num_obs_input], dtype=np.float32)
    obs_all[0, 0:22] = obs_single[0, 0:22].copy()       # actions (current)
    obs_all[0, 22:25] = obs_single[0, 22:25].copy()     # base_ang_vel (current)
    obs_all[0, 25:47] = obs_single[0, 25:47].copy()     # dof_pos (current)
    obs_all[0, 47:69] = obs_single[0, 47:69].copy()     # dof_vel (current)
    hist_len = cfg["frame_stack"] * cfg["num_single_obs"]  # 4 * 73 = 292
    obs_all[0, 69:69 + hist_len] = hist_obs_c[0] * cfg["obs_scale_hist"]  # history (PREVIOUS)
    obs_all[0, 69 + hist_len:69 + hist_len + 3] = obs_single[0, 69:72].copy()  # projected_gravity (current)
    obs_all[0, 69 + hist_len + 3] = obs_single[0, 72].copy()                   # ref_motion_phase (current)

    # NOW update history with current single-frame (for NEXT timestep)
    hist_obs_new = update_hist_obs(hist_dict, obs_single)

    obs_all = np.clip(obs_all, -cfg["clip_observations"], cfg["clip_observations"])
    return obs_all, hist_obs_new, obs_single


# ============================================================================
# Policy (fdd style)
# ============================================================================

def load_onnx_policy(policy_path, expected_obs_dim, expected_act_dim):
    """Load ONNX model and validate dims."""
    if not os.path.isfile(policy_path):
        raise FileNotFoundError(f"ONNX policy not found: {policy_path}")
    session = ort.InferenceSession(policy_path)
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]

    print(f"[ONNX] input:  name={inp.name}, shape={inp.shape}")
    print(f"[ONNX] output: name={out.name}, shape={out.shape}")

    onnx_in_dim = inp.shape[-1]
    onnx_out_dim = out.shape[-1]

    if onnx_in_dim != expected_obs_dim:
        raise ValueError(f"ONNX input dim {onnx_in_dim} != expected obs dim {expected_obs_dim}")
    if onnx_out_dim != expected_act_dim:
        raise ValueError(f"ONNX output dim {onnx_out_dim} != expected action dim {expected_act_dim}")

    # Dummy inference
    dummy = np.zeros((1, expected_obs_dim), dtype=np.float32)
    result = session.run([out.name], {inp.name: dummy})[0]
    assert result.shape == (1, expected_act_dim), \
        f"Dummy inference shape {result.shape} != (1, {expected_act_dim})"
    assert np.isfinite(result).all(), "Dummy inference returned NaN/Inf"
    print(f"  Dummy inference: shape={result.shape}, mean={result.mean():.4f}")

    return {
        "session": session,
        "input_name": inp.name,
        "output_name": out.name,
    }


def infer_policy_action(policy, obs_buff, clip_actions):
    """ONNX inference → clipped action."""
    raw_action = policy["session"].run(
        [policy["output_name"]], {policy["input_name"]: obs_buff}
    )[0]
    raw_action = np.asarray(raw_action).reshape(-1).astype(np.float32)
    return np.clip(raw_action, -clip_actions, clip_actions)


def apply_action_postprocess(raw_action, prev_action, prev_target_dof_pos, cfg):
    """Action filter + rate limit + scale → PD target. fdd style."""
    alpha = cfg["action_filter_alpha"]
    action = alpha * raw_action + (1.0 - alpha) * prev_action
    target_from_action = action * cfg["action_scale"] + cfg["default_dof_pos"]
    if cfg["target_pos_rate_limit"] > 0:
        max_delta = cfg["target_pos_rate_limit"] * cfg["simulation_dt"] * cfg["control_decimation"]
        target_dof_pos = np.clip(target_from_action,
                                 prev_target_dof_pos - max_delta,
                                 prev_target_dof_pos + max_delta)
    else:
        target_dof_pos = target_from_action
    return action.astype(np.float32), target_dof_pos.astype(np.float32)


# ============================================================================
# PD control (fdd style)
# ============================================================================

def pd_control(target_pos, dof_pos, target_vel, dof_vel, cfg):
    """PD torque: kp * delta_pos + kd * delta_vel. fdd style."""
    tau = (target_pos - dof_pos) * cfg["kps"] + (target_vel - dof_vel) * cfg["kds"]
    return np.clip(tau, -cfg["tau_limit"], cfg["tau_limit"])


# ============================================================================
# Main simulation loop (fdd style)
# ============================================================================

def run_mujoco(cfg,
               policy_path,
               headless=False,
               video_out="",
               video_fps=50,
               cam_follow_base=True,
               cam_distance=3.5,
               cam_azimuth=135.0,
               cam_elevation=-12.0,
               cam_lookat_z=0.78,
               cam_smoothing=0.1,
               allow_xml_augmentation=False,
               dump_obs_debug=False,
               debug_first_n=20,
               motion_file=None,
               zero_action=False):
    """Run MuJoCo sim2sim. Fdd-style loop: physics steps with decimated policy."""

    # ---- 1. Load MuJoCo ----
    model, data = load_mujoco_model(
        cfg["xml_path"], cfg["simulation_dt"],
        cfg["solver_iterations"], cfg["solver_ls_iterations"],
        allow_augmentation=allow_xml_augmentation)

    dof_names = cfg["dof_names"]
    assert len(dof_names) == NUM_ACTIONS, \
        f"Expected {NUM_ACTIONS} dof_names, got {len(dof_names)}"
    index_map = build_joint_index_map(model, dof_names)

    # ---- 2. Optional reference robot for split-screen playback ----
    ref_motion = load_reference_motion(motion_file, cfg) if motion_file else None
    if ref_motion is not None:
        pkl_cycle_time = len(ref_motion["root_pos"]) / max(ref_motion["fps"], 1e-9)
        old_cycle_time = cfg["cycle_time"]
        cfg["cycle_time"] = float(pkl_cycle_time)
        print(f"[REF_MOTION]   cycle_time from pkl={cfg['cycle_time']:.4f}s "
              f"(yaml fallback was {old_cycle_time:.4f}s)")
    split_screen = ref_motion is not None
    ref_model = None
    ref_data = None
    ref_index_map = None
    if split_screen:
        ref_model, ref_data = load_mujoco_model(
            cfg["xml_path"], cfg["simulation_dt"],
            cfg["solver_iterations"], cfg["solver_ls_iterations"],
            allow_augmentation=allow_xml_augmentation)
        ref_index_map = build_joint_index_map(ref_model, dof_names)
        write_reference_qpos(ref_model, ref_data, ref_index_map, ref_motion, 0)

    # ---- 3. Reset robot (always default pose) ----
    init_dof_pos = cfg["default_dof_pos"].astype(np.float64)
    data.qpos[0:3] = np.array([0.0, 0.0, 0.42], dtype=np.float64)
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # upright
    for i, qpos_id in enumerate(index_map["qpos_ids"]):
        data.qpos[qpos_id] = init_dof_pos[i]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    print(f"[RESET] Using default_dof_pos, root_pos=[0, 0, 0.42]")

    # ---- 3. Camera (free camera, fdd style) ----
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = float(cam_distance)
    cam.azimuth = float(cam_azimuth)
    cam.elevation = float(cam_elevation)

    # ---- 4. Renderer ----
    renderer = None
    renderer_left = None
    renderer_right = None
    frames = []
    if video_out or headless:
        if split_screen:
            renderer_left = mujoco.Renderer(ref_model, height=720, width=1280)
            renderer_right = mujoco.Renderer(model, height=720, width=1280)
        else:
            renderer = mujoco.Renderer(model, height=720, width=1280)
        _width = 1280
        _height = 720
    else:
        try:
            import mujoco_viewer
        except ImportError:
            raise RuntimeError("mujoco_viewer not available; use --headless --video-out")
        viewer = mujoco_viewer.MujocoViewer(model, data)
        viewer.cam.distance = cam_distance
        viewer.cam.azimuth = cam_azimuth
        viewer.cam.elevation = cam_elevation
        viewer.cam.lookat[:] = np.array([0.0, -0.25, cam_lookat_z])

    viewer = None  # headless mode

    # ---- 5. Policy ----
    expected_obs_dim = (cfg["frame_stack"] + 1) * cfg["num_single_obs"]
    if zero_action:
        print(f"[ZERO_ACTION] Bypassing ONNX policy — using action=0.0")
        policy = None
    else:
        policy = load_onnx_policy(policy_path, expected_obs_dim, cfg["num_actions"])

    print(f"[OBS_LAYOUT]")
    print(f"  actions:          slice(0, 22)    dim=22")
    print(f"  base_ang_vel:     slice(22, 25)   dim=3")
    print(f"  dof_pos:          slice(25, 47)   dim=22")
    print(f"  dof_vel:          slice(47, 69)   dim=22")
    hist_len = cfg["frame_stack"] * cfg["num_single_obs"]
    print(f"  history_actor:    slice(69, {69 + hist_len}) dim={hist_len}")
    print(f"  projected_gravity:slice({69 + hist_len}, {69 + hist_len + 3}) dim=3")
    print(f"  ref_motion_phase: slice({69 + hist_len + 3}, {69 + hist_len + 4}) dim=1")
    print(f"  final_obs_dim={expected_obs_dim}")
    if not zero_action:
        print(f"  onnx_input_dim={policy['session'].get_inputs()[0].shape[-1]}")
    print(f"[HISTORY_MODE] use_previous_history_then_update=True")
    print(f"[ZERO_ACTION] {zero_action}")

    # ---- 6. State init ----
    target_dof_pos = init_dof_pos.copy().astype(np.float32)  # match init qpos
    action = np.zeros(cfg["num_actions"], dtype=np.float32)
    hist_dict, hist_obs_c = create_history(cfg)
    print(f"[STATE_INIT] target_dof_pos = init_dof_pos (not default_dof_pos)")

    # ---- 7. Timing ----
    render_every_steps = max(1, int(round(
        1.0 / max(cfg["simulation_dt"] * float(video_fps), 1e-9))))
    sim_steps = int(cfg["simulation_duration"] / cfg["simulation_dt"])
    output_base_dir = os.path.dirname(video_out) if video_out else "sim2sim_outputs/q1_cr7"
    os.makedirs(output_base_dir, exist_ok=True)

    print(f"[SIM2SIM] sim_steps={sim_steps} ({cfg['simulation_duration']:.1f}s @ "
          f"{1.0/cfg['simulation_dt']:.0f}Hz), policy @ "
          f"{1.0/(cfg['simulation_dt']*cfg['control_decimation']):.0f}Hz, "
          f"render_stride={render_every_steps}")

    # ---- 8. CSV setup ----
    csv_rows = []
    stop_reason = "duration_reached"
    counter = 0
    cam_base_x = float(data.qpos[0])
    cam_base_y = float(data.qpos[1])
    smoothing = max(0.0, min(1.0, float(cam_smoothing)))

    # ---- 9. Main loop ----
    t0_wall = time.time()
    for _ in range(sim_steps):
        # PD control EVERY physics step
        mj = get_mujoco_data(data, index_map, cfg["default_dof_pos"])
        tau = pd_control(target_dof_pos, mj["dof_pos"],
                         np.zeros_like(cfg["kds"]), mj["dof_vel"], cfg)
        for i, act_id in enumerate(index_map["actuator_ids"]):
            data.ctrl[act_id] = tau[i]
        mujoco.mj_step(model, data)

        # Safety checks
        if cfg["safety_check_nonfinite"]:
            if (not np.isfinite(data.qpos).all()) or \
               (not np.isfinite(data.qvel).all()) or \
               (not np.isfinite(data.qacc).all()):
                stop_reason = "numerical_instability_nonfinite"
                print(f"[WARN] Non-finite state at step={counter}; stop")
                break
            max_abs_qacc = float(np.max(np.abs(data.qacc)))
            if max_abs_qacc > cfg["max_abs_qacc"]:
                stop_reason = "numerical_instability_qacc"
                print(f"[WARN] Excessive qacc at step={counter}: "
                      f"max_abs_qacc={max_abs_qacc:.3e} > {cfg['max_abs_qacc']:.3e}")
                break

        counter += 1

        # Motion end check (per physics step, matches fdd)
        if cfg["stop_at_motion_end"] and \
           (cfg["time_offset"] + counter * cfg["simulation_dt"]) >= cfg["cycle_time"]:
            stop_reason = "motion_end"
            print(f"[INFO] Reached motion end at "
                  f"t={cfg['time_offset'] + counter * cfg['simulation_dt']:.3f}s "
                  f"(cycle_time={cfg['cycle_time']:.3f}s); stop")
            break

        # Policy inference every decimation steps
        if counter % cfg["control_decimation"] == 0:
            if zero_action:
                raw_action = np.zeros(cfg["num_actions"], dtype=np.float32)
                action, target_dof_pos = apply_action_postprocess(
                    raw_action, action, target_dof_pos, cfg)
            else:
                # Build obs (previous history, then update)
                obs_buff, hist_obs_c, obs_single = get_obs(hist_obs_c, hist_dict, mj, action, counter, cfg)
                raw_action = infer_policy_action(policy, obs_buff, cfg["clip_actions"])
                action, target_dof_pos = apply_action_postprocess(
                    raw_action, action, target_dof_pos, cfg)

            # CSV
            phase = compute_ref_motion_phase(counter, cfg)
            control_step = counter // cfg["control_decimation"]
            csv_rows.append(_make_csv_row(counter, cfg, phase, mj, data,
                                          action, raw_action, target_dof_pos, tau))

            # ---- Per-step debug (first N control steps) ----
            if control_step < debug_first_n:
                _print_action_debug(control_step, phase, raw_action, action,
                                    target_dof_pos, mj, tau, cfg)

            # ---- Obs component dump (first policy step only) ----
            if dump_obs_debug and control_step == 1 and not zero_action:
                _dump_obs_debug(obs_buff, obs_single, hist_obs_c,
                                mj, raw_action, phase, action, cfg, output_base_dir)

            # Progress
            control_step = counter // cfg["control_decimation"]
            if control_step % 50 == 0:
                t_sim = counter * cfg["simulation_dt"]
                tau_abs = np.abs(tau)
                tau_sat = float((tau_abs >= cfg["tau_limit"] * 0.95).mean()) * 100
                print(f"  t={t_sim:.2f}s step={counter} ctrl={control_step} "
                      f"z={float(data.qpos[2]):.3f} phase={phase:.3f} "
                      f"tau_sat={tau_sat:.0f}%")

        # Render (strided)
        if (renderer is not None or renderer_right is not None) and counter % render_every_steps == 0:
            if cam_follow_base:
                cam_base_x += smoothing * (float(data.qpos[0]) - cam_base_x)
                cam_base_y += smoothing * (float(data.qpos[1]) - cam_base_y)
                cam.lookat[:] = np.array([cam_base_x, cam_base_y, float(cam_lookat_z)],
                                         dtype=np.float64)
            if split_screen:
                sim_time = counter * cfg["simulation_dt"]
                motion_idx = int((cfg["time_offset"] + sim_time) * ref_motion["fps"])
                motion_idx = min(motion_idx, len(ref_motion["root_pos"]) - 1)
                ref_root_pos = ref_motion["root_pos"][motion_idx].copy()
                ref_root_pos[0:2] = data.qpos[0:2]
                ref_root_rot = align_quat_yaw_wxyz(ref_motion["root_rot"][motion_idx], data.qpos[3:7])
                write_reference_qpos(ref_model, ref_data, ref_index_map, ref_motion, motion_idx,
                                     root_pos_override=ref_root_pos,
                                     root_rot_override=ref_root_rot)

                renderer_left.update_scene(ref_data, camera=cam)
                left = draw_panel_label(renderer_left.render(), "Reference Motion")

                renderer_right.update_scene(data, camera=cam)
                right = draw_panel_label(renderer_right.render(), "Sim2Sim Policy")

                frame = np.concatenate([left, right], axis=1)
                frames.append(frame)
            else:
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render())

    elapsed = time.time() - t0_wall
    policy_dt = cfg["simulation_dt"] * cfg["control_decimation"]
    control_steps = counter // cfg["control_decimation"]

    # ====== Summary ======
    print(f"\n{'='*60}")
    print("Simulation Summary")
    print(f"{'='*60}")
    print(f"  Physics steps: {counter}")
    print(f"  Control steps: {control_steps}")
    print(f"  Sim time: {counter * cfg['simulation_dt']:.2f}s")
    print(f"  Wall time: {elapsed:.1f}s")
    print(f"  Stop reason: {stop_reason}")
    print(f"  Captured frames: {len(frames)}")
    rendering_enabled = renderer is not None or renderer_right is not None
    if rendering_enabled and len(frames) > 0:
        actual_fps = len(frames) / max(elapsed, 1e-6)
        print(f"  Actual render FPS: {actual_fps:.1f}")

    # ====== Save video ======
    if rendering_enabled and frames and video_out:
        os.makedirs(os.path.dirname(video_out), exist_ok=True)
        _save_video(frames, video_out, fps=video_fps)
        print(f"  render_stride={render_every_steps}")

    if renderer is not None:
        renderer.close()
    if renderer_left is not None:
        renderer_left.close()
    if renderer_right is not None:
        renderer_right.close()

    # ====== Save CSV ======
    if video_out and csv_rows:
        csv_path = video_out.replace(".mp4", "_debug.csv")
        _save_csv(csv_rows, csv_path)

    print("Done.")


def _make_csv_row(physics_step, cfg, phase, mj_data, data, action, raw_action,
                  target_dof_pos, tau):
    """CSV row per control step."""
    tau_abs = np.abs(tau)
    return OrderedDict([
        ("step", physics_step // cfg["control_decimation"]),
        ("physics_step", physics_step),
        ("time", f"{physics_step * cfg['simulation_dt']:.6f}"),
        ("phase", f"{phase:.6f}"),
        ("root_z", f"{float(data.qpos[2]):.6f}"),
        ("root_quat_w", f"{float(data.qpos[3]):.6f}"),
        ("root_quat_x", f"{float(data.qpos[4]):.6f}"),
        ("root_quat_y", f"{float(data.qpos[5]):.6f}"),
        ("root_quat_z", f"{float(data.qpos[6]):.6f}"),
        ("projected_gravity_0", f"{float(mj_data['gvec'][0]):.6f}"),
        ("projected_gravity_1", f"{float(mj_data['gvec'][1]):.6f}"),
        ("projected_gravity_2", f"{float(mj_data['gvec'][2]):.6f}"),
        ("action_max_abs", f"{float(np.max(np.abs(raw_action))):.6f}"),
        ("action_mean", f"{float(np.mean(raw_action)):.6f}"),
        ("action[0]", f"{float(action[0]):.6f}"),
        ("action[1]", f"{float(action[1]):.6f}"),
        ("action[2]", f"{float(action[2]):.6f}"),
        ("dof_pos[0]", f"{float(mj_data['dof_pos'][0]):.6f}"),
        ("dof_pos[1]", f"{float(mj_data['dof_pos'][1]):.6f}"),
        ("dof_pos[2]", f"{float(mj_data['dof_pos'][2]):.6f}"),
        ("dof_vel[0]", f"{float(mj_data['dof_vel'][0]):.6f}"),
        ("target_dof_pos[0]", f"{float(target_dof_pos[0]):.6f}"),
        ("target_dof_pos[1]", f"{float(target_dof_pos[1]):.6f}"),
        ("tau[0]", f"{float(tau[0]):.4f}"),
        ("tau[1]", f"{float(tau[1]):.4f}"),
        ("tau[2]", f"{float(tau[2]):.4f}"),
        ("max_abs_tau", f"{float(np.max(tau_abs)):.4f}"),
        ("stop_reason", ""),
    ])


def _save_video(frames, output_path, fps=50):
    """Save frames as MP4 via ffmpeg subprocess (H.264, VSCode-compatible)."""
    import subprocess
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}', '-pix_fmt', 'rgb24', '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-pix_fmt', 'yuv420p',
        str(out),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"Video saved: {output_path} ({len(frames)} frames)")


def _save_csv(rows, csv_path):
    """Save CSV debug file."""
    if not rows:
        return
    rows[-1]["stop_reason"] = "end"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {csv_path} ({len(rows)} rows)")


# ============================================================================
# Per-step action debug + obs dump
# ============================================================================

def _print_action_debug(control_step, phase, raw_action, filtered_action,
                        target_dof_pos, mj_data, tau, cfg):
    """Print per-joint action debug for first N control steps."""
    dof_names = cfg["dof_names"]
    default_dof_pos = cfg["default_dof_pos"]
    action_scale = cfg["action_scale"]

    abs_act = np.abs(raw_action)
    top8 = np.argsort(abs_act)[::-1][:8]

    print(f"\n[STEP_DEBUG] step={control_step} phase={phase:.4f} "
          f"action_max_abs={abs_act.max():.4f} action_mean={raw_action.mean():+.4f}")

    print("[TOP_ACTION]")
    for rank, i in enumerate(top8):
        target_i = target_dof_pos[i]
        dof_i = mj_data["dof_pos"][i]
        print(f"  rank={rank} idx={i:2d} joint={dof_names[i]:<30s} "
              f"raw_action={raw_action[i]:+.4f} "
              f"target={target_i:+.4f} dof={dof_i:+.4f} tau={tau[i]:+.4f}")

    # Shoulder/elbow detail
    arm_idx = {
        "left_shoulder_pitch": 14, "left_shoulder_roll": 15,
        "left_shoulder_yaw": 16, "left_elbow": 17,
        "right_shoulder_pitch": 18, "right_shoulder_roll": 19,
        "right_shoulder_yaw": 20, "right_elbow": 21,
    }
    print("[ARM_DEBUG]")
    for label, i in arm_idx.items():
        target_i = target_dof_pos[i]
        dof_i = mj_data["dof_pos"][i]
        print(f"  {label:<24s} idx={i:2d} "
              f"action={raw_action[i]:+.4f} "
              f"target={target_i:+.4f} dof={dof_i:+.4f} tau={tau[i]:+.4f}")

    # Leg detail
    leg_idx = {
        "left_hip_pitch": 0, "left_hip_roll": 1, "left_knee": 3,
        "left_ankle_pitch": 4, "left_ankle_roll": 5,
        "right_hip_pitch": 6, "right_hip_roll": 7, "right_knee": 9,
        "right_ankle_pitch": 10, "right_ankle_roll": 11,
    }
    print("[LEG_DEBUG]")
    for label, i in leg_idx.items():
        target_i = target_dof_pos[i]
        dof_i = mj_data["dof_pos"][i]
        print(f"  {label:<24s} idx={i:2d} "
              f"action={raw_action[i]:+.4f} "
              f"target={target_i:+.4f} dof={dof_i:+.4f} tau={tau[i]:+.4f}")

    # Diagnostic: are shoulder actions already large?
    shoulder_max = max(abs(raw_action[i]) for i in [14, 15, 16, 17, 18, 19, 20, 21])
    if shoulder_max > 0.8:
        print(f"[DIAG] shoulder raw_action max_abs={shoulder_max:.4f} (large) "
              f"-> likely actor_obs mismatch or expected ref tracking deviation")
    else:
        # Check if target/dof exceeds joint limits despite moderate action
        for i in [14, 15, 16, 17, 18, 19, 20, 21]:
            target_i = target_dof_pos[i]
            if abs(target_i - default_dof_pos[i]) > 1.0:
                print(f"[DIAG] {dof_names[i]} target offset large: "
                      f"target={target_i:+.4f} default={default_dof_pos[i]:+.4f}")
        print(f"[DIAG] shoulder raw_action max_abs={shoulder_max:.4f} (moderate) "
              f"-> action normal but actuator/joint mapping or PD/XML may be wrong")


def _dump_obs_debug(obs_buff, obs_single, hist_obs_c,
                    mj_data, raw_action, phase, action, cfg, output_dir):
    """Save obs components and first action as .npy/.npz for comparison."""
    dump_dir = output_dir
    os.makedirs(dump_dir, exist_ok=True)

    # Full obs
    np.save(os.path.join(dump_dir, "mujoco_actor_obs_step0.npy"),
            obs_buff.astype(np.float32))
    # Action
    np.save(os.path.join(dump_dir, "mujoco_action_step0.npy"),
            raw_action.astype(np.float32))

    # Decompose into components matching obs layout
    hist_len = cfg["frame_stack"] * cfg["num_single_obs"]  # 292

    comp = {}
    comp["actions"] = obs_buff[0, 0:22]
    comp["base_ang_vel"] = obs_buff[0, 22:25]
    comp["dof_pos"] = obs_buff[0, 25:47]
    comp["dof_vel"] = obs_buff[0, 47:69]
    comp["history_actor"] = obs_buff[0, 69:69 + hist_len]
    comp["projected_gravity"] = obs_buff[0, 69 + hist_len:69 + hist_len + 3]
    comp["ref_motion_phase"] = obs_buff[0, 69 + hist_len + 3:69 + hist_len + 4]

    np.savez(os.path.join(dump_dir, "mujoco_obs_components_step0.npz"), **comp)

    # Also save raw state values (unscaled) for reference
    raw_state = {
        "dof_pos_raw": mj_data["dof_pos"].astype(np.float32),
        "dof_vel_raw": mj_data["dof_vel"].astype(np.float32),
        "base_angvel_raw": mj_data["base_angvel"].astype(np.float32),
        "gvec_raw": mj_data["gvec"].astype(np.float32),
        "phase": np.float32(phase),
        "action_prev": action.astype(np.float32),
        "raw_action": raw_action.astype(np.float32),
        "default_dof_pos": cfg["default_dof_pos"].astype(np.float32),
    }
    np.savez(os.path.join(dump_dir, "mujoco_raw_state_step0.npz"), **raw_state)

    # Print component stats
    print(f"\n[OBS_DUMP] saved to {dump_dir}/")
    print(f"[OBS_COMPONENTS_STEP0]")
    for key in ["actions", "base_ang_vel", "dof_pos", "dof_vel",
                "projected_gravity", "ref_motion_phase"]:
        c = comp[key]
        print(f"  {key:<20s} shape={tuple(c.shape)} "
              f"mean={c.mean():+.6f} std={c.std():.6f} "
              f"min={c.min():+.6f} max={c.max():+.6f} "
              f"first8={np.array2string(c.flatten()[:8], precision=3, suppress_small=True)}")
    h = comp["history_actor"]
    nonzero_h = np.count_nonzero(h)
    print(f"  {'history_actor':<20s} shape={tuple(h.shape)} "
          f"mean={h.mean():+.6f} std={h.std():.6f} "
          f"nonzero={nonzero_h}/{h.size} "
          f"first8={np.array2string(h.flatten()[:8], precision=3, suppress_small=True)}")
    print(f"  obs_full_dim={obs_buff.shape[1]}  onnx_input=365")


# ============================================================================
# CLI (fdd style)
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Q1 CR7 Motion Tracking — MuJoCo sim2sim (fdd-style)")
    p.add_argument("--config", type=str, default="scripts/q1_sim2sim_config.yaml")
    p.add_argument("--policy-path", type=str, default="",
                   help="Path to exported ONNX policy. Not required with --zero-action.")
    p.add_argument("--xml-path", type=str, default=None)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--video-out", type=str, default="")
    p.add_argument("--video-fps", type=int, default=50)
    p.add_argument("--cam-follow-base", action="store_true", default=True)
    p.add_argument("--no-cam-follow-base", action="store_true")
    p.add_argument("--cam-distance", type=float, default=3.5)
    p.add_argument("--cam-azimuth", type=float, default=135.0)
    p.add_argument("--cam-elevation", type=float, default=-12.0)
    p.add_argument("--cam-lookat-z", type=float, default=0.78)
    p.add_argument("--cam-smoothing", type=float, default=0.1)
    p.add_argument("--allow-xml-augmentation", action="store_true",
                   help="Allow runtime augmentation of kinematic-only XML with "
                        "minimal physics (masses, STL meshes, collision boxes).")
    p.add_argument("--motion-file", type=str, default=None,
                   help="Path to reference motion .pkl for left split-screen playback. "
                        "Without it, video rendering stays single-screen sim2sim.")
    p.add_argument("--dump-obs-debug", action="store_true",
                   help="Save obs components, action, and raw state as .npy/.npz "
                        "for comparison with IsaacGym eval output.")
    p.add_argument("--debug-first-n", type=int, default=20,
                   help="Print per-joint action debug for first N control steps.")
    p.add_argument("--zero-action", action="store_true",
                   help="Bypass ONNX policy — use zero actions. Useful for "
                        "verifying physics/XML without policy influence.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.no_cam_follow_base:
        args.cam_follow_base = False

    # Resolve paths
    if not os.path.isabs(args.config):
        args.config = os.path.join(_REPO_ROOT, args.config)
    if args.video_out and not os.path.isabs(args.video_out):
        args.video_out = os.path.join(_REPO_ROOT, args.video_out)

    cfg = read_conf(args.config)

    # CLI overrides
    if args.xml_path:
        xml_path = args.xml_path
        if not os.path.isabs(xml_path):
            xml_path = os.path.join(_REPO_ROOT, xml_path)
        cfg["xml_path"] = xml_path
    if args.duration is not None and args.duration > 0:
        cfg["simulation_duration"] = float(args.duration)

    print("=" * 60)
    print("Q1 CR7 Motion Tracking — MuJoCo sim2sim")
    print("=" * 60)
    print(f"[INFO] config={args.config}")
    print(f"[INFO] xml_path={cfg['xml_path']}")
    print(f"[INFO] policy_path={args.policy_path}")
    print(f"[INFO] duration={cfg['simulation_duration']:.1f}s "
          f"dt={cfg['simulation_dt']:.4f}s decim={cfg['control_decimation']} "
          f"cycle_time={cfg['cycle_time']:.1f}s "
          f"stop_at_motion_end={int(cfg['stop_at_motion_end'])} "
          f"phase_wrap={int(cfg['phase_wrap'])}")
    print(f"[INFO] kp_scale={cfg['kp_scale']:.3f} kd_scale={cfg['kd_scale']:.3f} "
          f"action_scale={cfg['action_scale']:.3f} "
          f"action_filter_alpha={cfg['action_filter_alpha']:.3f}")
    print(f"[INFO] allow_xml_augmentation={args.allow_xml_augmentation}")
    print()

    # Resolve motion file path
    motion_file = args.motion_file
    if motion_file and not os.path.isabs(motion_file):
        motion_file = os.path.join(_REPO_ROOT, motion_file)

    run_mujoco(
        cfg,
        policy_path=args.policy_path,
        headless=args.headless,
        video_out=args.video_out,
        video_fps=args.video_fps,
        cam_follow_base=args.cam_follow_base,
        cam_distance=args.cam_distance,
        cam_azimuth=args.cam_azimuth,
        cam_elevation=args.cam_elevation,
        cam_lookat_z=args.cam_lookat_z,
        cam_smoothing=args.cam_smoothing,
        allow_xml_augmentation=args.allow_xml_augmentation,
        dump_obs_debug=args.dump_obs_debug,
        debug_first_n=args.debug_first_n,
        motion_file=motion_file,
        zero_action=args.zero_action,
    )
    print("-----done------")


if __name__ == "__main__":
    main()
