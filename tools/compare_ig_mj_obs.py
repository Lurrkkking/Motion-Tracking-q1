#!/usr/bin/env python3
"""Compare IsaacGym vs MuJoCo first-step obs by computing IG obs from known formulas."""
import numpy as np, joblib, yaml

# Load training config for reference
with open("logs/Q1_CR7/4400_success_cr7/config.yaml") as f:
    cfg = yaml.safe_load(f)

robot = cfg["robot"]
default_dof = np.array([robot["init_state"]["default_joint_angles"][n] for n in robot["dof_names"]])
obs_scales = cfg["obs"]["obs_scales"]

# IsaacGym initial state (robot at default pose, zero velocity)
ig_dof_pos = default_dof.copy()          # current = default
ig_dof_vel = np.zeros(22)
ig_base_ang_vel = np.zeros(3)            # stationary
ig_proj_gravity = np.array([0., 0., -1.])  # upright
ig_actions = np.zeros(22)                # no previous action
ig_root_z = 0.42
ig_root_vz = 0.0
ig_yaw = 0.0
ig_yaw_rate = 0.0

# Motion at t=dt (training uses (0+1)*dt)
dt = 0.02  # control_dt
cycle = 3.56
phase = dt / cycle  # 0.0056

# Load reference motion for first frame
data = joblib.load("humanoidverse/data/motions/q1/q1_cr7_scale045_rootz042.pkl")
k = sorted(data.keys())[0]
m = data[k]
dof_all = m["pose_aa"].sum(axis=-1)[:, 1:].astype(np.float64)
root_all = m["root_trans_offset"].astype(np.float64)
mdt = 1.0 / m["fps"]

# Reference at t=dt
frame = dt / mdt
f0, f1 = int(np.floor(frame)), min(int(np.floor(frame)) + 1, len(dof_all) - 1)
a = frame - f0
ref_dof = dof_all[f0] * (1 - a) + dof_all[f1] * a
ref_root = root_all[f0] * (1 - a) + root_all[f1] * a

# Compute velocities (finite diff)
T = len(dof_all)
dof_vel_all = np.zeros_like(dof_all)
root_vel_all = np.zeros_like(root_all)
dof_vel_all[1:-1] = (dof_all[2:] - dof_all[:-2]) / (2 * mdt)
root_vel_all[1:-1] = (root_all[2:] - root_all[:-2]) / (2 * mdt)
dof_vel_all[0] = (dof_all[1] - dof_all[0]) / mdt
root_vel_all[0] = (root_all[1] - root_all[0]) / mdt
ref_dvel = dof_vel_all[f0] * (1 - a) + dof_vel_all[f1] * a
ref_rvel = root_vel_all[f0] * (1 - a) + root_vel_all[f1] * a

# Compute IsaacGym obs components
ig_parts = {}

# base_ang_vel: same as MJ, zero
ig_parts["base_ang_vel"] = ig_base_ang_vel * obs_scales["base_ang_vel"]

# projected_gravity
ig_parts["projected_gravity"] = ig_proj_gravity * obs_scales["projected_gravity"]

# dof_pos: (current - default) = 0
ig_parts["dof_pos"] = (ig_dof_pos - default_dof) * obs_scales["dof_pos"]

# dof_vel: 0
ig_parts["dof_vel"] = ig_dof_vel * obs_scales["dof_vel"]

# actions: 0
ig_parts["actions"] = ig_actions * obs_scales["actions"]

# ref_motion_phase
ig_parts["ref_motion_phase"] = np.array([phase], dtype=np.float32) * obs_scales["ref_motion_phase"]

# q1_root_error: [ref_z - act_z, ref_vz - act_vz, act_z, act_vz]
ig_parts["q1_root_error"] = np.array([
    ref_root[2] - ig_root_z,
    ref_rvel[2] - ig_root_vz,
    ig_root_z, ig_root_vz
], dtype=np.float32) * obs_scales["q1_root_error"]

# q1_yaw_error: [sin(err), cos(err), rate_err, act_rate]
ig_parts["q1_yaw_error"] = np.array([
    0., 1., 0., 0.  # yaw error = 0 at start
], dtype=np.float32) * obs_scales["q1_yaw_error"]

# q1_flight_phase: [crouch, takeoff, flight, landing, left_contact, right_contact]
ig_parts["q1_flight_phase"] = np.array([
    0, 0, 0, 0, 1, 1  # both feet in contact, no active phase
], dtype=np.float32) * obs_scales["q1_flight_phase"]

# q1_ref_dof_error: [ref_pos - act_pos, ref_vel - act_vel]
ig_parts["q1_ref_dof_error"] = np.concatenate([
    ref_dof - ig_dof_pos,
    ref_dvel - ig_dof_vel
], dtype=np.float32) * obs_scales["q1_ref_dof_error"]

# history_actor: all zeros
ig_parts["history_actor"] = np.zeros(292, dtype=np.float32) * obs_scales["history_actor"]

# Concatenate
OBS_ORDER = ["base_ang_vel", "projected_gravity", "dof_pos", "dof_vel", "actions",
             "ref_motion_phase", "q1_root_error", "q1_yaw_error", "q1_flight_phase",
             "q1_ref_dof_error", "history_actor"]
ig_obs = np.concatenate([ig_parts[k] for k in OBS_ORDER]).astype(np.float32)
ig_obs = np.clip(ig_obs, -100, 100)

# Load MuJoCo obs
mj_obs = np.load("debug_outputs/mujoco_first_obs.npy")

# Compare
print("=== OBS COMPARISON: IsaacGym (computed) vs MuJoCo ===")
print(f"IG obs: shape={ig_obs.shape} sum={ig_obs.sum():.2f}")
print(f"MJ obs: shape={mj_obs.shape} sum={mj_obs.sum():.2f}")
print(f"L2 diff: {np.sqrt(np.sum((ig_obs - mj_obs)**2)):.4f}")

COMP_DIM = dict(base_ang_vel=3, projected_gravity=3, dof_pos=22, dof_vel=22,
                actions=22, ref_motion_phase=1, q1_root_error=4, q1_yaw_error=4,
                q1_flight_phase=6, q1_ref_dof_error=44, history_actor=292)
off = 0
for k in OBS_ORDER:
    d = COMP_DIM[k]
    ig, mj = ig_obs[off:off+d], mj_obs[off:off+d]
    diff = np.sqrt(np.sum((ig-mj)**2))
    max_err = np.max(np.abs(ig-mj))
    m = "⚠" if max_err > 0.01 else " "
    print(f"{m} {k:25s} dim={d:3d} L2={diff:8.4f} max_err={max_err:.4f}")
    if max_err > 0.01:
        print(f"   IG[:6]={np.array2string(ig[:min(6,d)], precision=4)}")
        print(f"   MJ[:6]={np.array2string(mj[:min(6,d)], precision=4)}")
    off += d
