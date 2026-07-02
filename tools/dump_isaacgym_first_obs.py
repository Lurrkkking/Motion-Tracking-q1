#!/usr/bin/env python3
"""Dump first-step actor_obs + action from IsaacGym eval for sim2sim comparison."""
import sys, os
# isaacgym MUST be imported before torch
from isaacgym import gymapi, gymtorch
import torch
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from humanoidverse.envs.q1_motion_tracking.q1_cr7_motion_tracking import Q1CR7MotionTracking
from easydict import EasyDict

# Load config — manually resolve references
import yaml
with open("logs/Q1_CR7/4400_success_cr7/config.yaml") as f:
    raw = yaml.safe_load(f)

robot_cfg = raw["robot"]
# resolve $${...} references in dof_obs_size etc
# most are just direct values after Hydra resolves them in training

# Resolve asset paths
robot_cfg["asset"]["robot_type"] = "q1_22dof_box"
robot_cfg["asset"]["urdf_file"] = "q1/q1_22dof_box.urdf"
robot_cfg["asset"]["xml_file"] = "q1/q1_22dof_box.xml"
robot_cfg["motion"]["asset"]["assetFileName"] = "q1_22dof_box.xml"
robot_cfg["motion"]["asset"]["urdfFileName"] = "q1_22dof_box.urdf"

device = "cuda:0"
env_cfg = EasyDict(raw["env"]["config"])
env_cfg.robot = EasyDict(robot_cfg)
env_cfg.simulator = EasyDict(raw["simulator"])
env_cfg.obs = EasyDict(raw["obs"])
env_cfg.rewards = EasyDict(raw["rewards"])
env_cfg.domain_rand = EasyDict(raw["domain_rand"])
# terrain and simulator sections need direct assignment (env cfg references them as $${terrain} etc)
env_cfg.terrain = EasyDict(raw["terrain"])
env_cfg.simulator = EasyDict(raw["simulator"])
env_cfg.simulator.config.terrain = EasyDict(raw["terrain"])
env_cfg.headless = True
env_cfg.num_envs = 1

# Fix obs_dims: resolve eval expressions and convert list-of-dicts to flat dict
obs_dims_dict = {}
for item in raw["obs"]["obs_dims"]:
    for k, v in item.items():
        if isinstance(v, str) and "dof_obs_size" in v:
            obs_dims_dict[k] = 22
        elif isinstance(v, str) and "num_bodies" in v:
            obs_dims_dict[k] = 69
        elif isinstance(v, str) and "eval:" in v:
            expr = v.split("eval:")[1].strip("'")
            expr = expr.replace("${robot.dof_obs_size}", "22")
            expr = expr.replace("${robot.num_bodies}", "23")
            expr = expr.replace("${robot.motion.nums_extend_bodies}", "0")
            obs_dims_dict[k] = eval(expr)
        else:
            obs_dims_dict[k] = int(v) if not isinstance(v, str) else 22
env_cfg.obs.obs_dims = obs_dims_dict

env = Q1CR7MotionTracking(env_cfg, device)
# After init, buffers are set up but no obs computed yet.
# Run one step with zero action to trigger observation computation
obs = torch.zeros(1, 22, device=device)
env.step(obs)
# After step, obs_buf_dict should be populated
print(f"Available attrs: {[k for k in dir(env) if 'obs' in k.lower()]}")
if hasattr(env, 'obs_buf_dict'):
    print("obs_buf_dict available")
else:
    # Try computing observations manually
    env._pre_compute_observations_callback()
    env._compute_observations()

actor_obs = env.obs_buf_dict["actor_obs"].cpu().numpy()[0]
print(f"IG actor_obs: shape={actor_obs.shape} sum={actor_obs.sum():.2f}")

with torch.no_grad():
    action = env.actor_critic.actor(env.obs_buf_dict["actor_obs"]).cpu().numpy()[0]

print(f"IG action: {np.array2string(action, precision=3, max_line_width=200)}")
print(f"IG action[knee_L]={action[3]:.4f}")

os.makedirs("debug_outputs", exist_ok=True)
np.save("debug_outputs/isaacgym_first_obs.npy", actor_obs)
np.save("debug_outputs/isaacgym_first_action.npy", action)
print("Saved debug_outputs/isaacgym_first_obs.npy")
print("Saved debug_outputs/isaacgym_first_action.npy")

# Compare with MuJoCo
mj_obs = np.load("debug_outputs/mujoco_first_obs.npy")
mj_action = None
import onnxruntime
sess = onnxruntime.InferenceSession("logs/Q1_CR7/4400_success_cr7/exported/model_4400.onnx")
mj_action = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: mj_obs.reshape(1,-1)})[0][0]

COMP_DIM = dict(base_ang_vel=3, projected_gravity=3, dof_pos=22, dof_vel=22,
                actions=22, ref_motion_phase=1, q1_root_error=4, q1_yaw_error=4,
                q1_flight_phase=6, q1_ref_dof_error=44)
OBS_ORDER = ["base_ang_vel","projected_gravity","dof_pos","dof_vel","actions",
             "ref_motion_phase","q1_root_error","q1_yaw_error","q1_flight_phase",
             "q1_ref_dof_error","history_actor"]
HIST_KEYS = ["base_ang_vel","projected_gravity","dof_pos","dof_vel","actions","ref_motion_phase"]

print(f"\n{'='*80}")
print(f"OBS COMPARISON: IsaacGym vs MuJoCo")
print(f"IG obs shape={actor_obs.shape} sum={actor_obs.sum():.2f}")
print(f"MJ obs shape={mj_obs.shape} sum={mj_obs.sum():.2f}")
l2 = np.sqrt(np.sum((actor_obs - mj_obs)**2))
print(f"Total L2 diff: {l2:.4f}")

off_ig = off_mj = 0
for k in OBS_ORDER:
    d = COMP_DIM.get(k, sum(COMP_DIM[hk]*4 for hk in HIST_KEYS))
    ig, mj = actor_obs[off_ig:off_ig+d], mj_obs[off_mj:off_mj+d]
    diff = np.sqrt(np.sum((ig-mj)**2))
    max_err = np.max(np.abs(ig-mj))
    m = "⚠" if max_err > 0.1 else " "
    print(f"{m} {k:25s} dim={d:3d} L2={diff:8.4f} max_err={max_err:.4f}")
    if max_err > 0.1:
        print(f"   IG[:6]={np.array2string(ig[:min(6,d)], precision=3)}")
        print(f"   MJ[:6]={np.array2string(mj[:min(6,d)], precision=3)}")
    off_ig += d; off_mj += d

print(f"\nACTION COMPARISON:")
print(f"IG action: {np.array2string(action, precision=3, max_line_width=200)}")
print(f"MJ action: {np.array2string(mj_action, precision=3, max_line_width=200)}")
act_l2 = np.sqrt(np.sum((action - mj_action)**2))
print(f"Action L2 diff: {act_l2:.4f}")
per_joint = np.abs(action - mj_action)
for i in np.argsort(-per_joint)[:5]:
    print(f"  top{np.argsort(-per_joint).tolist().index(i)+1}: joint[{i}] IG={action[i]:.4f} MJ={mj_action[i]:.4f} diff={per_joint[i]:.4f}")
