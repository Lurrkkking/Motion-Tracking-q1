#!/usr/bin/env python
"""
Q1 CR7 rollout debug script — produces debug_outputs/q1_cr7_rollout_debug.csv

Records per-step metrics to diagnose:
1. Does crouch phase still terminate?
2. Does root_vz show positive peaks?
3. Flight phase foot contact rate?
4. Does yaw error decrease?
5. Is policy outputting pushing actions?

Usage (after training):
    cd /root/autodl-tmp/ASAP_official
    python scripts/eval_q1_cr7_debug.py \
        --checkpoint logs/<run>/model_<iter>.pt \
        --output debug_outputs/q1_cr7_rollout_debug.csv

Or run directly (will use random policy for quick test):
    python scripts/eval_q1_cr7_debug.py
"""

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

# isaacgym must be imported before torch in some setups
try:
    from isaacgym import gymapi, gymtorch
except ImportError:
    pass


def parse_args():
    p = argparse.ArgumentParser(description="Q1 CR7 rollout debug")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to trained checkpoint. If None, uses random actions.")
    p.add_argument("--output", type=str, default="debug_outputs/q1_cr7_rollout_debug.csv",
                   help="Output CSV path")
    p.add_argument("--num_steps", type=int, default=200,
                   help="Number of steps to run")
    p.add_argument("--log_interval", type=int, default=5,
                   help="Log every N steps")
    return p.parse_args()


def load_checkpoint(path, device):
    """Load a PPO checkpoint and return the actor network."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Try to extract actor
    if 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    elif 'actor' in ckpt:
        return ckpt['actor']
    else:
        # The checkpoint structure varies — try common keys
        for k in ['actor_state_dict', 'actor_critic_state_dict']:
            if k in ckpt:
                return ckpt[k]
        raise KeyError(f"Cannot find actor in checkpoint. Keys: {list(ckpt.keys())}")
    return sd


def run_rollout(env, num_steps, log_interval, output_path, policy=None):
    """Run rollout and log metrics to CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fieldnames = [
        'step', 'phase', 'is_crouch', 'is_takeoff', 'is_flight', 'is_landing',
        'ref_root_z', 'actual_root_z', 'root_z_error',
        'ref_root_vz', 'actual_root_vz', 'root_vz_error',
        'ref_yaw', 'actual_yaw', 'yaw_error_deg',
        'ref_yaw_rate', 'actual_yaw_rate', 'yaw_rate_error',
        'left_contact', 'right_contact', 'both_feet_air',
        'root_z_reward', 'root_vz_reward', 'yaw_reward', 'yaw_rate_reward',
        'flight_contact_reward', 'lower_body_joint_reward',
        'terminate_low_height', 'terminate_gravity', 'terminate_contact',
        'reset_buf_any', 'episode_length',
        'action_max_abs', 'knee_action_mean', 'ankle_action_mean', 'hip_action_mean',
        'knee_torque_mean', 'ankle_torque_mean', 'hip_torque_mean',
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        obs_dict = env.obs_buf_dict
        actor_obs = obs_dict.get('actor_obs', None)

        for step_i in range(num_steps):
            # Get action
            if policy is not None:
                with torch.no_grad():
                    action = policy(actor_obs)
            else:
                action = torch.randn(env.num_envs, env.num_actions, device=env.device) * 0.1

            obs_dict, rew_buf, reset_buf, extras = env.step(action)

            if step_i % log_interval != 0:
                continue

            # Collect metrics (all per-env averaged over batch)
            env_id = 0  # use first env

            row = {
                'step': step_i,
                'phase': env._ref_motion_phase[env_id].item(),
                'is_crouch': env.q1_phase_masks['crouch'][env_id].item(),
                'is_takeoff': env.q1_phase_masks['takeoff'][env_id].item(),
                'is_flight': env.q1_phase_masks['flight'][env_id].item(),
                'is_landing': env.q1_phase_masks['landing'][env_id].item(),
                'ref_root_z': env.q1_ref_root_z[env_id].item(),
                'actual_root_z': env.q1_actual_root_z[env_id].item(),
                'root_z_error': env.q1_root_z_error[env_id].item(),
                'ref_root_vz': env.q1_ref_root_vz[env_id].item(),
                'actual_root_vz': env.q1_actual_root_vz[env_id].item(),
                'root_vz_error': env.q1_root_vz_error[env_id].item(),
                'ref_yaw': env.q1_ref_yaw[env_id].item(),
                'actual_yaw': env.q1_actual_yaw[env_id].item(),
                'yaw_error_deg': np.rad2deg(env.q1_yaw_error_rad[env_id].item()),
                'ref_yaw_rate': env.q1_ref_yaw_rate[env_id].item(),
                'actual_yaw_rate': env.q1_actual_yaw_rate[env_id].item(),
                'yaw_rate_error': env.q1_yaw_rate_error[env_id].item(),
                'left_contact': env.q1_left_contact[env_id].item(),
                'right_contact': env.q1_right_contact[env_id].item(),
                'both_feet_air': env.q1_both_feet_air[env_id].item(),
                'root_z_reward': env._reward_q1_root_z_tracking()[env_id].item(),
                'root_vz_reward': env._reward_q1_root_vz_tracking()[env_id].item(),
                'yaw_reward': env._reward_q1_root_yaw_tracking()[env_id].item(),
                'yaw_rate_reward': env._reward_q1_root_yaw_rate_tracking()[env_id].item(),
                'flight_contact_reward': env._reward_q1_flight_contact()[env_id].item(),
                'lower_body_joint_reward': env._reward_q1_lower_body_joint_tracking()[env_id].item(),
                'terminate_low_height': env.log_dict.get('q1_terminate_low_height', torch.tensor(0.0)).item(),
                'terminate_gravity': env.log_dict.get('q1_terminate_gravity', torch.tensor(0.0)).item(),
                'terminate_contact': env.log_dict.get('q1_terminate_contact', torch.tensor(0.0)).item(),
                'reset_buf_any': reset_buf.any().item() if isinstance(reset_buf, torch.Tensor) else 0,
                'episode_length': env.episode_length_buf[env_id].item(),
                'action_max_abs': action[env_id].abs().max().item(),
            }

            # Knee/ankle/hip action/torque means
            knee_idx = env.q1_knee_indices
            ankle_idx = env.q1_ankle_indices
            hip_idx = env.q1_hip_indices

            row['knee_action_mean'] = action[env_id, knee_idx].mean().item() if len(knee_idx) > 0 else 0
            row['ankle_action_mean'] = action[env_id, ankle_idx].mean().item() if len(ankle_idx) > 0 else 0
            row['hip_action_mean'] = action[env_id, hip_idx].mean().item() if len(hip_idx) > 0 else 0

            if hasattr(env, 'torques') and env.torques is not None:
                torques = env.torques
                row['knee_torque_mean'] = torques[env_id, knee_idx].mean().item() if len(knee_idx) > 0 else 0
                row['ankle_torque_mean'] = torques[env_id, ankle_idx].mean().item() if len(ankle_idx) > 0 else 0
                row['hip_torque_mean'] = torques[env_id, hip_idx].mean().item() if len(hip_idx) > 0 else 0
            else:
                row['knee_torque_mean'] = 0
                row['ankle_torque_mean'] = 0
                row['hip_torque_mean'] = 0

            writer.writerow(row)

    print(f"Rollout CSV written to {output_path}")


def main():
    args = parse_args()

    # This function requires a running env — best called from within the training loop
    # or from eval_agent.py
    print("Q1 CR7 Eval Debug Script")
    print("=" * 50)
    print("This script is designed to be imported and used from eval_agent.py")
    print("or called within a training context.")
    print()
    print("To run directly, use the full eval command:")
    print()
    print("  cd /root/autodl-tmp/ASAP_official")
    print("  python humanoidverse/eval_agent.py \\")
    print("    +checkpoint=<path> \\")
    print("    +exp=q1_cr7_motion_tracking \\")
    print("    +robot=q1/q1_22dof \\")
    print("    +obs=motion_tracking/q1_cr7_tracking_obs \\")
    print("    +rewards=motion_tracking/reward_q1_cr7_tracking \\")
    print("    robot.motion.motion_file=humanoidverse/data/motions/q1/cr7_motion.pkl \\")
    print("    headless=False")


if __name__ == "__main__":
    main()
