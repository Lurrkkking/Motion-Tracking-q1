#!/usr/bin/env python3
"""
Debug rollout logger for Q1 CR7 motion tracking.

Intended to be called from eval_agent.py or a standalone rollout script.
Logs per-step debug information to CSV for offline analysis.

Usage (after training):
    python scripts/debug_q1_cr7_rollout.py \
        --checkpoint logs/Q1_CR7/q1_cr7_deployable_actor_v1/model_<iter>.pt \
        --output debug_outputs/q1_cr7_rollout_debug.csv \
        --motion-file humanoidverse/data/motions/q1/cr7_motion.pkl \
        --record-interval 5

This script does NOT start training. It requires a trained checkpoint.
"""

import argparse
import csv
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# CSV Column definition — matches the spec exactly
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "step",
    "phase",
    "is_crouch",
    "is_takeoff",
    "is_flight",
    "is_landing",
    "ref_root_z",
    "actual_root_z",
    "ref_root_vz",
    "actual_root_vz",
    "ref_yaw",
    "actual_yaw",
    "yaw_error_deg",
    "ref_yaw_rate",
    "actual_yaw_rate",
    "left_contact",
    "right_contact",
    "both_feet_air",
    "ref_max_foot_z",
    "actual_max_foot_z",
    "foot_max_height_error",
    "ref_left_foot_z",
    "actual_left_foot_z",
    "ref_right_foot_z",
    "actual_right_foot_z",
    "root_z_reward",
    "root_vz_reward",
    "foot_height_reward",
    "yaw_reward",
    "yaw_rate_reward",
    "flight_contact_reward",
    "q1_terminate_low_height",
    "q1_terminate_gravity",
    "q1_terminate_contact",
    "reset_buf",
    "action_max_abs",
    "knee_action",
    "ankle_action",
    "knee_torque",
    "ankle_torque",
]


def record_step(writer, step, env, record_interval=5):
    """
    Record one step of debug info for a single environment.

    Args:
        writer: csv.DictWriter
        step: int, current simulation step
        env: Q1CR7MotionTracking instance (after _post_physics_step)
        record_interval: int, only record every N steps
    """
    if step % record_interval != 0:
        return

    env_id = 0  # record env 0 by default

    phase_masks = env.q1_phase_masks

    row = {
        "step": step,
        "phase": env._ref_motion_phase[env_id, 0].item(),
        "is_crouch": int(phase_masks['crouch'][env_id].item()),
        "is_takeoff": int(phase_masks['takeoff'][env_id].item()),
        "is_flight": int(phase_masks['flight'][env_id].item()),
        "is_landing": int(phase_masks['landing'][env_id].item()),
        "ref_root_z": env.q1_ref_root_z[env_id].item(),
        "actual_root_z": env.q1_actual_root_z[env_id].item(),
        "ref_root_vz": env.q1_ref_root_vz[env_id].item(),
        "actual_root_vz": env.q1_actual_root_vz[env_id].item(),
        "ref_yaw": env.q1_ref_yaw[env_id].item(),
        "actual_yaw": env.q1_actual_yaw[env_id].item(),
        "yaw_error_deg": float(
            torch.rad2deg(env.q1_yaw_error_rad[env_id]).item()
        ) if hasattr(env, 'q1_yaw_error_rad') else 0.0,
        "ref_yaw_rate": env.q1_ref_yaw_rate[env_id].item(),
        "actual_yaw_rate": env.q1_actual_yaw_rate[env_id].item(),
        "left_contact": int(env.q1_left_contact[env_id].item()),
        "right_contact": int(env.q1_right_contact[env_id].item()),
        "both_feet_air": int(env.q1_both_feet_air[env_id].item()),
        # Foot height
        "ref_max_foot_z": env.q1_ref_max_foot_z[env_id].item(),
        "actual_max_foot_z": env.q1_actual_max_foot_z[env_id].item(),
        "foot_max_height_error": env.q1_foot_max_height_error[env_id].item(),
        "ref_left_foot_z": env.q1_ref_foot_z[env_id, 0].item(),
        "actual_left_foot_z": env.q1_actual_foot_z[env_id, 0].item(),
        "ref_right_foot_z": env.q1_ref_foot_z[env_id, 1].item(),
        "actual_right_foot_z": env.q1_actual_foot_z[env_id, 1].item(),
        # Rewards
        "root_z_reward": env.log_dict.get("q1_root_z_error", 0.0),
        "root_vz_reward": env.log_dict.get("q1_root_vz_error", 0.0),
        "foot_height_reward": env.log_dict.get("q1_foot_max_height_error", 0.0),
        "yaw_reward": env.log_dict.get("q1_yaw_error_deg", 0.0),
        "yaw_rate_reward": env.log_dict.get("q1_yaw_rate_error", 0.0),
        "flight_contact_reward": env.log_dict.get("q1_flight_contact_rate", 0.0),
        "q1_terminate_low_height": float(
            env.log_dict.get("q1_terminate_low_height", 0.0)
        ),
        "q1_terminate_gravity": float(
            env.log_dict.get("q1_terminate_gravity", 0.0)
        ),
        "q1_terminate_contact": float(
            env.log_dict.get("q1_terminate_contact", 0.0)
        ),
        "reset_buf": int(env.reset_buf[env_id].item()),
        "action_max_abs": env.actions[env_id].abs().max().item(),
        # Joint group action/torque averages
        "knee_action": float(
            env.actions[env_id][env.q1_knee_indices].abs().mean().item()
        ) if hasattr(env, 'q1_knee_indices') else 0.0,
        "ankle_action": float(
            env.actions[env_id][env.q1_ankle_indices].abs().mean().item()
        ) if hasattr(env, 'q1_ankle_indices') else 0.0,
        "knee_torque": float(
            env.torques[env_id][env.q1_knee_indices].abs().mean().item()
        ) if hasattr(env, 'q1_knee_indices') else 0.0,
        "ankle_torque": float(
            env.torques[env_id][env.q1_ankle_indices].abs().mean().item()
        ) if hasattr(env, 'q1_ankle_indices') else 0.0,
    }

    writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Q1 CR7 Debug Rollout — records per-step CSV"
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained checkpoint (.pt)')
    parser.add_argument('--output', type=str,
                        default='debug_outputs/q1_cr7_rollout_debug.csv',
                        help='Output CSV path')
    parser.add_argument('--motion-file', type=str,
                        default='humanoidverse/data/motions/q1/cr7_motion.pkl',
                        help='Path to Q1 CR7 motion pkl')
    parser.add_argument('--record-interval', type=int, default=5,
                        help='Record every N steps')
    parser.add_argument('--num-steps', type=int, default=1000,
                        help='Maximum rollout steps')
    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[DEBUG_ROLLOUT] Will output to {output_path}")
    print(f"[DEBUG_ROLLOUT] Columns: {CSV_COLUMNS}")
    print(f"[DEBUG_ROLLOUT] Record interval: {args.record_interval} steps")
    print()
    print("NOTE: This script requires a trained policy checkpoint to run.")
    print("Integrate with eval_agent.py or a standalone rollout script.")
    print("The record_step() function can be imported and called from:")
    print("  - humanoidverse/eval_agent.py (add hook after _post_physics_step)")
    print("  - Or a custom rollout script that loads the checkpoint and env.")
    print()
    print("This is a SKELETON — not runnable standalone without checkpoint.")

    # Initialize CSV with headers (even if empty for now)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

    print(f"[DEBUG_ROLLOUT] CSV header written to {output_path}")
    print("[DEBUG_ROLLOUT] Done (no actual rollout — no checkpoint provided or loaded)")


if __name__ == '__main__':
    # Allow import of record_step without requiring torch at import time
    try:
        import torch
    except ImportError:
        pass
    main()
