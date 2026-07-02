"""
Eval script: run a Q1 CR7 checkpoint and report per-phase root_z / root_vz errors.
Diagnose why mean q1_root_vz_error looks small but robot doesn't jump.

Usage:
    cd /root/autodl-tmp/ASAP_official
    python scripts/eval_q1_cr7_per_phase.py \
        --checkpoint logs/Q1_CR7/<run>/model_500.pt \
        --motion humanoidverse/data/motions/q1/q1_cr7_scale045_rootz042.pkl
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--motion", type=str,
                   default="humanoidverse/data/motions/q1/q1_cr7_scale045_rootz042.pkl")
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--num_steps", type=int, default=400)
    p.add_argument("--headless", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    # Use subprocess to call eval_agent with proper Hydra setup
    # But first, let's try a simpler approach: manual env + checkpoint
    import torch
    import numpy as np

    # Load checkpoint
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Use train_agent.py's hydra main to properly set up everything.
    # Simplest approach: call eval_agent.py as a subprocess to get the env,
    # but for quick diagnosis, let's just analyze the log_dict behavior.
    #
    # Instead of fighting hydra config resolution, let's explain the
    # root cause from the training metrics and the reward function design.
    print("=" * 80)
    print("DIAGNOSIS: Why q1_root_vz_error=0.0024 but robot doesn't jump")
    print("=" * 80)
    print("""
The logged 'q1_root_vz_error' is the BATCH MEAN over all 2048 envs
and ALL timesteps within the logging window:

    q1_root_vz_error = mean(ref_root_vz - actual_root_vz)

MOST timesteps are in crouch/stand phases where root_vz ≈ 0.
The takeoff phase (~5% of motion) has ref_root_vz ≈ 1.0+ m/s but
if the robot doesn't jump, actual_root_vz ≈ 0.0, so the error ≈ 1.0.

Example with simplified numbers:
  - 95% of timesteps: error = 0.0  (both near zero)
  -  5% of timesteps: error = 1.0  (ref=1.0, act=0.0)
  - Mean error = 0.95*0.0 + 0.05*1.0 = 0.05

The mean IS misleading. Let's compute per-phase from the reward values.
""")

    # Analyze from reward values instead.
    # The reward logs give us more info:
    # q1_root_vz_tracking reward = exp(-(ref_vz - act_vz)^2 / sigma)
    # If reward is high, error is low. If reward is low, error is high.
    #
    # But we need the actual rollout. Let me just run eval_agent.py with a
    # post-processing hook.

    # For now, manually instantiate with a workaround for the eval resolver
    from omegaconf import OmegaConf
    log_dir = os.path.dirname(args.checkpoint)
    saved_cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))

    # Manually resolve the eval interpolations for obs_dims
    num_bodies = saved_cfg.robot.num_bodies
    nums_extend = saved_cfg.robot.motion.nums_extend_bodies
    dof_obs_size = saved_cfg.robot.dof_obs_size

    # Fix obs_dims: convert list of single-key dicts to proper format
    resolved_obs_dims = {}
    for item in saved_cfg.obs.obs_dims:
        for k, v in item.items():
            if k == 'dif_local_rigid_body_pos' or k == 'local_ref_rigid_body_pos':
                resolved_obs_dims[k] = 3 * num_bodies + 3 * nums_extend
            elif k == 'q1_ref_dof_error':
                resolved_obs_dims[k] = 2 * dof_obs_size
            elif isinstance(v, int):
                resolved_obs_dims[k] = v

    # Patch the obs dims manually
    from humanoidverse.utils.helpers import pre_process_config
    print(f"Resolved obs_dims: {resolved_obs_dims}")
    print("Creating env...")
    from humanoidverse.envs.q1_motion_tracking.q1_cr7_motion_tracking import Q1CR7MotionTracking
    env = Q1CR7MotionTracking(config=saved_cfg.env.config, device=device)

    # Build actor from checkpoint
    print("Building actor...")
    print(f"  dim_obs={env.dim_obs}, dim_actions={env.dim_actions}")
    print(f"  actor_hidden_dims={cfg.algo.config.actor_hidden_dims}")
    from humanoidverse.agents.modules.ppo_modules import Actor
    actor = Actor(
        num_actor_obs=env.dim_obs,
        num_actions=env.dim_actions,
        actor_hidden_dims=cfg.algo.config.actor_hidden_dims,
        fixed_std=False,
        init_noise_std=cfg.algo.config.init_noise_std,
    ).to(device)
    actor.load_state_dict(ckpt["actor_model_state_dict"], strict=False)

    # Reset env
    print("Resetting env...")
    obs_dict = env.reset_all()
    actor_obs = obs_dict["actor_obs"]

    # Run rollout
    print(f"Running {args.num_steps} steps...")
    records = []
    motion_len_s = env.motion_len[0].item() if len(env.motion_len) > 0 else 5.9

    for step_i in range(args.num_steps):
        with torch.no_grad():
            action = actor(actor_obs)

        obs_dict, rew_buf, reset_buf, extras = env.step({"actions": action})
        actor_obs = obs_dict["actor_obs"]

        env_id = 0
        phase_val = env._ref_motion_phase[env_id].item()
        record = {
            "step": step_i,
            "phase": phase_val,
            "motion_time_s": phase_val * motion_len_s,
            "ref_root_z": env.q1_ref_root_z[env_id].item(),
            "actual_root_z": env.q1_actual_root_z[env_id].item(),
            "root_z_error": env.q1_root_z_error[env_id].item(),
            "ref_root_vz": env.q1_ref_root_vz[env_id].item(),
            "actual_root_vz": env.q1_actual_root_vz[env_id].item(),
            "root_vz_error": env.q1_root_vz_error[env_id].item(),
            "ref_yaw_rate": env.q1_ref_yaw_rate[env_id].item(),
            "actual_yaw_rate": env.q1_actual_yaw_rate[env_id].item(),
            "yaw_rate_error": env.q1_yaw_rate_error[env_id].item(),
            "is_crouch": env.q1_phase_masks["crouch"][env_id].item(),
            "is_takeoff": env.q1_phase_masks["takeoff"][env_id].item(),
            "is_flight": env.q1_phase_masks["flight"][env_id].item(),
            "is_landing": env.q1_phase_masks["landing"][env_id].item(),
            "left_contact": env.q1_left_contact[env_id].item(),
            "right_contact": env.q1_right_contact[env_id].item(),
            "both_feet_air": env.q1_both_feet_air[env_id].item(),
            "reset": reset_buf[env_id].item() if isinstance(reset_buf, torch.Tensor) else 0,
        }
        records.append(record)

    # --- Per-phase analysis ---
    print("\n" + "=" * 80)
    print("PER-PHASE ANALYSIS")
    print("=" * 80)

    for phase_name, phase_key in [
        ("CROUCH", "is_crouch"),
        ("TAKEOFF", "is_takeoff"),
        ("FLIGHT", "is_flight"),
        ("LANDING", "is_landing"),
    ]:
        phase_records = [r for r in records if r[phase_key]]
        if not phase_records:
            print(f"\n{phase_name}: NO TIMESTEPS IN THIS PHASE")
            continue

        n = len(phase_records)
        # Root Z
        ref_z = np.mean([r["ref_root_z"] for r in phase_records])
        act_z = np.mean([r["actual_root_z"] for r in phase_records])
        z_rmse = np.sqrt(np.mean([r["root_z_error"] ** 2 for r in phase_records]))
        # Root VZ
        ref_vz = np.mean([r["ref_root_vz"] for r in phase_records])
        act_vz = np.mean([r["actual_root_vz"] for r in phase_records])
        vz_rmse = np.sqrt(np.mean([r["root_vz_error"] ** 2 for r in phase_records]))

        # Max ref/act vz
        max_ref_vz = max(r["ref_root_vz"] for r in phase_records)
        max_act_vz = max(r["actual_root_vz"] for r in phase_records)

        # Contact
        any_contact = np.mean([r["left_contact"] or r["right_contact"] for r in phase_records])
        both_air = np.mean([r["both_feet_air"] for r in phase_records])

        print(f"\n{phase_name}: {n} steps ({n/len(records)*100:.1f}% of episode)")
        print(f"  ref_root_z  : mean={ref_z:.4f}")
        print(f"  actual_root_z: mean={act_z:.4f}")
        print(f"  root_z RMSE : {z_rmse:.4f}")
        print(f"  ref_root_vz : mean={ref_vz:.4f}, max={max_ref_vz:.4f}")
        print(f"  actual_root_vz: mean={act_vz:.4f}, max={max_act_vz:.4f}")
        print(f"  root_vz RMSE: {vz_rmse:.4f}")
        print(f"  any_contact : {any_contact:.3f}, both_feet_air: {both_air:.3f}")

    # --- Overall RMSE vs mean error ---
    print(f"\n{'='*60}")
    print("WHY MEAN ERROR HIDES THE PROBLEM")
    all_vz_error = [r["root_vz_error"] for r in records]
    print(f"  Overall mean root_vz_error    : {np.mean(all_vz_error):.5f}  <-- what log_dict shows")
    print(f"  Overall RMSE root_vz_error    : {np.sqrt(np.mean([e**2 for e in all_vz_error])):.5f}  <-- better metric")
    print(f"  Max abs root_vz_error         : {max(abs(e) for e in all_vz_error):.4f}")

    # Overall root_z
    all_z_error = [r["root_z_error"] for r in records]
    print(f"  Overall mean root_z_error     : {np.mean(all_z_error):.5f}")
    print(f"  Overall RMSE root_z_error     : {np.sqrt(np.mean([e**2 for e in all_z_error])):.5f}")
    print(f"  Max abs root_z_error          : {max(abs(e) for e in all_z_error):.4f}")

    # Phase coverage
    print(f"\n{'='*60}")
    print("PHASE COVERAGE")
    for k in ["is_crouch", "is_takeoff", "is_flight", "is_landing"]:
        ratio = np.mean([r[k] for r in records])
        print(f"  {k}: {ratio:.4f}")

    # Latest step root comparison
    print(f"\n{'='*60}")
    print("LAST 10 STEPS (raw values)")
    for r in records[-10:]:
        print(f"  step={r['step']:3d} phase={r['phase']:.3f} "
              f"ref_z={r['ref_root_z']:.3f} act_z={r['actual_root_z']:.3f} "
              f"ref_vz={r['ref_root_vz']:.3f} act_vz={r['actual_root_vz']:.3f} "
              f"air={r['both_feet_air']}")


if __name__ == "__main__":
    main()
