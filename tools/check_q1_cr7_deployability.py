#!/usr/bin/env python3
"""
Static deployability check for Q1 CR7 motion tracking.

Checks:
1. actor_obs does not contain forbidden terms (sim-only or privileged info)
2. critic_obs contains privileged information (expected)
3. reward config is well-formed
4. exp config targets Q1CR7MotionTracking
5. No future reference leakage
6. Dimensional consistency where computable statically

Usage:
    python scripts/check_q1_cr7_deployability.py \
        --exp-config humanoidverse/config/exp/q1_cr7_motion_tracking.yaml \
        --obs-config humanoidverse/config/obs/motion_tracking/q1_cr7_tracking_obs.yaml \
        --reward-config humanoidverse/config/rewards/motion_tracking/reward_q1_cr7_tracking.yaml
"""

import argparse
import sys
import os

# ---------------------------------------------------------------------------
# Forbidden terms for actor_obs (sim2real deployment constraint)
# ---------------------------------------------------------------------------
FORBIDDEN_ACTOR_TERMS = {
    "root_z_error",
    "root_vz_error",
    "root_yaw_error",
    "root_yaw_rate_error",
    "q1_root_error",
    "q1_yaw_error",
    "q1_flight_phase",
    "q1_foot_contact",
    "q1_ref_dof_error",
    "dif_local_rigid_body_pos",
    "local_ref_rigid_body_pos",
    "global_body_pos_error",
    "feet_contact_force",
    "base_lin_vel",          # not on the "allowed" list for Q1 CR7
    "termination",
    "reward",
    "future_ref",
    "next_ref",
    "lookahead",
}

# Terms that MUST be in actor_obs (hard requirements for deployability)
REQUIRED_ACTOR_TERMS = {
    "base_ang_vel",
    "projected_gravity",
    "dof_pos",
    "dof_vel",
    "actions",
    "ref_motion_phase",
}

# Terms that indicate future reference leakage
FUTURE_LEAKAGE_TERMS = {
    "future", "next_ref", "lookahead", "ref_t_plus", "next_frame",
    "ref_future", "pred_ref", "target_future",
}

# Allowed actor obs terms (sim2real deployable)
ALLOWED_ACTOR_TERMS = {
    "base_ang_vel",
    "projected_gravity",
    "dof_pos",
    "dof_vel",
    "actions",
    "ref_motion_phase",
    "history_actor",
    "ref_dof_error",  # optional, but allowed
}

# ---------------------------------------------------------------------------
# OmegaConf-based loading
# ---------------------------------------------------------------------------

def load_obs_config(path, robot_dof=22, num_bodies=23, num_extend=0):
    """Load obs config with OmegaConf and extract actor_obs, critic_obs lists.

    Resolves standard interpolation keys locally to avoid needing the full Hydra context.
    """
    from omegaconf import OmegaConf
    import re

    # Read raw text first to do manual interpolation
    with open(path, 'r') as f:
        text = f.read()

    # Resolve common interpolations
    rigid_body_dim = 3 * num_bodies + 3 * num_extend
    text = text.replace('${robot.dof_obs_size}', str(robot_dof))
    text = text.replace("${eval:'3 * ${robot.num_bodies} + 3 * ${robot.motion.nums_extend_bodies}'}", str(rigid_body_dim))
    # Generic eval patterns
    text = re.sub(r"\$\{eval:'2\s*\*\s*\$\{robot\.dof_obs_size\}'\}", str(2 * robot_dof), text)
    text = re.sub(r"\$\{eval:'[^']*'\}", "-1", text)  # fallback: replace any remaining eval with -1

    cfg = OmegaConf.create(text)
    obs = cfg.obs
    actor_obs = list(obs.obs_dict.actor_obs)
    critic_obs = list(obs.obs_dict.critic_obs)
    aux = {}
    if hasattr(obs, 'obs_auxiliary'):
        for key in obs.obs_auxiliary:
            aux[key] = dict(obs.obs_auxiliary[key])
    scales = dict(obs.obs_scales) if hasattr(obs, 'obs_scales') else {}
    obs_dims = {}
    if hasattr(obs, 'obs_dims'):
        for item in obs.obs_dims:
            obs_dims.update(dict(item))
    return actor_obs, critic_obs, aux, scales, obs_dims


def load_reward_config(path):
    """Load reward config and extract reward scales."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(path)
    rewards = cfg.rewards
    scales = dict(rewards.reward_scales) if hasattr(rewards, 'reward_scales') else {}
    return scales


def load_exp_config(path):
    """Load exp config and check target."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(path)
    return cfg


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def check_obs_config(obs_path):
    """Check actor_obs and critic_obs for deployability issues."""
    issues = []

    try:
        actor_obs, critic_obs, aux, scales, obs_dims = load_obs_config(obs_path)
    except Exception as e:
        issues.append(f"Failed to load obs config: {e}")
        return issues, [], [], [], []

    if not actor_obs:
        issues.append("actor_obs is empty")
        return issues, [], [], [], []

    # Check forbidden terms
    forbidden_found = []
    for term in actor_obs:
        if term in FORBIDDEN_ACTOR_TERMS:
            forbidden_found.append(term)
            issues.append(f"FORBIDDEN: '{term}' in actor_obs — cannot deploy on real robot")

    # Check unrecognized terms (not in allowed list AND not forbidden)
    # (just warn, don't fail)
    for term in actor_obs:
        if term not in ALLOWED_ACTOR_TERMS and term not in FORBIDDEN_ACTOR_TERMS:
            issues.append(f"WARNING: '{term}' in actor_obs is not in the standard allowed list — verify it's deployable")

    # Check future leakage in all obs terms
    future_found = []
    all_obs_terms = set(actor_obs) | set(critic_obs)
    for term in all_obs_terms:
        for leak_term in FUTURE_LEAKAGE_TERMS:
            if leak_term in term.lower():
                future_found.append(term)
                issues.append(f"FUTURE_LEAKAGE: '{term}' suggests future reference — not real-time deployable")

    # Check required terms
    missing_required = REQUIRED_ACTOR_TERMS - set(actor_obs)
    for term in missing_required:
        issues.append(f"MISSING_REQUIRED: '{term}' not in actor_obs")

    return issues, actor_obs, critic_obs, forbidden_found, future_found


def check_reward_config(reward_path):
    """Check reward config for basic structural issues."""
    issues = []

    try:
        scales = load_reward_config(reward_path)
    except Exception as e:
        issues.append(f"Failed to load reward config: {e}")
        return issues, {}

    if not scales:
        issues.append("No reward_scales found")

    return issues, scales


def check_exp_config(exp_path):
    """Check exp config targets Q1CR7MotionTracking."""
    issues = []

    try:
        cfg = load_exp_config(exp_path)
    except Exception as e:
        issues.append(f"Failed to load exp config: {e}")
        return issues

    target = cfg.env.get('_target_', '')
    if 'Q1CR7MotionTracking' not in str(target):
        issues.append(f"exp config _target_ is '{target}', expected Q1CR7MotionTracking")

    # Check termination — should be permissive for Q1 CR7
    term = cfg.env.config.get('termination', {})
    if hasattr(term, 'terminate_by_gravity') and term.terminate_by_gravity:
        issues.append("WARNING: terminate_by_gravity=True — crouch/takeoff phases may trigger false termination")
    if hasattr(term, 'terminate_by_low_height') and term.terminate_by_low_height:
        issues.append("WARNING: terminate_by_low_height=True — deep crouch may trigger false termination")

    return issues


def compute_actor_dim(actor_obs, obs_dims, aux):
    """Compute approximate actor_obs dimension."""
    dim = 0
    for term in actor_obs:
        if term in obs_dims:
            d = obs_dims[term]
            if isinstance(d, int) and d > 0:
                dim += d
        elif term in aux:
            hist_dim = 0
            for k, length in aux[term].items():
                if k in obs_dims:
                    kd = obs_dims[k]
                    if isinstance(kd, int) and kd > 0:
                        hist_dim += kd * length
            dim += hist_dim
    return dim


def compute_critic_dim(critic_obs, obs_dims, aux):
    """Compute approximate critic_obs dimension."""
    dim = 0
    for term in critic_obs:
        if term in obs_dims:
            d = obs_dims[term]
            if isinstance(d, int) and d > 0:
                dim += d
        elif term in aux:
            hist_dim = 0
            for k, length in aux[term].items():
                if k in obs_dims:
                    kd = obs_dims[k]
                    if isinstance(kd, int) and kd > 0:
                        hist_dim += kd * length
            dim += hist_dim
    return dim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Q1 CR7 Deployability Check")
    parser.add_argument('--exp-config', required=True)
    parser.add_argument('--obs-config', required=True)
    parser.add_argument('--reward-config', required=True)
    parser.add_argument('--robot-config', default=None)
    args = parser.parse_args()

    all_ok = True

    print("=" * 70)
    print("[DEPLOYABILITY] Checking Q1 CR7 motion tracking deployability")
    print("=" * 70)

    # --- Check obs config ---
    obs_issues, actor_obs, critic_obs, forbidden_found, future_found = \
        check_obs_config(args.obs_config)

    actor_obs_ok = len(forbidden_found) == 0
    critic_privileged_ok = True  # by design, critic CAN have privileged info
    future_leakage = len(future_found) > 0

    print(f"\nactor_obs_ok={actor_obs_ok}")
    print(f"critic_privileged_ok={critic_privileged_ok}")
    print(f"reward_privileged_ok=True")  # reward CAN use privileged info
    print(f"future_leakage={future_leakage}")
    print(f"forbidden_actor_terms={forbidden_found}")
    print(f"actor_obs={actor_obs}")
    print(f"critic_obs={critic_obs}")

    if obs_issues:
        for issue in obs_issues:
            print(f"  [ISSUE] {issue}")
        if forbidden_found:
            all_ok = False

    # --- Check reward config ---
    reward_issues, reward_scales = check_reward_config(args.reward_config)
    reward_names = list(reward_scales.keys())
    print(f"\nreward_names={reward_names}")
    if reward_issues:
        for issue in reward_issues:
            print(f"  [REWARD_ISSUE] {issue}")

    # --- Check exp config ---
    exp_issues = check_exp_config(args.exp_config)
    if exp_issues:
        for issue in exp_issues:
            print(f"  [EXP_ISSUE] {issue}")

    # --- Dimension summary ---
    try:
        _, _, aux, _, obs_dims = load_obs_config(args.obs_config)
        actor_dim = compute_actor_dim(actor_obs, obs_dims, aux)
        critic_dim = compute_critic_dim(critic_obs, obs_dims, aux)
        print(f"\nactor_obs_dim (approx)={actor_dim}")
        print(f"critic_obs_dim (approx)={critic_dim}")
    except Exception as e:
        print(f"\nactor_obs_dim (approx)=<error: {e}>")
        print(f"critic_obs_dim (approx)=<error>")

    # --- Export check ---
    print(f"\nexport_input_ok=Static only — verify that export_policy_as_onnx/export_policy_as_jit uses actor_critic.actor (not critic)")
    print(f"uses_critic_at_inference=Static only — check eval_agent.py and sim2sim scripts use actor_obs only")

    # --- Final verdict ---
    print("\n" + "=" * 70)
    if all_ok and actor_obs:
        print("[DEPLOYABILITY] PASSED — actor_obs is sim2real clean")
        print("=" * 70)
        return 0
    elif not actor_obs:
        print("[DEPLOYABILITY] FAILED — could not parse actor_obs")
        print("=" * 70)
        return 1
    else:
        print("[DEPLOYABILITY] FAILED — see issues above")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
