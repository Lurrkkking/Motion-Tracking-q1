"""
Diagnostic script for Q1 CR7 jump + 180° turn tracking.
Loads model_4800.pt and runs eval with detailed per-step debug prints.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial.transform import Rotation as sRot
from omegaconf import OmegaConf
import joblib
import pickle


def analyze_checkpoint(ckpt_path, config_path, num_steps=300):
    """Load checkpoint and run eval with debug prints."""

    # Load config
    conf = OmegaConf.load(config_path)
    conf.num_envs = 1
    conf.headless = False  # set to True if no display

    # Skip torch-dependent checkpoint loading in analysis mode
    print(f"Checkpoint path: {ckpt_path}")
    print(f"Model analysis: config-only mode (no GPU/torch available)")

    # We need to instantiate the env and policy.
    # Since the env creation is complex with hydra, let's use a simpler approach:
    # Directly load the model and run inference through the env

    # For now, let's just analyze what we can from the checkpoint and config
    print(f"\n{'='*80}")
    print("CONFIG ANALYSIS: Reward Structure")
    print(f"{'='*80}")

    reward_names = list(conf.rewards.reward_scales.keys())
    for name in reward_names:
        scale = conf.rewards.reward_scales[name]
        print(f"  {name}: scale={scale}")

    print(f"\n{'='*80}")
    print("CONFIG ANALYSIS: Observation Space")
    print(f"{'='*80}")
    actor_obs = conf.obs.obs_dict.actor_obs
    critic_obs = conf.obs.obs_dict.critic_obs
    print(f"  Actor obs: {actor_obs}")
    print(f"  Critic obs: {critic_obs}")

    print(f"\n{'='*80}")
    print("CONFIG ANALYSIS: Key Bodies & Tracking")
    print(f"{'='*80}")
    print(f"  motion_tracking_link: {conf.robot.motion.motion_tracking_link}")
    print(f"  lower_body_link: {conf.robot.motion.lower_body_link}")
    print(f"  upper_body_link: {conf.robot.motion.upper_body_link}")

    # Check if there are explicit jump/reward terms
    jump_related = ['root', 'jump', 'flight', 'height', 'yaw', 'heading', 'air', 'vel']
    print(f"\n{'='*80}")
    print("CONFIG ANALYSIS: Jump-related Reward Terms")
    print(f"{'='*80}")
    for keyword in jump_related:
        found = [n for n in reward_names if keyword in n.lower()]
        if found:
            for n in found:
                print(f"  [{keyword}] FOUND: {n} = {conf.rewards.reward_scales[n]}")
        else:
            print(f"  [{keyword}] MISSING: no reward term contains '{keyword}'")

    # Check termination config
    print(f"\n{'='*80}")
    print("CONFIG ANALYSIS: Termination")
    print(f"{'='*80}")
    env_cfg = conf.env.config if hasattr(conf.env, 'config') else conf.env
    term_cfg = env_cfg.termination
    term_scales = env_cfg.termination_scales
    for k, v in term_cfg.items():
        print(f"  {k}: {v}")
    print(f"  termination_min_base_height: {term_scales.termination_min_base_height}")
    print(f"  termination_gravity_x: {term_scales.termination_gravity_x}")
    print(f"  termination_gravity_y: {term_scales.termination_gravity_y}")

    # Check control config
    print(f"\n{'='*80}")
    print("CONFIG ANALYSIS: Control")
    print(f"{'='*80}")
    ctrl = conf.robot.control
    print(f"  control_type: {ctrl.control_type}")
    print(f"  action_scale: {ctrl.action_scale}")
    print(f"  clamp_actions: {ctrl.clamp_actions}")
    print(f"  stiffness: {dict(ctrl.stiffness)}")
    print(f"  damping: {dict(ctrl.damping)}")

    # Check torque limits
    dof_names = conf.robot.dof_names
    effort_limits = conf.robot.dof_effort_limit_list
    print(f"\n{'='*80}")
    print("CONFIG ANALYSIS: Torque Limits")
    print(f"{'='*80}")
    for name, limit in zip(dof_names, effort_limits):
        marker = " <-- KNEE" if "knee" in name else ""
        marker = " <-- ANKLE" if "ankle" in name else marker
        marker = " <-- HIP" if "hip" in name else marker
        print(f"  {name}: {limit:.1f} Nm{marker}")

    # Motion analysis from pkl
    print(f"\n{'='*80}")
    print("REFERENCE MOTION ANALYSIS (from PKL)")
    print(f"{'='*80}")
    motion_data = joblib.load(conf.robot.motion.motion_file)
    key = list(motion_data.keys())[0]
    md = motion_data[key]
    pose_aa = md['pose_aa']
    root_trans = md['root_trans_offset']
    fps = md['fps']

    rz = root_trans[:, 2]
    rz_min, rz_max = rz.min(), rz.max()
    rz_min_idx, rz_max_idx = rz.argmin(), rz.argmax()

    # Get yaw from pelvis rotation
    pelvis_aa = pose_aa[:, 0, :]
    pelvis_rot = sRot.from_rotvec(pelvis_aa)
    pelvis_euler = pelvis_rot.as_euler('XYZ', degrees=True)
    yaw_raw = pelvis_euler[:, 2]
    yaw_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(yaw_raw)))

    dt = 1.0 / fps
    rvz = np.gradient(rz, dt)

    # Find takeoff/landing
    takeoff_frame = None
    for i in range(rz_min_idx, rz_max_idx):
        if rz[i] > rz_min + 0.05:
            takeoff_frame = i
            break

    landing_frame = None
    for i in range(rz_max_idx, len(rz)):
        if rz[i] < rz_min + 0.03:
            landing_frame = i
            break

    print(f"[REF_CR7] FPS: {fps}, frames: {len(rz)}, duration: {len(rz)/fps:.2f}s")
    print(f"[REF_CR7] squat_frame={rz_min_idx}  (t={rz_min_idx/fps:.2f}s, rz={rz_min:.4f})")
    print(f"[REF_CR7] takeoff_frame={takeoff_frame} (t={takeoff_frame/fps:.2f}s, rz={rz[takeoff_frame]:.4f})")
    print(f"[REF_CR7] apex_frame={rz_max_idx}     (t={rz_max_idx/fps:.2f}s, rz={rz_max:.4f})")
    print(f"[REF_CR7] landing_frame={landing_frame} (t={landing_frame/fps:.2f}s, rz={rz[landing_frame]:.4f})")
    print(f"[REF_CR7] root_z min/max/delta = {rz_min:.4f}/{rz_max:.4f}/{rz_max-rz_min:.4f}m = {(rz_max-rz_min)*100:.1f}cm")
    print(f"[REF_CR7] root_yaw start/end/delta = {yaw_unwrapped[0]:.1f}/{yaw_unwrapped[-1]:.1f}/{yaw_unwrapped[-1]-yaw_unwrapped[0]:.1f} deg")
    print(f"[REF_CR7] yaw_during_flight ({takeoff_frame}-{landing_frame}) = {yaw_unwrapped[landing_frame]-yaw_unwrapped[takeoff_frame]:.1f} deg")
    print(f"[REF_CR7] max_root_vz = {rvz.max():.3f} m/s at frame {rvz.argmax()}")
    print(f"[REF_CR7] flight_frames = {takeoff_frame}-{landing_frame} ({landing_frame-takeoff_frame} frames = {(landing_frame-takeoff_frame)/fps:.2f}s)")

    # Check if root_z / root_yaw has explicit reward
    print(f"\n{'='*80}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'='*80}")

    has_root_z_reward = any('root' in n.lower() and ('z' in n.lower() or 'height' in n.lower() or 'pos' in n.lower()) for n in reward_names)
    has_root_yaw_reward = any(('yaw' in n.lower() or 'heading' in n.lower()) for n in reward_names)
    has_flight_reward = any(('flight' in n.lower() or 'air' in n.lower() or 'jump' in n.lower()) for n in reward_names)

    # Body position tracking analysis
    print(f"\n[DIAG] body_pos tracking is in WORLD frame for REWARD (dif_global_body_pos = ref_pos - actual_pos)")
    print(f"[DIAG] BUT body_pos observation is HEADING-INVARIANT: dif_local_body_pos = heading_inv * global_diff")
    print(f"[DIAG] Policy cannot directly observe global root height or global root yaw from body_pos features")
    print(f"[DIAG] base_lin_vel is in LOCAL frame (rotated by base_quat inverse)")
    print(f"[DIAG] projected_gravity is in LOCAL frame - gives orientation info, NOT height info")
    print(f"[DIAG] root_z explicit reward: {'EXISTS' if has_root_z_reward else 'MISSING'}")
    print(f"[DIAG] root_yaw/heading explicit reward: {'EXISTS' if has_root_yaw_reward else 'MISSING'}")
    print(f"[DIAG] flight/jump/air explicit reward: {'EXISTS' if has_flight_reward else 'MISSING'}")

    # Rotation analysis
    print(f"\n[DIAG] Rotation tracking: quat diff is GLOBAL (includes yaw) for REWARD")
    print(f"[DIAG] But rotation reward weight=0.8, sigma=3.0 — 180° yaw error gives exp(-pi²/3.0)=0.037")
    print(f"[DIAG] heading/yaw is NOT removed in reward rotation, but reward is WEAK for large yaw")

    # Penalty analysis
    penalty_names = conf.rewards.reward_penalty_reward_names
    print(f"\n[DIAG] Penalty terms: {penalty_names}")
    print(f"[DIAG] action_rate penalty scale: {conf.rewards.reward_scales.penalty_action_rate}")
    print(f"[DIAG] torque penalty scale: {conf.rewards.reward_scales.penalty_torques}")
    print(f"[DIAG] termination penalty: {conf.rewards.reward_scales.termination}")

    # Check if terminate_by_low_height is enabled
    min_h = term_scales.termination_min_base_height
    low_h_terminate = term_cfg.terminate_by_low_height
    print(f"[DIAG] terminate_by_low_height: {low_h_terminate} (threshold={min_h}m)")
    print(f"[DIAG] NOTE: min_base_height={min_h}m — if robot squats below this, it terminates")
    print(f"[DIAG] terminate_by_gravity: {term_cfg.terminate_by_gravity} (x={term_scales.termination_gravity_x}, y={term_scales.termination_gravity_y})")

    # Check randomize_pd_gain
    pd_rand = conf.domain_rand.randomize_pd_gain
    print(f"[DIAG] randomize_pd_gain: {pd_rand}")

    # Key concern areas
    print(f"\n{'='*80}")
    print("KEY CONCERNS")
    print(f"{'='*80}")

    concerns = []

    if not has_root_z_reward:
        concerns.append(
            "1. NO EXPLICIT ROOT Z REWARD: The body_pos reward includes pelvis z, but it's "
            "averaged over 23+ bodies. A 21cm pelvis z error is diluted by well-tracked lower limbs. "
            "The policy may not get enough gradient to learn jumping."
        )

    if not has_root_yaw_reward:
        concerns.append(
            "2. NO EXPLICIT YAW REWARD: Body rotation reward includes yaw, but weight=0.8 and "
            "sigma=3.0 means large yaw errors only weakly penalized. Policy can achieve decent "
            "rotation reward without the full 180° turn."
        )

    if not has_flight_reward:
        concerns.append(
            "3. NO FLIGHT PHASE REWARD: Nothing explicitly rewards feet leaving the ground during "
            "the flight phase. The policy can keep feet on ground, use small steps to rotate, "
            "and still get decent body_pos reward (since most body positions match well when "
            "the local pose is correct)."
        )

    concerns.append(
        "4. HEADING-INVARIANT OBS: Policy observations remove heading information. The policy "
        "cannot 'see' that a 180° yaw turn is needed — it only sees local body position errors. "
        "This makes it harder to plan the angular momentum needed."
    )

    concerns.append(
        "5. PENALTY PRESSURE: action_rate penalty (-0.4) and torque penalty (-1e-6) penalize "
        "the explosive movements needed for jumping. Without strong positive reward for jumping, "
        "the policy is incentivized to be conservative."
    )

    concerns.append(
        "6. PHYSICS CONSTRAINTS: PD stiffness (knee=30, ankle=20) and action_scale=0.25 may "
        "limit the explosive force needed. Torque limits (knee=36Nm, ankle=22Nm) may also "
        "constrain jump impulse."
    )

    for c in concerns:
        print(c)

    # Recommendation table
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS (in priority order)")
    print(f"{'='*80}")

    print("""
P1. ADD ROOT VERTICAL VELOCITY TRACKING REWARD:
    Add reward term that tracks ref root_vz. Scale ~2.0-5.0.
    This directly rewards jumping up and penalizes staying on ground.

P2. ADD ROOT YAW TRACKING REWARD:
    Add reward term that tracks ref root_yaw or root_ang_vel yaw.
    Scale ~1.0-2.0. This directly rewards the 180° turn.

P3. ADD FLIGHT PHASE CONTACT REWARD:
    During reference flight frames (where ref feet are off ground),
    penalize any foot contact. Scale ~-2.0 to -5.0.
    This forces the policy to actually leave the ground.

P4. PHASE-DEPENDENT PENALTY SCALING:
    During takeoff/flight phase (ref frames 57-102), reduce action_rate
    and torque penalties by 50-70%. This allows explosive movements.

P5. INCREASE PD GAINS / ACTION SCALE for jump motion:
    Consider increasing action_scale from 0.25 to 0.5-1.0 for knee/ankle,
    or increasing PD stiffness for knee from 30 to 50-80.

P6. CONSIDER ADDING ROOT HEIGHT TO OBSERVATION:
    Add global root_z (or pelvis height above ground) to the observation
    so the policy can see its own height. This helps with jump planning.
""")

    return conf


if __name__ == "__main__":
    ckpt = "/root/autodl-tmp/ASAP_official/logs/TEST_Q1/20260606_121606-MotionTracking_Q1_CR7_045_pkl-motion_tracking-q1_22dof_box/model_4800.pt"
    cfg = "/root/autodl-tmp/ASAP_official/logs/TEST_Q1/20260606_121606-MotionTracking_Q1_CR7_045_pkl-motion_tracking-q1_22dof_box/config.yaml"
    analyze_checkpoint(ckpt, cfg)
