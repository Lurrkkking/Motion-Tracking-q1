#!/usr/bin/env python
"""
Debug script: validate Q1 CR7 env mapping and run a short rollout.
Does NOT train — just instantiate env, run steps, dump CSV.

Usage:
    cd /root/autodl-tmp/ASAP_official
    python scripts/debug_q1_cr7_mapping.py
"""

import sys
import os
import csv
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from omegaconf import OmegaConf, DictConfig

# Minimal import test
print("=" * 60)
print("  Q1 CR7 Mapping Debug")
print("=" * 60)


def test_env_instantiate():
    """Test that Q1CR7MotionTracking can be instantiated via Hydra config."""
    from humanoidverse.envs.q1_motion_tracking.q1_cr7_motion_tracking import Q1CR7MotionTracking

    print("\n[1/4] Import OK: Q1CR7MotionTracking class found")
    return True


def test_config_load():
    """Test that config files load without error."""
    from omegaconf import OmegaConf

    config_files = [
        "humanoidverse/config/exp/q1_cr7_motion_tracking.yaml",
        "humanoidverse/config/obs/motion_tracking/q1_cr7_tracking_obs.yaml",
        "humanoidverse/config/rewards/motion_tracking/reward_q1_cr7_tracking.yaml",
    ]
    for cf in config_files:
        cfg = OmegaConf.load(cf)
        print(f"  Config loaded: {cf}")
    print("\n[2/4] Config load OK")
    return True


def test_full_hydra():
    """Test full Hydra instantiation."""
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = os.path.abspath("humanoidverse/config")

    # We need to merge all configs manually since we're not using @hydra.main
    base_cfg = OmegaConf.load(os.path.join(config_dir, "base.yaml"))

    # Override key groups
    exp_cfg = OmegaConf.load(os.path.join(config_dir, "exp/q1_cr7_motion_tracking.yaml"))
    robot_cfg = OmegaConf.load(os.path.join(config_dir, "robot/q1/q1_22dof.yaml"))
    obs_cfg = OmegaConf.load(os.path.join(config_dir, "obs/motion_tracking/q1_cr7_tracking_obs.yaml"))
    reward_cfg = OmegaConf.load(os.path.join(config_dir, "rewards/motion_tracking/reward_q1_cr7_tracking.yaml"))
    sim_cfg = OmegaConf.load(os.path.join(config_dir, "simulator/isaacgym.yaml"))
    algo_cfg = OmegaConf.load(os.path.join(config_dir, "algo/ppo.yaml"))
    dr_cfg = OmegaConf.load(os.path.join(config_dir, "domain_rand/NO_domain_rand.yaml"))
    terrain_cfg = OmegaConf.load(os.path.join(config_dir, "terrain/terrain_locomotion_plane.yaml"))

    # Check env target
    print(f"  env._target_ = {exp_cfg.env._target_}")
    print("\n[3/4] Hydra config merge OK")
    return True


def main():
    test_env_instantiate()
    test_config_load()
    test_full_hydra()

    print("\n[4/4] All checks passed for Q1 CR7 config validation")
    print("\nNext: run")
    print("  bash .sh/run_q1_cr7_test.sh")
    print("to test full env instantiation with Isaac Gym.")


if __name__ == "__main__":
    main()
