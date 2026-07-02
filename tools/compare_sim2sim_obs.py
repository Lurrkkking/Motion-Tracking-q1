#!/usr/bin/env python3
"""
Compare first-obs between IsaacGym eval and MuJoCo sim2sim.

Usage:
  python scripts/compare_sim2sim_obs.py \
      --isaacgym-obs debug_outputs/isaacgym_first_obs.npy \
      --mujoco-obs debug_outputs/mujoco_first_obs.npy \
      --component-names base_ang_vel,projected_gravity,dof_pos,dof_vel,actions,ref_motion_phase,q1_root_error,q1_yaw_error,q1_flight_phase,q1_ref_dof_error,history_actor \
      --component-dims 3,3,22,22,22,1,4,4,6,44,292
"""

import argparse
import numpy as np
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Compare sim2sim/IsaacGym first obs")
    p.add_argument("--isaacgym-obs", type=str, default="debug_outputs/isaacgym_first_obs.npy")
    p.add_argument("--mujoco-obs", type=str, default="debug_outputs/mujoco_first_obs.npy")
    p.add_argument("--component-names", type=str,
                   default="base_ang_vel,projected_gravity,dof_pos,dof_vel,actions,"
                           "ref_motion_phase,q1_root_error,q1_yaw_error,q1_flight_phase,"
                           "q1_ref_dof_error,history_actor")
    p.add_argument("--component-dims", type=str,
                   default="3,3,22,22,22,1,4,4,6,44,292")
    p.add_argument("--atol", type=float, default=1e-3)
    return p.parse_args()


def main():
    args = parse_args()

    try:
        ig_obs = np.load(args.isaacgym_obs).flatten()
        mj_obs = np.load(args.mujoco_obs).flatten()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run sim2sim with --dump-first-obs first, and eval_agent with obs dumping.")
        sys.exit(1)

    names = args.component_names.split(",")
    dims = [int(x) for x in args.component_dims.split(",")]

    assert len(names) == len(dims), f"{len(names)} names != {len(dims)} dims"
    assert sum(dims) == len(ig_obs), f"Sum dims {sum(dims)} != ig_obs len {len(ig_obs)}"
    assert sum(dims) == len(mj_obs), f"Sum dims {sum(dims)} != mj_obs len {len(mj_obs)}"

    l2_total = np.sqrt(np.sum((ig_obs - mj_obs) ** 2))
    print(f"Total L2 diff: {l2_total:.6f}")
    print()

    offset = 0
    for name, dim in zip(names, dims):
        ig = ig_obs[offset:offset + dim]
        mj = mj_obs[offset:offset + dim]
        l2 = np.sqrt(np.sum((ig - mj) ** 2))
        max_err = np.max(np.abs(ig - mj))
        print(f"  {name:25s} (dim={dim:3d}): L2={l2:.6f}, max_err={max_err:.6f}")
        if max_err > args.atol:
            print(f"    WARNING: large error! ig[:5]={ig[:5]}, mj[:5]={mj[:5]}")
        offset += dim

    if l2_total < args.atol:
        print("\n✓ Obs match within tolerance")
    else:
        print(f"\n✗ Obs mismatch (L2={l2_total:.6f} > {args.atol})")
        max_err_idx = np.argmax(np.abs(ig_obs - mj_obs))
        print(f"  Max error at index {max_err_idx}: ig={ig_obs[max_err_idx]:.6f}, mj={mj_obs[max_err_idx]:.6f}")


if __name__ == "__main__":
    main()
