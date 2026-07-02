#!/usr/bin/env python3
"""
Compare actor_obs components between IsaacGym eval and MuJoCo sim2sim.

Usage:
  python scripts/compare_actor_obs_npys.py \
      --isaacgym-npz debug_outputs/isaacgym_obs_components_step0.npz \
      --mujoco-npz sim2sim_outputs/q1_cr7/mujoco_obs_components_step0.npz
"""

import argparse
import numpy as np
import sys


COMPONENT_KEYS = [
    "actions", "base_ang_vel", "dof_pos", "dof_vel",
    "history_actor", "projected_gravity", "ref_motion_phase",
]


def load_components(npz_path):
    data = np.load(npz_path)
    comp = {}
    for key in COMPONENT_KEYS:
        if key in data:
            comp[key] = data[key]
        else:
            print(f"[WARN] {npz_path}: missing component '{key}'")
            comp[key] = None
    return comp


def compare_components(comp_a, comp_b, label_a, label_b, top_k=10):
    print(f"{'='*80}")
    print(f"Component comparison: {label_a}  vs  {label_b}")
    print(f"{'='*80}")

    for key in COMPONENT_KEYS:
        a = comp_a.get(key)
        b = comp_b.get(key)

        if a is None or b is None:
            status = "MISSING" if a is None and b is None else "ONLY_ONE"
            print(f"\n[{key}] {status}")
            continue

        if a.shape != b.shape:
            print(f"\n[{key}] SHAPE_MISMATCH: {a.shape} vs {b.shape}")
            continue

        diff = a.ravel() - b.ravel()
        abs_diff = np.abs(diff)
        l2 = np.sqrt(np.sum(diff ** 2))
        mean_abs = np.mean(abs_diff)
        max_abs = np.max(abs_diff)

        # Top-k diff indices
        top_idx = np.argsort(abs_diff)[::-1][:top_k]

        print(f"\n[{key}] shape={a.shape} shape_match=True")
        print(f"  L2_diff={l2:.6f}  mean_abs_diff={mean_abs:.6f}  max_abs_diff={max_abs:.6f}")
        print(f"  a:  mean={a.mean():+.6f} std={a.std():.6f} "
              f"min={a.min():+.6f} max={a.max():+.6f}")
        print(f"  b:  mean={b.mean():+.6f} std={b.std():.6f} "
              f"min={b.min():+.6f} max={b.max():+.6f}")

        if mean_abs > 0.01 or max_abs > 0.1:
            print(f"  ** SIGNIFICANT DIFFERENCE **")
            print(f"  top-{top_k} diff indices (flattened):")
            for rank, idx in enumerate(top_idx):
                # Map flat index to multi-dim if needed
                if a.ndim <= 1:
                    coord = f"[{idx}]"
                elif a.ndim == 2:
                    row, col = divmod(idx, a.shape[1])
                    coord = f"[{row},{col}]"
                else:
                    coord = f"[flat={idx}]"
                print(f"    #{rank}: {coord}  a={a.ravel()[idx]:+.6f}  b={b.ravel()[idx]:+.6f}  "
                      f"diff={diff.ravel()[idx]:+.6f}")
        else:
            print(f"  ✓ values match closely")

    # Summary
    print(f"\n{'='*80}")
    big_diffs = []
    for key in COMPONENT_KEYS:
        a = comp_a.get(key)
        b = comp_b.get(key)
        if a is not None and b is not None and a.shape == b.shape:
            d = np.abs(a.ravel() - b.ravel())
            if np.mean(d) > 0.01 or np.max(d) > 0.1:
                big_diffs.append((key, np.mean(d), np.max(d)))
    if big_diffs:
        print("Components with significant differences:")
        for key, mean_d, max_d in big_diffs:
            print(f"  {key:<20s}  mean_abs_diff={mean_d:.6f}  max_abs_diff={max_d:.6f}")
    else:
        print("All components match closely. ✓")
    print(f"{'='*80}")


def main():
    p = argparse.ArgumentParser(
        description="Compare actor_obs components between IsaacGym and MuJoCo")
    p.add_argument("--isaacgym-npz", type=str, default=None,
                   help="Path to IsaacGym obs components .npz")
    p.add_argument("--mujoco-npz", type=str, required=True,
                   help="Path to MuJoCo obs components .npz")
    args = p.parse_args()

    comp_mujoco = load_components(args.mujoco_npz)

    if args.isaacgym_npz:
        comp_isaacgym = load_components(args.isaacgym_npz)
        compare_components(comp_isaacgym, comp_mujoco,
                           "isaacgym", "mujoco")
    else:
        print("[INFO] No IsaacGym dump provided — printing MuJoCo components only.")
        print("Run IsaacGym eval with obs dump, then:")
        print(f"  python {__file__} --isaacgym-npz <path> --mujoco-npz {args.mujoco_npz}")
        print()
        for key in COMPONENT_KEYS:
            c = comp_mujoco.get(key)
            if c is None:
                print(f"[{key}] MISSING")
            else:
                print(f"[{key}] shape={c.shape} mean={c.mean():+.6f} std={c.std():.6f} "
                      f"min={c.min():+.6f} max={c.max():+.6f}")


if __name__ == "__main__":
    main()
