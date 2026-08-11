#!/usr/bin/env python3
"""Show Q1 training default joint pose in MuJoCo.

This script is offline-only: it does not import ROS2, subscribe to robot topics,
or publish joint commands. It only loads the configured default_dof_pos and
opens a MuJoCo viewer.
"""

import argparse
import importlib.util
import os
import time
from pathlib import Path

import numpy as np
import yaml

try:
    import mujoco
    import mujoco.viewer
except ImportError as exc:
    raise RuntimeError("mujoco is required. Install with: python3 -m pip install mujoco") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return str(REPO_ROOT / path)


def load_builder():
    builder_path = Path(__file__).resolve().with_name("visualize_q1_live_mujoco_ros2.py")
    spec = importlib.util.spec_from_file_location("q1_live_mujoco_builder", builder_path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder.build_model_xml


def read_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    joint_names = list(cfg.get("joint_names", cfg.get("dof_names", [])))
    default_dof_pos = np.asarray(cfg["default_dof_pos"], dtype=np.float64)
    if len(joint_names) != len(default_dof_pos):
        raise ValueError(
            f"joint_names length {len(joint_names)} != default_dof_pos length {len(default_dof_pos)}"
        )
    return joint_names, default_dof_pos


def apply_default_pose(model, data, joint_names, default_dof_pos, root_height):
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    try:
        root_qposadr = int(model.joint("root").qposadr[0])
        data.qpos[root_qposadr:root_qposadr + 3] = [0.0, 0.0, root_height]
        data.qpos[root_qposadr + 3:root_qposadr + 7] = [1.0, 0.0, 0.0, 0.0]
    except KeyError:
        pass
    for idx, name in enumerate(joint_names):
        try:
            qposadr = int(model.joint(name).qposadr[0])
        except KeyError as exc:
            raise RuntimeError(f"MuJoCo model is missing joint {name}") from exc
        data.qpos[qposadr] = float(default_dof_pos[idx])
    mujoco.mj_forward(model, data)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Q1 default_dof_pos in MuJoCo")
    parser.add_argument("--config", default="config/q1_sim2real_base.yaml")
    parser.add_argument("--urdf", default="/home/hoho/q1_wordandexample/q1_robot/q1_22dof_box.urdf")
    parser.add_argument("--root-height", type=float, default=0.42)
    parser.add_argument("--visual-geometry", choices=("auto", "mesh", "simplified"), default="auto")
    parser.add_argument("--viewer-hz", type=float, default=60.0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.config = resolve_path(args.config)
    args.urdf = resolve_path(args.urdf)
    joint_names, default_dof_pos = read_config(args.config)
    build_model_xml = load_builder()
    mjcf_xml, actual_geometry = build_model_xml(
        args.urdf,
        args.root_height,
        fixed_base=False,
        visual_geometry=args.visual_geometry,
    )
    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    data = mujoco.MjData(model)
    apply_default_pose(model, data, joint_names, default_dof_pos, args.root_height)

    print("=" * 72)
    print("Q1 Default Pose MuJoCo Viewer")
    print("=" * 72)
    print(f"config={args.config}")
    print(f"urdf={args.urdf}")
    print(f"visual_geometry={args.visual_geometry} actual_geometry={actual_geometry}")
    print("default_dof_pos:")
    for name, value in zip(joint_names, default_dof_pos):
        print(f"  {name}: {value:.6f}")
    print()

    sleep_s = 1.0 / max(float(args.viewer_hz), 1.0)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 140
        viewer.cam.elevation = -15
        while viewer.is_running():
            apply_default_pose(model, data, joint_names, default_dof_pos, args.root_height)
            viewer.sync()
            time.sleep(sleep_s)


if __name__ == "__main__":
    main()
