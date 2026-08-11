#!/usr/bin/env python3
"""MuJoCo smoke test for the q1_asap_2real ROS2 deployment policy path.

This script intentionally reuses the observation and action post-processing
functions from sim2real_q1_motion_tracking_ros2.py, so the dry-run matches the
real deployment path as closely as possible without ROS2.
"""

import argparse
import csv
import importlib.util
import os
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml


PKG_ROOT = Path(__file__).resolve().parents[1]


def load_deploy_module():
    path = PKG_ROOT / "scripts" / "sim2real_q1_motion_tracking_ros2.py"
    spec = importlib.util.spec_from_file_location("q1_deploy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PKG_ROOT / path


def load_motion(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        key = sorted(data.keys())[0]
        motion = data[key]
    else:
        key = "motion"
        motion = data
    pose_aa = np.asarray(motion["pose_aa"], dtype=np.float64)
    fps = float(motion["fps"])
    dof_pos = pose_aa.sum(axis=-1)[:, 1:23].astype(np.float64)
    return key, motion, dof_pos, fps


def joint_maps(model, dof_names):
    actuator_by_joint = {}
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        actuator_by_joint[name] = i

    qpos_ids, qvel_ids, act_ids = [], [], []
    for name in dof_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"joint not found in MuJoCo model: {name}")
        if name not in actuator_by_joint:
            raise RuntimeError(f"actuator not found for joint: {name}")
        qpos_ids.append(int(model.jnt_qposadr[jid]))
        qvel_ids.append(int(model.jnt_dofadr[jid]))
        act_ids.append(int(actuator_by_joint[name]))
    return np.array(qpos_ids), np.array(qvel_ids), np.array(act_ids)


def robot_state(deploy, model, data, qpos_ids, qvel_ids):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    quat = data.xquat[body_id].copy()  # wxyz
    vel = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, vel, 0)
    ang_world = vel[:3].copy()
    base_ang_vel = deploy.quat_rotate_inverse_wxyz(quat, ang_world).astype(np.float32)
    projected_gravity = deploy.quat_rotate_inverse_wxyz(quat, [0.0, 0.0, -1.0]).astype(np.float32)
    return {
        "dof_pos": data.qpos[qpos_ids].copy().astype(np.float32),
        "dof_vel": data.qvel[qvel_ids].copy().astype(np.float32),
        "base_ang_vel": base_ang_vel,
        "projected_gravity": projected_gravity,
        "root_z": float(data.qpos[2]),
        "quat": quat,
    }


def pd_torque(target, q, dq, kp, kd, limit):
    tau = kp * (target - q) - kd * dq
    return np.clip(tau, -limit, limit)


def main():
    parser = argparse.ArgumentParser(description="MuJoCo smoke test for Q1 2real ONNX deployment path")
    parser.add_argument("--config", default="config/q1_sim2real_base.yaml")
    parser.add_argument("--policy-path", default="policies/stand/model_5000.onnx")
    parser.add_argument("--motion-file", default="motions/stand/q1_stand_still.pkl")
    parser.add_argument("--mujoco-xml", default="/root/autodl-tmp/ASAP_official/scripts/q1_sim2sim_physics.xml")
    parser.add_argument("--num-control-steps", type=int, default=150)
    parser.add_argument("--command-step-rad", type=float, default=0.005)
    parser.add_argument("--target-step-rad", type=float, default=0.08)
    parser.add_argument("--output-csv", default="outputs/mujoco_stand_smoke.csv")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-output", default="outputs/mujoco_stand_smoke.mp4")
    args = parser.parse_args()

    deploy = load_deploy_module()
    cfg_path = resolve(args.config)
    policy_path = resolve(args.policy_path)
    motion_path = resolve(args.motion_file)
    output_csv = resolve(args.output_csv)

    cfg = deploy.read_conf(str(cfg_path))
    cycle_time, cycle_source = deploy.infer_cycle_time_from_motion(str(motion_path))
    cfg["cycle_time"] = float(cycle_time)
    cfg["simulation_duration"] = max(float(cfg.get("simulation_duration", 0.0)), cfg["cycle_time"])

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    torque_limits = np.asarray(raw_cfg.get("tau_limit", [36.0] * cfg["num_actions"]), dtype=np.float64)

    expected_obs_dim = (cfg["frame_stack"] + 1) * cfg["num_single_obs"]
    policy = deploy.load_onnx_policy(str(policy_path), expected_obs_dim, cfg["num_actions"])

    model = mujoco.MjModel.from_xml_path(str(args.mujoco_xml))
    model.opt.timestep = float(cfg["simulation_dt"])
    data = mujoco.MjData(model)

    # Ensure a fixed camera exists for offscreen rendering
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "render_cam")
    if cam_id < 0:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_cam")
    if cam_id < 0 and model.ncam > 0:
        cam_id = 0
    use_fixed_cam = cam_id >= 0

    renderer = None
    frame_dir = None
    if args.record_video:
        width, height = 1280, 720
        try:
            gl_context = mujoco.GLContext(max_width=width, max_height=height)
            gl_context.make_current()
        except Exception as e:
            print(f"[WARN] GLContext failed ({e}), trying xvfb-run fallback")
            renderer = None
        else:
            renderer = mujoco.Renderer(model, height=height, width=width)
            if use_fixed_cam:
                data.cam.distance = 1.8
                data.cam.azimuth = 90
                data.cam.elevation = -15
                data.cam.lookat[:] = [0.0, 0.0, 0.55]
            frame_dir = tempfile.mkdtemp(prefix="q1_smoke_frames_")

    qpos_ids, qvel_ids, act_ids = joint_maps(model, cfg["dof_names"])

    motion_key, motion, motion_dof_pos, fps = load_motion(motion_path)
    init_dof = motion_dof_pos[0]
    if init_dof.shape[0] != cfg["num_actions"]:
        raise RuntimeError(f"motion dof dim {init_dof.shape[0]} != {cfg['num_actions']}")

    data.qpos[:3] = np.array([0.0, 0.0, 0.42], dtype=np.float64)
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qpos[qpos_ids] = init_dof
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    hist_dict, hist_obs = deploy.create_history(cfg)
    last_action = np.zeros(cfg["num_actions"], dtype=np.float32)
    latest_target = init_dof.astype(np.float32).copy()
    command_target = latest_target.copy()
    policy_dt = cfg["simulation_dt"] * cfg["control_decimation"]

    rows = []
    stop_reason = "completed"
    for step in range(args.num_control_steps):
        state = robot_state(deploy, model, data, qpos_ids, qvel_ids)
        obs, hist_obs = deploy.get_obs(hist_obs, hist_dict, state, last_action, step * cfg["control_decimation"], cfg)
        raw_action = deploy.infer_policy_action(policy, obs, cfg["clip_actions"])
        action, target = deploy.apply_action_postprocess(raw_action, last_action, latest_target, cfg, policy_dt)
        target = np.clip(target, latest_target - args.target_step_rad, latest_target + args.target_step_rad)
        latest_target = target.astype(np.float32)
        last_action = action.astype(np.float32)

        for _ in range(cfg["control_decimation"]):
            desired = np.clip(latest_target, command_target - args.command_step_rad, command_target + args.command_step_rad)
            command_target = desired.astype(np.float32)
            q = data.qpos[qpos_ids].copy()
            dq = data.qvel[qvel_ids].copy()
            tau = pd_torque(command_target, q, dq, cfg["kps"], cfg["kds"], torque_limits)
            data.ctrl[act_ids] = tau
            mujoco.mj_step(model, data)

        rows.append({
            "step": step,
            "time": step * policy_dt,
            "phase": deploy.compute_ref_motion_phase(step * cfg["control_decimation"], cfg),
            "root_z": float(data.qpos[2]),
            "action_max": float(np.max(np.abs(raw_action))),
            "target_delta_max": float(np.max(np.abs(latest_target - state["dof_pos"]))),
            "command_delta_max": float(np.max(np.abs(command_target - state["dof_pos"]))),
            "dof_vel_max": float(np.max(np.abs(data.qvel[qvel_ids]))),
            "tau_max": float(np.max(np.abs(tau))),
            "proj_g_x": float(state["projected_gravity"][0]),
            "proj_g_y": float(state["projected_gravity"][1]),
            "proj_g_z": float(state["projected_gravity"][2]),
        })

        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            stop_reason = "nonfinite_state"
            break
        if data.qpos[2] < 0.05:
            stop_reason = f"root_z_low({data.qpos[2]:.3f})"
            break

        # Render frame after each control step
        if renderer is not None:
            renderer.update_scene(data, camera=cam_id if use_fixed_cam else -1)
            pixels = renderer.render()
            frame_path = os.path.join(frame_dir, f"frame_{step:06d}.png")
            cv2.imwrite(frame_path, cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))

    if renderer is not None:
        renderer.close()
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    root_z = [r["root_z"] for r in rows]
    action_max = [r["action_max"] for r in rows]
    tau_max = [r["tau_max"] for r in rows]
    print("=" * 72)
    print("Q1 2real MuJoCo smoke test")
    print("=" * 72)
    print(f"policy={policy_path}")
    print(f"motion={motion_path} key={motion_key} cycle_time={cfg['cycle_time']:.3f}s source={cycle_source}")
    print(f"steps={len(rows)} sim_time={len(rows) * policy_dt:.3f}s stop={stop_reason}")
    print(f"root_z=[{min(root_z):.4f}, {max(root_z):.4f}]")
    print(f"action_max=[{min(action_max):.4f}, {max(action_max):.4f}]")
    print(f"tau_max=[{min(tau_max):.4f}, {max(tau_max):.4f}]")
    if args.record_video and frame_dir:
        video_path = resolve(args.video_output)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-framerate", "50",
            "-i", os.path.join(frame_dir, "frame_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "23", "-preset", "fast",
            str(video_path),
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            print(f"video={video_path}")
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg error: {e.stderr.decode()}")
        except FileNotFoundError:
            print("ffmpeg not found; frames saved in", frame_dir)
            print("Run: ffmpeg -framerate 50 -i " + frame_dir + "/frame_%06d.png -c:v libx264 output.mp4")
        else:
            shutil.rmtree(frame_dir)

    print(f"csv={output_csv}")


if __name__ == "__main__":
    main()
