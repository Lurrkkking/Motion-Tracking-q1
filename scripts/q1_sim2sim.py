#!/usr/bin/env python3
"""
Q1 CR7 Motion Tracking MuJoCo sim2sim — ~300 lines, config-driven.
Mimics /root/autodl-tmp/mujoco_simulation/fdd_asap_sim2sim.py.

Builds 423-dim actor_obs matching training, PD control, video + CSV output.

Usage:
  python scripts/q1_sim2sim.py \
      --config scripts/q1_sim2sim_config.yaml \
      --motion-file humanoidverse/data/motions/q1/q1_cr7_scale045_rootz042.pkl \
      --video-out sim2sim_outputs/q1_cr7/rollout.mp4 \
      --debug-csv
"""
import os, sys
if "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import argparse, time, csv
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_roll_joint", "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
]
N_DOF = 22

# Obs component dims (matching q1_cr7_tracking_obs.yaml)
COMP_DIM = dict(base_ang_vel=3, projected_gravity=3, dof_pos=22, dof_vel=22,
                actions=22, ref_motion_phase=1, q1_root_error=4, q1_yaw_error=4,
                q1_flight_phase=6, q1_ref_dof_error=44)
OBS_ORDER = ["base_ang_vel", "projected_gravity", "dof_pos", "dof_vel",
             "actions", "ref_motion_phase", "q1_root_error", "q1_yaw_error",
             "q1_flight_phase", "q1_ref_dof_error", "history_actor"]
HIST_KEYS = ["base_ang_vel", "projected_gravity", "dof_pos", "dof_vel",
             "actions", "ref_motion_phase"]
HIST_DEPTH = 4


# ══════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════
def read_config(path):
    import yaml
    with open(path) as f:
        d = yaml.safe_load(f)
    cfg = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
            v = np.array(v, dtype=np.float32)
        setattr(cfg, k, v)
    return cfg


# ══════════════════════════════════════════════════════════════════════
#  Motion
# ══════════════════════════════════════════════════════════════════════
def load_motion(path):
    """Load pkl → (T,22) dof_pos, (T,3) root_pos, fps."""
    import joblib
    data = joblib.load(path)
    k = sorted(data.keys())[0]
    m = data[k]
    dof = m["pose_aa"].sum(axis=-1)[:, 1:].astype(np.float64)
    root = m["root_trans_offset"].astype(np.float64)
    return dof, root, float(m["fps"])


def interp(arr, t, dt):
    f = np.clip(t / dt, 0, len(arr) - 1.001)
    f0, f1 = int(np.floor(f)), min(int(np.floor(f)) + 1, len(arr) - 1)
    a = f - f0
    return arr[f0] * (1 - a) + arr[f1] * a


def yaw_xyzw(q):
    x, y, z, w = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ══════════════════════════════════════════════════════════════════════
#  MuJoCo state
# ══════════════════════════════════════════════════════════════════════
def get_state(data):
    q, dq = data.qpos.astype(np.float64), data.qvel.astype(np.float64)
    q_xyzw = np.array([q[4], q[5], q[6], q[3]])
    r = R.from_quat(q_xyzw)
    return (r.apply(dq[3:6], inverse=True),
            r.apply([0., 0., -1.], inverse=True),
            q[7:N_DOF+7].copy(), dq[6:N_DOF+6].copy(),
            q_xyzw, dq[3:6].copy())


# ══════════════════════════════════════════════════════════════════════
#  XML generation (add physics if missing)
# ══════════════════════════════════════════════════════════════════════
def ensure_physics_xml(xml_path):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()

    if root.find(".//inertial") is not None:
        return ET.tostring(root, encoding="unicode")

    wb = root.find("worldbody") or ET.SubElement(root, "worldbody")

    # Ground + light
    ET.SubElement(wb, "geom", dict(name="floor", type="plane", size="0 0 0.05",
        rgba="0.2 0.3 0.4 1", contype="1", conaffinity="1"))
    ET.SubElement(wb, "light", dict(directional="true", diffuse="0.7 0.7 0.7",
        pos="3 2 4", dir="-3 -2 -4"))

    # Inertial + geom per body
    body_mass = dict(pelvis=1.7, torso_link=1.5, left_knee_link=1.2, right_knee_link=1.2)
    for body in wb.iter("body"):
        name = body.get("name", "")
        m = body_mass.get(name, 0.5)
        is_foot = "ankle_roll" in name
        if body.find("inertial") is None:
            d = f"{m*0.01:.6f}"
            body.insert(0, ET.Element("inertial", dict(pos="0 0 0", mass=str(m),
                diaginertia=f"{d} {d} {d}")))
        if body.find("geom") is None:
            body.insert(1, ET.Element("geom", dict(type="box", size="0.04 0.04 0.04",
                rgba="0.6 0.6 0.6 1", contype="0", conaffinity="0")))
        # Foot collision spheres
        if is_foot:
            for ys in ["0.02", "-0.02"]:
                body.insert(len(list(body)), ET.Element("geom", dict(
                    type="sphere", size="0.025", pos=f"0.0 {ys} -0.04",
                    contype="1", conaffinity="1", rgba="0 0 0 0",
                    friction="1.0 0.005 0.0001")))

    return ET.tostring(root, encoding="unicode")


# ══════════════════════════════════════════════════════════════════════
#  Main sim loop
# ══════════════════════════════════════════════════════════════════════
def run(cfg, args):
    base = Path(__file__).resolve().parent.parent
    for a in ["xml_path", "policy_path"]:
        p = getattr(cfg, a); setattr(cfg, a, str(base / p) if not os.path.isabs(p) else p)
    assert os.path.isfile(cfg.xml_path), f"xml: {cfg.xml_path}"
    assert os.path.isfile(cfg.policy_path), f"policy: {cfg.policy_path}"

    # Motion
    ref_dof_all, ref_root_all, motion_fps = load_motion(args.motion_file)
    mdt = 1.0 / motion_fps

    # Model
    xml_str = ensure_physics_xml(cfg.xml_path)
    model = mujoco.MjModel.from_xml_string(xml_str)
    model.opt.timestep = cfg.simulation_dt
    model.opt.iterations = cfg.solver_iterations
    if hasattr(model.opt, "ls_iterations"):
        model.opt.ls_iterations = cfg.solver_ls_iterations
    data = mujoco.MjData(model)

    # Init
    data.qpos[:3] = [0., 0., 0.42]
    data.qpos[3:7] = [1., 0., 0., 0.]
    data.qpos[7:N_DOF+7] = cfg.default_dof_pos
    mujoco.mj_step(model, data)

    # Policy
    import onnxruntime
    sess = onnxruntime.InferenceSession(cfg.policy_path)
    iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name

    # State
    action = np.zeros(N_DOF, dtype=np.float32)
    target = cfg.default_dof_pos.copy()
    hist = {k: np.zeros((HIST_DEPTH, COMP_DIM[k]), dtype=np.float32)
            for k in HIST_KEYS}

    # Camera
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.distance = args.cam_distance
    cam.azimuth = args.cam_azimuth
    cam.elevation = args.cam_elevation

    renderer = None; frames = []
    if args.video_out:
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    csv_rows = []
    sim_steps = int(cfg.simulation_duration / cfg.simulation_dt)
    render_every = max(1, int(round(1.0 / max(cfg.simulation_dt * args.video_fps, 1e-9))))
    ct, counter, stop = 0, 0, "duration"

    t0 = time.time()
    for _ in range(sim_steps):
        mj = get_state(data)
        tau = cfg.kps * (target - mj[2]) - cfg.kds * mj[3]
        tau = np.clip(tau, -cfg.tau_limit, cfg.tau_limit)
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        counter += 1

        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            stop = "nan"; break
        if np.abs(data.qacc).max() > cfg.max_abs_qacc:
            stop = "qacc"; break

        t_sim = cfg.time_offset + counter * cfg.simulation_dt
        if cfg.stop_at_motion_end and t_sim >= cfg.cycle_time:
            stop = "motion_end"; break
        if counter % cfg.control_decimation != 0:
            continue

        # ── build 423-dim obs ──────────────────────────────────
        base_ang, proj_g, dof_pos, dof_vel, base_q, ang_w = mj
        phase = min(float(np.clip(t_sim / cfg.cycle_time, 0, 1)), 1.0)
        tm = min(t_sim, cfg.cycle_time - 1e-6)

        ref_dof = interp(ref_dof_all, tm, mdt)
        ref_root = interp(ref_root_all, tm, mdt)

        ref_yaw = yaw_xyzw(np.array([0., 0., 0., 1.]))
        act_yaw = yaw_xyzw(base_q)
        ye = ref_yaw - act_yaw

        dof_err = ref_dof - dof_pos
        act_z = float(data.qpos[2])

        p = {}
        p["base_ang_vel"]      = base_ang * 0.25
        p["projected_gravity"] = proj_g * 1.0
        p["dof_pos"]           = (dof_pos - cfg.default_dof_pos) * 1.0
        p["dof_vel"]           = dof_vel * 0.05
        p["actions"]           = action * 1.0
        p["ref_motion_phase"]  = np.array([phase], dtype=np.float32)
        p["q1_root_error"]     = np.array([ref_root[2] - act_z, 0., act_z, ang_w[2]], dtype=np.float32)
        p["q1_yaw_error"]      = np.array([np.sin(ye), np.cos(ye), 0., ang_w[2]], dtype=np.float32)
        p["q1_flight_phase"]   = np.zeros(6, dtype=np.float32)
        p["q1_ref_dof_error"]  = np.concatenate([dof_err, np.zeros(22)], dtype=np.float32)

        hv = dict(base_ang_vel=base_ang, projected_gravity=proj_g,
                  dof_pos=dof_pos - cfg.default_dof_pos, dof_vel=dof_vel,
                  actions=action, ref_motion_phase=np.array([phase], dtype=np.float32))
        for k in HIST_KEYS:
            hist[k][1:] = hist[k][:-1]
            hist[k][0] = hv[k].astype(np.float32)
        p["history_actor"] = np.concatenate([hist[k].flatten() for k in HIST_KEYS])

        obs = np.clip(np.concatenate([p[k] for k in OBS_ORDER]).astype(np.float32), -100, 100)

        # Policy
        raw = sess.run([oname], {iname: obs.reshape(1, -1)})[0][0]
        raw = np.clip(raw, -cfg.clip_actions, cfg.clip_actions)
        action = raw
        target = action * cfg.action_scale + cfg.default_dof_pos
        ct += 1

        if ct % 50 == 0 or ct == 1:
            print(f"  t={t_sim:.2f}s step={ct} z={act_z:.3f} ref_z={ref_root[2]:.3f} "
                  f"phase={phase:.3f} τ_max={np.abs(tau).max():.0f} "
                  f"yaw_err={np.rad2deg(ye):.1f}° dof_err={np.abs(dof_err).mean():.3f}")

        if args.debug_csv:
            csv_rows.append(dict(
                step=ct, time=f"{t_sim:.4f}", phase=f"{phase:.4f}",
                root_z=f"{act_z:.4f}", ref_z=f"{ref_root[2]:.4f}",
                yaw_err=f"{np.rad2deg(ye):.2f}", dof_err_mean=f"{np.abs(dof_err).mean():.4f}",
                action_max=f"{np.abs(raw).max():.3f}", tau_max=f"{np.abs(tau).max():.1f}",
                knee_L=f"{dof_pos[3]:.4f}", ref_knee_L=f"{ref_dof[3]:.4f}",
            ))

        if renderer and counter % render_every == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())

    elapsed = time.time() - t0
    print(f"\n[SUMMARY] steps={ct} stop={stop} wall={elapsed:.1f}s z={data.qpos[2]:.3f}")

    if renderer and frames:
        os.makedirs(os.path.dirname(args.video_out) or ".", exist_ok=True)
        try:
            import imageio.v2 as imageio
            imageio.mimsave(args.video_out, frames, fps=args.video_fps)
        except ImportError:
            import cv2
            h, w = frames[0].shape[:2]
            vw = cv2.VideoWriter_fourcc(*"mp4v")
            wr = cv2.VideoWriter(args.video_out, vw, args.video_fps, (w, h))
            for f in frames: wr.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            wr.release()
        print(f"[VIDEO] {args.video_out} ({len(frames)}f)")

    if csv_rows:
        cp = args.video_out.replace(".mp4", ".csv") if args.video_out else "sim2sim.csv"
        with open(cp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_rows[0].keys()); w.writeheader(); w.writerows(csv_rows)
        print(f"[CSV] {cp} ({len(csv_rows)}r)")

    if renderer: renderer.close()


# ══════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser("Q1 CR7 MuJoCo sim2sim")
    p.add_argument("--config", default="scripts/q1_sim2sim_config.yaml")
    p.add_argument("--policy-path", default="")
    p.add_argument("--xml-path", default="")
    p.add_argument("--motion-file", default="humanoidverse/data/motions/q1/q1_cr7_scale045_rootz042.pkl")
    p.add_argument("--video-out", default="")
    p.add_argument("--video-fps", type=int, default=50)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--cam-distance", type=float, default=3.5)
    p.add_argument("--cam-azimuth", type=float, default=160.0)
    p.add_argument("--cam-elevation", type=float, default=-12.0)
    p.add_argument("--debug-csv", action="store_true")
    p.add_argument("--headless", action="store_true")
    args = p.parse_args()

    cfg = read_config(args.config)
    if args.policy_path: cfg.policy_path = args.policy_path
    if args.xml_path: cfg.xml_path = args.xml_path

    print(f"[CONFIG] xml={cfg.xml_path}")
    print(f"[CONFIG] policy={cfg.policy_path}")
    print(f"[CONFIG] dt={cfg.simulation_dt} decim={cfg.control_decimation} "
          f"cycle={cfg.cycle_time}s dofs={cfg.num_actions}")

    run(cfg, args)
    print("----- done -----")


if __name__ == "__main__":
    main()
