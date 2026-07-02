#!/usr/bin/env python3
"""
Q1 CR7 Motion Tracking — MuJoCo sim2sim.
All joint access is name-based (no hardcoded qpos slicing).
"""
import os, sys
if "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import argparse, time, csv, yaml
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

N_DOF = 22
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

OBS_ORDER = ["base_ang_vel", "projected_gravity", "dof_pos", "dof_vel", "actions",
             "ref_motion_phase", "q1_root_error", "q1_yaw_error", "q1_flight_phase",
             "q1_ref_dof_error", "history_actor"]
COMP_DIM = dict(base_ang_vel=3, projected_gravity=3, dof_pos=22, dof_vel=22,
                actions=22, ref_motion_phase=1, q1_root_error=4, q1_yaw_error=4,
                q1_flight_phase=6, q1_ref_dof_error=44)
HIST_KEYS = ["base_ang_vel", "projected_gravity", "dof_pos", "dof_vel", "actions", "ref_motion_phase"]
HIST_DEPTH = 4
PHASE_CFG = dict(crouch_start=0.20, crouch_end=0.45, takeoff_start=0.38, takeoff_end=0.55,
                 flight_start=0.50, flight_end=0.72, landing_start=0.72, landing_end=0.90)


# ── config ───────────────────────────────────────────────────────────
def read_config(path):
    with open(path) as f:
        d = yaml.safe_load(f)
    cfg = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
            v = np.array(v, dtype=np.float32)
        setattr(cfg, k, v)
    return cfg


# ── motion ───────────────────────────────────────────────────────────
def load_motion(path):
    import joblib
    data = joblib.load(path)
    k = sorted(data.keys())[0]
    m = data[k]
    fps = float(m["fps"]); dt = 1.0 / fps
    dof = m["pose_aa"].sum(axis=-1)[:, 1:].astype(np.float64)
    root = m["root_trans_offset"].astype(np.float64)
    # finite-diff velocities
    dof_vel = np.zeros_like(dof); root_vel = np.zeros_like(root)
    T = len(dof)
    if T >= 3:
        dof_vel[1:-1] = (dof[2:] - dof[:-2]) / (2 * dt)
        root_vel[1:-1] = (root[2:] - root[:-2]) / (2 * dt)
    if T >= 2:
        dof_vel[0] = (dof[1] - dof[0]) / dt; dof_vel[-1] = (dof[-1] - dof[-2]) / dt
        root_vel[0] = (root[1] - root[0]) / dt; root_vel[-1] = (root[-1] - root[-2]) / dt
    return dof, dof_vel, root, root_vel, fps, dt


def interp(arr, t, dt):
    f = np.clip(t / dt, 0, len(arr) - 1.001)
    f0, f1 = int(np.floor(f)), min(int(np.floor(f)) + 1, len(arr) - 1)
    return arr[f0] * (1 - (f - f0)) + arr[f1] * (f - f0)


def yaw_xyzw(q):
    x, y, z, w = q[:4]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ── name-based MuJoCo access ────────────────────────────────────────
def build_joint_maps(model):
    """返回 name→qpos/dof/actuator 的映射字典"""
    qpos_a, qvel_a, act_a = {}, {}, {}
    for name in JOINT_NAMES:
        jid = model.joint(name).id
        qpos_a[name] = model.jnt_qposadr[jid]
        qvel_a[name] = model.jnt_dofadr[jid]
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jname in JOINT_NAMES:
            act_a[jname] = i
    return qpos_a, qvel_a, act_a


def get_dof_pos(data, qpos_a):
    return np.array([data.qpos[qpos_a[n]] for n in JOINT_NAMES], dtype=np.float64)


def get_dof_vel(data, qvel_a):
    return np.array([data.qvel[qvel_a[n]] for n in JOINT_NAMES], dtype=np.float64)


def set_dof_pos(data, qpos_a, vals):
    for n, v in zip(JOINT_NAMES, vals):
        data.qpos[qpos_a[n]] = float(v)


def apply_torques(data, act_a, tau):
    ctrl = np.zeros(data.ctrl.shape)
    for i, n in enumerate(JOINT_NAMES):
        ctrl[act_a[n]] = tau[i]
    data.ctrl[:] = ctrl


def get_state(data, qpos_a, qvel_a):
    q = data.qpos.astype(np.float64)
    dq = data.qvel.astype(np.float64)
    q_xyzw = np.array([q[4], q[5], q[6], q[3]])
    rot = R.from_quat(q_xyzw)
    return (rot.apply(dq[3:6], inverse=True),
            rot.apply([0., 0., -1.], inverse=True),
            get_dof_pos(data, qpos_a),
            get_dof_vel(data, qvel_a),
            q_xyzw, dq[3:6].copy(), dq[:3].copy())


def get_foot_contact(data, model):
    lc = rc = False
    for ci in range(data.ncon):
        g1, g2 = data.contact[ci].geom1, data.contact[ci].geom2
        b1 = model.body(int(model.geom_bodyid[g1])).name if g1 < model.ngeom else "world"
        b2 = model.body(int(model.geom_bodyid[g2])).name if g2 < model.ngeom else "world"
        bodies = {b1, b2}
        if "world" in bodies:
            if "left_ankle_roll" in b1 or "left_ankle_roll" in b2: lc = True
            if "right_ankle_roll" in b1 or "right_ankle_roll" in b2: rc = True
    return lc, rc


def compute_phase_masks(phase):
    pc = PHASE_CFG
    cr = pc["crouch_start"] <= phase < pc["crouch_end"]
    to = pc["takeoff_start"] <= phase < pc["takeoff_end"]
    fl = pc["flight_start"] <= phase < pc["flight_end"]
    la = pc["landing_start"] <= phase < pc["landing_end"]
    fl = fl and not to; la = la and not to and not fl
    cr = cr and not to and not fl and not la
    return cr, to, fl, la


# ── main ─────────────────────────────────────────────────────────────
def run(cfg, args):
    base = Path(__file__).resolve().parent.parent
    xml_path = args.xml_path or cfg.xml_path
    policy_path = args.policy_path or cfg.policy_path
    if not os.path.isabs(xml_path): xml_path = str(base / xml_path)
    if not os.path.isabs(policy_path): policy_path = str(base / policy_path)
    motion_path = args.motion_file if os.path.isabs(args.motion_file) else str(base / args.motion_file)
    assert os.path.isfile(xml_path), f"XML: {xml_path}"
    assert os.path.isfile(policy_path) or args.mode != "policy", f"Policy: {policy_path}"

    # Motion
    ref_dof_all, ref_dvel_all, ref_root_all, ref_rvel_all, fps, mdt = load_motion(motion_path)
    print(f"[MOTION] frames={len(ref_dof_all)} fps={fps} root_z=[{ref_root_all[:,2].min():.3f},{ref_root_all[:,2].max():.3f}]")
    print(f"[MODE] {args.mode}")

    # MuJoCo
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = cfg.simulation_dt
    model.opt.iterations = getattr(cfg, "solver_iterations", 100)
    if hasattr(model.opt, "ls_iterations") and hasattr(cfg, "solver_ls_iterations"):
        model.opt.ls_iterations = cfg.solver_ls_iterations
    data = mujoco.MjData(model)
    qpos_a, qvel_a, act_a = build_joint_maps(model)
    print(f"[MAPPING] {len(act_a)}/{len(JOINT_NAMES)} actuators mapped")

    # Init
    if args.mode == "ref_replay":
        # Start from motion frame 0
        set_dof_pos(data, qpos_a, ref_dof_all[0])
        data.qpos[:3] = ref_root_all[0].copy()
    else:
        set_dof_pos(data, qpos_a, cfg.default_dof_pos)
        data.qpos[:3] = [0., 0., 0.42]
    data.qpos[3:7] = [1., 0., 0., 0.]
    data.qvel[:] = 0.
    mujoco.mj_forward(model, data)

    # Foot compensation (direct qpos adjust, no mj_step)
    min_z = min(data.geom_xpos[i][2] for i in range(model.ngeom)
                if model.geom_contype[i] != 0 and model.geom_bodyid[i] != 0)
    if abs(min_z) > 5e-4:
        data.qpos[2] -= min_z
        mujoco.mj_forward(model, data)
        print(f"[INIT] foot_off={min_z:.4f} z={data.qpos[2]:.4f}")
    else:
        print(f"[INIT] z={data.qpos[2]:.4f} feet grounded ok")

    # PD
    tau_lim = cfg.tau_limit
    kps = cfg.kps * getattr(cfg, "kp_scale", 1.0)
    kds = cfg.kds * getattr(cfg, "kd_scale", 1.0)

    # Policy / action
    target = cfg.default_dof_pos.copy()
    action = np.zeros(N_DOF, dtype=np.float32)
    sess = None
    if args.mode == "policy":
        import onnxruntime
        sess = onnxruntime.InferenceSession(policy_path)
        iname = sess.get_inputs()[0].name; oname = sess.get_outputs()[0].name
        print(f"[ONNX] {iname} {sess.get_inputs()[0].shape} → {oname} {sess.get_outputs()[0].shape}")

    # History for obs
    hist = {k: np.zeros((HIST_DEPTH, COMP_DIM[k]), dtype=np.float32) for k in HIST_KEYS}

    # Camera / render
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.distance = args.cam_distance; cam.azimuth = args.cam_azimuth
    cam.elevation = args.cam_elevation; cam.lookat[:] = [0., 0., 0.4]
    renderer = None; frames = []
    if args.video_out:
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    # Loop
    csv_rows = []
    sim_steps = int(cfg.simulation_duration / cfg.simulation_dt)
    render_stride = max(1, int(round(1.0 / max(cfg.simulation_dt * args.video_fps, 1e-9))))
    ct = counter = 0; stop = "timeout"
    cycle = cfg.cycle_time
    tgt_delta = np.zeros(N_DOF)
    stats = dict(raw_max=0, tgt_jump_max=0, tau_peak=0, qacc_peak=0)

    t0 = time.time()
    for _ in range(sim_steps):
        mj = get_state(data, qpos_a, qvel_a)
        dof_pos = mj[2]

        # Target
        if args.mode == "zero_action":
            target = cfg.default_dof_pos
        elif args.mode == "ref_replay":
            tm = min(counter * cfg.simulation_dt, cycle - 1e-6)
            target = interp(ref_dof_all, tm, mdt)

        tau = kps * (target - dof_pos) - kds * mj[3]
        tau = np.clip(tau, -tau_lim, tau_lim)
        stats['tau_peak'] = max(stats['tau_peak'], float(np.abs(tau).max()))
        apply_torques(data, act_a, tau)
        mujoco.mj_step(model, data)
        counter += 1

        qa = float(np.abs(data.qacc).max())
        stats['qacc_peak'] = max(stats['qacc_peak'], qa)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            stop = "nan"; break
        if qa > cfg.max_abs_qacc:
            stop = f"qacc({np.abs(data.qacc).max():.0f})"; break

        t_sim = cfg.time_offset + counter * cfg.simulation_dt
        if cfg.stop_at_motion_end and t_sim >= cycle:
            stop = "motion_end"; break
        if counter % cfg.control_decimation != 0:
            continue

        # ── policy step ──
        if args.mode == "policy":
            base_ang, proj_g, dp, dv, base_q, ang_w, lin_w = mj
            phase = min(float(np.clip(t_sim / cycle, 0, 1)), 1.0)
            tm = min(t_sim, cycle - 1e-6)
            ref_dof = interp(ref_dof_all, tm, mdt)
            ref_dvel = interp(ref_dvel_all, tm, mdt)
            ref_root = interp(ref_root_all, tm, mdt)
            ref_rvel = interp(ref_rvel_all, tm, mdt)
            act_z = float(data.qpos[2])
            act_yaw = yaw_xyzw(base_q)
            ye = 0.0 - act_yaw
            cr, to, fl, la = compute_phase_masks(phase)
            lc, rc = get_foot_contact(data, model)

            p = {}
            p["base_ang_vel"] = base_ang * 0.25
            p["projected_gravity"] = proj_g * 1.0
            p["dof_pos"] = (dp - cfg.default_dof_pos) * 1.0
            p["dof_vel"] = dv * 0.05
            p["actions"] = action * 1.0
            p["ref_motion_phase"] = np.array([phase], dtype=np.float32)
            p["q1_root_error"] = np.array([ref_root[2]-act_z, ref_rvel[2]-lin_w[2], act_z, lin_w[2]], dtype=np.float32)
            p["q1_yaw_error"] = np.array([np.sin(ye), np.cos(ye), 0.0-ang_w[2], ang_w[2]], dtype=np.float32)
            p["q1_flight_phase"] = np.array([float(cr),float(to),float(fl),float(la),float(lc),float(rc)], dtype=np.float32)
            p["q1_ref_dof_error"] = np.concatenate([ref_dof-dp, ref_dvel-dv], dtype=np.float32)

            p["history_actor"] = np.concatenate([hist[k].flatten() for k in HIST_KEYS])
            hv = dict(base_ang_vel=base_ang, projected_gravity=proj_g,
                      dof_pos=dp-cfg.default_dof_pos, dof_vel=dv,
                      actions=action, ref_motion_phase=np.array([phase], dtype=np.float32))
            for k in HIST_KEYS:
                hist[k][1:] = hist[k][:-1]; hist[k][0] = hv[k].astype(np.float32)

            obs = np.clip(np.concatenate([p[k] for k in OBS_ORDER]).astype(np.float32), -100, 100)

            raw = sess.run([oname], {iname: obs.reshape(1, -1)})[0][0]
            # sim2sim action clip (much tighter than training's 100)
            raw = np.clip(raw, -args.sim_action_clip, args.sim_action_clip)
            action = raw
            # target = default + action_scale * action, with rate limit
            target_raw = cfg.default_dof_pos + args.alpha * cfg.action_scale * action
            tgt_delta = np.clip(target_raw - target, -args.target_max_delta, args.target_max_delta)
            target = target + tgt_delta
            # stats
            stats['raw_max'] = max(stats['raw_max'], float(np.abs(raw).max()))
            stats['tgt_jump_max'] = max(stats['tgt_jump_max'], float(np.abs(tgt_delta).max()))

        ct += 1
        if ct == 1:
            print(f"[STEP1] t={t_sim:.3f} |dof_pos|={np.abs(dof_pos-cfg.default_dof_pos).max():.4f} "
                  f"|dof_vel|={np.abs(mj[3]).max():.4f}")
            if args.mode == "policy":
                print(f"[STEP1] |ref_dof_err|={np.abs(ref_dof-dof_pos).max():.4f}")
                print(f"[STEP1] raw_action: max={np.abs(raw).max():.3f} mean={np.abs(raw).mean():.3f}")
                print(f"[STEP1] target_jump: max={np.abs(tgt_delta).max():.4f}")
            worst = np.argmax(np.abs(tau))
            print(f"[STEP1] max_tau joint: {JOINT_NAMES[worst]} tau={tau[worst]:.1f} target={target[worst]:.3f} pos={dof_pos[worst]:.3f}")

        if ct % 50 == 0 or ct == 1:
            tjd = np.abs(tgt_delta).max() if args.mode == "policy" else 0
            print(f"  t={t_sim:.2f}s step={ct:3d} z={data.qpos[2]:.3f} τ_max={np.abs(tau).max():.0f} "
                  f"raw_max={np.abs(action).max():.1f} tgt_jump={tjd:.3f}")

        if args.debug_csv:
            dof_err_str = f"{np.abs(ref_dof-dof_pos).max():.4f}" if args.mode=="policy" else "0"
            csv_rows.append(dict(step=ct, time=f"{t_sim:.4f}", z=f"{data.qpos[2]:.4f}",
                                 tau_max=f"{np.abs(tau).max():.1f}", dof_err=dof_err_str))

        if renderer and counter % render_stride == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())

    elapsed = time.time() - t0
    msg = (f"\n[SUMMARY] mode={args.mode} steps={ct} stop={stop} wall={elapsed:.1f}s z=[{data.qpos[2]:.3f}]\n"
           f"[STATS] raw_action_max={stats['raw_max']:.1f} target_jump_max={stats['tgt_jump_max']:.4f} "
           f"tau_peak={stats['tau_peak']:.0f} qacc_peak={stats['qacc_peak']:.0f}")
    print(msg, flush=True)

    if renderer and frames:
        os.makedirs(os.path.dirname(args.video_out) or ".", exist_ok=True)
        try:
            import imageio.v2 as imageio
            imageio.mimsave(args.video_out, frames, fps=args.video_fps)
        except:
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
        print(f"[CSV] {cp}")

    if renderer:
        try: renderer.close()
        except: pass


def main():
    p = argparse.ArgumentParser("Q1 MuJoCo sim2sim")
    p.add_argument("--config", default="scripts/q1_sim2sim_config.yaml")
    p.add_argument("--policy-path", default="")
    p.add_argument("--xml-path", default="scripts/q1_motion_tracking_mj.xml")
    p.add_argument("--motion-file", default="humanoidverse/data/motions/q1/q1_cr7_scale045_rootz042.pkl")
    p.add_argument("--mode", default="policy", choices=["policy", "zero_action", "ref_replay"])
    p.add_argument("--video-out", default="")
    p.add_argument("--video-fps", type=int, default=50)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--cam-distance", type=float, default=3.5)
    p.add_argument("--cam-azimuth", type=float, default=160.0)
    p.add_argument("--cam-elevation", type=float, default=-12.0)
    p.add_argument("--debug-csv", action="store_true")
    p.add_argument("--alpha", type=float, default=1.0, help="Action scale multiplier")
    p.add_argument("--sim-action-clip", type=float, default=5.0, help="Sim2sim raw action clip")
    p.add_argument("--target-max-delta", type=float, default=0.05, help="Max target change per control step (rad)")
    p.add_argument("--max-delta", type=float, default=0.0, help="(deprecated) use --target-max-delta")
    args = p.parse_args()
    # backward compat
    if args.max_delta > 0 and args.target_max_delta == 0.05:
        args.target_max_delta = args.max_delta

    cfg = read_config(args.config)
    print(f"[CONFIG] xml={args.xml_path} policy={args.policy_path or cfg.policy_path}")
    print(f"[CONFIG] dt={cfg.simulation_dt} decim={cfg.control_decimation} cycle={cfg.cycle_time}s")

    run(cfg, args)
    print("----- done -----")


if __name__ == "__main__":
    main()
