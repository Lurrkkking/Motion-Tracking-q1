#!/usr/bin/env python3
"""综合诊断：alpha sweep + per-joint trace + rate limit"""
import os; os.environ['MUJOCO_GL']='egl'
import sys; sys.path.insert(0,'.')
import mujoco, numpy as np, onnxruntime, time
from scripts.q1_mujoco_sim2sim import *
from pathlib import Path

base = Path('.').resolve()
cfg = read_config('scripts/q1_sim2sim_config.yaml')
cfg.control_dt = cfg.simulation_dt * cfg.control_decimation

MODEL = str(base/'logs/Q1_CR7/20260608_143745-Q1_CR7_Tracking_v1-q1_cr7_motion_tracking-q1_22dof_box/exported/model_5500.onnx')
XML = str(base/'scripts/q1_motion_tracking_mj.xml')
MOTION = str(base/'humanoidverse/data/motions/q1/q1_cr7_scale045_rootz042.pkl')

ref_dof_all, ref_dvel_all, ref_root_all, ref_rvel_all, fps, mdt = load_motion(MOTION)

def make_env():
    model = mujoco.MjModel.from_xml_path(XML)
    model.opt.timestep = cfg.simulation_dt
    model.opt.iterations = 100; model.opt.ls_iterations = 50
    data = mujoco.MjData(model)
    qpos_a, qvel_a, act_a = build_joint_maps(model)
    set_dof_pos(data, qpos_a, cfg.default_dof_pos)
    data.qpos[:3]=[0.,0.,0.42]; data.qpos[3:7]=[1.,0.,0.,0.]; data.qvel[:]=0.
    mujoco.mj_forward(model, data)
    min_z=min(data.geom_xpos[i][2] for i in range(model.ngeom) if model.geom_contype[i]!=0)
    if abs(min_z)>5e-4: data.qpos[2]-=min_z; mujoco.mj_forward(model, data)
    return model, data, qpos_a, qvel_a, act_a

def run_trial(alpha=1.0, max_delta=0.0, arm_zero=False, upper_zero=False, leg_zero=False,
              hist_warmup=False, hist_first_obs=None, n_steps=40):
    """Run policy sim for n_steps control cycles, return diagnostics."""
    model, data, qpos_a, qvel_a, act_a = make_env()
    sess = onnxruntime.InferenceSession(MODEL)
    iname = sess.get_inputs()[0].name; oname = sess.get_outputs()[0].name

    target = cfg.default_dof_pos.copy()
    prev_target = cfg.default_dof_pos.copy()
    action = np.zeros(22, dtype=np.float32)

    hist = {k: np.zeros((HIST_DEPTH, COMP_DIM[k]), dtype=np.float32) for k in HIST_KEYS}
    if hist_warmup and hist_first_obs is not None:
        # Fill history with first-obs values (simulating warmstart)
        pass  # TODO

    tau_lim = cfg.tau_limit; kps = cfg.kps; kds = cfg.kds
    cycle = cfg.cycle_time; cdt = cfg.control_dt; decim = cfg.control_decimation
    sim_steps = n_steps * decim
    ct = counter = 0
    records = []
    qacc_max = 0; tau_max_h = 0; action_max_h = 0; target_jump_max = 0
    collapsed = False

    for _ in range(sim_steps):
        mj = get_state(data, qpos_a, qvel_a)
        tau = kps*(target - mj[2]) - kds*mj[3]
        tau = np.clip(tau, -tau_lim, tau_lim)
        apply_torques(data, act_a, tau)
        mujoco.mj_step(model, data)
        counter += 1
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()): collapsed = True; break
        qa = np.abs(data.qacc).max(); qacc_max = max(qacc_max, qa)
        tau_max_h = max(tau_max_h, np.abs(tau).max())
        if qa > cfg.max_abs_qacc: collapsed = True; break
        if data.qpos[2] < 0.05: collapsed = True; break
        if counter % decim != 0: continue

        base_ang, proj_g, dp, dv, base_q, ang_w, lin_w = get_state(data, qpos_a, qvel_a)
        t_sim = (ct+1)*cdt; phase = min(t_sim/cycle, 1.0); tm = min(t_sim, cycle-1e-6)
        ref_dof = interp(ref_dof_all, tm, mdt); ref_dvel = interp(ref_dvel_all, tm, mdt)
        ref_root = interp(ref_root_all, tm, mdt)
        act_z = float(data.qpos[2]); act_yaw = yaw_xyzw(base_q); ye = 0.0 - act_yaw
        cr,to,fl,la = compute_phase_masks(phase); lc,rc = get_foot_contact(data, model)

        p = {}
        p['base_ang_vel'] = base_ang*0.25; p['projected_gravity'] = proj_g
        p['dof_pos'] = dp - cfg.default_dof_pos; p['dof_vel'] = dv*0.05; p['actions'] = action
        p['ref_motion_phase'] = np.array([phase], dtype=np.float32)
        p['q1_root_error'] = np.array([ref_root[2]-act_z, 0., act_z, lin_w[2]], dtype=np.float32)
        p['q1_yaw_error'] = np.array([np.sin(ye), np.cos(ye), -ang_w[2], ang_w[2]], dtype=np.float32)
        p['q1_flight_phase'] = np.array([float(cr),float(to),float(fl),float(la),float(lc),float(rc)], dtype=np.float32)
        p['q1_ref_dof_error'] = np.concatenate([ref_dof-dp, np.zeros(22)], dtype=np.float32)
        p['history_actor'] = np.concatenate([hist[k].flatten() for k in HIST_KEYS])
        hv = dict(base_ang_vel=base_ang, projected_gravity=proj_g, dof_pos=dp-cfg.default_dof_pos,
                  dof_vel=dv, actions=action, ref_motion_phase=np.array([phase], dtype=np.float32))
        for k in HIST_KEYS: hist[k][1:]=hist[k][:-1]; hist[k][0]=hv[k].astype(np.float32)
        obs = np.clip(np.concatenate([p[k] for k in OBS_ORDER]).astype(np.float32), -100, 100)

        raw = sess.run([oname], {iname: obs.reshape(1, -1)})[0][0]
        raw = np.clip(raw, -cfg.clip_actions, cfg.clip_actions)
        action_max_h = max(action_max_h, np.abs(raw).max())

        # Apply masks
        if arm_zero:
            raw[14:22] = 0.0  # arms
        if upper_zero:
            raw[12:22] = 0.0  # waist + arms
        if leg_zero:
            raw[0:12] = 0.0   # legs

        # Apply alpha
        target_raw = cfg.default_dof_pos + alpha * cfg.action_scale * raw

        # Rate limit
        if max_delta > 0:
            target_raw = np.clip(target_raw, prev_target - max_delta, prev_target + max_delta)

        target_jump = np.abs(target_raw - prev_target).max()
        target_jump_max = max(target_jump_max, target_jump)
        prev_target = target_raw.copy()
        target = target_raw
        action = raw

        ct += 1
        if ct <= 10:
            worst_joint = np.argmax(np.abs(target - dp))
            records.append(dict(
                step=ct, joint=JOINT_NAMES[worst_joint],
                action_val=raw[worst_joint], dof_pos=dp[worst_joint],
                target_val=target[worst_joint], target_diff=target[worst_joint]-dp[worst_joint],
                tau_val=np.abs(tau).max(), knee_a=raw[3], hip_a=raw[0], ank_a=raw[4],
            ))

    return dict(
        collapsed=collapsed, steps=ct, qacc_max=qacc_max, tau_max=tau_max_h,
        action_max=action_max_h, target_jump_max=target_jump_max,
        records=records, final_z=data.qpos[2],
    )


print("="*80)
print("EXPERIMENT 1: Alpha sweep")
print("="*80)
for alpha in [0.0, 0.1, 0.25, 0.5, 1.0]:
    r = run_trial(alpha=alpha, n_steps=50)
    status = "✗炸" if r['collapsed'] else "✓稳"
    print(f"  alpha={alpha:.2f} {status} steps={r['steps']} qacc={r['qacc_max']:.0f} tau={r['tau_max']:.0f} action={r['action_max']:.1f} jump={r['target_jump_max']:.3f} z={r['final_z']:.3f}")

print()
print("="*80)
print("EXPERIMENT 4: Per-joint trace (alpha=1.0, first 10 steps)")
print("="*80)
r = run_trial(alpha=1.0, n_steps=10)
for rec in r['records']:
    print(f"  step={rec['step']:2d} worst={rec['joint']:28s} a={rec['action_val']:7.2f} pos={rec['dof_pos']:7.3f} tgt={rec['target_val']:7.3f} diff={rec['target_diff']:7.3f} τ={rec['tau_val']:5.0f} knee_a={rec['knee_a']:6.2f} hip_a={rec['hip_a']:6.2f} ank_a={rec['ank_a']:6.2f}")

print()
print("="*80)
print("EXPERIMENT 5: Action mask ablation")
print("="*80)
for label, kwargs in [
    ("full", {}),
    ("arms=0", dict(arm_zero=True)),
    ("upper=0", dict(upper_zero=True)),
    ("legs=0", dict(leg_zero=True)),
]:
    r = run_trial(**kwargs, n_steps=50)
    status = "✗炸" if r['collapsed'] else "✓稳"
    print(f"  {label:10s} {status} steps={r['steps']} qacc={r['qacc_max']:.0f} tau={r['tau_max']:.0f} action={r['action_max']:.1f}")

print()
print("="*80)
print("EXPERIMENT 6: Target rate limit (alpha=1.0)")
print("="*80)
for max_delta in [0.03, 0.05, 0.08, 0.15, 0.0]:
    r = run_trial(alpha=1.0, max_delta=max_delta, n_steps=50)
    status = "✗炸" if r['collapsed'] else "✓稳"
    print(f"  max_delta={max_delta:.2f} {status} steps={r['steps']} qacc={r['qacc_max']:.0f} tau={r['tau_max']:.0f} jump={r['target_jump_max']:.3f} action={r['action_max']:.1f}")

print()
print("done")
