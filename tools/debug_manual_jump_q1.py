#!/usr/bin/env python3
"""
Q1 manual jump physics verification — no policy, no training.
Two modes:
  A: ACTION_SPACE_TEST — actions clamped to [-1,1], uses env.step()
  B: DIRECT_PD_TARGET_TEST — direct PD target, bypasses action_scale

Run from repo root:
  cd /root/autodl-tmp/ASAP_official
  python scripts/debug_manual_jump_q1.py +simulator=isaacgym +exp=motion_tracking ...
"""
import os, sys, csv
from pathlib import Path
import numpy as np

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Register OmegaConf resolvers (required for config interpolation)
import math
try:
    OmegaConf.register_new_resolver("eval", eval)
except Exception: pass
try:
    OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
except Exception: pass
try:
    OmegaConf.register_new_resolver("eq", lambda x, y: str(x).lower() == str(y).lower())
except Exception: pass
try:
    OmegaConf.register_new_resolver("sqrt", lambda x: math.sqrt(float(x)))
except Exception: pass
try:
    OmegaConf.register_new_resolver("sum", lambda x: sum(x))
except Exception: pass
try:
    OmegaConf.register_new_resolver("ceil", lambda x: math.ceil(x))
except Exception: pass
try:
    OmegaConf.register_new_resolver("int", lambda x: int(x))
except Exception: pass
try:
    OmegaConf.register_new_resolver("len", lambda x: len(x))
except Exception: pass
try:
    OmegaConf.register_new_resolver("sum_list", lambda lst: sum(lst))
except Exception: pass


# =========== helpers ===========
def find_joint_indices(dof_names, patterns):
    result = {}
    for pat in patterns:
        result[pat] = [i for i, n in enumerate(dof_names) if pat in n]
    return result

def get_side(name):
    if 'left' in name: return 'L'
    if 'right' in name: return 'R'
    return 'C'

def pick_LR(indices, dof_names, joint_type=''):
    """Pick L and R indices from a list. Prefer pitch joints."""
    lefts = [i for i in indices if get_side(dof_names[i]) == 'L']
    rights = [i for i in indices if get_side(dof_names[i]) == 'R']
    if joint_type:
        lefts_p = [i for i in lefts if joint_type in dof_names[i]]
        rights_p = [i for i in rights if joint_type in dof_names[i]]
        if lefts_p: lefts = lefts_p
        if rights_p: rights = rights_p
    L = lefts[0] if lefts else indices[0]
    R = rights[0] if rights else indices[-1]
    return L, R

HEADER = [
    "step","mode",
    "root_z","root_vz","root_roll","root_pitch","root_yaw","root_ang_vel_z",
    "left_foot_force_z","right_foot_force_z","total_contact_force_z",
    "left_foot_height","right_foot_height","left_in_contact","right_in_contact",
    "action_knee_L","action_knee_R","target_knee_L","target_knee_R",
    "dof_pos_knee_L","dof_pos_knee_R","dof_vel_knee_L","dof_vel_knee_R",
    "torque_knee_L","torque_knee_R","torque_ratio_knee_L","torque_ratio_knee_R",
    "action_hip_L","action_hip_R","target_hip_L","target_hip_R",
    "dof_pos_hip_L","dof_pos_hip_R","torque_hip_L","torque_hip_R",
    "torque_ratio_hip_L","torque_ratio_hip_R",
    "action_ankle_L","action_ankle_R","target_ankle_L","target_ankle_R",
    "dof_pos_ankle_L","dof_pos_ankle_R","torque_ankle_L","torque_ankle_R",
    "torque_ratio_ankle_L","torque_ratio_ankle_R",
    "projected_gravity_x","projected_gravity_y","reset_buf",
]

def record_step(writer, data, step, mode, env):
    s = env.simulator; e = 0; tl = env.torque_limits
    rz   = s.robot_root_states[e,2].item()
    rvz  = s.robot_root_states[e,9].item()
    rpy  = env.rpy[e]
    rangz= s.robot_root_states[e,12].item()
    lfi  = env.feet_indices[0].item(); rfi = env.feet_indices[1].item()
    lfz  = s.contact_forces[e,lfi,2].item(); rfz = s.contact_forces[e,rfi,2].item()
    gz   = env.env_origins[e,2].item()
    lfh  = s._rigid_body_pos[e,lfi,2].item()-gz; rfh = s._rigid_body_pos[e,rfi,2].item()-gz
    lfc  = 1 if lfz>1.0 else 0; rfc = 1 if rfz>1.0 else 0
    kL,kR = data["kL"],data["kR"]; hL,hR = data["hL"],data["hR"]; aL,aR = data["aL"],data["aR"]
    row = [step,mode, rz,rvz, rpy[0].item(),rpy[1].item(),rpy[2].item(),rangz,
           lfz,rfz,lfz+rfz, lfh,rfh, lfc,rfc,
           env.actions[e,kL].item(),env.actions[e,kR].item(),
           data["pd_target"][e,kL].item(),data["pd_target"][e,kR].item(),
           s.dof_pos[e,kL].item(),s.dof_pos[e,kR].item(),
           s.dof_vel[e,kL].item(),s.dof_vel[e,kR].item(),
           env.torques[e,kL].item(),env.torques[e,kR].item(),
           abs(env.torques[e,kL].item())/max(tl[kL].item(),1e-6),
           abs(env.torques[e,kR].item())/max(tl[kR].item(),1e-6),
           env.actions[e,hL].item(),env.actions[e,hR].item(),
           data["pd_target"][e,hL].item(),data["pd_target"][e,hR].item(),
           s.dof_pos[e,hL].item(),s.dof_pos[e,hR].item(),
           env.torques[e,hL].item(),env.torques[e,hR].item(),
           abs(env.torques[e,hL].item())/max(tl[hL].item(),1e-6),
           abs(env.torques[e,hR].item())/max(tl[hR].item(),1e-6),
           env.actions[e,aL].item(),env.actions[e,aR].item(),
           data["pd_target"][e,aL].item(),data["pd_target"][e,aR].item(),
           s.dof_pos[e,aL].item(),s.dof_pos[e,aR].item(),
           env.torques[e,aL].item(),env.torques[e,aR].item(),
           abs(env.torques[e,aL].item())/max(tl[aL].item(),1e-6),
           abs(env.torques[e,aR].item())/max(tl[aR].item(),1e-6),
           env.projected_gravity[e,0].item(),env.projected_gravity[e,1].item(),
           env.reset_buf[e].item()]
    writer.writerow(row)

def print_step(data, step, mode, env):
    if step%5!=0: return
    s=env.simulator; e=0; lfi=env.feet_indices[0].item(); rfi=env.feet_indices[1].item()
    kL,kR = data["kL"],data["kR"]
    print(f"[JUMP_TEST] mode={mode:22s} step={step:3d} root_z={s.robot_root_states[e,2].item():.4f} "
          f"root_vz={s.robot_root_states[e,9].item():+.4f} "
          f"Fz_L={s.contact_forces[e,lfi,2].item():6.1f} Fz_R={s.contact_forces[e,rfi,2].item():6.1f} "
          f"τ_kL={env.torques[e,kL].item():+.1f} τ_kR={env.torques[e,kR].item():+.1f} "
          f"a_kL={env.actions[e,kL].item():+.2f} a_kR={env.actions[e,kR].item():+.2f}")

def reset_env(env):
    """Trigger full env reset."""
    env.reset_buf[:] = 1
    env_ids = env.reset_buf.nonzero(as_tuple=False).flatten()
    env.reset_envs_idx(env_ids)

def compute_summary(csv_path, mode_name):
    with open(csv_path,'r') as f:
        rows = list(csv.DictReader(f))
    if not rows: return {}
    rz=[float(r['root_z']) for r in rows]; rvz=[float(r['root_vz']) for r in rows]
    contact=[float(r['total_contact_force_z']) for r in rows]
    flight_steps=sum(1 for r in rows if int(r['left_in_contact'])==0 and int(r['right_in_contact'])==0)
    takeoff=rows[50:70] if len(rows)>70 else rows
    ktr=max(float(r['torque_ratio_knee_L']) for r in takeoff) if takeoff else 0
    ktr=max(ktr, max(float(r['torque_ratio_knee_R']) for r in takeoff)) if takeoff else 0
    atr=max(float(r['torque_ratio_ankle_L']) for r in takeoff) if takeoff else 0
    htr=max(float(r['torque_ratio_hip_L']) for r in takeoff) if takeoff else 0
    kact=np.mean([float(r['action_knee_L']) for r in takeoff]) if takeoff else 0
    lags=[abs(float(r['target_knee_L'])-float(r['dof_pos_knee_L'])) for r in rows]
    lags+=[abs(float(r['target_knee_R'])-float(r['dof_pos_knee_R'])) for r in rows]
    return {"mode":mode_name,"root_z_start":rz[0],"root_z_max":max(rz),"root_z_delta":max(rz)-rz[0],
            "root_vz_max":max(rvz),"max_total_contact_force_z":max(contact),
            "flight_steps":flight_steps,"both_feet_off_ground":flight_steps>0,
            "max_knee_torque_ratio":ktr,"max_ankle_torque_ratio":atr,"max_hip_torque_ratio":htr,
            "mean_takeoff_knee_action":kact,"max_dof_tracking_lag":max(lags) if lags else 0,
            "termination_happened":any(int(r['reset_buf'])==1 for r in rows)}

def make_plot(csv_path, png_path, title):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    with open(csv_path,'r') as f:
        rows=list(csv.DictReader(f))
    steps=[int(r['step']) for r in rows]
    rz=[float(r['root_z']) for r in rows]; rvz=[float(r['root_vz']) for r in rows]
    contact=[float(r['total_contact_force_z']) for r in rows]
    ktr=[max(float(r['torque_ratio_knee_L']),float(r['torque_ratio_knee_R'])) for r in rows]
    lc=[int(r['left_in_contact']) for r in rows]; rc=[int(r['right_in_contact']) for r in rows]
    fig,axes=plt.subplots(5,1,figsize=(10,12),sharex=True)
    fig.suptitle(title,fontsize=13)
    axes[0].plot(steps,rz,'b-'); axes[0].set_ylabel('root_z (m)'); axes[0].grid(True)
    axes[1].plot(steps,rvz,'r-'); axes[1].axhline(y=0,color='gray',ls='--'); axes[1].set_ylabel('root_vz (m/s)'); axes[1].grid(True)
    axes[2].plot(steps,contact,'g-'); axes[2].set_ylabel('total contact Fz (N)'); axes[2].grid(True)
    axes[3].plot(steps,ktr,'orange'); axes[3].axhline(y=1.0,color='red',ls='--',alpha=0.5)
    axes[3].set_ylabel('knee torque ratio'); axes[3].set_ylim(0,1.5); axes[3].grid(True)
    axes[4].fill_between(steps,0,1,where=np.array(lc)==1,alpha=0.4,color='blue',label='L contact')
    axes[4].fill_between(steps,0,1,where=np.array(rc)==1,alpha=0.4,color='red',label='R contact')
    axes[4].set_ylabel('foot contact'); axes[4].set_xlabel('step'); axes[4].legend(); axes[4].set_ylim(0,1.2)
    plt.tight_layout(); plt.savefig(png_path,dpi=120); plt.close()
    print(f"[PLOT] saved {png_path}")


# =========== main ===========
@hydra.main(config_path="/root/autodl-tmp/ASAP_official/humanoidverse/config", config_name="base", version_base="1.1")
def main(config: OmegaConf):
    import isaacgym  # noqa
    import torch

    # cd back to repo root (Hydra changes cwd)
    os.chdir(hydra.utils.get_original_cwd())

    outdir = Path("/root/autodl-tmp/ASAP_official/debug_outputs")
    outdir.mkdir(exist_ok=True, parents=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[SETUP] device={device}")

    # Force 1 env, headless, enable offscreen video recording
    config.env.config.num_envs = 1
    config.headless = True
    config.env.config.headless = True

    from humanoidverse.utils.helpers import pre_process_config
    pre_process_config(config)

    print("[SETUP] Instantiating env...")
    env = instantiate(config=config.env, device=device)

    # Offscreen frame capture (manual, no video writer dependency)
    from isaacgym import gymapi
    outdir_frames = outdir / "frames_action_space"
    outdir_frames.mkdir(exist_ok=True, parents=True)
    outdir_frames_b = outdir / "frames_direct_pd"
    outdir_frames_b.mkdir(exist_ok=True, parents=True)
    env._debug_capture_frame = False
    env._debug_frame_dir = outdir_frames
    env._debug_frame_idx = 0

    # ==================== PART 1: CONTROL PARAMS ====================
    print("\n"+"="*70)
    print("[Q1_CONTROL]")
    print("="*70)
    print(f"dof_names ({len(env.dof_names)}): {env.dof_names}")
    print(f"action_scale = {env.config.robot.control.action_scale}")
    print(f"control_type = {env.config.robot.control.control_type}")

    joint_idx = find_joint_indices(env.dof_names, ['knee','hip','ankle'])
    kL,kR = pick_LR(joint_idx['knee'], env.dof_names)
    hL,hR = pick_LR(joint_idx['hip'], env.dof_names, 'pitch')
    aL,aR = pick_LR(joint_idx['ankle'], env.dof_names, 'pitch')

    print(f"hip indices   = {joint_idx['hip']}  → L={hL} R={hR}")
    print(f"knee indices  = {joint_idx['knee']}  → L={kL} R={kR}")
    print(f"ankle indices = {joint_idx['ankle']}  → L={aL} R={aR}")

    for tag, idxs in [('hip',joint_idx['hip']),('knee',joint_idx['knee']),('ankle',joint_idx['ankle'])]:
        print(f"\n--- {tag} ---")
        for i in idxs:
            print(f"  {env.dof_names[i]:30s} default={env.default_dof_pos[0,i].item():+.3f}  "
                  f"kp={env.p_gains[i].item():.1f}  kd={env.d_gains[i].item():.2f}  "
                  f"τ_lim={env.torque_limits[i].item():.1f}  "
                  f"pos_lim=[{env.simulator.dof_pos_limits[i,0].item():.3f},{env.simulator.dof_pos_limits[i,1].item():.3f}]")

    print(f"\nfeet indices = {env.feet_indices.tolist()}")
    print(f"feet names   = {[env.simulator._body_list[i] for i in env.feet_indices.tolist()]}")
    print(f"terminate_min_base_height = {env.config.termination_scales.termination_min_base_height}")

    # ==================== PART 2: READ REFERENCE ====================
    print("\n"+"="*70)
    print("[REF_MOTION] Reading CR7 reference dof_pos...")

    reset_env(env)
    ml = env._motion_lib
    motion_len = float(ml.get_motion_length(env.motion_ids)[0])
    motion_dt = float(ml._motion_dt)
    num_frames = int(motion_len / motion_dt)
    print(f"  motion length={motion_len:.2f}s  dt={motion_dt:.3f}s  fps={1.0/motion_dt:.0f}  frames={num_frames}")

    motion_ids = torch.zeros(1, dtype=torch.long, device=device)
    offset = env.env_origins[:1]

    # Sample frames to find crouch (min root_z) and takeoff
    n_samples = min(100, int(num_frames * 0.7))
    sample_times = torch.linspace(motion_dt, motion_len * 0.7, n_samples, device=device)
    samples = {"root_z": [], "dof_pos": []}
    for t in sample_times:
        res = ml.get_motion_state(motion_ids, t.unsqueeze(0), offset=offset)
        samples["root_z"].append(res['root_pos'][0,2].item())
        samples["dof_pos"].append(res['dof_pos'][0].cpu().numpy())

    rz_arr = np.array(samples["root_z"])
    crouch_idx = int(np.argmin(rz_arr))
    crouch_dof = samples["dof_pos"][crouch_idx]
    crouch_time = float(sample_times[crouch_idx])

    # Takeoff: look where root_vz is max (rise), ~5-25 frames after crouch
    t_step = float(sample_times[1] - sample_times[0])
    rvz_arr = np.gradient(rz_arr, motion_dt * t_step)
    search_start = min(crouch_idx + 5, n_samples - 1)
    search_end = min(crouch_idx + 25, n_samples - 1)
    takeoff_idx = int(search_start + np.argmax(rvz_arr[search_start:search_end]))
    takeoff_dof = samples["dof_pos"][takeoff_idx]
    takeoff_time = float(sample_times[takeoff_idx])

    print(f"  crouch  idx={crouch_idx} t={crouch_time:.2f}s root_z={rz_arr[crouch_idx]:.4f} rvz={rvz_arr[crouch_idx]:+.3f}")
    print(f"  takeoff idx={takeoff_idx} t={takeoff_time:.2f}s root_z={rz_arr[takeoff_idx]:.4f} rvz={rvz_arr[takeoff_idx]:+.3f}")
    print(f"  root_z min={rz_arr.min():.4f} max={rz_arr.max():.4f}")

    default_np = env.default_dof_pos[0].cpu().numpy()
    ascale = env.config.robot.control.action_scale

    crouch_action = np.clip((crouch_dof - default_np) / ascale, -1.0, 1.0)
    extend_action = np.clip((takeoff_dof - default_np) / ascale, -1.0, 1.0)
    zero_action = np.zeros(env.dim_actions)

    # Clamp PD targets to joint limits
    lim_lo = env.simulator.dof_pos_limits[:,0].cpu().numpy()
    lim_hi = env.simulator.dof_pos_limits[:,1].cpu().numpy()
    crouch_target_raw = crouch_dof.copy()
    extend_target_raw = takeoff_dof.copy()
    crouch_target = np.clip(crouch_target_raw, lim_lo, lim_hi)
    extend_target = np.clip(extend_target_raw, lim_lo, lim_hi)
    default_target = np.clip(default_np, lim_lo, lim_hi)

    print("\n--- Key joint targets vs actions ---")
    for tag, idxs in [('hip',joint_idx['hip']),('knee',joint_idx['knee']),('ankle',joint_idx['ankle'])]:
        for i in idxs:
            print(f"  {env.dof_names[i]:30s} def={default_np[i]:+.3f} lim=[{lim_lo[i]:+.3f},{lim_hi[i]:+.3f}]  "
                  f"cr_dof={crouch_dof[i]:+.3f} cr_act={crouch_action[i]:+.2f}  "
                  f"ex_dof={takeoff_dof[i]:+.3f} ex_act={extend_action[i]:+.2f}  "
                  f"ex_target={extend_target[i]:+.3f}")

    # Warn about action clipping
    n_clipped_crouch = np.sum(np.abs(crouch_action) >= 0.99)
    n_clipped_extend = np.sum(np.abs(extend_action) >= 0.99)
    print(f"\n  crouch actions clipped: {n_clipped_crouch}/{env.dim_actions}")
    print(f"  extend actions clipped: {n_clipped_extend}/{env.dim_actions}")
    if n_clipped_extend > 4:
        print("  ⚠ MANY EXTEND ACTIONS CLIPPED → action_scale may be too small!")

    # ==================== MODE A ====================
    print("\n"+"="*70)
    print("[MODE_A] ACTION_SPACE_TEST")
    print("="*70)

    phases_A = [(0,20,zero_action,"stabilize"),(20,50,crouch_action,"crouch"),
                (50,70,extend_action,"extend"),(70,130,zero_action,"aerial/hold")]

    csv_A = outdir / "q1_manual_jump_action_space.csv"
    reset_env(env)
    pd_target = env.default_dof_pos.clone()
    data_A = {"kL":kL,"kR":kR,"hL":hL,"hR":hR,"aL":aL,"aR":aR,"pd_target":pd_target}

    with open(csv_A,'w',newline='') as f:
        w = csv.writer(f); w.writerow(HEADER); step=0
        for ss,ee,act_np,phase in phases_A:
            act_t = torch.tensor(act_np,dtype=torch.float,device=device).unsqueeze(0)
            for s in range(ss,ee):
                data_A["pd_target"] = env.default_dof_pos + env.actions * ascale
                env.step({"actions": act_t})
                print_step(data_A,step,f"A:{phase}",env)
                record_step(w,data_A,step,f"A:{phase}",env)
                step+=1
                if env.reset_buf[0].item()>0:
                    print(f"  [TERM] step {step} in {phase}")
                    break
    print(f"[CSV] {csv_A}")

    sum_A = compute_summary(str(csv_A),"ACTION_SPACE_TEST")
    print("\n[SUMMARY_ACTION_SPACE]")
    for k,v in sum_A.items(): print(f"  {k} = {v}")

    make_plot(str(csv_A),str(outdir/"q1_manual_jump_action_space.png"),"Q1 Manual Jump — Action Space Test")

    # ==================== MODE B ====================
    print("\n"+"="*70)
    print("[MODE_B] DIRECT_PD_TARGET_TEST")
    print("="*70)

    phases_B = [(0,20,default_target,"stabilize"),(20,50,crouch_target,"crouch"),
                (50,70,extend_target,"extend"),(70,130,default_target,"aerial/hold")]

    csv_B = outdir / "q1_manual_jump_direct_pd.csv"
    reset_env(env)
    data_B = {"kL":kL,"kR":kR,"hL":hL,"hR":hR,"aL":aL,"aR":aR,"pd_target":env.default_dof_pos.clone()}

    with open(csv_B,'w',newline='') as f:
        w = csv.writer(f); w.writerow(HEADER); step=0
        for ss,ee,tgt_np,phase in phases_B:
            tgt_t = torch.tensor(tgt_np,dtype=torch.float,device=device).unsqueeze(0)
            for s in range(ss,ee):
                env._debug_direct_pd_target = tgt_t
                data_B["pd_target"] = tgt_t.clone()
                env.step({"actions": torch.zeros(1,env.dim_actions,device=device)})
                print_step(data_B,step,f"B:{phase}",env)
                record_step(w,data_B,step,f"B:{phase}",env)
                step+=1
                if env.reset_buf[0].item()>0:
                    print(f"  [TERM] step {step} in {phase}")
                    break
    print(f"[CSV] {csv_B}")

    sum_B = compute_summary(str(csv_B),"DIRECT_PD_TARGET_TEST")
    print("\n[SUMMARY_DIRECT_PD]")
    for k,v in sum_B.items(): print(f"  {k} = {v}")

    make_plot(str(csv_B),str(outdir/"q1_manual_jump_direct_pd.png"),"Q1 Manual Jump — Direct PD Target Test")

    # ==================== FINAL DIAGNOSIS ====================
    print("\n"+"="*70)
    print("[DIAGNOSIS]")
    print("="*70)

    a_ok = sum_A["both_feet_off_ground"] or sum_A["root_vz_max"] > 0.2
    b_ok = sum_B["both_feet_off_ground"] or sum_B["root_vz_max"] > 0.2

    if not a_ok and b_ok:
        print("ACTION_SPACE can't jump, DIRECT_PD can → action_scale too restrictive.")
    elif not a_ok and not b_ok:
        if sum_B["max_knee_torque_ratio"] > 0.8:
            print("Neither mode jumps. Torque saturated → physics/contact/mass issue.")
        elif sum_B["max_knee_torque_ratio"] < 0.3:
            print("Neither mode jumps. Torque very low → target/joint mapping issue.")
        else:
            print(f"Neither mode jumps. Mixed issue (torque_ratio={sum_B['max_knee_torque_ratio']:.2f}).")
    else:
        print("Both modes can jump → NOT a physics problem. Focus on reward/obs/penalty.")

    print(f"\n  A: rzΔ={sum_A['root_z_delta']:.4f} vz_max={sum_A['root_vz_max']:.3f} flight={sum_A['both_feet_off_ground']} term={sum_A['termination_happened']}")
    print(f"  B: rzΔ={sum_B['root_z_delta']:.4f} vz_max={sum_B['root_vz_max']:.3f} flight={sum_B['both_feet_off_ground']} term={sum_B['termination_happened']}")
    print(f"\n  CSV: {csv_A}  |  {csv_B}")
    print(f"  PNG: {outdir/'q1_manual_jump_action_space.png'}  |  {outdir/'q1_manual_jump_direct_pd.png'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
