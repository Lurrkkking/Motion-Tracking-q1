#!/usr/bin/env python3
import os; os.environ['MUJOCO_GL']='egl'
import sys; sys.path.insert(0,'.')
import mujoco, numpy as np, onnxruntime, traceback
from scripts.q1_mujoco_sim2sim import *
from pathlib import Path

cfg = read_config('scripts/q1_sim2sim_config.yaml')
cfg.control_dt = cfg.simulation_dt * cfg.control_decimation  # computed, not in yaml
base = Path('.').resolve()

print(f'cfg: decim={cfg.control_decimation} action_scale={cfg.action_scale} cycle={cfg.cycle_time} cdt={cfg.control_dt}', flush=True)

ref_dof_all, ref_dvel_all, ref_root_all, ref_rvel_all, fps, mdt = load_motion(str(base/'humanoidverse/data/motions/q1/q1_cr7_scale045_rootz042.pkl'))
model = mujoco.MjModel.from_xml_path(str(base/'scripts/q1_motion_tracking_mj.xml'))
model.opt.timestep = cfg.simulation_dt
model.opt.iterations = 100; model.opt.ls_iterations = 50
data = mujoco.MjData(model)
qpos_a, qvel_a, act_a = build_joint_maps(model)

set_dof_pos(data, qpos_a, cfg.default_dof_pos)
data.qpos[:3]=[0.,0.,0.42]; data.qpos[3:7]=[1.,0.,0.,0.]; data.qvel[:]=0.
mujoco.mj_forward(model, data)
min_z=min(data.geom_xpos[i][2] for i in range(model.ngeom) if model.geom_contype[i]!=0)
if abs(min_z)>5e-4: data.qpos[2]-=min_z; mujoco.mj_forward(model, data)

print(f'init z={data.qpos[2]:.3f}', flush=True)

sess=onnxruntime.InferenceSession(str(base/'logs/Q1_CR7/20260608_143745-Q1_CR7_Tracking_v1-q1_cr7_motion_tracking-q1_22dof_box/exported/model_5500.onnx'))
iname=sess.get_inputs()[0].name; oname=sess.get_outputs()[0].name
print(f'ONNX loaded', flush=True)

target=cfg.default_dof_pos.copy(); action=np.zeros(22,dtype=np.float32)
hist={k:np.zeros((HIST_DEPTH,COMP_DIM[k]),dtype=np.float32) for k in HIST_KEYS}
tau_lim=cfg.tau_limit; kps=cfg.kps; kds=cfg.kds
cycle=cfg.cycle_time; cdt=cfg.control_dt; decim=cfg.control_decimation

for si in range(10):
    try:
        mj=get_state(data,qpos_a,qvel_a)
        tau=kps*(target-mj[2])-kds*mj[3]; tau=np.clip(tau,-tau_lim,tau_lim)
        apply_torques(data,act_a,tau)
        for _ in range(decim):
            mujoco.mj_step(model,data)
            mj2=get_state(data,qpos_a,qvel_a)
            tau2=kps*(target-mj2[2])-kds*mj2[3]; tau2=np.clip(tau2,-tau_lim,tau_lim)
            apply_torques(data,act_a,tau2)

        base_ang,proj_g,dp,dv,base_q,ang_w,lin_w=get_state(data,qpos_a,qvel_a)
        # DAMPEN velocity feedback for first 5 steps to prevent initial oscillation
        if si < 5:
            dv = dv * 0.1
        t_sim=(si+1)*cdt; phase=min(t_sim/cycle,1.0); tm=min(t_sim,cycle-1e-6)
        ref_dof=interp(ref_dof_all,tm,mdt); ref_root=interp(ref_root_all,tm,mdt)
        act_z=float(data.qpos[2]); act_yaw=yaw_xyzw(base_q); ye=0.0-act_yaw
        cr,to,fl,la=compute_phase_masks(phase); lc,rc=get_foot_contact(data,model)

        p={}; p['base_ang_vel']=base_ang*0.25; p['projected_gravity']=proj_g
        p['dof_pos']=dp-cfg.default_dof_pos; p['dof_vel']=dv*0.05; p['actions']=action
        p['ref_motion_phase']=np.array([phase],dtype=np.float32)
        p['q1_root_error']=np.array([ref_root[2]-act_z,0.,act_z,lin_w[2]],dtype=np.float32)
        p['q1_yaw_error']=np.array([np.sin(ye),np.cos(ye),-ang_w[2],ang_w[2]],dtype=np.float32)
        p['q1_flight_phase']=np.array([float(cr),float(to),float(fl),float(la),float(lc),float(rc)],dtype=np.float32)
        p['q1_ref_dof_error']=np.concatenate([ref_dof-dp,np.zeros(22)],dtype=np.float32)
        p['history_actor']=np.concatenate([hist[k].flatten() for k in HIST_KEYS])
        hv=dict(base_ang_vel=base_ang,projected_gravity=proj_g,dof_pos=dp-cfg.default_dof_pos,dof_vel=dv,actions=action,ref_motion_phase=np.array([phase],dtype=np.float32))
        for k in HIST_KEYS: hist[k][1:]=hist[k][:-1]; hist[k][0]=hv[k].astype(np.float32)
        obs=np.clip(np.concatenate([p[k] for k in OBS_ORDER]).astype(np.float32),-100,100)

        raw=sess.run([oname],{iname:obs.reshape(1,-1)})[0][0]
        raw=np.clip(raw,-cfg.clip_actions,cfg.clip_actions)
        action=raw; target=action*cfg.action_scale+cfg.default_dof_pos

        kni,hip,ank=3,0,4
        print(f's={si+1:2d} z={act_z:.3f} knee(tgt={target[kni]:.2f} cur={dp[kni]:.2f} ref={ref_dof[kni]:.2f} a={action[kni]:.2f}) hip(cur={dp[hip]:.2f}) τmax={np.abs(tau).max():.0f} amax={np.abs(action).max():.1f} dvmax={np.abs(dv).max():.1f}', flush=True)
    except Exception as e:
        print(f'ERROR at step {si}: {e}', flush=True)
        traceback.print_exc()
        break

print('done', flush=True)
