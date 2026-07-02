"""Minimal jump video capture — single env, manual action, PNG frames."""
import os, sys, cv2, numpy as np
from pathlib import Path

# -- hydra + env setup --
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import math
try:
    OmegaConf.register_new_resolver("eval", eval)
except: pass
try:
    OmegaConf.register_new_resolver("if", lambda p,a,b: a if p else b)
except: pass
try:
    OmegaConf.register_new_resolver("eq", lambda x,y: str(x).lower()==str(y).lower())
except: pass
try:
    OmegaConf.register_new_resolver("sqrt", lambda x: math.sqrt(float(x)))
except: pass
try:
    OmegaConf.register_new_resolver("sum", lambda x: sum(x))
except: pass
try:
    OmegaConf.register_new_resolver("ceil", lambda x: math.ceil(x))
except: pass
try:
    OmegaConf.register_new_resolver("int", lambda x: int(x))
except: pass
try:
    OmegaConf.register_new_resolver("len", lambda x: len(x))
except: pass
try:
    OmegaConf.register_new_resolver("sum_list", lambda l: sum(l))
except: pass

@hydra.main(config_path="/root/autodl-tmp/ASAP_official/humanoidverse/config", config_name="base", version_base="1.1")
def main(config):
    import isaacgym
    from isaacgym import gymapi
    import torch

    os.chdir(hydra.utils.get_original_cwd())
    outdir = Path("/root/autodl-tmp/ASAP_official/debug_outputs/frames")
    outdir.mkdir(exist_ok=True, parents=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.env.config.num_envs = 1
    config.headless = True
    config.env.config.headless = True

    from humanoidverse.utils.helpers import pre_process_config
    pre_process_config(config)

    env = instantiate(config=config.env, device=device)
    gym = env.simulator.gym
    sim = env.simulator.sim
    env_ptr = env.simulator.envs[0]

    # -- setup camera after env creation --
    cam_props = gymapi.CameraProperties()
    cam_props.width = 1280
    cam_props.height = 720
    cam = gym.create_camera_sensor(env_ptr, cam_props)
    print(f"[CAMERA] handle={cam}")
    if cam == -1:
        print("ERROR: Camera creation failed — no graphics pipeline. Try running without headless.")
        return
    gym.set_camera_location(cam, env_ptr, gymapi.Vec3(3, 2, 1.5), gymapi.Vec3(0, 0.6, 0.5))

    # -- grab reference crouch/takeoff --
    env.reset_buf[:] = 1
    env_ids = env.reset_buf.nonzero(as_tuple=False).flatten()
    env.reset_envs_idx(env_ids)
    ml = env._motion_lib
    motion_len = float(ml.get_motion_length(env.motion_ids)[0])
    motion_dt = float(ml._motion_dt)
    motion_ids = torch.zeros(1, dtype=torch.long, device=device)
    offset = env.env_origins[:1]

    n_samples = 100
    sample_times = torch.linspace(motion_dt, motion_len*0.7, n_samples, device=device)
    rz_samples = []
    dof_samples = []
    for t in sample_times:
        res = ml.get_motion_state(motion_ids, t.unsqueeze(0), offset=offset)
        rz_samples.append(res['root_pos'][0,2].item())
        dof_samples.append(res['dof_pos'][0].cpu().numpy())
    crouch_idx = int(np.argmin(rz_samples))
    crouch_dof = dof_samples[crouch_idx]
    takeoff_idx = min(crouch_idx+18, len(dof_samples)-1)
    takeoff_dof = dof_samples[takeoff_idx]

    default_np = env.default_dof_pos[0].cpu().numpy()
    ascale = env.config.robot.control.action_scale
    crouch_action = np.clip((crouch_dof - default_np)/ascale, -1, 1)
    extend_action = np.clip((takeoff_dof - default_np)/ascale, -1, 1)
    zero_action = np.zeros(env.dim_actions)

    # -- jump sequence: stabilize(20) + crouch(30) + extend(20) --
    phases = [
        (0, 20, zero_action),
        (20, 50, crouch_action),
        (50, 70, extend_action),
        (70, 90, zero_action),
    ]

    # Re-reset for clean start
    env.reset_envs_idx(env_ids)

    step = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = str(outdir / "q1_manual_jump.mp4")
    writer = cv2.VideoWriter(video_path, fourcc, 30, (1280, 720))

    for ss, ee, act_np in phases:
        act_t = torch.tensor(act_np, dtype=torch.float, device=device).unsqueeze(0)
        for s in range(ss, ee):
            env.step({"actions": act_t})
            # capture frame
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
            img = gym.get_camera_image(sim, env_ptr, cam, gymapi.IMAGE_COLOR)
            if img is not None and img.size > 0:
                frame = img.reshape(720, 1280, 4)[:, :, :3]  # RGBA → RGB
                writer.write(frame)
            step += 1
            rz = env.simulator.robot_root_states[0,2].item()
            print(f"[VIDEO] step={step} root_z={rz:.3f}", end='\r')
            if env.reset_buf[0].item() > 0:
                break

    writer.release()
    print(f"\n[DONE] Video: {video_path}  ({step} frames)")

if __name__ == "__main__":
    main()
