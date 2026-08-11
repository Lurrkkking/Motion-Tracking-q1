#!/usr/bin/env python3
"""Headless, ROS-free regression test for the Q1 stand + mimic policy handoff.

This deliberately imports the pure policy helpers and ``MujocoSimBackend`` from
sim2real_q1_motion_tracking_ros2.py so the observation/history layout remains
identical to the deployment path.  It never instantiates its ROS node.
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

# Renderer initialization must happen after this, so EGL works without DISPLAY.
os.environ.setdefault("MUJOCO_GL", "egl")


ROOT = Path(__file__).resolve().parents[1]
NUM_ACTIONS = 22
LOWER_BODY_INDICES = np.arange(0, 12, dtype=np.int64)
UPPER_BODY_INDICES = np.arange(12, 22, dtype=np.int64)
POST_HOLD_S = 0.5
POST_INTERP_S = 1.5
STATE_ORDER = (
    "HOLD_CURRENT", "GET_READY", "STAND_POLICY", "PRE_MIMIC_INTERP",
    "PRE_MIMIC_HOLD", "MIMIC_POLICY", "POST_MIMIC_HOLD",
    "POST_MIMIC_INTERP", "DONE",
)


def load_deployment_helpers():
    """Load only helpers; the source has optional ROS imports and needs no ROS install."""
    path = Path(__file__).resolve().with_name("sim2real_q1_motion_tracking_ros2.py")
    spec = importlib.util.spec_from_file_location("q1_deployment_helpers", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RuntimeError(f"cannot load deployment helpers from {path}: {exc}") from exc
    required = (
        "read_conf", "PolicyRuntime", "create_history", "get_obs",
        "load_onnx_policy", "infer_policy_action", "apply_action_postprocess",
        "compute_runtime_phase", "load_q1_joint_limits", "MujocoSimBackend",
        "infer_cycle_time_from_motion", "resolve_start_upper_pose",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"deployment helper module missing: {missing}")
    return module


def repo_path(value):
    value = Path(value)
    return str(value if value.is_absolute() else ROOT / value)


def finite(name, value):
    if not np.isfinite(value).all():
        raise RuntimeError(f"nonfinite {name}")


class FfmpegVideoWriter:
    """MuJoCo EGL frames encoded directly to MP4; no viewer or DISPLAY required."""
    def __init__(self, mujoco, live_model, live_data, dof_names, args):
        self.width, self.height = int(args.video_width), int(args.video_height)
        self.frames = 0
        self.mujoco = mujoco
        self.external_model = bool(args.render_mjcf)
        if self.external_model:
            if not Path(args.render_mjcf).is_file():
                raise FileNotFoundError(f"render MJCF not found: {args.render_mjcf}")
            self.render_model = mujoco.MjModel.from_xml_path(args.render_mjcf)
            self.render_data = mujoco.MjData(self.render_model)
            self.render_qposadr = {}
            for name in dof_names:
                try:
                    self.render_qposadr[name] = int(self.render_model.joint(name).qposadr[0])
                except KeyError as exc:
                    raise RuntimeError(f"render MJCF is missing policy joint {name}") from exc
            mujoco.mj_resetData(self.render_model, self.render_data)
            mujoco.mj_forward(self.render_model, self.render_data)
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.camera.lookat[:] = [0.0, 0.0, 0.45]
            self.camera.distance = 2.0
            self.camera.azimuth = 140.0
            self.camera.elevation = -20.0
        else:
            self.render_model, self.render_data = live_model, live_data
            self.render_qposadr = None
            if live_model.ncam:
                self.camera = "overview"
            else:
                self.camera = mujoco.MjvCamera()
                self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                self.camera.lookat[:] = [0.0, 0.0, 0.45]
                self.camera.distance = 2.0
                self.camera.azimuth = 140.0
                self.camera.elevation = -20.0
        self.dof_names = list(dof_names)
        Path(args.video_output).parent.mkdir(parents=True, exist_ok=True)
        command = [
            args.ffmpeg_path, "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", f"{self.width}x{self.height}",
            "-framerate", str(args.video_fps), "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", args.video_output,
        ]
        self.render_model.vis.global_.offwidth = max(int(self.render_model.vis.global_.offwidth), self.width)
        self.render_model.vis.global_.offheight = max(int(self.render_model.vis.global_.offheight), self.height)
        try:
            self.renderer = mujoco.Renderer(self.render_model, height=self.height, width=self.width)
            self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as exc:
            raise RuntimeError(f"cannot initialize EGL/FFmpeg video export: {exc}") from exc

    def write(self, live_data, dof_pos):
        if self.external_model:
            for idx, name in enumerate(self.dof_names):
                self.render_data.qpos[self.render_qposadr[name]] = float(dof_pos[idx])
            self.mujoco.mj_forward(self.render_model, self.render_data)
        else:
            self.render_data = live_data
        self.renderer.update_scene(self.render_data, camera=self.camera)
        frame = self.renderer.render()
        try:
            self.proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        except BrokenPipeError as exc:
            detail = self.proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg video pipe closed: {detail}") from exc
        self.frames += 1

    def close(self):
        if getattr(self, "proc", None) is None:
            return
        self.proc.stdin.close()
        detail = self.proc.stderr.read().decode("utf-8", errors="replace")
        code = self.proc.wait()
        self.renderer.close()
        self.proc = None
        if code:
            raise RuntimeError(f"FFmpeg exited with code {code}: {detail}")


class MujocoPhysicsBackend:
    """Free-root MuJoCo PD backend: ground contacts are solved by mj_step."""
    def __init__(self, mujoco, helper, args, cfg):
        if not Path(args.physics_mjcf).is_file():
            raise FileNotFoundError(f"physics MJCF not found: {args.physics_mjcf}")
        self.mujoco, self.helper, self.cfg = mujoco, helper, cfg
        self.model = mujoco.MjModel.from_xml_path(args.physics_mjcf)
        self.model.opt.timestep = float(args.timestep)
        self.model.opt.iterations = int(args.solver_iterations)
        self.model.opt.ls_iterations = int(args.solver_ls_iterations)
        self.data = mujoco.MjData(self.model)
        self.dof_names = list(cfg["dof_names"])
        self.joint_qposadr, self.joint_dofadr, self.actuator_ids = {}, {}, []
        for name in self.dof_names:
            try:
                joint = self.model.joint(name)
                actuator = self.model.actuator(name)
            except KeyError as exc:
                raise RuntimeError(f"physics MJCF is missing joint or motor {name}") from exc
            self.joint_qposadr[name] = int(joint.qposadr[0])
            self.joint_dofadr[name] = int(joint.dofadr[0])
            self.actuator_ids.append(int(actuator.id))
        self.root_qposadr = self.root_dofadr = None
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                self.root_qposadr = int(self.model.jnt_qposadr[joint_id])
                self.root_dofadr = int(self.model.jnt_dofadr[joint_id])
                break
        if self.root_qposadr is None:
            raise RuntimeError("physics MJCF needs a free root for ground-contact simulation")
        self.pelvis_body_id = int(self.model.body("pelvis").id)
        self.total_mass = float(self.model.body_mass.sum())
        self.assist_enabled = True
        self.assist_release_s = None
        self.assist_anchor = np.array([0.0, 0.0, float(args.root_height)], dtype=np.float64)
        self.assist_kp_pos = float(args.assist_kp_pos)
        self.assist_kd_pos = float(args.assist_kd_pos)
        self.assist_kp_rot = float(args.assist_kp_rot)
        self.assist_kd_rot = float(args.assist_kd_rot)
        self.max_contacts = 0
        self.min_root_z = float("inf")
        self.reset()

    def reset(self):
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.root_qposadr:self.root_qposadr + 3] = [0.0, 0.0, float(self.cfg.get("root_height", 0.42))]
        self.data.qpos[self.root_qposadr + 3:self.root_qposadr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        for idx, name in enumerate(self.dof_names):
            self.data.qpos[self.joint_qposadr[name]] = float(self.cfg["default_dof_pos"][idx])
        self.mujoco.mj_forward(self.model, self.data)

    def release_assist(self, sim_time):
        if self.assist_enabled:
            self.assist_enabled = False
            self.assist_release_s = float(sim_time)
            print(f"PHYSICS ASSIST released at {sim_time:.3f}s after first stand ONNX inference")

    def _apply_assist(self):
        self.data.xfrc_applied[:] = 0.0
        if not self.assist_enabled:
            return
        pos = np.asarray(self.data.xpos[self.pelvis_body_id], dtype=np.float64)
        linvel = np.asarray(self.data.qvel[self.root_dofadr:self.root_dofadr + 3], dtype=np.float64)
        quat = np.asarray(self.data.xquat[self.pelvis_body_id], dtype=np.float64)
        if quat[0] < 0.0:
            quat = -quat
        force = (self.assist_kp_pos * (self.assist_anchor - pos)
                 - self.assist_kd_pos * linvel
                 + np.array([0.0, 0.0, self.total_mass * 9.81]))
        force = np.clip(force, -1000.0, 1000.0)
        rot_error = 2.0 * quat[1:4]
        angvel = np.asarray(self.data.qvel[self.root_dofadr + 3:self.root_dofadr + 6], dtype=np.float64)
        torque = np.clip(-self.assist_kp_rot * rot_error - self.assist_kd_rot * angvel, -100.0, 100.0)
        self.data.xfrc_applied[self.pelvis_body_id, 0:3] = force
        self.data.xfrc_applied[self.pelvis_body_id, 3:6] = torque

    def step(self, target_dof_pos, kps, kds):
        target = np.asarray(target_dof_pos, dtype=np.float64).reshape(NUM_ACTIONS)
        kp = np.asarray(kps, dtype=np.float64).reshape(NUM_ACTIONS)
        kd = np.asarray(kds, dtype=np.float64).reshape(NUM_ACTIONS)
        q = np.array([self.data.qpos[self.joint_qposadr[name]] for name in self.dof_names])
        dq = np.array([self.data.qvel[self.joint_dofadr[name]] for name in self.dof_names])
        tau = kp * (target - q) - kd * dq
        for idx, actuator_id in enumerate(self.actuator_ids):
            self.data.ctrl[actuator_id] = tau[idx]
        self._apply_assist()
        self.mujoco.mj_step(self.model, self.data)
        self.max_contacts = max(self.max_contacts, int(self.data.ncon))
        self.min_root_z = min(self.min_root_z, float(self.data.qpos[self.root_qposadr + 2]))

    def snapshot(self):
        q = np.array([self.data.qpos[self.joint_qposadr[name]] for name in self.dof_names], dtype=np.float32)
        dq = np.array([self.data.qvel[self.joint_dofadr[name]] for name in self.dof_names], dtype=np.float32)
        quat = np.asarray(self.data.qpos[self.root_qposadr + 3:self.root_qposadr + 7], dtype=np.float64)
        ang = np.asarray(self.data.qvel[self.root_dofadr + 3:self.root_dofadr + 6], dtype=np.float32)
        gravity = self.helper.quat_rotate_inverse_wxyz(quat, [0.0, 0.0, -1.0]).astype(np.float32)
        return q, dq, ang, gravity


class HeadlessDualPolicyTest:
    def __init__(self, args, h):
        self.args, self.h = args, h
        self.stand_cfg = h.read_conf(args.stand_config)
        self.mimic_cfg = h.read_conf(args.mimic_config)
        self._validate_config()
        # Consume the same per-mimic transition settings as the ROS2 bridge.
        # The headless runner has no terminal keyboard, so stand_warmup_s remains
        # an explicit virtual '[' event used only to schedule a regression run.
        model_cfg = self.mimic_cfg.get("mimic_models", {}).get(args.mimic_name, {})
        for arg_name, yaml_key, default in (
            ("pre_interp_s", "pre_interp_s", 1.5),
            ("pre_hold_s", "pre_hold_s", 1.0),
            ("post_hold_s", "post_hold_s", 0.5),
            ("post_interp_s", "post_interp_s", 1.5),
        ):
            if getattr(args, arg_name) is None:
                setattr(args, arg_name, float(model_cfg.get(yaml_key, default)))
        if args.post_handoff_s <= 0.0 or args.post_handoff_s > args.post_hold_s:
            raise ValueError("--post-handoff-s must satisfy 0 < post-handoff-s <= post-hold-s")
        self.policy_dt = self.stand_cfg["simulation_dt"] * self.stand_cfg["control_decimation"]
        self.sim_dt = float(args.timestep)
        if not np.isclose(self.sim_dt, self.stand_cfg["simulation_dt"]):
            raise ValueError("--timestep must equal config simulation_dt to preserve policy phase semantics")
        if int(round(self.policy_dt / self.sim_dt)) != 10 or not np.isclose(self.policy_dt, 0.02):
            raise ValueError("first-pass headless test requires policy_dt=0.02 and exactly 10 MuJoCo steps/policy")
        self.steps_per_policy = 10
        self.lower, self.upper, self.limit_source = h.load_q1_joint_limits(
            self.stand_cfg["dof_names"], args.urdf, args.robot_yaml)
        self.limit_margin = float(args.joint_limit_margin_rad)
        self.clamp_limit_count = 0
        self.clamp_step_count = 0
        self.state = "HOLD_CURRENT"
        self.log_state = self.state
        self.state_enter_s = 0.0
        self.enter_times = {self.state: 0.0}
        self.stand_stage = 0
        self.last_error = ""
        self.failure_reason = ""
        self.nan_seen = False
        self.mimic_start_counter = None
        self.mimic_counter_zero_verified = False
        self.mimic_all_22_verified = True
        self.first_stand_onnx_done = False
        # A stand policy is not a safe shadow controller while the robot is
        # executing an unrelated mimic trajectory.  Its history/action may
        # leave the stand policy's training distribution, so it is reset at
        # the mimic-to-stand handoff instead.
        self.stand_shadow_paused_during_mimic = True
        self.stand_runtime_reentry_resets = 0
        self.prev_stand_counter = None
        self.last_final_target = self.stand_cfg["default_dof_pos"].copy()
        self.command_target = self.last_final_target.copy()
        self.last_command_step_max = 0.0
        self.max_command_step_seen = 0.0
        self.final_target = self.last_final_target.copy()
        self.pre_interp_start_upper = None
        self.post_hold_upper = None
        self.post_handoff_start_target = None
        self.handoff_start_lower_gap_max = 0.0
        self.handoff_max_single_tick_lower_delta = 0.0
        self.handoff_limit_clamp_count = 0
        self.post_hold_to_interp_lower_jump = 0.0
        self.upper_transition_jump = 0.0
        self.handoff_blend = 0.0
        self.handoff_raw_lower_delta = 0.0
        self.handoff_final_lower_delta = 0.0
        self.handoff_limited_mask = np.zeros(NUM_ACTIONS, dtype=bool)
        self.logs = defaultdict(list)
        self.times = {"stand": [], "mimic": [], "callback": []}
        self.max_abs_vel = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self._make_runtimes()
        backend_args = SimpleNamespace(
            urdf=args.urdf, mujoco_root_height=args.root_height,
            mujoco_free_root=args.free_root, mujoco_control_mode=args.control_mode,
            mujoco_kinematic_tau=args.kinematic_tau, mujoco_kp_scale=args.kp_scale,
            mujoco_kd_scale=args.kd_scale, mujoco_joint_armature=args.joint_armature,
            mujoco_joint_damping=args.joint_damping,
            mujoco_visual_geometry=args.visual_geometry, command_period_s=self.sim_dt,
        )
        if args.physics_mjcf:
            if args.control_mode != "pd" or not args.free_root:
                raise ValueError("--physics-mjcf requires --control-mode pd --free-root for real ground contact")
            self.stand_cfg["root_height"] = float(args.root_height)
            self.backend = MujocoPhysicsBackend(h.mujoco, h, args, self.stand_cfg)
        else:
            self.backend = h.MujocoSimBackend(backend_args, self.stand_cfg)
            self.backend.model.opt.timestep = self.sim_dt
        self.backend.reset()
        self.video = None
        self.video_frames = 0
        q, _, _, _ = self.backend.snapshot()
        self.latest_q = q
        self.latest_target = q.copy()
        self.command_target = q.copy()
        self.last_final_target = q.copy()

    def _validate_config(self):
        for role, cfg in (("stand", self.stand_cfg), ("mimic", self.mimic_cfg)):
            bad = []
            if cfg["num_actions"] != NUM_ACTIONS: bad.append("num_actions")
            if len(cfg["dof_names"]) != NUM_ACTIONS: bad.append("joint_names")
            for key in ("default_dof_pos", "kps", "kds"):
                if np.asarray(cfg[key]).shape != (NUM_ACTIONS,): bad.append(key)
            if bad:
                raise ValueError(f"{role} config invalid 22-DoF fields: {bad}")
            finite(f"{role} config vectors", np.concatenate([cfg["default_dof_pos"], cfg["kps"], cfg["kds"]]))
        if self.stand_cfg["dof_names"] != self.mimic_cfg["dof_names"]:
            raise ValueError("stand/mimic joint names and order must be identical")
        stand_cycle, stand_source = self.h.infer_cycle_time_from_motion(self.args.stand_motion_file)
        mimic_cycle, mimic_source = self.h.infer_cycle_time_from_motion(self.args.mimic_motion_file)
        if stand_cycle is not None: self.stand_cfg["cycle_time"] = float(stand_cycle)
        if mimic_cycle is not None: self.mimic_cfg["cycle_time"] = float(mimic_cycle)
        self.stand_cycle_source, self.mimic_cycle_source = stand_source, mimic_source
        for role, cfg in (("stand", self.stand_cfg), ("mimic", self.mimic_cfg)):
            if not np.isfinite(cfg["cycle_time"]) or cfg["cycle_time"] <= 0:
                raise ValueError(f"{role} motion cycle time must be positive")
        a = self.stand_cfg["simulation_dt"] * self.stand_cfg["control_decimation"]
        b = self.mimic_cfg["simulation_dt"] * self.mimic_cfg["control_decimation"]
        if not np.isclose(a, b): raise ValueError(f"stand/mimic policy_dt mismatch: {a} vs {b}")

    def _runtime(self, name, policy_path, motion_file, cfg, wrap):
        expected = (cfg["frame_stack"] + 1) * cfg["num_single_obs"]
        policy = self.h.load_onnx_policy(policy_path, expected, NUM_ACTIONS)
        runtime = self.h.PolicyRuntime(name=name, policy=policy, cfg=cfg, motion_file=motion_file,
            cycle_time=float(cfg["cycle_time"]), phase_wrap=wrap, policy_path=policy_path)
        self.reset_runtime(runtime)
        return runtime

    def _make_runtimes(self):
        self.stand = self._runtime("stand", self.args.stand_policy_path, self.args.stand_motion_file, self.stand_cfg, True)
        self.mimic = self._runtime("mimic", self.args.mimic_policy_path, self.args.mimic_motion_file, self.mimic_cfg, False)
        model_cfg = self.mimic_cfg.get("mimic_models", {}).get(self.args.mimic_name, {})
        self.mimic_start_upper, self.mimic_start_source = self.h.resolve_start_upper_pose(
            self.args.mimic_motion_file, self.mimic_cfg["dof_names"], UPPER_BODY_INDICES,
            model_cfg.get("start_upper_body_dof_pos"), model_cfg.get("motion_joint_order", self.mimic_cfg.get("motion_joint_order")))

    def reset_runtime(self, runtime):
        runtime.counter = 0
        runtime.last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        runtime.latest_raw_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        runtime.hist_dict, runtime.hist_obs = self.h.create_history(runtime.cfg)
        runtime.latest_target = getattr(self, "latest_q", runtime.cfg["default_dof_pos"]).copy()

    def transition(self, state, now, reason):
        if state == self.state: return
        old_state = self.state
        print(f"STATE {old_state} -> {state} at {now:.3f}s: {reason}")
        self.state, self.state_enter_s = state, now
        self.enter_times.setdefault(state, float(now))
        if state == "GET_READY":
            self.get_ready_start_q = self.latest_q.copy()
        elif state == "STAND_POLICY" and old_state == "HOLD_CURRENT":
            # Match version3's deliberate HOLD_CURRENT -> STAND_POLICY reset.
            self.reset_runtime(self.stand)
        elif state == "PRE_MIMIC_INTERP":
            # Version3 starts the blend from the actual 500 Hz-ramped command,
            # not from the un-ramped 50 Hz policy target.
            self.pre_interp_start_upper = self.command_target[UPPER_BODY_INDICES].copy()
        elif state == "MIMIC_POLICY":
            self.reset_runtime(self.mimic)
            self.mimic_start_counter = self.mimic.counter
            self.mimic_counter_zero_verified = self.mimic.counter == 0
        elif state == "POST_MIMIC_HOLD":
            # This is the 500 Hz-ramped target actually applied on the
            # preceding physics step, matching version3's handoff source.
            self.post_handoff_start_target = self.command_target.copy()
            self.post_hold_upper = self.post_handoff_start_target[UPPER_BODY_INDICES].copy()
            # Restart phase/history/action from the measured post-mimic state.
            # This prevents stale shadow rollout from saturating the stand ONNX.
            self.reset_runtime(self.stand)
            self.stand_runtime_reentry_resets += 1
            self.handoff_start_lower_gap_max = float(np.max(np.abs(
                self.post_handoff_start_target[LOWER_BODY_INDICES]
                - self.stand.latest_target[LOWER_BODY_INDICES]
            )))
            self.upper_transition_jump = float(np.max(np.abs(
                self.post_hold_upper - self.post_handoff_start_target[UPPER_BODY_INDICES]
            )))
            print(
                "POST handoff start (stand runtime re-primed): "
                f"start_lower={np.array2string(self.post_handoff_start_target[LOWER_BODY_INDICES], precision=3)} "
                f"stand_lower={np.array2string(self.stand.latest_target[LOWER_BODY_INDICES], precision=3)} "
                f"gap_max={self.handoff_start_lower_gap_max:.4f} post_handoff_s={self.args.post_handoff_s:.3f}"
            )

    def safe_target(self, target, reference):
        target = np.asarray(target, dtype=np.float32).reshape(NUM_ACTIONS).copy()
        finite("target", target)
        self.last_step_limited_mask = np.zeros(NUM_ACTIONS, dtype=bool)
        if self.args.max_target_step_rad > 0:
            before = target.copy()
            target = np.clip(target, reference - self.args.max_target_step_rad, reference + self.args.max_target_step_rad)
            self.last_step_limited_mask = np.abs(before - target) > 1e-7
            self.clamp_step_count += int(self.last_step_limited_mask.any())
        before = target.copy()
        target = np.clip(target, self.lower + self.limit_margin, self.upper - self.limit_margin)
        self.clamp_limit_count += int(not np.array_equal(before, target))
        return target.astype(np.float32)

    def clamp_command_target_to_limits(self, target):
        before = np.asarray(target, dtype=np.float32)
        clamped = np.clip(before, self.lower + self.limit_margin, self.upper - self.limit_margin)
        if not np.allclose(before, clamped):
            self.clamp_limit_count += 1
        return clamped.astype(np.float32)

    def update_command_ramp(self):
        # Same 500 Hz command ramp as version3.command_loop_step().
        desired = self.clamp_command_target_to_limits(self.latest_target)
        before = self.command_target.copy()
        max_step = float(self.args.max_command_step_rad)
        if max_step > 0.0:
            self.command_target = np.clip(desired, before - max_step, before + max_step).astype(np.float32)
        else:
            self.command_target = desired.copy()
        self.command_target = self.clamp_command_target_to_limits(self.command_target)
        self.last_command_step_max = float(np.max(np.abs(self.command_target - before)))
        self.max_command_step_seen = max(self.max_command_step_seen, self.last_command_step_max)

    def infer(self, runtime, state):
        obs, runtime.hist_obs = self.h.get_obs(runtime.hist_obs, runtime.hist_dict, state,
            runtime.last_action, runtime.counter, runtime.cfg, ref_motion_phase=self.h.compute_runtime_phase(runtime))
        finite(f"{runtime.name} obs", obs)
        t0 = time.perf_counter_ns()
        raw = self.h.infer_policy_action(runtime.policy, obs, runtime.cfg["clip_actions"])
        ms = (time.perf_counter_ns() - t0) / 1e6
        action, target = self.h.apply_action_postprocess(raw, runtime.last_action, runtime.latest_target, runtime.cfg, self.policy_dt)
        finite(f"{runtime.name} action", action); finite(f"{runtime.name} target", target)
        runtime.last_action, runtime.latest_raw_action, runtime.latest_target = action, raw, target
        runtime.counter += runtime.cfg["control_decimation"]
        return target.astype(np.float32), ms

    def policy_tick(self, now):
        active_state = self.state
        self.log_state = active_state
        q, dq, ang, gravity = self.backend.snapshot()
        state = {"dof_pos": q, "dof_vel": dq, "base_ang_vel": ang, "projected_gravity": gravity}
        for name, value in state.items(): finite(name, value)
        t0 = time.perf_counter_ns()
        if active_state == "MIMIC_POLICY":
            # Do not accumulate out-of-distribution mimic state in stand's
            # action/history buffers.  It is freshly re-primed at handoff.
            stand_raw, stand_ms = self.stand.latest_target.copy(), 0.0
        else:
            stand_raw, stand_ms = self.infer(self.stand, state)
        # Match deployment: compose with the current raw stand target, then apply target safety once to the final command.
        stand_target = stand_raw
        if active_state == "STAND_POLICY" and self.stand_stage == 0 and not self.first_stand_onnx_done:
            self.first_stand_onnx_done = True
            if hasattr(self.backend, "release_assist"):
                self.backend.release_assist(now)
        mimic_target = self.mimic.latest_target.copy()
        elapsed = now - self.state_enter_s
        if self.state == "STAND_POLICY":
            final = stand_target
        elif self.state == "PRE_MIMIC_INTERP":
            a = np.clip(elapsed / self.args.pre_interp_s, 0., 1.)
            final = stand_target.copy(); final[UPPER_BODY_INDICES] = (1-a)*self.pre_interp_start_upper + a*self.mimic_start_upper
            if a >= 1: self.transition("PRE_MIMIC_HOLD", now, "pre mimic interpolation complete")
        elif self.state == "PRE_MIMIC_HOLD":
            final = stand_target.copy(); final[UPPER_BODY_INDICES] = self.mimic_start_upper
            if elapsed >= self.args.pre_hold_s: self.transition("MIMIC_POLICY", now, "pre mimic hold complete")
        elif self.state == "MIMIC_POLICY":
            mimic_raw, mimic_ms = self.infer(self.mimic, state)
            mimic_target = self.safe_target(mimic_raw, self.latest_target)
            final = mimic_target.copy()  # all 22 DoF are mimic-owned here
            if not np.array_equal(final, mimic_target): self.mimic_all_22_verified = False
            if self.mimic.counter * self.mimic.cfg["simulation_dt"] >= self.mimic.cycle_time:
                self.transition("POST_MIMIC_HOLD", now, "mimic phase reached end")
        elif self.state == "POST_MIMIC_HOLD":
            if self.post_handoff_start_target is None:
                raise RuntimeError("POST_MIMIC_HOLD is missing handoff start target")
            u = float(np.clip(elapsed / self.args.post_handoff_s, 0., 1.))
            self.handoff_blend = 10.0*u**3 - 15.0*u**4 + 6.0*u**5
            final = stand_target.copy()
            final[LOWER_BODY_INDICES] = (
                (1.0 - self.handoff_blend) * self.post_handoff_start_target[LOWER_BODY_INDICES]
                + self.handoff_blend * stand_target[LOWER_BODY_INDICES]
            )
            final[UPPER_BODY_INDICES] = self.post_hold_upper
            if elapsed >= self.args.post_hold_s: self.transition("POST_MIMIC_INTERP", now, "post mimic hold complete")
        elif self.state == "POST_MIMIC_INTERP":
            a = np.clip(elapsed / self.args.post_interp_s, 0., 1.)
            final = stand_target.copy(); final[UPPER_BODY_INDICES] = (1-a)*self.post_hold_upper + a*stand_target[UPPER_BODY_INDICES]
            if a >= 1: self.transition("STAND_POLICY", now, "post mimic interpolation complete")
        else:
            raise RuntimeError(f"policy_tick in unexpected state {self.state}")
        mimic_ms = locals().get("mimic_ms", 0.0)
        previous_target = self.latest_target.copy()
        self.handoff_raw_lower_delta = float(np.max(np.abs(
            final[LOWER_BODY_INDICES] - previous_target[LOWER_BODY_INDICES]
        ))) if active_state == "POST_MIMIC_HOLD" else 0.0
        final = self.safe_target(final, previous_target)
        self.handoff_final_lower_delta = float(np.max(np.abs(
            final[LOWER_BODY_INDICES] - previous_target[LOWER_BODY_INDICES]
        ))) if active_state == "POST_MIMIC_HOLD" else 0.0
        self.handoff_limited_mask = self.last_step_limited_mask.copy() if active_state == "POST_MIMIC_HOLD" else np.zeros(NUM_ACTIONS, dtype=bool)
        if active_state == "POST_MIMIC_HOLD":
            self.handoff_max_single_tick_lower_delta = max(
                self.handoff_max_single_tick_lower_delta, self.handoff_final_lower_delta
            )
            self.handoff_limit_clamp_count += int(self.handoff_limited_mask[LOWER_BODY_INDICES].any())
            if elapsed >= self.args.post_hold_s:
                self.post_hold_to_interp_lower_jump = self.handoff_final_lower_delta
        if active_state == "MIMIC_POLICY" and not np.array_equal(final, mimic_target):
            raise RuntimeError("MIMIC_POLICY final_target differs from all-22 mimic_target after safety")
        self.prev_stand_counter = self.stand.counter
        self.final_target, self.latest_target = final, final.copy()
        self.times["stand"].append(stand_ms)
        if active_state == "MIMIC_POLICY": self.times["mimic"].append(mimic_ms)
        self.times["callback"].append((time.perf_counter_ns()-t0)/1e6)
        return stand_target, mimic_target, q, dq, ang, gravity, stand_ms, mimic_ms

    def log_step(self, now, stand_target, mimic_target, q, dq, ang, gravity, stand_ms, mimic_ms):
        finite("final_target", self.final_target)
        self.max_abs_vel = np.maximum(self.max_abs_vel, np.abs(dq))
        value = {
            "sim_time": now, "state_name": getattr(self, "log_state", self.state), "measured_q": q, "measured_dq": dq,
            "final_target": self.final_target, "command_target": self.command_target,
            "command_step_max": float(self.last_command_step_max),
            "stand_target": stand_target, "mimic_target": mimic_target,
            "stand_phase": self.h.compute_runtime_phase(self.stand), "mimic_phase": self.h.compute_runtime_phase(self.mimic),
            "stand_counter": self.stand.counter, "mimic_counter": self.mimic.counter,
            "target_delta_max": float(np.max(np.abs(self.final_target-self.last_final_target))),
            "handoff_blend": float(self.handoff_blend) if getattr(self, "log_state", self.state) == "POST_MIMIC_HOLD" else 0.0,
            "handoff_lower_raw_target_delta": float(self.handoff_raw_lower_delta),
            "handoff_lower_final_tick_delta": float(self.handoff_final_lower_delta),
            "handoff_limit_clamped": bool(self.handoff_limited_mask[LOWER_BODY_INDICES].any()),
            "projected_gravity": gravity, "base_ang_vel": ang, "stand_inference_ms": stand_ms,
            "mimic_inference_ms": mimic_ms, "policy_callback_ms": self.times["callback"][-1] if self.times["callback"] else 0.,
        }
        for key, item in value.items(): self.logs[key].append(np.asarray(item).copy() if isinstance(item, np.ndarray) else item)
        self.last_final_target = self.final_target.copy()

    def run(self):
        now, sim_step = 0.0, 0
        next_video_s = 0.0
        if self.args.video_output:
            self.video = FfmpegVideoWriter(self.h.mujoco, self.backend.model, self.backend.data, self.stand_cfg["dof_names"], self.args)
        zero = np.zeros(NUM_ACTIONS, dtype=np.float32)
        placeholder = self.latest_q.copy()
        latest = (placeholder, placeholder, *self.backend.snapshot(), 0., 0.)
        while self.state != "DONE":
            if sim_step % self.steps_per_policy == 0:
                if self.state == "HOLD_CURRENT" and now >= self.args.mujoco_start_delay_s:
                    self.transition("GET_READY", now, "mujoco auto get-ready")
                if self.state == "GET_READY":
                    a = np.clip((now-self.state_enter_s) / self.args.get_ready_duration_s, 0., 1.)
                    # version3 applies get-ready target safety relative to the
                    # ramped command target, then the command loop ramps again.
                    self.final_target = self.safe_target(
                        (1-a)*self.get_ready_start_q + a*self.stand_cfg["default_dof_pos"],
                        self.command_target,
                    )
                    self.latest_target = self.final_target.copy()
                    if a >= 1:
                        self.transition("HOLD_CURRENT", now, "get ready complete")
                        self.transition("STAND_POLICY", now, "mujoco auto start after get-ready")
                    latest = (placeholder, placeholder, *self.backend.snapshot(), 0., 0.)
                elif self.state == "HOLD_CURRENT":
                    self.final_target = self.latest_q.copy(); latest = (placeholder, placeholder, *self.backend.snapshot(), 0., 0.)
                else:
                    latest = self.policy_tick(now)
                    if self.state == "STAND_POLICY":
                        if self.stand_stage == 0 and now-self.state_enter_s >= self.args.stand_warmup_s:
                            self.stand_stage = 1; self.transition("PRE_MIMIC_INTERP", now, "automatic first stand interval complete")
                        elif self.stand_stage == 1 and now-self.state_enter_s >= self.args.post_stand_s:
                            self.transition("DONE", now, "automatic post-mimic stand interval complete")
            self.update_command_ramp()
            self.backend.step(self.command_target, self.stand_cfg["kps"], self.stand_cfg["kds"])
            self.latest_q, sim_dq, _, _ = self.backend.snapshot()
            if self.args.physics_mjcf and float(np.max(np.abs(sim_dq))) > self.args.physics_max_abs_dq:
                raise RuntimeError(f"physics instability: |dq|={float(np.max(np.abs(sim_dq))):.3f} > --physics-max-abs-dq {self.args.physics_max_abs_dq:.3f}")
            if self.video is not None and now + 1e-9 >= next_video_s:
                self.video.write(self.backend.data, self.latest_q)
                self.video_frames = self.video.frames
                next_video_s += 1.0 / self.args.video_fps
            self.log_step(now, *latest)
            sim_step += 1; now = sim_step*self.sim_dt
            if now > self.args.max_sim_s: raise RuntimeError(f"state machine timeout at {now:.3f}s")
        if self.video is not None:
            self.video.close()
            self.video = None
        return now

    def save(self, success):
        if self.video is not None:
            try:
                self.video.close()
                self.video_frames = self.video.frames
            except Exception as exc:
                if not self.failure_reason:
                    self.failure_reason = f"video finalization failed: {exc}"
            finally:
                self.video = None
        Path(self.args.output).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.args.output, **{k: np.asarray(v, dtype="U32") if k == "state_name" else np.asarray(v) for k,v in self.logs.items()})
        def stats(name):
            x = np.asarray(self.times[name], dtype=float)
            return {"avg_ms": float(x.mean()) if x.size else 0., "max_ms": float(x.max()) if x.size else 0., "count": int(x.size)}
        summary = {
            "success": bool(success), "reached_done": self.state == "DONE", "final_state": self.state,
            "failure_reason": self.failure_reason, "state_enter_time_s": {k: float(v) for k,v in self.enter_times.items()},
            "stand_cycle_time_s": float(self.stand.cycle_time), "mimic_cycle_time_s": float(self.mimic.cycle_time),
            "stand_cycle_source": self.stand_cycle_source, "mimic_cycle_source": self.mimic_cycle_source,
            "stand_onnx_io": [365, 22], "mimic_onnx_io": [365, 22], "policy_dt_s": float(self.policy_dt),
            "max_target_jump_rad": float(max((np.max(x) for x in np.abs(np.diff(np.asarray(self.logs["final_target"]), axis=0))), default=0.)),
            "max_command_step_rad": float(self.args.max_command_step_rad),
            "max_observed_command_step_rad": float(self.max_command_step_seen),
            "max_joint_velocity_rad_s": [float(x) for x in self.max_abs_vel], "inference_ms": {k: stats(k) for k in self.times},
            "nan_seen": bool(self.nan_seen), "joint_limit_clamp_count": int(self.clamp_limit_count), "target_step_clamp_count": int(self.clamp_step_count),
            "handoff_duration": float(self.args.post_handoff_s),
            "handoff_start_lower_gap_max": float(self.handoff_start_lower_gap_max),
            "handoff_max_single_tick_lower_delta": float(self.handoff_max_single_tick_lower_delta),
            "handoff_limit_clamp_count": int(self.handoff_limit_clamp_count),
            "post_hold_to_interp_lower_jump": float(self.post_hold_to_interp_lower_jump),
            "upper_transition_jump": float(self.upper_transition_jump),
            "mimic_counter_started_at_zero": bool(self.mimic_counter_zero_verified),
            "stand_shadow_paused_during_mimic": bool(self.stand_shadow_paused_during_mimic),
            "stand_runtime_reentry_resets": int(self.stand_runtime_reentry_resets),
            "mimic_strictly_controls_all_22_dof": bool(self.mimic_all_22_verified),
            "fixed_root": not self.args.free_root, "control_mode": self.args.control_mode,
            "physics_mjcf": self.args.physics_mjcf, "max_ground_contacts": int(getattr(self.backend, "max_contacts", 0)),
            "stand_assist_used": bool(self.args.physics_mjcf), "stand_assist_released_after_first_onnx": bool(self.first_stand_onnx_done),
            "stand_assist_release_time_s": getattr(self.backend, "assist_release_s", None),
            "min_root_z_m": float(getattr(self.backend, "min_root_z", self.args.root_height)),
            "ros_imported": bool(sys.modules.get("rclpy")), "output": self.args.output,
            "video_output": self.args.video_output, "render_mjcf": self.args.render_mjcf, "video_fps": int(self.args.video_fps), "video_frames": int(self.video_frames),
        }
        Path(self.args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        with open(self.args.summary_output, "w", encoding="utf-8") as f: yaml.safe_dump(summary, f, sort_keys=False)
        return summary


def parse_args():
    p = argparse.ArgumentParser(description="ROS-free headless Q1 dual-policy MuJoCo regression")
    p.add_argument("--stand-policy-path", required=True); p.add_argument("--stand-motion-file", required=True); p.add_argument("--stand-config", required=True)
    p.add_argument("--mimic-policy-path", required=True); p.add_argument("--mimic-motion-file", required=True); p.add_argument("--mimic-config", required=True)
    p.add_argument("--urdf", required=True); p.add_argument("--robot-yaml", required=True)
    p.add_argument("--output", default="logs/q1_dual_policy_headless.npz"); p.add_argument("--summary-output", default="logs/q1_dual_policy_headless_summary.yaml")
    p.add_argument("--video-output", default=""); p.add_argument("--render-mjcf", default=""); p.add_argument("--physics-mjcf", default=""); p.add_argument("--video-fps", type=int, default=30); p.add_argument("--video-width", type=int, default=960); p.add_argument("--video-height", type=int, default=720); p.add_argument("--ffmpeg-path", default="ffmpeg")
    p.add_argument("--mujoco-start-delay-s", type=float, default=1.0); p.add_argument("--get-ready-duration-s", type=float, default=10.0)
    p.add_argument("--stand-warmup-s", type=float, default=5.0, help="virtual '[' event after this much stand time; headless has no keyboard")
    p.add_argument("--post-stand-s", type=float, default=5.0, help="regression-only terminal stand window; version3 itself keeps running")
    p.add_argument("--pre-interp-s", type=float, default=None); p.add_argument("--pre-hold-s", type=float, default=None)
    p.add_argument("--post-hold-s", type=float, default=None); p.add_argument("--post-interp-s", type=float, default=None)
    p.add_argument("--post-handoff-s", type=float, default=0.5)
    p.add_argument("--control-mode", choices=("kinematic", "pd"), default="kinematic"); p.add_argument("--free-root", action="store_true")
    p.add_argument("--solver-iterations", type=int, default=100); p.add_argument("--solver-ls-iterations", type=int, default=50)
    p.add_argument("--assist-kp-pos", type=float, default=300.0); p.add_argument("--assist-kd-pos", type=float, default=100.0); p.add_argument("--assist-kp-rot", type=float, default=50.0); p.add_argument("--assist-kd-rot", type=float, default=10.0); p.add_argument("--physics-max-abs-dq", type=float, default=100.0)
    p.add_argument("--timestep", type=float, default=0.002); p.add_argument("--root-height", type=float, default=0.42); p.add_argument("--kinematic-tau", type=float, default=0.04)
    p.add_argument("--kp-scale", type=float, default=0.05); p.add_argument("--kd-scale", type=float, default=1.0); p.add_argument("--joint-armature", type=float, default=0.01); p.add_argument("--joint-damping", type=float, default=0.2)
    p.add_argument("--visual-geometry", choices=("auto", "mesh", "simplified"), default="auto"); p.add_argument("--joint-limit-margin-rad", type=float, default=0.02); p.add_argument("--max-target-step-rad", type=float, default=0.2); p.add_argument("--max-command-step-rad", type=float, default=0.01); p.add_argument("--mimic-name", default="lateral_raise"); p.add_argument("--max-sim-s", type=float, default=120.)
    a = p.parse_args()
    for key in ("stand_policy_path", "stand_motion_file", "stand_config", "mimic_policy_path", "mimic_motion_file", "mimic_config", "urdf", "robot_yaml", "output", "summary_output"):
        setattr(a, key, repo_path(getattr(a, key)) )
    if a.video_output:
        a.video_output = repo_path(a.video_output)
    if a.render_mjcf:
        a.render_mjcf = repo_path(a.render_mjcf)
    if a.physics_mjcf:
        a.physics_mjcf = repo_path(a.physics_mjcf)
    if a.video_fps <= 0 or a.video_width <= 0 or a.video_height <= 0:
        raise ValueError("--video-fps, --video-width, and --video-height must be positive")
    if a.mujoco_start_delay_s < 0.0 or a.get_ready_duration_s <= 0.0:
        raise ValueError("--mujoco-start-delay-s must be >= 0 and --get-ready-duration-s must be > 0")
    if a.max_command_step_rad <= 0.0:
        raise ValueError("--max-command-step-rad must be > 0")
    if a.post_handoff_s <= 0.0:
        raise ValueError("--post-handoff-s must be > 0")
    if a.physics_mjcf:
        print("INFO: free-root PD physics/contact mode enabled; evaluate dynamic stability separately.")
    elif a.free_root or a.control_mode != "kinematic":
        print("WARNING: only fixed-root + kinematic is the first-pass validated mode.")
    return a


def main():
    args, test, success = parse_args(), None, False
    try:
        h = load_deployment_helpers()
        test = HeadlessDualPolicyTest(args, h)
        print("Running ROS-free MuJoCo dual-policy state machine (no viewer, no sleep).")
        test.run(); success = True
    except Exception as exc:
        message = f"HEADLESS DUAL-POLICY TEST FAILED: {type(exc).__name__}: {exc}"
        print(message, file=sys.stderr)
        if test is not None: test.failure_reason = message
    finally:
        if test is not None:
            summary = test.save(success)
            print(f"summary={args.summary_output} output={args.output}")
            print(f"result: success={summary['success']} final_state={summary['final_state']} max_target_jump_rad={summary['max_target_jump_rad']:.6f}")
    if not success: sys.exit(1)


if __name__ == "__main__":
    main()
