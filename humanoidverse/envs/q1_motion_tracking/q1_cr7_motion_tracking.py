"""
Q1 CR7 Motion Tracking env — dedicated subclass of LeggedRobotMotionTracking.

Provides:
- Strict Q1 dof/body mapping validation
- Phase-aware termination (crouch/takeoff/flight/landing)
- Q1-specific obs getters (root_z, root_vz, yaw, flight phase)
- Q1-specific reward functions (root tracking, flight contact, joint weighting)
"""

import torch
import numpy as np
from humanoidverse.envs.motion_tracking.motion_tracking import LeggedRobotMotionTracking
from isaac_utils.rotations import (
    quat_to_angle_axis, calc_heading_quat, calc_heading_quat_inv,
    quat_rotate_inverse,
)
from loguru import logger
from termcolor import colored


class Q1CR7MotionTracking(LeggedRobotMotionTracking):
    """Q1 CR7 specialized motion tracking env.

    Inherits all base motion tracking logic from LeggedRobotMotionTracking.
    Only overrides what's necessary for Q1 CR7 jump task.
    """

    def __init__(self, config, device):
        # Q1-specific phase config (defaults, overridden by config if present)
        # config is the inner env.config dict (already unwrapped by Hydra instantiate)
        self._q1_cr7_config = dict(getattr(config, 'q1_cr7', {}))
        self._init_q1_cr7_defaults()

        super().__init__(config, device)

    def _init_q1_cr7_defaults(self):
        """Set Q1 CR7 defaults if not provided in config."""
        defaults = {
            'takeoff_phase_start': 0.38,
            'takeoff_phase_end': 0.50,
            'flight_phase_start': 0.50,
            'flight_phase_end': 0.72,
            'landing_phase_start': 0.72,
            'landing_phase_end': 0.88,
            'crouch_phase_start': 0.20,
            'crouch_phase_end': 0.45,
            'reference_relative_termination': True,
            'crouch_height_margin': 0.18,
            'flight_height_margin': 0.18,
            'landing_height_margin': 0.12,
            'catastrophic_root_z': 0.08,
            'auto_infer_phase': False,
        }
        for k, v in defaults.items():
            if k not in self._q1_cr7_config:
                self._q1_cr7_config[k] = v

    # ------------------------------------------------------------------
    #  Mapping validation
    # ------------------------------------------------------------------

    def _init_tracking_config(self):
        super()._init_tracking_config()
        self._validate_q1_mapping()

    def _validate_q1_mapping(self):
        """Strict validation of Q1 dof/body/motion mapping. Raises ValueError on mismatch."""
        cfg = self.config
        robot_cfg = cfg.robot
        motion_cfg = robot_cfg.motion

        sim = self.simulator
        num_dofs = sim.num_dof
        body_list = sim._body_list

        # --- 1. DOF count ---
        assert num_dofs == 22, f"Q1 CR7 requires 22 dofs, got {num_dofs}"

        # --- 2. Check all configured dof/body names exist in simulator ---
        def _check_names(names, label, source_list):
            missing = [n for n in names if n not in source_list]
            if missing:
                raise ValueError(
                    f"[Q1CR7_MAPPING] {label} not found in simulator: {missing}"
                )

        # robot.dof_names are JOINT names — check against simulator dof_names
        dof_names_cfg = list(robot_cfg.dof_names)
        _check_names(dof_names_cfg, "config robot.dof_names", sim.dof_names)

        # robot.body_names are body/link names — check against simulator _body_list
        body_names_cfg = list(robot_cfg.body_names)
        _check_names(body_names_cfg, "config robot.body_names", body_list)

        # motion.dof_names are skeleton node names (link-based, not joint-based)
        # They are used by MotionLib for skeleton mapping, not for simulator DOF matching.
        # We still log a mismatch warning if they differ from robot.dof_names.
        motion_dof_names = list(motion_cfg.dof_names)
        logger.info(f"  motion.dof_names count={len(motion_dof_names)} (skeleton nodes, not checked against sim joints)")

        # motion.body_names are body/link names — check against simulator _body_list
        motion_body_names = list(motion_cfg.body_names)
        _check_names(motion_body_names, "config robot.motion.body_names", body_list)

        # --- 3. Check tracking/body link groups ---
        for link_group, group_name in [
            (motion_cfg.get('lower_body_link', []), 'lower_body_link'),
            (motion_cfg.get('upper_body_link', []), 'upper_body_link'),
            (motion_cfg.get('motion_tracking_link', []), 'motion_tracking_link'),
        ]:
            _check_names(link_group, group_name, body_list)

        # --- 4. Check feet ---
        left_foot = robot_cfg.get('left_foot_name', 'left_ankle_roll_link')
        right_foot = robot_cfg.get('right_foot_name', 'right_ankle_roll_link')
        _check_names([left_foot, right_foot], 'feet_names', body_list)

        # --- 5. Check pelvis/torso ---
        pelvis_name = motion_cfg.get('pelvis_link', 'pelvis')
        torso_name = motion_cfg.get('base_link', 'torso_link')
        _check_names([pelvis_name, torso_name], 'pelvis/torso', body_list)

        # --- 6. Collect indices ---
        self.q1_feet_indices = torch.tensor(
            [body_list.index(left_foot), body_list.index(right_foot)],
            dtype=torch.long, device=self.device
        )

        self.q1_lower_body_indices = [
            body_list.index(n) for n in motion_cfg.get('lower_body_link', [])
        ]
        self.q1_upper_body_indices = [
            body_list.index(n) for n in motion_cfg.get('upper_body_link', [])
        ]
        self.q1_motion_tracking_indices = [
            body_list.index(n) for n in motion_cfg.get('motion_tracking_link', [])
        ]
        self.q1_pelvis_index = body_list.index(pelvis_name)
        self.q1_torso_index = body_list.index(torso_name)

        # --- 7. Joint groups for lower-body reward weighting ---
        # Build index maps for hip/knee/ankle/waist
        self.q1_knee_indices = []
        self.q1_ankle_indices = []
        self.q1_hip_indices = []
        self.q1_waist_indices = []

        knee_patterns = ['knee']
        ankle_patterns = ['ankle']
        hip_patterns = ['hip']
        waist_patterns = ['waist']

        for i, name in enumerate(sim.dof_names):
            if any(p in name for p in knee_patterns):
                self.q1_knee_indices.append(i)
            elif any(p in name for p in ankle_patterns):
                self.q1_ankle_indices.append(i)
            elif any(p in name for p in hip_patterns):
                self.q1_hip_indices.append(i)
            elif any(p in name for p in waist_patterns):
                self.q1_waist_indices.append(i)

        self.q1_knee_indices = torch.tensor(self.q1_knee_indices, dtype=torch.long, device=self.device)
        self.q1_ankle_indices = torch.tensor(self.q1_ankle_indices, dtype=torch.long, device=self.device)
        self.q1_hip_indices = torch.tensor(self.q1_hip_indices, dtype=torch.long, device=self.device)
        self.q1_waist_indices = torch.tensor(self.q1_waist_indices, dtype=torch.long, device=self.device)

        # The shoot-specific reward tracks every actuated joint on the right
        # leg: hip pitch/roll/yaw, knee, and ankle pitch/roll.  Resolve these
        # by name so the reward remains correct if the simulator DOF order is
        # changed in the future.
        right_leg_dof_names = [
            'right_hip_pitch_joint', 'right_hip_roll_joint',
            'right_hip_yaw_joint', 'right_knee_joint',
            'right_ankle_pitch_joint', 'right_ankle_roll_joint',
        ]
        _check_names(right_leg_dof_names, 'right_leg_dof_names', sim.dof_names)
        self.q1_right_leg_indices = torch.tensor(
            [sim.dof_names.index(name) for name in right_leg_dof_names],
            dtype=torch.long, device=self.device,
        )

        # GK-dive-specific reward tracks all six LEFT-leg joints (the reaching
        # leg for q1_gk_low_left_2_extend).  Resolved by name like the right.
        left_leg_dof_names = [
            'left_hip_pitch_joint', 'left_hip_roll_joint',
            'left_hip_yaw_joint', 'left_knee_joint',
            'left_ankle_pitch_joint', 'left_ankle_roll_joint',
        ]
        _check_names(left_leg_dof_names, 'left_leg_dof_names', sim.dof_names)
        self.q1_left_leg_indices = torch.tensor(
            [sim.dof_names.index(name) for name in left_leg_dof_names],
            dtype=torch.long, device=self.device,
        )

        # --- 8. Build lower-body joint weight vector ---
        self.q1_lower_body_joint_weights = torch.ones(num_dofs, device=self.device)
        knee_w = 2.0
        ankle_w = 1.5
        hip_w = 1.5
        hip_yaw_w = 1.0
        waist_w = 1.0

        for i, name in enumerate(sim.dof_names):
            if 'knee' in name:
                self.q1_lower_body_joint_weights[i] = knee_w
            elif 'ankle_pitch' in name:
                self.q1_lower_body_joint_weights[i] = ankle_w
            elif 'ankle_roll' in name:
                self.q1_lower_body_joint_weights[i] = ankle_w
            elif 'hip_pitch' in name:
                self.q1_lower_body_joint_weights[i] = hip_w
            elif 'hip_roll' in name:
                self.q1_lower_body_joint_weights[i] = hip_w
            elif 'hip_yaw' in name:
                self.q1_lower_body_joint_weights[i] = hip_yaw_w
            elif 'waist' in name:
                self.q1_lower_body_joint_weights[i] = waist_w
            elif 'shoulder' in name or 'elbow' in name:
                self.q1_lower_body_joint_weights[i] = 0.3  # low weight for upper body

        # Strictly the two legs: six hip DOFs, two knees and four ankles.
        # Unlike q1_lower_body_joint_weights this excludes waist and arms,
        # because the CR7 event reward must concentrate on takeoff/landing.
        self.q1_leg_joint_indices = torch.cat([
            self.q1_hip_indices, self.q1_knee_indices, self.q1_ankle_indices,
        ])
        self.q1_leg_joint_weights = self.q1_lower_body_joint_weights[
            self.q1_leg_joint_indices
        ]

        # --- Print mapping summary ---
        logger.info(colored("[Q1CR7_MAPPING]", "cyan"))
        logger.info(f"  num_dofs={num_dofs}")
        logger.info(f"  num_bodies={len(body_list)}")
        logger.info(f"  num_extend_bodies={getattr(self, 'num_extend_bodies', 0)}")
        logger.info(f"  dof_names={sim.dof_names}")
        logger.info(f"  feet_indices={self.q1_feet_indices.tolist()}")
        logger.info(f"  lower_body_indices={self.q1_lower_body_indices}")
        logger.info(f"  upper_body_indices={self.q1_upper_body_indices}")
        logger.info(f"  motion_tracking_indices={self.q1_motion_tracking_indices}")
        logger.info(f"  pelvis_index={self.q1_pelvis_index}")
        logger.info(f"  torso_index={self.q1_torso_index}")
        logger.info(f"  knee_indices={self.q1_knee_indices.tolist()}")
        logger.info(f"  ankle_indices={self.q1_ankle_indices.tolist()}")
        logger.info(f"  hip_indices={self.q1_hip_indices.tolist()}")
        logger.info(f"  waist_indices={self.q1_waist_indices.tolist()}")
        logger.info(f"  right_leg_indices={self.q1_right_leg_indices.tolist()}")
        logger.info(f"  leg_joint_indices={self.q1_leg_joint_indices.tolist()}")
        logger.info(colored("[Q1CR7_MAPPING] Validation PASSED", "green"))

    # ------------------------------------------------------------------
    #  Phase helpers
    # ------------------------------------------------------------------

    def _get_phase_masks(self):
        """Return boolean masks for each motion phase based on reference phase."""
        phase = self._ref_motion_phase.squeeze(-1)  # [num_envs] in [0, 1]
        cfg = self._q1_cr7_config

        crouch = (phase >= cfg['crouch_phase_start']) & (phase < cfg['crouch_phase_end'])
        takeoff = (phase >= cfg['takeoff_phase_start']) & (phase < cfg['takeoff_phase_end'])
        flight = (phase >= cfg['flight_phase_start']) & (phase < cfg['flight_phase_end'])
        landing = (phase >= cfg['landing_phase_start']) & (phase < cfg['landing_phase_end'])

        # Handle overlap: priority takeoff > flight > landing > crouch
        flight = flight & ~takeoff
        landing = landing & ~takeoff & ~flight
        crouch = crouch & ~takeoff & ~flight & ~landing

        # After landing: phase >= landing_phase_end
        after_landing = phase >= cfg['landing_phase_end']

        return {
            'crouch': crouch,
            'takeoff': takeoff,
            'flight': flight,
            'landing': landing,
            'after_landing': after_landing,
        }

    def _get_q1_cr7_phase_masks(self):
        """Public alias — returns is_crouch, is_takeoff, is_flight, is_landing, is_after_landing."""
        return self._get_phase_masks()

    def _get_q1_shoot_phase_weight(self):
        """Smooth [0, 1] gate for the shoot-specific right-leg rewards.

        The settings live in the selected reward YAML, so the generic Q1
        tracking configuration carries no shoot-motion assumption.
        """
        cfg = self.config.rewards.get('q1_shoot_phase', {})
        start = float(cfg.get('start', 0.50))
        end = float(cfg.get('end', 0.72))
        transition = float(cfg.get('transition', 0.02))
        if not 0.0 <= start < end <= 1.0:
            raise ValueError(f"Invalid q1_shoot_phase window: start={start}, end={end}")

        phase = self._ref_motion_phase.squeeze(-1)
        if transition <= 0.0:
            return ((phase >= start) & (phase <= end)).float()

        rise = ((phase - start) / transition).clamp(min=0.0, max=1.0)
        fall = ((end - phase) / transition).clamp(min=0.0, max=1.0)
        return rise * fall

    def _get_q1_gk_dive_phase_weight(self):
        """Smooth [0, 1] gate for the GK-dive-specific left-leg rewards.

        Settings live in the selected reward YAML (q1_gk_dive_phase), so the
        generic Q1 tracking configuration carries no dive assumption.
        """
        cfg = self.config.rewards.get('q1_gk_dive_phase', {})
        start = float(cfg.get('start', 0.10))
        end = float(cfg.get('end', 0.35))
        transition = float(cfg.get('transition', 0.02))
        if not 0.0 <= start < end <= 1.0:
            raise ValueError(f"Invalid q1_gk_dive_phase window: start={start}, end={end}")

        phase = self._ref_motion_phase.squeeze(-1)
        if transition <= 0.0:
            return ((phase >= start) & (phase <= end)).float()

        rise = ((phase - start) / transition).clamp(min=0.0, max=1.0)
        fall = ((end - phase) / transition).clamp(min=0.0, max=1.0)
        return rise * fall

    def _get_q1_cr7_jump_phase_weight(self):
        """Smooth gate for the crouch-to-landing event of a CR7 jump motion."""
        cfg = self.config.rewards.get('q1_cr7_jump_phase', {})
        start = float(cfg.get('start', 0.22))
        end = float(cfg.get('end', 0.62))
        transition = float(cfg.get('transition', 0.02))
        if not 0.0 <= start < end <= 1.0:
            raise ValueError(f"Invalid q1_cr7_jump_phase window: start={start}, end={end}")

        phase = self._ref_motion_phase.squeeze(-1)
        if transition <= 0.0:
            return ((phase >= start) & (phase <= end)).float()

        rise = ((phase - start) / transition).clamp(min=0.0, max=1.0)
        fall = ((end - phase) / transition).clamp(min=0.0, max=1.0)
        return rise * fall

    # ------------------------------------------------------------------
    #  Pre-compute observations — add Q1 CR7-specific buffers
    # ------------------------------------------------------------------

    def _pre_compute_observations_callback(self):
        super()._pre_compute_observations_callback()

        # --- Motion reference data ---
        offset = self.env_origins
        motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times
        motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)

        # Runtime sanity: motion dof count must match simulator
        if not hasattr(self, '_q1_motion_dof_checked') or not self._q1_motion_dof_checked:
            m_dof = motion_res['dof_pos'].shape[1]
            assert m_dof == self.num_dofs, \
                f"Motion dof_pos dim {m_dof} != simulator dof count {self.num_dofs}"
            self._q1_motion_dof_checked = True

        # --- Root state ---
        # actual
        self.q1_actual_root_pos = self.simulator.robot_root_states[:, 0:3].clone()
        self.q1_actual_root_z = self.q1_actual_root_pos[:, 2]
        self.q1_actual_root_rot = self.simulator.robot_root_states[:, 3:7].clone()  # xyzw
        self.q1_actual_root_vel = self.simulator.robot_root_states[:, 7:10].clone()
        self.q1_actual_root_vz = self.q1_actual_root_vel[:, 2]
        self.q1_actual_root_ang_vel = self.simulator.robot_root_states[:, 10:13].clone()
        self.q1_actual_yaw_rate = self.q1_actual_root_ang_vel[:, 2]

        # reference
        self.q1_ref_root_pos = motion_res['root_pos'].clone()
        self.q1_ref_root_z = self.q1_ref_root_pos[:, 2]
        self.q1_ref_root_rot = motion_res['root_rot'].clone()  # xyzw
        self.q1_ref_root_vel = motion_res['root_vel'].clone()
        self.q1_ref_root_vz = self.q1_ref_root_vel[:, 2]
        self.q1_ref_root_ang_vel = motion_res['root_ang_vel'].clone()
        self.q1_ref_yaw_rate = self.q1_ref_root_ang_vel[:, 2]

        # --- Yaw computation ---
        self.q1_actual_yaw = self._compute_yaw_from_quat_xyzw(self.q1_actual_root_rot)
        self.q1_ref_yaw = self._compute_yaw_from_quat_xyzw(self.q1_ref_root_rot)

        # Yaw error (sin/cos representation)
        yaw_err = self.q1_ref_yaw - self.q1_actual_yaw
        self.q1_yaw_error_sin = torch.sin(yaw_err)
        self.q1_yaw_error_cos = torch.cos(yaw_err)
        self.q1_yaw_error_rad = torch.atan2(self.q1_yaw_error_sin, self.q1_yaw_error_cos)

        # --- Root error ---
        self.q1_root_z_error = self.q1_ref_root_z - self.q1_actual_root_z
        self.q1_root_vz_error = self.q1_ref_root_vz - self.q1_actual_root_vz
        self.q1_yaw_rate_error = self.q1_ref_yaw_rate - self.q1_actual_yaw_rate

        # --- Foot contact detection ---
        contact_forces = self.simulator.contact_forces[:, self.q1_feet_indices, :]
        contact_force_norm = torch.norm(contact_forces, dim=-1)  # [num_envs, 2]
        self.q1_foot_contact = contact_force_norm > 1.0  # threshold 1N
        self.q1_left_contact = self.q1_foot_contact[:, 0]
        self.q1_right_contact = self.q1_foot_contact[:, 1]
        self.q1_any_contact = self.q1_left_contact | self.q1_right_contact
        self.q1_both_feet_air = ~self.q1_left_contact & ~self.q1_right_contact

        # --- Phase masks ---
        self.q1_phase_masks = self._get_phase_masks()

        # --- DOF error ---
        self.q1_ref_dof_pos = motion_res['dof_pos'].clone()
        self.q1_ref_dof_vel = motion_res['dof_vel'].clone()
        self.q1_dof_pos_error = self.q1_ref_dof_pos - self.simulator.dof_pos
        self.q1_dof_vel_error = self.q1_ref_dof_vel - self.simulator.dof_vel

        # --- Logging ---
        self._log_q1_debug_info()

    @staticmethod
    def _compute_yaw_from_quat_xyzw(q_xyzw):
        """Compute yaw angle from xyzw quaternion."""
        # q = [x, y, z, w]
        w = q_xyzw[:, 3]
        z = q_xyzw[:, 2]
        x = q_xyzw[:, 0]
        y = q_xyzw[:, 1]
        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

    def _log_q1_debug_info(self):
        """Minimal Q1-specific debug log."""
        pass

    # ------------------------------------------------------------------
    #  Phase-aware termination
    # ------------------------------------------------------------------

    def _check_termination(self):
        """Q1 CR7 termination — config-controlled + catastrophic.

        Standard termination (gravity/height/contact etc.) controlled by
        env.config.termination.* — can be overridden from CLI.

        Additionally, always terminates on:
        - root_z < 0.0 — pelvis below ground, truly broken
        - Motion end / episode length timeout
        """
        # Standard config-controlled termination (reads self.config.termination.*)
        self.reset_buf[:] = 0
        self.time_out_buf[:] = 0
        self._update_reset_buf()  # parent: gravity, height, contact, dof_pos, etc.
        self._update_timeout_buf()
        self.reset_buf |= self.time_out_buf

        # Extra catastrophic guard: pelvis below ground always resets
        terminate_catastrophic = self.q1_actual_root_z < 0.0
        self.reset_buf |= terminate_catastrophic

        # Log
        self.log_dict["q1_terminate_low_height"] = (self.reset_buf & ~terminate_catastrophic & ~self.time_out_buf).float().mean()
        self.log_dict["q1_terminate_gravity"] = torch.zeros(1, device=self.device)
        self.log_dict["q1_terminate_contact"] = torch.zeros(1, device=self.device)
        self.log_dict["q1_terminate_catastrophic"] = terminate_catastrophic.float().mean()
        self.log_dict["q1_terminate_motion_end"] = (self.time_out_buf & ~self.reset_buf).float().mean()

    def _update_timeout_buf(self):
        """Timeout from episode length + motion end."""
        self.time_out_buf |= self.episode_length_buf > self.max_episode_length
        if hasattr(self.config.termination, 'terminate_when_motion_end') and \
           self.config.termination.terminate_when_motion_end:
            current_time = (self.episode_length_buf) * self.dt + self.motion_start_times
            self.time_out_buf |= current_time > self.motion_len

    # ------------------------------------------------------------------
    #  Obs getters for Q1 CR7-specific observations
    # ------------------------------------------------------------------

    def _get_obs_q1_root_error(self):
        """Root z and vz error + actual values."""
        return torch.stack([
            self.q1_root_z_error,
            self.q1_root_vz_error,
            self.q1_actual_root_z,
            self.q1_actual_root_vz,
        ], dim=-1)

    def _get_obs_q1_yaw_error(self):
        """Yaw error (sin/cos) + yaw rate error + actual yaw rate."""
        return torch.stack([
            self.q1_yaw_error_sin,
            self.q1_yaw_error_cos,
            self.q1_yaw_rate_error,
            self.q1_actual_yaw_rate,
        ], dim=-1)

    def _get_obs_q1_flight_phase(self):
        """Phase indicators + foot contact status."""
        masks = self.q1_phase_masks
        return torch.stack([
            masks['crouch'].float(),
            masks['takeoff'].float(),
            masks['flight'].float(),
            masks['landing'].float(),
            self.q1_left_contact.float(),
            self.q1_right_contact.float(),
        ], dim=-1)

    def _get_obs_q1_ref_dof_error(self):
        """Reference DOF error (pos + vel)."""
        return torch.cat([
            self.q1_dof_pos_error,
            self.q1_dof_vel_error,
        ], dim=-1)

    # ------------------------------------------------------------------
    #  Q1 CR7 Reward functions — v2: generic frame-by-frame joint tracking
    # ------------------------------------------------------------------
    # Active (in reward_scales):
    #   q1_joint_position_tracking, q1_joint_velocity_tracking,
    #   q1_lower_body_joint_position_tracking, q1_lower_body_joint_velocity_tracking
    #
    # Inactive / removed from reward_scales (kept for possible future use):
    #   q1_root_z_tracking, q1_root_vz_tracking, q1_foot_max_height_tracking,
    #   q1_root_yaw_tracking, q1_root_yaw_rate_tracking, q1_flight_no_contact,
    #   q1_stance_contact, q1_takeoff_height_progress, q1_landing_stability
    # ------------------------------------------------------------------

    # ---- Active: generic joint tracking ----

    def _reward_q1_joint_position_tracking(self):
        """All 22 joints — uniform position tracking.  exp(-mse / sigma)."""
        err = self.dif_joint_angles  # ref - actual, [N, 22]
        mse = torch.mean(err ** 2, dim=-1)  # [N]
        sigma = self.config.rewards.reward_tracking_sigma.q1_joint_pos
        return torch.exp(-mse / sigma)

    def _reward_q1_joint_velocity_tracking(self):
        """All 22 joints — uniform velocity tracking.  exp(-mse / sigma)."""
        err = self.dif_joint_velocities  # ref - actual, [N, 22]
        mse = torch.mean(err ** 2, dim=-1)  # [N]
        sigma = self.config.rewards.reward_tracking_sigma.q1_joint_vel
        return torch.exp(-mse / sigma)

    def _reward_q1_lower_body_joint_position_tracking(self):
        """Lower-body + waist joints — weighted position tracking.  exp(-weighted_mse / sigma)."""
        weights = self.q1_lower_body_joint_weights  # [22]
        err = self.dif_joint_angles  # [N, 22]
        weighted_sq = weights * err ** 2
        weighted_mse = weighted_sq.sum(dim=-1) / (weights.sum() + 1e-8)  # [N]
        sigma = self.config.rewards.reward_tracking_sigma.q1_lower_body_joint_pos
        return torch.exp(-weighted_mse / sigma)

    def _reward_q1_lower_body_joint_velocity_tracking(self):
        """Lower-body + waist joints — weighted velocity tracking.  exp(-weighted_mse / sigma)."""
        weights = self.q1_lower_body_joint_weights  # [22]
        err = self.dif_joint_velocities  # [N, 22]
        weighted_sq = weights * err ** 2
        weighted_mse = weighted_sq.sum(dim=-1) / (weights.sum() + 1e-8)  # [N]
        sigma = self.config.rewards.reward_tracking_sigma.q1_lower_body_joint_vel
        return torch.exp(-weighted_mse / sigma)

    def _reward_q1_shoot_right_leg_position_tracking(self):
        """Track all six right-leg joints during the shooting swing only."""
        err = self.dif_joint_angles[:, self.q1_right_leg_indices]
        mse = torch.mean(err ** 2, dim=-1)
        sigma = self.config.rewards.reward_tracking_sigma.q1_shoot_right_leg_pos
        return self._get_q1_shoot_phase_weight() * torch.exp(-mse / sigma)

    def _reward_q1_shoot_right_leg_velocity_tracking(self):
        """Track all six right-leg joint velocities during the shooting swing."""
        err = self.dif_joint_velocities[:, self.q1_right_leg_indices]
        mse = torch.mean(err ** 2, dim=-1)
        sigma = self.config.rewards.reward_tracking_sigma.q1_shoot_right_leg_vel
        return self._get_q1_shoot_phase_weight() * torch.exp(-mse / sigma)

    def _reward_q1_shoot_right_foot_height_tracking(self):
        """Match the reference right-ankle world height during the shooting swing."""
        right_foot_z_err = self.dif_global_body_pos[:, self.q1_feet_indices[1], 2]
        sigma = self.config.rewards.reward_tracking_sigma.q1_shoot_right_foot_z
        return self._get_q1_shoot_phase_weight() * torch.exp(
            -(right_foot_z_err ** 2) / sigma
        )

    def _reward_q1_shoot_left_leg_position_tracking(self):
        """Track all six left-leg joints during the shooting swing only."""
        err = self.dif_joint_angles[:, self.q1_left_leg_indices]
        mse = torch.mean(err ** 2, dim=-1)
        sigma = self.config.rewards.reward_tracking_sigma.q1_shoot_left_leg_pos
        return self._get_q1_shoot_phase_weight() * torch.exp(-mse / sigma)

    def _reward_q1_shoot_left_leg_velocity_tracking(self):
        """Track all six left-leg joint velocities during the shooting swing."""
        err = self.dif_joint_velocities[:, self.q1_left_leg_indices]
        mse = torch.mean(err ** 2, dim=-1)
        sigma = self.config.rewards.reward_tracking_sigma.q1_shoot_left_leg_vel
        return self._get_q1_shoot_phase_weight() * torch.exp(-mse / sigma)

    def _reward_q1_shoot_left_foot_height_tracking(self):
        """Match the reference left-ankle world height during the shooting swing."""
        left_foot_z_err = self.dif_global_body_pos[:, self.q1_feet_indices[0], 2]
        sigma = self.config.rewards.reward_tracking_sigma.q1_shoot_left_foot_z
        return self._get_q1_shoot_phase_weight() * torch.exp(
            -(left_foot_z_err ** 2) / sigma
        )

    def _reward_q1_gk_dive_left_leg_position_tracking(self):
        """Track all six left-leg joints during the goalkeeper dive window only."""
        err = self.dif_joint_angles[:, self.q1_left_leg_indices]
        mse = torch.mean(err ** 2, dim=-1)
        sigma = self.config.rewards.reward_tracking_sigma.q1_gk_dive_left_leg_pos
        return self._get_q1_gk_dive_phase_weight() * torch.exp(-mse / sigma)

    def _reward_q1_gk_dive_left_leg_velocity_tracking(self):
        """Track all six left-leg joint velocities during the goalkeeper dive window."""
        err = self.dif_joint_velocities[:, self.q1_left_leg_indices]
        mse = torch.mean(err ** 2, dim=-1)
        sigma = self.config.rewards.reward_tracking_sigma.q1_gk_dive_left_leg_vel
        return self._get_q1_gk_dive_phase_weight() * torch.exp(-mse / sigma)

    def _reward_q1_gk_dive_left_foot_position_tracking(self):
        """Match the reference left-ankle world position (reach) during the dive."""
        left_foot_err = self.dif_global_body_pos[:, self.q1_feet_indices[0], :]
        mse = torch.mean(left_foot_err ** 2, dim=-1)
        sigma = self.config.rewards.reward_tracking_sigma.q1_gk_dive_left_foot_pos
        return self._get_q1_gk_dive_phase_weight() * torch.exp(-mse / sigma)

    def _reward_q1_gk_dive_right_leg_position_tracking(self):
        """Track all six right-leg joints during the goalkeeper dive window only."""
        err = self.dif_joint_angles[:, self.q1_right_leg_indices]
        mse = torch.mean(err ** 2, dim=-1)
        sigma = self.config.rewards.reward_tracking_sigma.q1_gk_dive_right_leg_pos
        return self._get_q1_gk_dive_phase_weight() * torch.exp(-mse / sigma)

    def _reward_q1_gk_dive_right_leg_velocity_tracking(self):
        """Track all six right-leg joint velocities during the goalkeeper dive window."""
        err = self.dif_joint_velocities[:, self.q1_right_leg_indices]
        mse = torch.mean(err ** 2, dim=-1)
        sigma = self.config.rewards.reward_tracking_sigma.q1_gk_dive_right_leg_vel
        return self._get_q1_gk_dive_phase_weight() * torch.exp(-mse / sigma)

    def _reward_q1_gk_dive_right_foot_position_tracking(self):
        """Match the reference right-ankle world position (reach) during the dive."""
        right_foot_err = self.dif_global_body_pos[:, self.q1_feet_indices[1], :]
        mse = torch.mean(right_foot_err ** 2, dim=-1)
        sigma = self.config.rewards.reward_tracking_sigma.q1_gk_dive_right_foot_pos
        return self._get_q1_gk_dive_phase_weight() * torch.exp(-mse / sigma)

    def _reward_q1_cr7_root_z_tracking(self):
        """Track reference pelvis height for the whole motion, including crouch."""
        sigma = self.config.rewards.reward_tracking_sigma.q1_cr7_root_z
        return torch.exp(-(self.q1_root_z_error ** 2) / sigma)

    def _reward_q1_cr7_jump_root_z_tracking(self):
        """Increase root-height tracking emphasis from crouch through landing."""
        sigma = self.config.rewards.reward_tracking_sigma.q1_cr7_root_z
        tracking = torch.exp(-(self.q1_root_z_error ** 2) / sigma)
        return self._get_q1_cr7_jump_phase_weight() * tracking

    def _reward_q1_cr7_root_vz_tracking(self):
        """Track signed reference vertical velocity, with extra jump-window weight."""
        cfg = self.config.rewards.q1_cr7_jump_phase
        outside_weight = float(cfg.get('outside_vz_weight', 0.25))
        jump_weight = float(cfg.get('vz_weight', 1.5))
        phase_weight = self._get_q1_cr7_jump_phase_weight()
        weight = outside_weight + (jump_weight - outside_weight) * phase_weight
        sigma = self.config.rewards.reward_tracking_sigma.q1_cr7_root_vz
        return weight * torch.exp(-(self.q1_root_vz_error ** 2) / sigma)

    def _reward_q1_cr7_jump_leg_position_tracking(self):
        """Strong bilateral hip/knee/ankle position tracking in the jump window."""
        err = self.dif_joint_angles[:, self.q1_leg_joint_indices]
        weights = self.q1_leg_joint_weights
        weighted_mse = (weights * err.square()).sum(dim=-1) / (weights.sum() + 1e-8)
        sigma = self.config.rewards.reward_tracking_sigma.q1_cr7_jump_leg_pos
        return self._get_q1_cr7_jump_phase_weight() * torch.exp(-weighted_mse / sigma)

    def _reward_q1_cr7_jump_leg_velocity_tracking(self):
        """Strong bilateral hip/knee/ankle velocity tracking in the jump window."""
        err = self.dif_joint_velocities[:, self.q1_leg_joint_indices]
        weights = self.q1_leg_joint_weights
        weighted_mse = (weights * err.square()).sum(dim=-1) / (weights.sum() + 1e-8)
        sigma = self.config.rewards.reward_tracking_sigma.q1_cr7_jump_leg_vel
        return self._get_q1_cr7_jump_phase_weight() * torch.exp(-weighted_mse / sigma)

    def _reward_q1_cr7_landing_feet_level(self):
        """Penalize foot roll/pitch only for feet contacting during landing."""
        feet_quat = self.simulator._rigid_body_rot[:, self.q1_feet_indices, :]
        gravity = self.gravity_vec[:, None, :].expand(-1, 2, -1)
        feet_gravity = quat_rotate_inverse(
            feet_quat.reshape(-1, 4), gravity.reshape(-1, 3), w_last=True
        ).reshape(self.num_envs, 2, 3)
        tilt = torch.norm(feet_gravity[:, :, :2], dim=-1)
        landing_contact = self.q1_phase_masks['landing'].unsqueeze(-1) & self.q1_foot_contact
        return torch.sum(tilt * landing_contact.float(), dim=-1)

    def _reward_q1_cr7_landing_impact(self):
        """Softly penalize excessive upward ground force during reference landing.

        The threshold makes ordinary support force free; only the excess is
        charged, which avoids discouraging the contact required to land.
        """
        threshold = float(self.config.rewards.q1_cr7_landing_contact_force_threshold)
        if threshold <= 0.0:
            raise ValueError('q1_cr7_landing_contact_force_threshold must be positive')
        normal_force = self.simulator.contact_forces[:, self.q1_feet_indices, 2].clamp_min(0.0)
        excess = (normal_force / threshold - 1.0).clamp_min(0.0)
        landing = self.q1_phase_masks['landing'].float().unsqueeze(-1)
        return torch.sum(landing * excess.square(), dim=-1)

    # ---- Inactive (not in reward_scales) — kept for future use ----

    def _reward_q1_root_z_tracking(self):
        """[INACTIVE] Track reference root_z."""
        error = self.q1_root_z_error
        sigma = 0.02
        return torch.exp(-error ** 2 / sigma)

    def _reward_q1_root_vz_tracking(self):
        """[INACTIVE] Track reference root_vz."""
        masks = self.q1_phase_masks
        error = self.q1_root_vz_error
        sigma = 0.15
        is_aerial = masks['takeoff'] | masks['flight'] | masks['landing']
        weight = torch.where(is_aerial, torch.tensor(1.5, device=self.device),
                             torch.tensor(0.5, device=self.device))
        return weight * torch.exp(-error ** 2 / sigma)

    def _reward_q1_foot_max_height_tracking(self):
        """[INACTIVE] Track reference max foot z."""
        masks = self.q1_phase_masks
        error = self.q1_foot_max_height_error
        sigma = 0.03
        is_aerial = masks['takeoff'] | masks['flight'] | masks['landing']
        weight = torch.where(is_aerial, torch.tensor(2.0, device=self.device),
                             torch.tensor(0.5, device=self.device))
        return weight * torch.exp(-error ** 2 / sigma)

    def _reward_q1_root_yaw_tracking(self):
        """[INACTIVE] Track reference yaw."""
        yaw_err_rad = self.q1_yaw_error_rad.abs()
        sigma = 0.8
        return torch.exp(-yaw_err_rad ** 2 / sigma)

    def _reward_q1_root_yaw_rate_tracking(self):
        """[INACTIVE] Track reference yaw rate."""
        masks = self.q1_phase_masks
        error = self.q1_yaw_rate_error
        sigma = 1.5
        is_aerial = masks['takeoff'] | masks['flight']
        weight = torch.where(is_aerial, torch.tensor(2.0, device=self.device),
                             torch.tensor(0.3, device=self.device))
        return weight * torch.exp(-error ** 2 / sigma)

    def _reward_q1_flight_no_contact(self):
        """[INACTIVE] Penalize foot contact during flight phase."""
        flight_mask = self.q1_phase_masks['flight'].float()
        no_contact = (~self.q1_any_contact).float()
        return flight_mask * no_contact

    def _reward_q1_stance_contact(self):
        """[INACTIVE] Reward foot contact in non-flight phases."""
        not_flight = (~self.q1_phase_masks['flight']).float()
        any_contact = self.q1_any_contact.float()
        return not_flight * any_contact

    def _reward_q1_takeoff_height_progress(self):
        """[INACTIVE] Encourage progress toward takeoff height."""
        masks = self.q1_phase_masks
        takeoff_crouch = masks['crouch'] | masks['takeoff']
        error = self.q1_root_z_error
        sigma = 0.05
        tracking = torch.exp(-error ** 2 / sigma)
        return takeoff_crouch.float() * tracking

    def _reward_q1_landing_stability(self):
        """[INACTIVE] Reward stability during landing."""
        masks = self.q1_phase_masks
        landing = masks['landing']
        grav_xy_norm = torch.norm(self.projected_gravity[:, :2], dim=-1)
        roll_pitch_ok = torch.exp(-grav_xy_norm ** 2 / 0.01)
        contact_ok = self.q1_any_contact.float()
        vz_ok = torch.exp(-self.q1_actual_root_vz ** 2 / 0.1)
        stability = roll_pitch_ok * contact_ok * vz_ok
        return landing.float() * stability
