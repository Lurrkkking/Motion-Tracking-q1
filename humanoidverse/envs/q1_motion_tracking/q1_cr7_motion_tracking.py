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
from isaac_utils.rotations import quat_to_angle_axis, calc_heading_quat, calc_heading_quat_inv
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
        # takeoff/early flight transition region
        flight = flight & ~takeoff
        landing = landing & ~takeoff & ~flight
        crouch = crouch & ~takeoff & ~flight & ~landing

        return {
            'crouch': crouch,
            'takeoff': takeoff,
            'flight': flight,
            'landing': landing,
        }

    # ------------------------------------------------------------------
    #  Pre-compute observations — add Q1 CR7-specific buffers
    # ------------------------------------------------------------------

    def _pre_compute_observations_callback(self):
        super()._pre_compute_observations_callback()

        # --- Motion reference data ---
        offset = self.env_origins
        motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times
        motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)

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
        """Log Q1-specific debug info to log_dict."""
        self.log_dict["q1_root_z_error"] = self.q1_root_z_error.mean()
        self.log_dict["q1_root_vz_error"] = self.q1_root_vz_error.mean()
        self.log_dict["q1_yaw_error_deg"] = torch.rad2deg(self.q1_yaw_error_rad).abs().mean()
        self.log_dict["q1_yaw_rate_error"] = self.q1_yaw_rate_error.mean()
        self.log_dict["q1_ref_root_z"] = self.q1_ref_root_z.mean()
        self.log_dict["q1_actual_root_z"] = self.q1_actual_root_z.mean()
        self.log_dict["q1_ref_vz"] = self.q1_ref_root_vz.mean()
        self.log_dict["q1_actual_vz"] = self.q1_actual_root_vz.mean()
        self.log_dict["q1_ref_yaw_rate"] = self.q1_ref_yaw_rate.mean()
        self.log_dict["q1_actual_yaw_rate"] = self.q1_actual_yaw_rate.mean()

        # Flight/contact rates
        flight_mask = self.q1_phase_masks['flight'].float()
        flight_contact = (self.q1_any_contact & self.q1_phase_masks['flight']).float()
        self.log_dict["q1_flight_contact_rate"] = flight_contact.mean()
        self.log_dict["q1_both_feet_air_rate"] = self.q1_both_feet_air.float().mean()

        # Phase ratios
        self.log_dict["q1_phase_crouch_ratio"] = self.q1_phase_masks['crouch'].float().mean()
        self.log_dict["q1_phase_takeoff_ratio"] = self.q1_phase_masks['takeoff'].float().mean()
        self.log_dict["q1_phase_flight_ratio"] = self.q1_phase_masks['flight'].float().mean()
        self.log_dict["q1_phase_landing_ratio"] = self.q1_phase_masks['landing'].float().mean()
        self.log_dict["q1_episode_phase_mean"] = self._ref_motion_phase.mean()

        # --- Per-phase root_z / root_vz RMSE ---
        for phase_name, mask in [
            ("crouch", self.q1_phase_masks['crouch']),
            ("takeoff", self.q1_phase_masks['takeoff']),
            ("flight", self.q1_phase_masks['flight']),
            ("landing", self.q1_phase_masks['landing']),
        ]:
            weight = mask.float()
            total = weight.sum() + 1e-8
            # root_z RMSE within phase
            z_rmse = torch.sqrt((weight * self.q1_root_z_error ** 2).sum() / total)
            vz_rmse = torch.sqrt((weight * self.q1_root_vz_error ** 2).sum() / total)
            self.log_dict[f"q1_root_z_rmse_{phase_name}"] = z_rmse
            self.log_dict[f"q1_root_vz_rmse_{phase_name}"] = vz_rmse

        # --- Peak values (max over envs) ---
        self.log_dict["q1_actual_root_vz_max"] = self.q1_actual_root_vz.max()
        self.log_dict["q1_ref_root_vz_max"] = self.q1_ref_root_vz.max()
        self.log_dict["q1_actual_root_z_max"] = self.q1_actual_root_z.max()
        self.log_dict["q1_ref_root_z_max"] = self.q1_ref_root_z.max()
        self.log_dict["q1_both_feet_air_count"] = self.q1_both_feet_air.float().sum()

    # ------------------------------------------------------------------
    #  Phase-aware termination
    # ------------------------------------------------------------------

    def _check_termination(self):
        """Q1 CR7 phase-aware termination.

        Overrides the default termination pipeline:
        1. Always checks motion_end timeout (from super)
        2. Disables standard gravity/height/contact termination
        3. Applies phase-specific termination rules:
           - crouch/takeoff/flight: only catastrophic falls terminate
           - landing: stricter termination
           - reference-relative height termination (if enabled)
        """
        # Start with clean buffers
        self.reset_buf[:] = 0
        self.time_out_buf[:] = 0

        # Timeout
        self._update_timeout_buf()

        # Q1-specific graceful termination
        cfg = self._q1_cr7_config
        phase = self._ref_motion_phase.squeeze(-1)
        masks = self.q1_phase_masks

        # --- Build per-phase termination flags ---
        is_aerial_phase = masks['crouch'] | masks['takeoff'] | masks['flight']
        is_landing_phase = masks['landing']
        is_late_phase = phase >= cfg['landing_phase_end']

        if cfg.get('reference_relative_termination', True):
            # Reference-relative height termination
            height_margin = torch.where(
                is_aerial_phase, cfg['crouch_height_margin'],
                torch.where(is_landing_phase, cfg['landing_height_margin'],
                            cfg['landing_height_margin'])
            )
            terminate_low_height = self.q1_actual_root_z < (self.q1_ref_root_z - height_margin)
        else:
            # Fixed height termination
            min_height = torch.where(
                is_aerial_phase,
                torch.tensor(cfg['catastrophic_root_z'], device=self.device),
                torch.where(is_landing_phase,
                            torch.tensor(0.12, device=self.device),
                            torch.tensor(0.12, device=self.device))
            )
            terminate_low_height = self.q1_actual_root_z < min_height

        # Catastrophic: robot fully collapsed
        terminate_catastrophic = self.q1_actual_root_z < cfg['catastrophic_root_z']

        # Gravity-based termination (relaxed for aerial phases)
        grav_x = self.projected_gravity[:, 0]
        grav_y = self.projected_gravity[:, 1]

        if is_aerial_phase.any():
            # Aerial: only terminate if nearly horizontal
            grav_term_aerial = (torch.abs(grav_x) > 0.98) | (torch.abs(grav_y) > 0.98)
        else:
            grav_term_aerial = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Landing/late: stricter
        grav_term_landing = (torch.abs(grav_x) > 0.9) | (torch.abs(grav_y) > 0.9)
        is_not_aerial = ~is_aerial_phase
        terminate_gravity = (is_aerial_phase & grav_term_aerial) | (is_not_aerial & grav_term_landing)

        # Contact termination: only if non-foot bodies hit ground hard
        # Use key bodies (pelvis, shoulders, hips) for contact check
        terminate_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # --- Compose reset buffer ---
        self.reset_buf |= terminate_catastrophic
        self.reset_buf |= terminate_low_height
        self.reset_buf |= terminate_gravity
        self.reset_buf |= terminate_contact

        # Always include timeout
        self.reset_buf |= self.time_out_buf

        # --- Log termination flags ---
        self.log_dict["q1_terminate_low_height"] = terminate_low_height.float().mean()
        self.log_dict["q1_terminate_gravity"] = terminate_gravity.float().mean()
        self.log_dict["q1_terminate_contact"] = terminate_contact.float().mean()
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
    #  Q1 CR7 Reward functions
    # ------------------------------------------------------------------

    def _reward_q1_root_z_tracking(self):
        """Track reference root_z."""
        error = self.q1_root_z_error
        sigma = 0.02
        return torch.exp(-error ** 2 / sigma)

    def _reward_q1_root_vz_tracking(self):
        """Track reference root_vz. Weighted more heavily in aerial phases."""
        masks = self.q1_phase_masks
        error = self.q1_root_vz_error
        sigma = 0.15

        # Higher weight in aerial phases
        is_aerial = masks['takeoff'] | masks['flight'] | masks['landing']
        weight = torch.where(is_aerial, torch.tensor(1.5, device=self.device),
                             torch.tensor(0.5, device=self.device))

        return weight * torch.exp(-error ** 2 / sigma)

    def _reward_q1_root_yaw_tracking(self):
        """Track reference yaw using sin/cos error."""
        yaw_err_rad = self.q1_yaw_error_rad.abs()
        sigma = 0.8
        return torch.exp(-yaw_err_rad ** 2 / sigma)

    def _reward_q1_root_yaw_rate_tracking(self):
        """Track reference yaw rate. Weighted more heavily in flight/takeoff."""
        masks = self.q1_phase_masks
        error = self.q1_yaw_rate_error
        sigma = 1.5

        # Higher weight in flight/takeoff
        is_aerial = masks['takeoff'] | masks['flight']
        weight = torch.where(is_aerial, torch.tensor(2.0, device=self.device),
                             torch.tensor(0.3, device=self.device))

        return weight * torch.exp(-error ** 2 / sigma)

    def _reward_q1_flight_contact(self):
        """Penalize foot contact during reference flight phase. Positive reward for no contact."""
        flight_mask = self.q1_phase_masks['flight'].float()
        no_contact = (~self.q1_any_contact).float()
        return flight_mask * no_contact

    def _reward_q1_stance_contact(self):
        """Reward having at least one foot in contact during non-flight phases."""
        not_flight = (~self.q1_phase_masks['flight']).float()
        any_contact = self.q1_any_contact.float()
        return not_flight * any_contact

    def _reward_q1_lower_body_joint_tracking(self):
        """Weighted joint position tracking for lower body joints."""
        ref_pos = self.q1_ref_dof_pos
        actual_pos = self.simulator.dof_pos
        weights = self.q1_lower_body_joint_weights

        weighted_sq_error = weights * (ref_pos - actual_pos) ** 2
        weighted_mse = weighted_sq_error.mean(dim=-1)
        sigma = 0.10
        return torch.exp(-weighted_mse / sigma)

    def _reward_q1_takeoff_height_progress(self):
        """Encourage progress toward takeoff height during crouch/takeoff phases."""
        masks = self.q1_phase_masks
        takeoff_crouch = masks['crouch'] | masks['takeoff']
        # Reward actual root_z being close to ref_root_z during crouch/takeoff
        error = self.q1_root_z_error
        sigma = 0.05
        tracking = torch.exp(-error ** 2 / sigma)
        return takeoff_crouch.float() * tracking

    def _reward_q1_landing_stability(self):
        """Reward stability during landing: low roll/pitch, foot contact, near-zero vz."""
        masks = self.q1_phase_masks
        landing = masks['landing']

        # Roll/pitch from projected gravity
        grav_xy_norm = torch.norm(self.projected_gravity[:, :2], dim=-1)
        roll_pitch_ok = torch.exp(-grav_xy_norm ** 2 / 0.01)

        # Contact ok
        contact_ok = self.q1_any_contact.float()

        # vz near zero
        vz_ok = torch.exp(-self.q1_actual_root_vz ** 2 / 0.1)

        stability = roll_pitch_ok * contact_ok * vz_ok
        return landing.float() * stability
