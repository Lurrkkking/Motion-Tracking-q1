#!/bin/bash
set -euo pipefail

# G1 Motion Tracking
#
# Motion files:
#   humanoidverse/data/motions/g1_29dof_anneal_23dof/v1/amass_all.pkl  (默认)
#   humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-own_kickball_eg_gvhmr.pkl
#
# Usage:
#   bash .sh/run_g1_motion_tracking.sh
#   MOTION_FILE=humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-own_kickball_eg_gvhmr.pkl bash .sh/run_g1_motion_tracking.sh

MOTION_FILE="${MOTION_FILE:-humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-own_kickball_eg_gvhmr.pkl}"
EXP_NAME="${EXP_NAME:-G1_Motion_Tracking_v1}"
NUM_ENVS="${NUM_ENVS:-2048}"
CHECKPOINT="${CHECKPOINT:-}"

echo "====================================================="
echo "G1 Motion Tracking"
echo "motion       : ${MOTION_FILE}"
echo "exp          : ${EXP_NAME}"
echo "envs         : ${NUM_ENVS}"
echo "checkpoint   : ${CHECKPOINT:-none}"
echo "====================================================="

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/q1_motion
cd /root/autodl-tmp/ASAP_official

CHECKPOINT_ARGS=""
if [ -n "${CHECKPOINT}" ]; then
  CHECKPOINT_ARGS="checkpoint=${CHECKPOINT} auto_load_latest=False algo.config.load_optimizer=False"
fi

HYDRA_FULL_ERROR=1 exec python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  +robot=g1/g1_29dof_anneal_23dof \
  +terrain=terrain_locomotion_plane \
  +domain_rand=domain_rand_base \
  +rewards=motion_tracking/reward_motion_tracking_dm_2real \
  +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
  "robot.motion.motion_file=${MOTION_FILE}" \
  experiment_name="${EXP_NAME}" \
  num_envs="${NUM_ENVS}" \
  headless=True \
  algo.config.num_mini_batches=4 \
  ++algo.config.learn_sigma=False \
  algo.config.init_noise_std=0.8 \
  domain_rand.push_interval_s=[5,10] \
  domain_rand.max_push_vel_xy=1.0 \
  ++domain_rand.base_com_range.x=[-0.03,0.03] \
  ++domain_rand.base_com_range.y=[-0.03,0.03] \
  ++domain_rand.base_com_range.z=[-0.05,0.05] \
  ++domain_rand.link_mass_range=[0.8,1.2] \
  ++domain_rand.kp_range=[0.75,1.25] \
  ++domain_rand.kd_range=[0.75,1.25] \
  ++domain_rand.friction_range=[0.5,1.25] \
  ++domain_rand.added_mass_range=[-2.0,2.0] \
  ++domain_rand.rfi_lim=0.1 \
  ++domain_rand.rfi_lim_range=[0.5,1.5] \
  ++domain_rand.ctrl_delay_step_range=[0,2] \
  ${CHECKPOINT_ARGS}
