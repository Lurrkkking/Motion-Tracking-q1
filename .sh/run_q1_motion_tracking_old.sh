#!/bin/bash
set -euo pipefail

# Q1 22dof motion_tracking — 对齐 G1 jump2 配置（固定探索噪声 0.8）
#
# Conda env: /root/autodl-tmp/conda_envs/q1_motion
#
# Motion files (in humanoidverse/data/motions/q1/):
#   q1_bolt.pkl  q1_cr7_motion.pkl  q1_cr7_scale045.pkl  q1_stand_still.pkl
#
# Usage:
#   bash .sh/run_q1_motion_tracking.sh
#   MOTION_FILE=humanoidverse/data/motions/q1/q1_cr7_motion.pkl bash .sh/run_q1_motion_tracking.sh

MOTION_FILE="${MOTION_FILE:-humanoidverse/data/motions/q1/q1_cr7_scale045_rootz042.pkl}"
EXP_NAME="${EXP_NAME:-MotionTracking_Q1_CR7_045_pkl}"

echo "====================================================="
echo "Q1 22dof Motion Tracking — 固定探索噪声 0.8"
echo "conda : /root/autodl-tmp/conda_envs/q1_motion"
echo "motion: ${MOTION_FILE}"
echo "exp   : ${EXP_NAME}"
echo "====================================================="

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/q1_motion
cd /root/autodl-tmp/ASAP_official

HYDRA_FULL_ERROR=1 exec python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  +robot=q1/q1_22dof \
  +obs=motion_tracking/q1_deepmimic_a2c_nolinvel_LARGEnoise_history \
  +rewards=motion_tracking/reward_motion_tracking_q1_cr7 \
  +domain_rand=NO_domain_rand \
  +terrain=terrain_locomotion_plane \
  "robot.motion.motion_file=${MOTION_FILE}" \
  project_name=TEST_Q1 \
  experiment_name="${EXP_NAME}" \
  num_envs=2048 \
  headless=True \
  env.config.termination.terminate_by_gravity=False \
  env.config.termination.terminate_by_low_height=True \
  env.config.termination_scales.termination_min_base_height=0.15 \
  algo.config.num_mini_batches=4 \
  ++algo.config.learn_sigma=False \
  algo.config.init_noise_std=0.8
