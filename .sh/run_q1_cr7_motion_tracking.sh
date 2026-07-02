#!/bin/bash
set -euo pipefail

# Q1 CR7 22dof motion_tracking — 专用 Q1CR7MotionTracking env
# 使用新 task/env/reward/obs: q1_cr7_motion_tracking
#
# Conda env: /root/autodl-tmp/conda_envs/q1_motion
#
# Motion files (in humanoidverse/data/motions/q1/):
#   q1_bolt.pkl  q1_cr7_motion.pkl  q1_cr7_scale045.pkl  q1_stand_still.pkl
#
# Usage:
#   bash .sh/run_q1_cr7_motion_tracking.sh
#   MOTION_FILE=humanoidverse/data/motions/q1/q1_cr7_motion.pkl bash .sh/run_q1_cr7_motion_tracking.sh

MOTION_FILE="${MOTION_FILE:-humanoidverse/data/motions/q1/Bolt_q1_2s.pkl}"
EXP_NAME="${EXP_NAME:-Q1_Bolt_Tracking_v1}"
NUM_ENVS="${NUM_ENVS:-2048}"
CHECKPOINT="${CHECKPOINT:-}"
DOMAIN_RAND="${DOMAIN_RAND:-domain_rand_base}"

echo "====================================================="
echo "Q1 CR7 Motion Tracking — 专用 Task/Env/Reward/Obs"
echo "conda     : /root/autodl-tmp/conda_envs/q1_motion"
echo "motion    : ${MOTION_FILE}"
echo "exp       : ${EXP_NAME}"
echo "envs      : ${NUM_ENVS}"
echo "domain_rand: ${DOMAIN_RAND}"
echo "checkpoint : ${CHECKPOINT:-none}"
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
  +exp=q1_cr7_motion_tracking \
  +robot=q1/q1_22dof \
  +obs=motion_tracking/q1_cr7_tracking_obs \
  +rewards=motion_tracking/reward_q1_cr7_tracking \
  +domain_rand="${DOMAIN_RAND}" \
  +terrain=terrain_locomotion_plane \
  "robot.motion.motion_file=${MOTION_FILE}" \
  project_name=Q1_Bolt \
  experiment_name="${EXP_NAME}" \
  num_envs="${NUM_ENVS}" \
  headless=True \
  ++algo.config.learn_sigma=False \
  algo.config.init_noise_std=0.8 \
  domain_rand.push_robots=False \
  env.config.termination.terminate_by_gravity=True \
  env.config.termination.terminate_by_low_height=True \
  ${CHECKPOINT_ARGS}
