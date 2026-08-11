#!/bin/bash
set -euo pipefail

# Q1 RIGHT-foot shooting motion tracking (BGI_Shooting_00021, 6.5s version:
# 3s action + 1s upper-body->default interp + 2.5s hold).
# Reward: q1_shoot_right_* window (phase 0.14-0.28, right-foot peak ~0.26).
#
# Usage:
#   bash .sh/run_q1_shooting_00021.sh
#   NUM_ENVS=4096 CHECKPOINT=... bash .sh/run_q1_shooting_00021.sh

MOTION_FILE="${MOTION_FILE:-humanoidverse/data/motions/q1/BGI_Shooting_00021.pkl}"
EXP_NAME="${EXP_NAME:-Q1_Shooting_00021}"
NUM_ENVS="${NUM_ENVS:-2048}"
CHECKPOINT="${CHECKPOINT:-}"
DOMAIN_RAND="${DOMAIN_RAND:-domain_rand_base}"
REWARD_CONFIG="${REWARD_CONFIG:-motion_tracking/reward_q1_shoot_tracking}"

echo "Q1 Motion Tracking"
echo "motion     : ${MOTION_FILE}"
echo "exp        : ${EXP_NAME}"
echo "envs       : ${NUM_ENVS}"
echo "domain_rand: ${DOMAIN_RAND}"
echo "rewards    : ${REWARD_CONFIG}"
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
  +rewards="${REWARD_CONFIG}" \
  +domain_rand="${DOMAIN_RAND}" \
  +terrain=terrain_locomotion_plane \
  "robot.motion.motion_file=${MOTION_FILE}" \
  project_name=Q1 \
  experiment_name="${EXP_NAME}" \
  num_envs="${NUM_ENVS}" \
  headless=True \
  ++algo.config.learn_sigma=True \
  algo.config.init_noise_std=0.8 \
  domain_rand.push_robots=False \
  env.config.termination.terminate_by_gravity=True \
  env.config.termination.terminate_by_low_height=True \
  ${CHECKPOINT_ARGS}
