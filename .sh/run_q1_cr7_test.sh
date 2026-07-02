#!/bin/bash
set -euo pipefail

# Q1 CR7 Motion Tracking — dedicated env/reward/obs
#
# For first test (Level 1: instantiation):
#   bash .sh/run_q1_cr7_test.sh test
#
# For training (Level 3: 128 env):
#   bash .sh/run_q1_cr7_test.sh train
#
# For debug (Level 2: step 100):
#   bash .sh/run_q1_cr7_test.sh debug

MODE="${1:-test}"

MOTION_FILE="humanoidverse/data/motions/q1/q1_cr7_scale045_rootz042.pkl"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/q1_motion
cd /root/autodl-tmp/ASAP_official

case "$MODE" in
  test)
    echo "=== Level 1: Instantiation test (num_envs=1) ==="
    CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 \
    python humanoidverse/train_agent.py \
      +simulator=isaacgym \
      +exp=q1_cr7_motion_tracking \
      +robot=q1/q1_22dof \
      +obs=motion_tracking/q1_cr7_tracking_obs \
      +rewards=motion_tracking/reward_q1_cr7_tracking \
      +domain_rand=NO_domain_rand \
      +terrain=terrain_locomotion_plane \
      headless=True \
      num_envs=1 \
      project_name=TEST_Q1_CR7_TASK \
      experiment_name=Q1_CR7_import_test \
      "robot.motion.motion_file=${MOTION_FILE}"
    ;;

  debug)
    echo "=== Level 2: 100-step debug (num_envs=1) ==="
    CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 \
    python humanoidverse/train_agent.py \
      +simulator=isaacgym \
      +exp=q1_cr7_motion_tracking \
      +robot=q1/q1_22dof \
      +obs=motion_tracking/q1_cr7_tracking_obs \
      +rewards=motion_tracking/reward_q1_cr7_tracking \
      +domain_rand=NO_domain_rand \
      +terrain=terrain_locomotion_plane \
      headless=True \
      num_envs=1 \
      project_name=TEST_Q1_CR7_TASK \
      experiment_name=Q1_CR7_100step_debug \
      "robot.motion.motion_file=${MOTION_FILE}" \
      env.config.max_episode_length_s=5
    ;;

  train)
    echo "=== Level 3: 128-env training ==="
    CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 \
    python humanoidverse/train_agent.py \
      +simulator=isaacgym \
      +exp=q1_cr7_motion_tracking \
      +robot=q1/q1_22dof \
      +obs=motion_tracking/q1_cr7_tracking_obs \
      +rewards=motion_tracking/reward_q1_cr7_tracking \
      +domain_rand=NO_domain_rand \
      +terrain=terrain_locomotion_plane \
      headless=True \
      num_envs=128 \
      project_name=TEST_Q1_CR7_TASK \
      experiment_name=Q1_CR7_v1_128env \
      "robot.motion.motion_file=${MOTION_FILE}"
    ;;

  *)
    echo "Usage: bash .sh/run_q1_cr7_test.sh [test|debug|train]"
    exit 1
    ;;
esac
