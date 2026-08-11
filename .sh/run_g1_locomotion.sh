#!/bin/bash
set -euo pipefail

# G1 Locomotion — velocity tracking (平地)
#
# Conda env: /root/autodl-tmp/conda_envs/q1_motion
#
# Usage:
#   bash .sh/run_g1_locomotion.sh
#   NUM_ENVS=4096 HEADLESS=True bash .sh/run_g1_locomotion.sh

EXP_NAME="${EXP_NAME:-G1_Locomotion_v1}"
NUM_ENVS="${NUM_ENVS:-2048}"
CHECKPOINT="${CHECKPOINT:-}"
HEADLESS="${HEADLESS:-True}"

echo "====================================================="
echo "G1 Locomotion — Velocity Tracking (Plane)"
echo "conda      : /root/autodl-tmp/conda_envs/q1_motion"
echo "exp        : ${EXP_NAME}"
echo "envs       : ${NUM_ENVS}"
echo "headless   : ${HEADLESS}"
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
  +exp=locomotion \
  +robot=g1/g1_29dof_anneal_23dof \
  +rewards=loco/reward_g1_locomotion \
  +obs=loco/leggedloco_obs_singlestep_wolinvel \
  +terrain=terrain_locomotion_plane \
  +domain_rand=domain_rand_base \
  project_name=G1_Locomotion \
  experiment_name="${EXP_NAME}" \
  num_envs="${NUM_ENVS}" \
  headless="${HEADLESS}" \
  ${CHECKPOINT_ARGS}
