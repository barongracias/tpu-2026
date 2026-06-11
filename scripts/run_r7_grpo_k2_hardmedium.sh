#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
VENV=${VENV:-$HOME/venvs/tunix}
ENV_FILE=${ENV_FILE:-$HOME/.env}

cd "$REPO"

source "$VENV/bin/activate"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

unset MAX_STEPS_OVERRIDE WANDB_RUN_ID
export WANDB_MODE=${WANDB_MODE:-online}

export RUN_ROOT=${RUN_ROOT:-"$HOME/tpu-runs/part-i/R7-grpo-k2-hardmedium-$(date +%Y%m%d_%H%M%S)"}
export WANDB_NAME=${WANDB_NAME:-R7-grpo-k2-hardmedium}
export DATA_SOURCE=${DATA_SOURCE:-manifest}
export TRAIN_MANIFEST=${TRAIN_MANIFEST:-"$HOME/tpu-runs/part-i/manifests/gsm8k_train_base_hard_medium.jsonl"}
export NUM_BATCHES=${NUM_BATCHES:-3836}
export NUM_GENERATIONS=${NUM_GENERATIONS:-2}
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
export RUN_SEED=${RUN_SEED:-0}
export EVAL_SEED=${EVAL_SEED:-0}
export TRAIN_DATA_DIR=${TRAIN_DATA_DIR:-"$RUN_ROOT/data/train"}
export TEST_DATA_DIR=${TEST_DATA_DIR:-"$RUN_ROOT/data/test"}
export CKPT_DIR=${CKPT_DIR:-"$RUN_ROOT/ckpts"}
export INTERMEDIATE_CKPT_DIR=${INTERMEDIATE_CKPT_DIR:-"$RUN_ROOT/intermediate_ckpt"}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-"$RUN_ROOT/tensorboard"}
export SAVE_INTERVAL_STEPS=${SAVE_INTERVAL_STEPS:-250}
export MAX_TO_KEEP=${MAX_TO_KEEP:-20}

mkdir -p "$RUN_ROOT"/{ckpts,intermediate_ckpt,tensorboard,logs,data/train,data/test}
echo "$RUN_ROOT" | tee "$RUN_ROOT/RUN_ROOT.txt"

python scripts/train.py --source manifest 2>&1 | tee "$RUN_ROOT/logs/train.log"
