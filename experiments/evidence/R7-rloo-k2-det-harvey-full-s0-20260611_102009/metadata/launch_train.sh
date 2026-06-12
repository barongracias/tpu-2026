#!/usr/bin/env bash
set -euo pipefail

export REPO=/home/harvey/tpu-2026
export VENV=/home/harvey/venvs/tunix
export RUN_ID=R7-rloo-k2-det-harvey-full-s0-20260611_102009
export RUN_ROOT=/home/harvey/tpu-runs/part-i/R7-rloo-k2-det-harvey-full-s0-20260611_102009

export ADV_ESTIMATOR=rloo
export NUM_GENERATIONS=2
export RUN_SEED=0
export EVAL_SEED=0
export EVAL_MANIFEST=/home/harvey/tpu-runs/part-i/manifests/gsm8k_test_seed0_n64.jsonl

export SAVE_INTERVAL_STEPS=250
export MAX_TO_KEEP=20
unset MAX_STEPS_OVERRIDE

export MODEL_REVISION=dcc83ea841ab6100d6b47a070329e1ba4cf78752
export JAX_REF=3ff6dc40cdb982921de0ef12c0bf8e5c64311a6f
export TUNIX_REF=7fadb3c81e4348f714b2b5a07f6cc2bd10da10de
export QWIX_REF=c2548f06eeca005090329530576623afcd48241c
export FLAX_REF=5ab9d1463550a7fc7fe13575af55081e68258dd7

export CKPT_DIR="$RUN_ROOT/ckpts"
export INTERMEDIATE_CKPT_DIR="$RUN_ROOT/intermediate_ckpt"
export TENSORBOARD_DIR="$RUN_ROOT/tensorboard"
export TRAIN_DATA_DIR="$RUN_ROOT/data/train"
export TEST_DATA_DIR="$RUN_ROOT/data/test"
export TMPDIR="$RUN_ROOT/tmp"
export WANDB_DIR="$RUN_ROOT/wandb"

export WANDB_PROJECT=agentic-ai-coursework
export WANDB_ENTITY=barongracias-university-of-cambridge
export WANDB_RUN_ID="$RUN_ID"
export WANDB_NAME="$RUN_ID"
export WANDB_MODE=online

mkdir -p "$RUN_ROOT"/{logs,ckpts,intermediate_ckpt,tensorboard,eval,data/train,data/test,tmp,wandb}

cd "$REPO/scripts"
source "$VENV/bin/activate"

set +e
python -u train.py --wandb-run-id "$WANDB_RUN_ID" 2>&1 | tee -a "$RUN_ROOT/logs/train.log"
status=${PIPESTATUS[0]}
set -e

echo
echo "--- process exited ($status) ---"
exec bash
