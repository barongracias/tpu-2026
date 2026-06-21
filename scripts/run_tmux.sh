#!/usr/bin/env bash
# Launch training inside a detached tmux session so closing your shell does
# NOT kill the run. Re-run this script and it just attaches to the session.
#
#   ./run_tmux.sh                # start (or attach)
#   ./run_tmux.sh resume         # resume the WANDB_RUN_ID from the environment
#   tmux attach -t tunix         # reattach manually
#   tmux kill-session -t tunix   # stop everything

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
SESSION=${SESSION:-tunix}
VENV=${VENV:-$HOME/venvs/tunix}

: "${RUN_ROOT:?Set RUN_ROOT to a persistent per-run directory, e.g. \$HOME/tpu-runs/part-i/R6-hard-s0.}"
: "${MODEL_REVISION:?Set MODEL_REVISION to an exact Hugging Face model commit SHA.}"
: "${JAX_REF:?Set JAX_REF to the exact jax git commit used in the venv.}"
: "${QWIX_REF:?Set QWIX_REF to the exact qwix git commit used in the venv.}"
: "${FLAX_REF:?Set FLAX_REF to the exact flax git commit used in the venv.}"

export CKPT_DIR=${CKPT_DIR:-$RUN_ROOT/ckpts}
export INTERMEDIATE_CKPT_DIR=${INTERMEDIATE_CKPT_DIR:-$RUN_ROOT/intermediate_ckpt}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-$RUN_ROOT/tensorboard}
export TRAIN_DATA_DIR=${TRAIN_DATA_DIR:-$RUN_ROOT/data/train}
export TEST_DATA_DIR=${TEST_DATA_DIR:-$RUN_ROOT/data/test}
export EVAL_SEED=${EVAL_SEED:-0}
export EVAL_MANIFEST=${EVAL_MANIFEST:-$HOME/tpu-runs/part-i/manifests/gsm8k_test_seed${EVAL_SEED}_n64.jsonl}
export RUN_SEED=${RUN_SEED:-0}
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(basename "$RUN_ROOT")}"

mkdir -p "$RUN_ROOT"/{ckpts,intermediate_ckpt,tensorboard,logs,eval,data/train,data/test}
mkdir -p "$(dirname "$EVAL_MANIFEST")"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists — attaching."
  exec tmux attach -t "$SESSION"
fi

INNER="cd '$REPO/scripts' && source '$VENV/bin/activate' && python -u train.py"
if [[ "${1:-}" == "resume" ]]; then
  INNER="cd '$REPO/scripts' && source '$VENV/bin/activate' && WANDB_RUN_ID='$WANDB_RUN_ID' python -u train.py --wandb-run-id '$WANDB_RUN_ID'"
fi

# Run under bash (not dash) so `source` works, and keep the shell alive on
# success/failure so we can read the output instead of tmux closing on us.
# Also tee output to a logfile in case we miss something on screen.
LOG="$RUN_ROOT/logs/train.log"
CMD="bash -lc '$INNER 2>&1 | tee -a $LOG; echo; echo \"--- process exited (\$?) ---\"; exec bash'"

tmux new-session -d -s "$SESSION" "$CMD"
echo "Started tmux session '$SESSION'. Attach with: tmux attach -t $SESSION"
echo "Log file:                                tail -f $LOG"
echo "Run root:                                $RUN_ROOT"
