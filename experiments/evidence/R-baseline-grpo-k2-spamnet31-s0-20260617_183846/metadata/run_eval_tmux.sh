#!/usr/bin/env bash
set +e

/home/harvey/tpu-runs/part-i/R-baseline-grpo-k2-spamnet31-s0-20260617_183846/metadata/launch_full_eval.sh \
  2>&1 | tee -a /home/harvey/tpu-runs/part-i/R-baseline-grpo-k2-spamnet31-s0-20260617_183846/logs/eval_pipeline.log
status=${PIPESTATUS[0]}

echo
echo "--- eval process exited (${status}) ---"
exec bash
