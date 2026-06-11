# Experiment Contract

This file records the launch contract for comparable TPU experiments.

## Required Pins

Set these before bootstrapping or launching:

```bash
export JAX_REF="<exact jax git commit sha>"
export QWIX_REF="<exact qwix git commit sha>"
export FLAX_REF="<exact flax git commit sha>"
export TUNIX_REF="683256db1a0919b5cfd46cee52cebc96331494fb"
export MODEL_REVISION="<exact google/gemma-3-1b-it Hugging Face commit sha>"
```

`bootstrap.sh` refuses to install JAX/Qwix/Flax from moving HEAD. `model.py`
refuses to download the model unless `MODEL_REVISION` is set.

Current resolved Hugging Face model revision for `google/gemma-3-1b-it`
queried from the HF model API on 2026-06-09:

```bash
export MODEL_REVISION="dcc83ea841ab6100d6b47a070329e1ba4cf78752"
```

This pins the current HF revision at query time. Prefer an existing
`run_metadata.json` value when reproducing a specific historical run.

## Required Run Paths

Every run must use a persistent run root:

```bash
export RUN_ROOT="$HOME/tpu-runs/part-i/<run-id>"
export CKPT_DIR="$RUN_ROOT/ckpts"
export INTERMEDIATE_CKPT_DIR="$RUN_ROOT/intermediate_ckpt"
export TENSORBOARD_DIR="$RUN_ROOT/tensorboard"
export TRAIN_DATA_DIR="$RUN_ROOT/data/train"
export TEST_DATA_DIR="$RUN_ROOT/data/test"
```

`scripts/run_tmux.sh` derives these defaults from `RUN_ROOT`.

## Required Seeds And Manifest

Training randomness and held-out evaluation order are separate:

```bash
export RUN_SEED=0
export EVAL_SEED=0
export EVAL_MANIFEST="$HOME/tpu-runs/part-i/manifests/gsm8k_test_seed0_n64.jsonl"
```

`RUN_SEED` controls train shuffling, rollout sampling, and learner shuffling.
`EVAL_SEED` controls the held-out eval prompt order only when a manifest does
not already exist. Once `EVAL_MANIFEST` exists, all checkpoint/base evals must
load it so seed repeats compare against the same prompts. Prefer a shared
manifest path outside any single run root for cross-run comparisons.

The first eval for a run may create the manifest:

```bash
python -u scripts/evaluate.py \
  --preset greedy \
  --source tfds \
  --ckpt-dir "$CKPT_DIR" \
  --eval-manifest "$EVAL_MANIFEST" \
  --output-csv "$RUN_ROOT/eval/<run-id>_greedy.csv"
```

Later evals should use the same `--eval-manifest` path.
