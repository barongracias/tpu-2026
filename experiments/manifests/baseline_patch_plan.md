# Baseline Patch Plan

Purpose: define the smallest code changes needed before TPU runs are scientifically usable for Part I. This is not an implementation record; update the iteration logs with what is actually changed and tested.

## Scope

Patch the `tpu-2026` training/evaluation workflow only enough to support:

- controlled GRPO vs RLOO comparison,
- reproducible seeds,
- persistent evidence artefacts,
- checkpoint-backed evaluation,
- per-prompt outputs for confidence intervals and plots.

Avoid extra algorithm changes until the baseline comparison is verified.

## P0. Configurable Output Directories

Current upstream state observed in `external/tpu-2026/scripts/config.py` and training imports:

- checkpoint and TensorBoard paths are configured centrally,
- upstream examples commonly use `/tmp` paths,
- TPU setup notes warn that `/tmp` may not persist across sessions.

Required change:

- allow `CKPT_DIR`, `INTERMEDIATE_CKPT_DIR`, and `TENSORBOARD_DIR` to be overridden from environment variables,
- print or save the resolved paths at run start.

Verification:

- launch a short run with all three paths under `$HOME/tpu-runs/...`,
- confirm files are created there and not under `/tmp`.

## P1. Estimator Selection

Current upstream state observed in `external/tpu-2026/scripts/train.py`:

- `GRPOConfig` is constructed with `num_generations`, `num_iterations`, `beta`, and `epsilon`,
- no estimator argument is passed from the training script.

Required change:

- add an `ADV_ESTIMATOR` control with default `grpo`,
- pass it into `GRPOConfig(advantage_estimator=...)`,
- accept at least `grpo` and `rloo`,
- log the resolved estimator at run start.

Verification:

- instantiate a debug GRPO run and a debug RLOO run,
- confirm the resolved estimator appears in logs/config,
- confirm both runs reach the same debug step count under otherwise identical controls.

## P2. Seed Controls

Current upstream state observed:

- Tunix `GRPOLearner` supports `data_shuffle_seed`,
- rollout configuration supports a `seed`,
- `external/tpu-2026/scripts/data.py` contains a hard-coded dataset shuffle seed.

Required change:

- add `RUN_SEED` with a default value,
- pass it to `GRPOLearner(data_shuffle_seed=...)`,
- pass it to rollout configuration,
- make dataset shuffle seed configurable instead of hard-coded.

Verification:

- logs/config include the seed values,
- two debug launches with the same seed use the same config,
- changing the seed changes only seed-controlled fields.

## P3. Short-Run Step Override

Current upstream state observed:

- `MAX_STEPS` is imported from config by `train.py`,
- debug runs need a small step count before full TPU spending.

Required change:

- add `MAX_STEPS_OVERRIDE` or equivalent,
- keep the upstream default unchanged when the override is absent,
- make `SAVE_INTERVAL_STEPS` environment-overridable so a short debug run can still save an evaluable checkpoint.

Verification:

- a 50-step debug run stops at the intended step budget,
- with `SAVE_INTERVAL_STEPS=50`, a 50-step debug run leaves a checkpoint suitable for the restore check.

## P4. Evaluation Checkpoint Restore

Current upstream state observed:

- `chat.py` contains checkpoint restore logic using Tunix checkpoint management,
- `evaluate.py` constructs the model and LoRA wrapper but appears not to restore a trained checkpoint before sampling.

Required change:

- add `--ckpt-dir` and optional `--step` to `evaluate.py`,
- restore trained LoRA parameters before evaluation,
- print the checkpoint directory and restored step,
- keep a no-restore path only for explicit base-model sanity checks.

Verification:

- evaluate a debug checkpoint,
- confirm the restored checkpoint step is printed,
- compare against an explicit no-restore run to ensure the path matters.

## P5. Per-Prompt Evaluation Output

Current upstream state observed:

- aggregate evaluation is useful but insufficient for bootstrap confidence intervals.

Required change:

- write a CSV or JSONL file with one row per evaluation prompt,
- include prompt id/source, model output, expected answer or score fields, correctness/reward, checkpoint id, estimator, seed, and preset.

Verification:

- aggregate accuracy or score can be recomputed exactly from the saved file,
- bootstrap confidence intervals can be generated without re-running model inference.

## P6. Run Metadata Snapshot

Required change:

- save a small run config file or print a parseable config block at run start,
- include commit hashes, estimator, seed, data source, step budget, output directories, W&B run id, and evaluation preset.

Verification:

- iteration log can reference an immutable metadata file for each run.

## Suggested Patch Order

1. P0 persistent paths.
2. P1 estimator selection.
3. P2 seed controls.
4. P3 debug step override.
5. P4 evaluation restore.
6. P5 per-prompt evaluation output.
7. P6 metadata snapshot.

This order gets the first debug training runs unblocked quickly, then prevents evaluation and reporting evidence from becoming ambiguous.

