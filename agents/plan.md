# TPU-2026 Coursework Branch Plan

## Objective

Prepare the `tpu-2026` baseline for reproducible Part I practical runs: baseline GRPO reproduction, controlled RLOO comparison at fixed compute, checkpoint-backed evaluation, and source-backed report evidence.

## Source Of Truth

- Coursework PDF and report planning live in `../agentic-ai-coursework`.
- Main runbook: `../agentic-ai-coursework/experiments/runbooks/tpu_day1_runbook.md`.
- Main patch plan: `../agentic-ai-coursework/experiments/manifests/baseline_patch_plan.md`.
- This repository contains the actual training/evaluation patches for the TPU code.

## Milestone 1: Verify Patch Scope

Status: complete for committed P0-P6 plus debug LR/W&B fix at `6a2aef7`; Fred-review follow-up edits pending commit.

Goal:
- Confirm P0-P6 are implemented without changing Tunix.

Checks:
- `git diff --name-only main..HEAD` shows only `bootstrap.sh`, `scripts/config.py`, `scripts/data.py`, `scripts/train.py`, `scripts/evaluate.py` for the 8 commits.
- `git diff --check` passes.
- Syntax compilation passes with `PYTHONPYCACHEPREFIX=/tmp/tpu2026-pycache`.

## Milestone 2: Commit/Share Local Follow-Up Fix

Status: pending user action.

Goal:
- Preserve the full-run learning-rate schedule during short debug runs, record the resolved W&B run id, and incorporate Fred's debug-checkpoint review.

Files:
- `scripts/config.py`: `FULL_MAX_STEPS`/`LR_DECAY_STEPS` are committed; local follow-up makes `SAVE_INTERVAL_STEPS` env-overridable.
- `scripts/train.py`: `LR_DECAY_STEPS` and resolved W&B id are committed; local follow-up records `lr_decay_steps` and `save_interval_steps` in metadata.
- `scripts/evaluate.py`: local follow-up auto-resolves an `actor/` checkpoint child and records requested/resolved checkpoint paths.

Verification:
- `git diff --check`.
- `env PYTHONPYCACHEPREFIX=/tmp/tpu2026-pycache python3 -m py_compile scripts/config.py scripts/data.py scripts/train.py scripts/evaluate.py`.

## Milestone 3: TPU Day-One Debug

Status: pending TPU access.

Goal:
- Run short GRPO and RLOO jobs before committing full TPU time.

Steps:
- Create persistent `$RUN_ROOT` under `$HOME/tpu-runs/...`.
- Export `CKPT_DIR`, `INTERMEDIATE_CKPT_DIR`, `TENSORBOARD_DIR`, `RUN_SEED=0`, `MAX_STEPS_OVERRIDE=50`, and `SAVE_INTERVAL_STEPS=50`.
- Run D1 with `ADV_ESTIMATOR=grpo`.
- Run D2 with `ADV_ESTIMATOR=rloo`.
- Record each run in an iteration log.

Checks:
- TPU backend is active.
- Output paths are persistent.
- Checkpoints are saved.
- TensorBoard/W&B metrics appear.
- `run_metadata.json` records estimator, seed, paths, commit, and W&B run id.

## Milestone 4: Evaluation Restore Check

Status: pending debug checkpoint.

Goal:
- Confirm evaluation uses trained LoRA parameters, not an un-restored LoRA wrapper.

Command shape:

```bash
python -u scripts/evaluate.py --preset greedy --source tfds --ckpt-dir "$CKPT_DIR" --output-csv "$RUN_ROOT/eval/d1_grpo_eval.csv"
```

Checks:
- `restored_step` and the resolved actor checkpoint root are printed.
- CSV contains one row per prompt.
- Aggregate accuracy can be recomputed from the CSV.

## Milestone 5: Full Controlled Runs

Status: blocked until debug/evaluation gates pass.

Goal:
- Execute the locked GRPO vs RLOO comparison with fixed data, seed controls, and compute budget.

Checks:
- Same held-out split and evaluation preset.
- Same total training step budget unless explicitly justified.
- Per-run metadata, logs, checkpoints, and eval CSVs preserved.

## Milestone 6: Evidence Extraction

Status: pending runs.

Goal:
- Provide report-ready evidence for I.1 and I.3.

Outputs:
- Reward curves.
- KL curves.
- Diagnostic curves such as `advantage/nonzero_frac` if logged.
- Accuracy/score table with bootstrap confidence intervals from per-prompt CSVs.
