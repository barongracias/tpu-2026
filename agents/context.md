# TPU-2026 Coursework Branch Context

This fork is the Part I practical training/evaluation codebase for the Multi-Agent Systems and Agentic AI coursework. It starts from upstream `borisbolliet/tpu-2026` commit `324abbe4b4e229ea812223856393547db4fbb53e` and the local `coursework` branch currently contains the P0-P6 preparation patches needed for reproducible GRPO/RLOO TPU runs. The main coursework/report repository is the sibling directory `../agentic-ai-coursework`.

## Current Status

- Branch: `coursework`.
- Upstream baseline: `324abbe4b4e229ea812223856393547db4fbb53e`.
- Current committed head: `6a2aef7` (`Add TPU handoff docs and preserve debug LR schedule`).
- Branch is aligned with `origin/coursework` before the local Fred-review follow-up edits.
- Only baseline-owned files were touched in the 8 commits: `bootstrap.sh`, `scripts/config.py`, `scripts/data.py`, `scripts/train.py`, and `scripts/evaluate.py`.
- No Tunix source files were edited.
- Local follow-up edits after Fred's review: `scripts/config.py` makes `SAVE_INTERVAL_STEPS` environment-overridable; `scripts/train.py` records `lr_decay_steps` and `save_interval_steps`; `scripts/evaluate.py` auto-resolves `$CKPT_DIR/actor` when present and records requested/resolved checkpoint paths. These edits are not committed yet.

## What Was Implemented

- P0: `CKPT_DIR`, `INTERMEDIATE_CKPT_DIR`, and `TENSORBOARD_DIR` are environment-overridable.
- P1: `ADV_ESTIMATOR` is passed to `GRPOConfig(advantage_estimator=...)` for `grpo`, `rloo`, or `drgrpo`.
- P2: `RUN_SEED` is threaded into Grain shuffle, rollout config, and `GRPOLearner.data_shuffle_seed`.
- P3: `MAX_STEPS_OVERRIDE` caps debug runs without changing the intended full-run LR schedule. Use `SAVE_INTERVAL_STEPS=50` with 50-step debug runs so the restore check has a checkpoint.
- P4: `evaluate.py` can restore trained LoRA checkpoints through `--ckpt-dir`, optional `--step`, and explicit `--no-restore` for base-model sanity checks.
- P5: `evaluate.py --output-csv` writes per-prompt rows for bootstrap confidence intervals and auditability.
- P6: `train.py` writes `run_metadata.json` into `CKPT_DIR` at run start.
- `bootstrap.sh` pins Tunix to `683256db1a0919b5cfd46cee52cebc96331494fb` to avoid HEAD drift.

## Current Experiment Intent

The planned I.3 controlled comparison is standard GRPO vs RLOO at fixed compute. This is motivated by the coursework theory around group-mean advantage normalisation at small group size `K=2`: RLOO keeps reward-difference magnitude where standard GRPO collapses to mostly sign information.

Target matrix:

| Run | Estimator | Seed | Purpose |
| --- | --- | ---: | --- |
| D1 | grpo | 0 | 50-step debug baseline. |
| D2 | rloo | 0 | 50-step debug estimator-switch check. |
| R1 | grpo | 0 | Full baseline reproduction. |
| R3 | rloo | 0 | Full controlled variant. |
| R4 | rloo | 1 | Second-seed variant if TPU time permits. |
| R2 | grpo | 1 | Second-seed baseline if TPU time permits. |

## What To Read First

1. `agents/context.md` in this repository.
2. `agents/plan.md` in this repository.
3. `agents/report_notes.md` in this repository.
4. Main coursework repo: `../agentic-ai-coursework/agents/context.md`.
5. Main coursework repo: `../agentic-ai-coursework/experiments/runbooks/tpu_day1_runbook.md`.
6. Patched code: `scripts/config.py`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/data.py`, `bootstrap.sh`.

## Safe Useful Commands

```bash
git status --short --branch
git log --oneline --decorate --max-count=12
git diff --check
env PYTHONPYCACHEPREFIX=/tmp/tpu2026-pycache python3 -m py_compile scripts/config.py scripts/data.py scripts/train.py scripts/evaluate.py
```

On a TPU VM after setup, use the main coursework runbook before any full run.

## Things Not To Do

- Do not push from an automated session.
- Do not edit Tunix unless a TPU/debug failure proves the pinned Tunix API itself is wrong.
- Do not start a full 5 hour run until the debug gates pass.
- Do not store checkpoints or TensorBoard logs only under `/tmp`.
- Do not claim numerical results until there are saved logs and per-prompt evaluation outputs.

## Open Checks Before Full Runs

- Confirm JAX backend reports TPU.
- Confirm `ADV_ESTIMATOR=rloo` is accepted by the pinned Tunix commit on the TPU VM.
- Confirm `MAX_STEPS_OVERRIDE=50` stops debug training at 50 steps while `LR_DECAY_STEPS` remains the full-run value.
- Confirm checkpoints and TensorBoard files are written to persistent `$HOME/tpu-runs/...` paths.
- Confirm a 50-step debug run writes a checkpoint when `SAVE_INTERVAL_STEPS=50`.
- Confirm `evaluate.py --ckpt-dir ... --output-csv ...` restores a trained checkpoint, resolves the actor checkpoint root, and writes per-prompt rows.
- Confirm W&B logs to the intended team/entity project, not the upstream default.
