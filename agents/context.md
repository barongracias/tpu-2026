# TPU-2026 Coursework Branch Context

This fork is the Part I practical training/evaluation codebase for the Multi-Agent Systems and Agentic AI coursework. It starts from upstream `borisbolliet/tpu-2026` commit `324abbe4b4e229ea812223856393547db4fbb53e` and the local `coursework` branch currently contains the P0-P6 preparation patches needed for reproducible GRPO/RLOO TPU runs. The main coursework/report repository is the sibling directory `../agentic-ai-coursework`.

## Current Status

- Branch: `coursework`.
- Upstream baseline: `324abbe4b4e229ea812223856393547db4fbb53e`.
- Current pulled head: `4339ba8` on `coursework` / `origin/coursework`.
- Branch is aligned with `origin/coursework`.
- Only baseline-owned files were touched in the 8 commits: `bootstrap.sh`, `scripts/config.py`, `scripts/data.py`, `scripts/train.py`, and `scripts/evaluate.py`.
- No Tunix source files were edited.
- D1 GRPO 50-step debug completed successfully: step 50 reached, actor checkpoint restored, TensorBoard/W&B emitted evidence, and greedy eval CSV was written.
- D2 RLOO 50-step debug completed successfully: step 50 reached, actor checkpoint restored, TensorBoard/W&B emitted evidence, and greedy eval CSV was written.
- R1 GRPO seed 0 full training completed; do not start R3/R4/R2 until Baron approves after R1 evaluation/review.

## What Was Implemented

- P0: `CKPT_DIR`, `INTERMEDIATE_CKPT_DIR`, and `TENSORBOARD_DIR` are environment-overridable.
- P1: `ADV_ESTIMATOR` is passed to `GRPOConfig(advantage_estimator=...)` for `grpo`, `rloo`, or `drgrpo`.
- P2: `RUN_SEED` is threaded into Grain shuffle, rollout config, and `GRPOLearner.data_shuffle_seed`.
- P3: `MAX_STEPS_OVERRIDE` caps debug runs without changing the intended full-run LR schedule. Use `SAVE_INTERVAL_STEPS=50` with 50-step debug runs so the restore check has a checkpoint.
- P4: `evaluate.py` can restore trained LoRA checkpoints through `--ckpt-dir`, optional `--step`, and explicit `--no-restore` for base-model sanity checks.
- P5: `evaluate.py --output-csv` writes per-prompt rows for bootstrap confidence intervals and auditability.
- P6: `train.py` writes `run_metadata.json` into `CKPT_DIR` at run start. Hygiene follow-up now records repo root, launch cwd, and train/test data dirs; git commit resolution is repo-root-aware for run-local launches.
- `bootstrap.sh` pins Tunix to `683256db1a0919b5cfd46cee52cebc96331494fb` to avoid HEAD drift.

## Current Experiment Intent

The planned I.3 controlled comparison is standard GRPO vs RLOO at fixed compute. This is motivated by the coursework theory around group-mean advantage normalisation at small group size `K=2`: RLOO keeps reward-difference magnitude where standard GRPO collapses to mostly sign information.

Target matrix:

| Run | Estimator | Seed | Purpose |
| --- | --- | ---: | --- |
| D1 | grpo | 0 | complete: 50-step debug baseline passed. |
| D2 | rloo | 0 | complete: 50-step debug variant passed. |
| R1 | grpo | 0 | complete: full baseline training finished; eval pending. |
| R3 | rloo | 0 | Full controlled variant. |
| R4 | rloo | 1 | Second-seed variant if TPU time permits. |
| R2 | grpo | 1 | Second-seed baseline if TPU time permits. |

R1 completed training:
- Session: `r1-grpo-full-s0` exited after training completion
- Run root: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231`
- Launched from repo cwd at HEAD `4339ba8` with `ADV_ESTIMATOR=grpo`, `RUN_SEED=0`, and no `MAX_STEPS_OVERRIDE`.
- Uses `TRAIN_DATA_DIR=/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231/data/train` and `TEST_DATA_DIR=/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231/data/test`.
- W&B: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R1-grpo-full-s0`
- Final monitor: tmux exited; `Training finished.` present; final checkpoint `ckpts/actor/3364` exists; TensorBoard event file exists; W&B emitted step-order warnings.

Debug evidence:
- D1 run root: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D1-grpo-debug-20260608_142021`; restored step 50; eval `correct=30/64`, `acc=46.88%`, `partial=50.00%`, `format=6.25%`.
- D2 run root: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D2-rloo-debug-20260608_144102`; restored step 50; eval `correct=30/64`, `acc=46.88%`, `partial=46.88%`, `format=4.69%`.
- D2 initial repo-cwd attempt at `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D2-rloo-debug-20260608_143942` failed before training due to the repo-local TFDS/protobuf metadata cache issue; the successful retry ran from run-local cwd.
- D2 `run_metadata.json` records `tpu2026_commit=unknown` because the successful retry ran outside the git repo; actual synced HEAD before launch was `be631b2`.
- Both debug runs emitted W&B step-order warnings.
- Hygiene patch is committed, pushed, pulled, and validated at `4339ba8`: `TRAIN_DATA_DIR` and `TEST_DATA_DIR` are env-overridable; metadata records repo root, launch cwd, data dirs, and the real git commit.

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
- Do not start any full run other than approved R1 GRPO seed 0.
- Do not store checkpoints or TensorBoard logs only under `/tmp`.
- Do not claim numerical results until there are saved logs and per-prompt evaluation outputs.

## Open Checks Before Full Runs

- JAX backend reports TPU.
- `ADV_ESTIMATOR=rloo` is accepted by the pinned Tunix commit on the TPU VM.
- `MAX_STEPS_OVERRIDE=50` stops debug training at 50 steps while `LR_DECAY_STEPS` remains the full-run value.
- Checkpoints and TensorBoard files are written to persistent `$HOME/tpu-runs/...` paths.
- A 50-step debug run writes a checkpoint when `SAVE_INTERVAL_STEPS=50`.
- `evaluate.py --ckpt-dir ... --output-csv ...` restores trained checkpoints, resolves the actor checkpoint root, and writes per-prompt rows for both D1 and D2.
- W&B logs to the intended team/entity project, not the upstream default.
- R1 launch requirement: start from repo cwd so metadata records the real commit, and set `TRAIN_DATA_DIR`/`TEST_DATA_DIR` to `$RUN_ROOT/data/train` and `$RUN_ROOT/data/test` to avoid repo-local TFDS/protobuf cache issues.
