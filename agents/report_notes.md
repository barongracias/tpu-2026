# TPU-2026 Coursework Branch Notes

## Current Headline State

The `coursework` branch contains the P0-P6 preparation patches for Part I TPU usage and is aligned with `origin/coursework` at `e3ebeef`. D1 GRPO 50-step debug has passed; D2 RLOO debug is next. Full 5h runs remain blocked until D2 passes.

## 2026-06-08: Patch Review

Verified branch state:
- Current branch: `coursework`.
- Upstream baseline and `origin/coursework`: `324abbe4b4e229ea812223856393547db4fbb53e`.
- Current committed head: `e3ebeef`.
- Branch aligned with `origin/coursework` before local review follow-up edits.
- 8-commit diff touches only `bootstrap.sh`, `scripts/config.py`, `scripts/data.py`, `scripts/train.py`, and `scripts/evaluate.py`.
- Working tree after Fred review has local modifications to `scripts/config.py`, `scripts/train.py`, `scripts/evaluate.py`, and handoff docs.

Patch mapping:
- P0 persistent paths: implemented in `scripts/config.py`.
- P1 estimator switch: implemented in `scripts/config.py` and `scripts/train.py`.
- P2 seed controls: implemented in `scripts/config.py`, `scripts/data.py`, and `scripts/train.py`.
- P3 step override: implemented in `scripts/config.py`; committed follow-up keeps LR schedule full-run shaped and makes `SAVE_INTERVAL_STEPS` env-overridable for checkpointed debug runs.
- P4 checkpoint restore: implemented in `scripts/evaluate.py`.
- P5 per-prompt CSV: implemented in `scripts/evaluate.py`.
- P6 metadata: implemented in `scripts/train.py`; committed follow-up records W&B-created run ids, LR decay, and save interval.
- Tunix pin: implemented in `bootstrap.sh`.

Verification run locally:
- `git diff --check`: passed.
- `env PYTHONPYCACHEPREFIX=/tmp/tpu2026-pycache python3 -m py_compile scripts/config.py scripts/data.py scripts/train.py scripts/evaluate.py`: passed.
- AST parse of the same files: passed before local follow-up; py_compile passed after local follow-up.

Caveats:
- D1 GRPO TPU debug passed on 2026-06-08.
- Final training/evaluation metrics do not exist yet; D1 numbers are debug-only.
- The `ADV_ESTIMATOR=rloo` path still needs a day-one TPU debug run against the pinned Tunix commit.
- The evaluation restore path still needs a real checkpoint to prove it restores the intended step.
- No Tunix edits are currently justified.

## Push Readiness

Current push-readiness note is superseded: Fred-review follow-up was committed and pushed as `e3ebeef`.

## 2026-06-08: Fred review follow-up

Accepted findings:
- `SAVE_INTERVAL_STEPS=500` is too high for a 50-step debug run if Tunix does not save at final step. Local code now makes `SAVE_INTERVAL_STEPS` env-overridable; debug runs should set `SAVE_INTERVAL_STEPS=50`.
- Actor checkpoints are likely under `$CKPT_DIR/actor`. Local `evaluate.py` now accepts either `$CKPT_DIR` or `$CKPT_DIR/actor` by resolving an `actor/` child if present.

No Tunix edit is justified from this review.

## 2026-06-08: D1 GRPO debug attempt failed before training

Run root:
- `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D1-grpo-debug-20260608_135758`

Outcome:
- Training did not reach step 50.
- Failure happened during Hugging Face model download.
- Error: gated repo 401 for `google/gemma-3-1b-it`.
- `~/.env` absent; `HF_TOKEN`, `WANDB_API_KEY`, `KAGGLE_USERNAME`, and `KAGGLE_KEY` missing.

Files created:
- `RUN_ROOT.txt`
- `ckpts/run_metadata.json`
- `logs/train.log`

Metadata sanity:
- `advantage_estimator=grpo`
- `max_steps=50`
- `lr_decay_steps=3364`
- `save_interval_steps=50`

Next action:
- Superseded on 2026-06-08: secrets were configured and D1 was rerun successfully.


## 2026-06-08: D1 GRPO debug passed

Run root:
- `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D1-grpo-debug-20260608_142021`

Outcome:
- Training reached step 50.
- `Training finished.` present in `logs/train.log`.
- Checkpoints: `ckpts/actor/1` and `ckpts/actor/50`.
- TensorBoard event file: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D1-grpo-debug-20260608_142021/tensorboard/events.out.tfevents.1780928465.t1v-n-0339f27d-w-0`.
- W&B: https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/D1-grpo-debug-seed0.

Metadata:
- `tpu2026_commit=e3ebeef4a3be9ad978959b66d4ecb16deecccefb`
- `advantage_estimator=grpo`
- `run_seed=0`
- `max_steps=50`
- `lr_decay_steps=3364`
- `save_interval_steps=50`

Evaluation:
- Restored `restored_step=50`.
- Resolved checkpoint root: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D1-grpo-debug-20260608_142021/ckpts/actor`.
- Greedy eval: `correct=30/64`, `acc=46.88%`, `partial=50.00%`, `format=6.25%`.
- CSV: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D1-grpo-debug-20260608_142021/eval/eval_greedy.csv`.

Caveats:
- Debug run only, not final baseline performance.
- Eval from repo cwd hit a TFDS metadata/protobuf issue; rerun from `RUN_ROOT/eval` completed.
- W&B emitted step-order warnings near the end; inspect scalar traces before using plots.

Decision:
- D2 RLOO 50-step debug is operationally safe to run next.
- Keep all controls identical except `ADV_ESTIMATOR=rloo` and run id/root.
- Full runs remain blocked until D2 passes.
