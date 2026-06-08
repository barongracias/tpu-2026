# TPU-2026 Coursework Branch Notes

## Current Headline State

The `coursework` branch contains the P0-P6 preparation patches for Part I TPU usage and is aligned with `origin/coursework` at `be631b2`. D1 GRPO and D2 RLOO 50-step debug runs have passed. Full runs remain blocked until Baron reviews the debug outcomes and approves them.

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
- The `ADV_ESTIMATOR=rloo` path passed a day-one TPU debug run against the pinned Tunix commit.
- The evaluation restore path restored real step-50 checkpoints for both D1 and D2.
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

Decision after D1:
- D1 passed the GRPO debug gate.
- D2 RLOO 50-step debug was operationally safe to run next.
- Full runs remained blocked until D2 passed.

## 2026-06-08: D2 RLOO debug passed

Repo state:
- Synced with `git pull --ff-only origin coursework` before D2.
- Clean status before launch.
- HEAD before launch: `be631b2`.

Initial D2 attempt:
- Run root: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D2-rloo-debug-20260608_143942`
- Failed before training due to the repo-local TFDS/protobuf metadata cache issue.
- No actor checkpoint was produced.

Successful D2 retry:
- Run root: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D2-rloo-debug-20260608_144102`
- Estimator: `rloo`
- Seed: `0`
- Max steps: `50`
- Save interval: `50`
- Training reached step 50 and logged `Training finished.`
- Persistent checkpoints: `ckpts/actor/1` and `ckpts/actor/50`
- TensorBoard event file: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D2-rloo-debug-20260608_144102/tensorboard/events.out.tfevents.1780929695.t1v-n-0339f27d-w-0`
- W&B run: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/D2-rloo-debug-seed0`

Metadata:
- `advantage_estimator`: `rloo`
- `run_seed`: `0`
- `max_steps`: `50`
- `lr_decay_steps`: `3364`
- `save_interval_steps`: `50`
- `data_source`: `tfds`
- `ckpt_dir`: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D2-rloo-debug-20260608_144102/ckpts`
- `intermediate_ckpt_dir`: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D2-rloo-debug-20260608_144102/intermediate_ckpt`
- `tensorboard_dir`: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D2-rloo-debug-20260608_144102/tensorboard`
- `wandb_project`: `agentic-ai-coursework`
- `wandb_entity`: `barongracias-university-of-cambridge`
- Caveat: `tpu2026_commit` is `unknown` because the successful retry ran from the run-local cwd to avoid the TFDS cache issue; the synced repo HEAD before launch was `be631b2`.

Evaluation:
- Launched from `$RUN_ROOT/eval`.
- Restored step: `50`
- Resolved checkpoint dir: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D2-rloo-debug-20260608_144102/ckpts/actor`
- Output CSV: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D2-rloo-debug-20260608_144102/eval/eval_greedy.csv`
- Final metrics: `correct=30/64`, `acc=46.88%`, `partial=46.88%`, `format=4.69%`

Warnings:
- W&B emitted step-order warnings such as attempts to log step 0 after current step 49 or 50.
- Successful W&B run reused the run id from the initial failed D2 attempt.

Decision after D2:
- D2 passes the RLOO debug gate.
- D1 and D2 debug gates are both passed.
- Do not start full runs until Baron reviews these results and explicitly approves.
