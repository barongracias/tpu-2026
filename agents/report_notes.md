# TPU-2026 Coursework Branch Notes

## Current Headline State

The `coursework` branch contains the P0-P6 preparation patches for Part I TPU usage and is aligned with `origin/coursework` at `6a2aef7`. Local follow-up edits from Fred's review make debug checkpoint saving and evaluation checkpoint-root handling more robust; these follow-up edits are not committed yet.

## 2026-06-08: Patch Review

Verified branch state:
- Current branch: `coursework`.
- Upstream baseline and `origin/coursework`: `324abbe4b4e229ea812223856393547db4fbb53e`.
- Current committed head: `6a2aef7`.
- Branch aligned with `origin/coursework` before local review follow-up edits.
- 8-commit diff touches only `bootstrap.sh`, `scripts/config.py`, `scripts/data.py`, `scripts/train.py`, and `scripts/evaluate.py`.
- Working tree after Fred review has local modifications to `scripts/config.py`, `scripts/train.py`, `scripts/evaluate.py`, and handoff docs.

Patch mapping:
- P0 persistent paths: implemented in `scripts/config.py`.
- P1 estimator switch: implemented in `scripts/config.py` and `scripts/train.py`.
- P2 seed controls: implemented in `scripts/config.py`, `scripts/data.py`, and `scripts/train.py`.
- P3 step override: implemented in `scripts/config.py`; committed follow-up keeps LR schedule full-run shaped; local Fred-review follow-up makes `SAVE_INTERVAL_STEPS` env-overridable for checkpointed debug runs.
- P4 checkpoint restore: implemented in `scripts/evaluate.py`.
- P5 per-prompt CSV: implemented in `scripts/evaluate.py`.
- P6 metadata: implemented in `scripts/train.py`; committed follow-up records W&B-created run ids; local Fred-review follow-up records LR decay and save interval.
- Tunix pin: implemented in `bootstrap.sh`.

Verification run locally:
- `git diff --check`: passed.
- `env PYTHONPYCACHEPREFIX=/tmp/tpu2026-pycache python3 -m py_compile scripts/config.py scripts/data.py scripts/train.py scripts/evaluate.py`: passed.
- AST parse of the same files: passed before local follow-up; py_compile passed after local follow-up.

Caveats:
- No TPU runtime test has been run yet.
- No training/evaluation metrics exist yet.
- The `ADV_ESTIMATOR=rloo` path still needs a day-one TPU debug run against the pinned Tunix commit.
- The evaluation restore path still needs a real checkpoint to prove it restores the intended step.
- No Tunix edits are currently justified.

## Push Readiness

Do not push the current working tree as-is if the Fred-review follow-up should be included, because `git push` would omit uncommitted modifications. The next human action is to review and commit the local `scripts/config.py`, `scripts/train.py`, `scripts/evaluate.py`, and agent-doc updates, then push the branch once syntax/diff checks still pass.

## 2026-06-08: Fred review follow-up

Accepted findings:
- `SAVE_INTERVAL_STEPS=500` is too high for a 50-step debug run if Tunix does not save at final step. Local code now makes `SAVE_INTERVAL_STEPS` env-overridable; debug runs should set `SAVE_INTERVAL_STEPS=50`.
- Actor checkpoints are likely under `$CKPT_DIR/actor`. Local `evaluate.py` now accepts either `$CKPT_DIR` or `$CKPT_DIR/actor` by resolving an `actor/` child if present.

No Tunix edit is justified from this review.
