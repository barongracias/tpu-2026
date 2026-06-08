# TPU-2026 Coursework Branch Notes

## Current Headline State

The `coursework` branch contains the P0-P6 preparation patches for Part I TPU usage. The branch is ahead of `origin/coursework` by 8 commits. A local follow-up fix has been applied after `35faf32` to preserve the full learning-rate schedule under `MAX_STEPS_OVERRIDE` and to save the resolved W&B run id in metadata; this follow-up is not committed yet.

## 2026-06-08: Patch Review

Verified branch state:
- Current branch: `coursework`.
- Upstream baseline and `origin/coursework`: `324abbe4b4e229ea812223856393547db4fbb53e`.
- Current committed head before local edits: `35faf3241f9a9b424e805e9f0e1ca97cf60f8101`.
- 8 commits ahead of `origin/coursework`.
- 8-commit diff touches only `bootstrap.sh`, `scripts/config.py`, `scripts/data.py`, `scripts/train.py`, and `scripts/evaluate.py`.
- Working tree after review has local modifications to `scripts/config.py` and `scripts/train.py`.

Patch mapping:
- P0 persistent paths: implemented in `scripts/config.py`.
- P1 estimator switch: implemented in `scripts/config.py` and `scripts/train.py`.
- P2 seed controls: implemented in `scripts/config.py`, `scripts/data.py`, and `scripts/train.py`.
- P3 step override: implemented in `scripts/config.py`; local follow-up keeps LR schedule full-run shaped.
- P4 checkpoint restore: implemented in `scripts/evaluate.py`.
- P5 per-prompt CSV: implemented in `scripts/evaluate.py`.
- P6 metadata: implemented in `scripts/train.py`; local follow-up records W&B-created run ids.
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

Do not push the current working tree as-is if the local follow-up fix should be included, because `git push` would omit uncommitted modifications. The next human action is to review and commit the local `scripts/config.py` and `scripts/train.py` fix, then push the branch once syntax/diff checks still pass.
