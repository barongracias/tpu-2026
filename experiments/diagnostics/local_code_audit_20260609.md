# Local Code Correctness Audit 2026-06-09

## Scope

Static/local audit of the sibling `../tpu-2026` fork at local commit `99cb7f8`. The local checkout does not yet include the TPU-side uncommitted D3 `NUM_GENERATIONS` env patch. Files inspected:

- `scripts/config.py`
- `scripts/train.py`
- `scripts/data.py`
- `scripts/rewards.py`
- `scripts/evaluate.py`
- `scripts/model.py`

Validation run locally:

- `git diff --check` in `../tpu-2026`: passed.
- `env PYTHONPYCACHEPREFIX=/tmp/tpu2026-pycache python3 -m py_compile scripts/config.py scripts/data.py scripts/train.py scripts/evaluate.py scripts/rewards.py scripts/model.py`: passed.

## Findings

### F1: Reward scale strongly favours format over numeric correctness

Severity: high for training quality; not an infrastructure bug.

Source:

- `scripts/rewards.py`: `match_format_exactly`, `match_format_approximately`, `check_answer`, `check_numbers`.

Local sanity check with stubbed data constants:

| Case | Reward components | Total |
| --- | ---: | ---: |
| Good format, correct number | `[3.0, 2.5, 3.0, 1.5]` | `10.0` |
| Good format, wrong number | `[3.0, 2.5, -1.0, 0.0]` | `4.5` |
| Good format, non-numeric answer | `[3.0, 2.5, -0.5, 0.0]` | `5.0` |
| `<answer>4</answer>` without reasoning wrapper | `[0.0, -0.5, 0.0, 1.5]` | `1.0` |
| Plain correct answer with no tags | `[0.0, -2.5, 0.0, 0.0]` | `-2.5` |
| Empty response | `[0.0, -2.5, 0.0, 0.0]` | `-2.5` |

Interpretation:

- A fully formatted wrong answer is rewarded much more than a numerically correct answer that lacks the full wrapper.
- A fully formatted non-numeric answer still receives a large positive reward because exact/approximate format rewards dominate the answer penalty.
- This directly matches the R1 diagnosis: format improves while numeric correctness deteriorates.
- The issue is an objective-design problem rather than evidence that checkpoint restore, TFDS split, or evaluation is fundamentally broken.

Recommended controls:

- Down-weight format rewards, especially `match_format_exactly` and `match_format_approximately`.
- Consider making exact numeric correctness the dominant positive term and treating format as a small auxiliary bonus.
- Add a reward sanity check script to the repo before further reward experiments.

### F2: Evaluation restore path appears correct, but `--no-restore` should be treated as zero-LoRA/base-wrapper unless verified

Severity: medium for interpretation.

Source:

- `scripts/evaluate.py`: constructs a LoRA-wrapped model, optionally restores LoRA, and always samples from `lora`.
- `scripts/model.py`: `get_lora_model(base, mesh)` wraps the base model with Qwix LoRA adapters.

Interpretation:

- Trained checkpoint eval correctly restores LoRA via `CheckpointManager(..., restore_only_lora_params=True)` and prints the restored step.
- Base eval with `--no-restore` skips checkpoint restore but still samples from the LoRA-wrapped model. This is probably equivalent to base if Qwix initializes LoRA as a no-op, but that should be verified on TPU or by source inspection.
- Current base result is stable and strong enough to act as a useful reference, but final report wording should say `--no-restore base-wrapper eval` unless no-op LoRA init is confirmed.

Recommended control:

- TPU-side: verify Qwix LoRA initialisation is no-op, or add a true-base evaluation path that samples from `base` rather than `lora` when `--no-restore` is set.

### F3: Data split and checkpoint restore are not the dominant suspected failures

Severity: low.

Source:

- `scripts/data.py`: train split uses shuffled GSM8K train, then fixed train/val split; eval uses shuffled GSM8K test with fixed `RUN_SEED` and `NUM_TEST_BATCHES=64`.
- `scripts/evaluate.py`: resolves `$CKPT_DIR/actor`, restores requested steps, and records requested/resolved checkpoint metadata in CSV.

Interpretation:

- Run-local TFDS caches were necessary because repo-local caches triggered protobuf metadata failures, but successful evals used fresh caches and stable restore markers.
- The base model remains strong on the same eval path, making a dominant cache/eval artefact unlikely.

### F4: K=8 patch is currently TPU-side only

Severity: bookkeeping.

Source:

- Local `scripts/config.py` at `99cb7f8` still has `NUM_GENERATIONS = 2`.
- TPU-side D3 report says `scripts/config.py` now has `NUM_GENERATIONS = int(os.environ.get("NUM_GENERATIONS", "2"))` as an uncommitted local patch.

Recommended control:

- If D3 confirms K=8 remains useful, commit the env override plus metadata/logging additions after review.
- Metadata should record `num_generations`, `beta`, `epsilon`, rollout temperature/top-k/top-p, and generation length for every run.

## Early Stopping Assessment

Early stopping helps explain R1 but is not sufficient as the main improvement claim:

- R1 step 2000 is best retained checkpoint at 24/64, but still below base at 31/64.
- R1 degradation after step 2000 supports overtraining/KL drift.
- Early stopping could be reported as a diagnostic or stabilisation control, but selecting a checkpoint by test accuracy would be invalid. Use validation curves/reward/KL/held-out protocol if early stopping becomes a method.

## Recommended Next Plan

Local follow-up:

- Keep static audit notes current.
- If TPU-side proposes patches, review them before commit: `NUM_GENERATIONS` env override, metadata logging, reward sanity checks, and any true-base eval change.
- Start report-side plotting/data extraction only after TPU diagnostic artefacts are copied or made accessible.

TPU follow-up:

- Finish D3/D4 diagnostics and report whether K=8 stays stable beyond 50 steps.
- Run reward sanity checks in the TPU environment.
- Verify no-op LoRA base behaviour or add a true-base eval path.
- Do not start another full run until reward weighting or K/group-size hypothesis is deliberately selected.

Likely next experiment after D3:

- Medium K=8 debug (`250`-`500` steps) with frequent checkpoints and empty-response checks, or
- Reward-weight debug that reduces format reward dominance while preserving numeric correctness as the main objective.
