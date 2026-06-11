# TPU-2026 Coursework Branch Notes

## Current Headline State

The `coursework` branch contains the P0-P6 preparation patches for Part I TPU usage and was aligned with `origin/coursework` at `820fad6` for earlier runs. D1/D2, R1/R3, D3/D4, and R5 have completed. R6 RLOO K=8 and chained RLOO K=2 full training were launched from non-deterministic branch `harvey` at W&B-recorded commit `8a0f7f266552cb2666710ac589cb6bda5cd40121`; the later note-time checkout was `harvey-grpo-k8-rerun`. The R6 runs still need retained-checkpoint evaluation before they are used as performance evidence. R7 is the deterministic rerun: K=2 is running on this VM from `harvey-grpo-k8-rerun` commit `71aab87dee2d2c78256384d084d063d8b40c9e0c`, while Baron has sent the matching K=8 launch prompt/command to a second TPU VM.

Local run notes, diagnostics, manifests, and runbooks for the current TPU work are now tracked under `experiments/`. Start with `experiments/README.md`, then use `experiments/baseline/`, `experiments/variants/`, and `experiments/diagnostics/` for run-specific summaries; TPU-side raw logs/CSVs remain under each `$RUN_ROOT`.

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
- R1 GRPO seed 0 is approved to start after the hygiene patch; do not start R3/R4/R2 yet.

## 2026-06-08: Pre-full-run hygiene patch

Purpose:
- Make `TRAIN_DATA_DIR` and `TEST_DATA_DIR` environment-overridable so full/debug runs can avoid repo-local TFDS/protobuf cache state.
- Make `train.py` resolve the git commit from the repository root, even when launched from a run-local cwd.
- Add `repo_root`, `launch_cwd`, `train_data_dir`, and `test_data_dir` to `run_metadata.json`.

Status:
- Patch committed, pushed, pulled, and validated at `4339ba8`.
- `git diff --check` passed.
- `py_compile` passed for `scripts/config.py`, `scripts/data.py`, `scripts/train.py`, and `scripts/evaluate.py`.
- R1 GRPO seed 0 is approved to start; do not start R3/R4/R2 yet.

R1 launch requirements:
- Launch from repo cwd so `run_metadata.json` records the real git commit.
- Set `TRAIN_DATA_DIR=$RUN_ROOT/data/train` and `TEST_DATA_DIR=$RUN_ROOT/data/test` to avoid repo-local TFDS/protobuf cache issues.
- Keep all run artefacts under persistent `$HOME/tpu-runs/part-i/...`.

## 2026-06-08: R1 GRPO full seed 0 training complete

Run root:
- `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231`

Launch state:
- tmux session: `r1-grpo-full-s0` exited after completion
- Repo HEAD: `4339ba8`
- Launch cwd: repo root `~/tpu-2026`
- Estimator: `grpo`
- Seed: `0`
- `MAX_STEPS_OVERRIDE` unset for full training.
- `TRAIN_DATA_DIR=/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231/data/train`
- `TEST_DATA_DIR=/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231/data/test`
- Artefacts are under persistent `$HOME/tpu-runs`.
- W&B run: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R1-grpo-full-s0`
- Final monitor: tmux exited; `Training finished.` present; final checkpoint `ckpts/actor/3364` exists; TensorBoard event file exists.
- Metadata records commit `4339ba84ba3396d9e6defaef218ecded435fc787`, estimator `grpo`, seed `0`, `max_steps=3364`, and run-local train/test data dirs.
- W&B emitted repeated step-order warnings at final step `3364`; plots should be inspected with care.
- R1 evaluation is pending; do not start R3/R4/R2 yet.

Restrictions:
- Do not start R3/R4/R2 until Baron approves after R1 review.

## 2026-06-08: R1 GRPO full seed 0 greedy evaluation

Trained checkpoint eval:
- Checkpoint requested: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231/ckpts`
- Resolved checkpoint dir: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231/ckpts/actor`
- Restored step: `3364`
- CSV: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231/eval/r1_grpo_step3364_greedy.csv`
- Log: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231/logs/eval_r1_grpo_step3364_greedy.log`
- Metrics: `correct=12/64`, `acc=18.75%`, `partial=18.75%`, `format=35.94%`

Base-model eval:
- Restore mode: `--no-restore`
- CSV: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231/eval/base_greedy_seed0.csv`
- Log: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231/logs/eval_base_greedy_seed0.log`
- Metrics: `correct=31/64`, `acc=48.44%`, `partial=48.44%`, `format=1.56%`

Warnings and failures:
- Initial trained eval restored `ckpts/actor/3364` but failed when reading the existing `$RUN_ROOT/data/train` TFDS cache with the protobuf `FieldDescriptor.label` issue.
- Trained eval succeeded after rerunning from `$RUN_ROOT/eval` with fresh eval-local TFDS caches.
- Initial base eval failed when reusing the trained eval cache; base eval succeeded with a separate fresh eval-local TFDS cache.
- Common nonfatal warnings included LoRA RNG initialization warning and TensorFlow CUDA/oneDNN messages.

Decision:
- R1 is valid as the completed I.1 GRPO baseline control from an operational/checkpoint-restore perspective.
- Numerically, trained R1 greedy underperforms the base model on the 64-prompt eval sample, so inspect outputs/curves before interpreting quality.
- Do not start R3/R4/R2 until Baron approves after review.

## 2026-06-08: R1 intermediate-checkpoint diagnostic

Metrics:
| Eval | Exact | Partial | Format |
| --- | ---: | ---: | ---: |
| Base greedy | 31/64 (48.44%) | 31/64 (48.44%) | 1/64 (1.56%) |
| R1 step 2000 | 24/64 (37.50%) | 26/64 (40.62%) | 28/64 (43.75%) |
| R1 step 2500 | 11/64 (17.19%) | 11/64 (17.19%) | 23/64 (35.94%) |
| R1 step 3000 | 8/64 (12.50%) | 8/64 (12.50%) | 16/64 (25.00%) |
| R1 step 3364 | 12/64 (18.75%) | 12/64 (18.75%) | 23/64 (35.94%) |

Trend:
- Degradation is gradual after step 2000 rather than isolated to the final step.
- Step 2000 is the best retained R1 checkpoint but remains below base exact accuracy.
- Steps 2500, 3000, and 3364 are materially worse than base and worse than step 2000.

TensorBoard evidence:
- Scalar summary: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231/eval/r1_tensorboard_scalar_summary.txt`
- Found KL, pg_clipfrac, reward component, and completion length tags.
- Did not find `advantage/nonzero_frac` or ratio-spread tags in the event file.
- Eval KL values: near step 1984 `0.324`, 2496 `0.545`, 3008 `0.859`, 3328 `0.488`.
- Train KL values: near step 1984 `0.290`, 2496 `0.830`, 3008 `0.977`, 3364 `0.499`.
- `actor/train/pg_clipfrac` and `actor/eval/pg_clipfrac` stayed `0.0`.
- Eval format rewards improved while `check_numbers` fell from initial `0.674` to about `0.285` near the end, consistent with format optimisation at the expense of numeric reliability.

Qualitative CSV pattern:
- Base correct / final trained wrong: 23 prompts.
- Final trained correct / base wrong: 4 prompts.
- Both wrong: 29 prompts.
- Trained responses more often use requested `<reasoning>`/`<answer>` structure, but arithmetic and problem interpretation often degrade.
- Response lengths are not a simple collapse to empty text: final trained responses average about `176.9` words vs base `143.7`, with max around `596` words.

Recommendation:
- Do not approve R3 immediately.
- First review R1 W&B/TensorBoard curves and the per-prompt CSVs.
- Consider changing R3/full-run controls before more TPU time: use an early checkpoint strategy, reduce run length, adjust reward weighting, or run a small controlled diagnostic with revised evaluation/training settings.

## 2026-06-09: R3 RLOO full seed 0 completed

Run root:
- `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R3-rloo-full-s0-20260608_230810`

Training:
- tmux session exited after completion.
- `Training finished.` present in `logs/train.log`.
- Metadata records commit `99cb7f88b0e49a7d65bb0c5274734a190833aa3f`, estimator `rloo`, seed `0`, `max_steps=3364`, and run-local train/test data dirs.
- W&B run: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R3-rloo-full-s0-20260608_230810`
- Retained checkpoints: `ckpts/actor/2000`, `2500`, `3000`, `3364`.

Auto-evaluation:
| Eval | Exact | Partial | Format |
| --- | ---: | ---: | ---: |
| R3 step 2000 | 1/64 (1.56%) | 1/64 (1.56%) | 10/64 (15.62%) |
| R3 step 2500 | 6/64 (9.38%) | 7/64 (10.94%) | 16/64 (25.00%) |
| R3 step 3000 | 1/64 (1.56%) | 2/64 (3.12%) | 4/64 (6.25%) |
| R3 step 3364 | 1/64 (1.56%) | 1/64 (1.56%) | 5/64 (7.81%) |

Logs and CSVs:
- Summary: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R3-rloo-full-s0-20260608_230810/eval/r3_eval_summary.txt`
- CSVs: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R3-rloo-full-s0-20260608_230810/eval/r3_rloo_step{2000,2500,3000,3364}_greedy.csv`
- Logs: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R3-rloo-full-s0-20260608_230810/logs/eval_r3_rloo_step{2000,2500,3000,3364}_greedy.log`

Warnings and interpretation:
- W&B step-order warnings occurred again at final step `3364`.
- Eval logs include fresh TFDS cache `Variant folder ... has no dataset_info.json` warnings; these are expected for new per-step caches and did not stop eval.
- No fatal training or eval failure was found in inspected completion markers.
- R3 is operationally complete as the RLOO comparison, but numerically poor: best retained checkpoint is step 2500 at only `6/64` exact.

Decision:
- Do not start R4/R2 or K=8 yet.
- Review R1/R3 reward curves and output CSVs before any further full run.
- The current evidence points to reward/training-control problems rather than a single checkpoint-selection issue.

## 2026-06-09: R1/R3 report-ready diagnosis

Summary metrics:
| Eval | Exact | Partial | Format | Mean words | Empty responses |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base greedy | 31/64 (48.44%) | 31/64 (48.44%) | 1/64 (1.56%) | 143.7 | 0/64 |
| R1 GRPO step 2000 | 24/64 (37.50%) | 26/64 (40.62%) | 28/64 (43.75%) | 174.5 | 0/64 |
| R1 GRPO step 3364 | 12/64 (18.75%) | 12/64 (18.75%) | 23/64 (35.94%) | 176.9 | 0/64 |
| R3 RLOO step 2500 | 6/64 (9.38%) | 7/64 (10.94%) | 16/64 (25.00%) | 77.0 | 45/64 |
| R3 RLOO step 3364 | 1/64 (1.56%) | 1/64 (1.56%) | 5/64 (7.81%) | 24.5 | 58/64 |

Evidence files:
- Markdown summary: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/report_diagnostics/r1_r3_report_diagnostic_summary.md`
- Metrics CSV: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/report_diagnostics/r1_r3_eval_metrics.csv`
- TensorBoard scalar samples CSV: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/report_diagnostics/r1_r3_tensorboard_scalar_samples.csv`
- R1 scalar summary: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231/eval/r1_tensorboard_scalar_summary.txt`
- R3 scalar summary: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R3-rloo-full-s0-20260608_230810/eval/r3_tensorboard_scalar_summary.txt`
- Qualitative examples: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/report_diagnostics/base_correct_r1_r3_wrong_examples.csv`

TensorBoard diagnosis:
- R1 eval KL rises from about `0.324` near step 1984 to `0.859` near step 3008, then `0.488` near 3328; train KL reaches about `0.977` near step 3008.
- R3 eval KL is lower/moderate in the sampled scalar table (`0.246` near 1984, `0.440` near 3008, `0.374` near 3328), but external greedy eval collapses, so KL alone is not sufficient.
- `actor/train/pg_clipfrac` and `actor/eval/pg_clipfrac` remain `0.0` for both runs, so clipping does not flag the instability.
- Reward/check_answer/check_numbers scalar signals do not align with external greedy accuracy, especially for R3, pointing to reward/eval mismatch.

Qualitative diagnosis:
- Base correct while both R1 final and R3 final are wrong on 23 prompts.
- R1 mostly keeps non-empty reasoning and often follows the requested tags, but makes arithmetic or problem-interpretation mistakes.
- R3 shows generation collapse: 45/64 empty responses at the best retained checkpoint and 58/64 empty at final.

Dominant failure classification:
- R1: reward-format over-optimisation + arithmetic degradation + checkpoint overtraining, with KL drift as a warning sign.
- R3: generation collapse / unstable decoding after RLOO training, with reward/eval mismatch; not primarily a TFDS cache artefact.

Recommendation:
- Do not start another full run yet.
- A cheap D3 K=8 debug is justified next, but only after exposing/configuring `NUM_GENERATIONS=8`; run it as a short debug with fresh eval caches and explicit empty-output diagnostics.
- Do not run K=8 full unattended until D3 debug results are reviewed.

## 2026-06-09: D3 GRPO K=8 debug

Run root:
- `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D3-grpo-k8-debug-20260609_094815`

Patch:
- `NUM_GENERATIONS` is now env-overridable in `scripts/config.py`, defaulting to `2`.
- D3 launched with `NUM_GENERATIONS=8` and `MAX_STEPS_OVERRIDE=50`.
- Uncommitted diff/status snapshot was saved under the run root `patches/` directory.

Training status:
- Completed without OOM or fatal error.
- Checkpoints include `ckpts/actor/1` and `ckpts/actor/50`.
- Metadata records commit `99cb7f88b0e49a7d65bb0c5274734a190833aa3f`, estimator `grpo`, seed `0`, max steps `50`, and run-local data/output dirs.
- W&B: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/D3-grpo-k8-debug-20260609_094815`
- Nonfatal warnings: repeated W&B step-order warnings near steps 49/50; expected fresh-cache TFDS variant warnings during eval.

Eval status:
- Restore succeeded from `ckpts/actor` at step `50`.
- CSV: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D3-grpo-k8-debug-20260609_094815/eval/d3_grpo_k8_step50_greedy.csv`
- Eval log: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D3-grpo-k8-debug-20260609_094815/logs/eval_d3_grpo_k8_step50_greedy.log`
- Summary file: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D3-grpo-k8-debug-20260609_094815/eval/d3_eval_summary.txt`

Metrics:
| Eval | Exact | Partial | Format | Empty responses |
| --- | ---: | ---: | ---: | ---: |
| D3 GRPO K=8 step 50 | 31/64 (48.44%) | 32/64 (50.00%) | 8/64 (12.50%) | 0/64 |

Interpretation:
- K=8 did not OOM and did not show the R3 empty-response collapse in this 50-step debug.
- The step-50 score matches the base greedy exact rate, which is encouraging for a cheap diagnostic but not enough evidence that a full K=8 run will avoid the R1/R3 full-horizon degradation.
- Recommendation: full K=8 is worth considering only after review; a slightly longer K=8 debug or added scalar/CSV diagnostics may be prudent before spending a full overnight run.

## 2026-06-09: Audit, reward sanity, and D4 K=8 medium debug

Code audit:
- No blocking implementation bug found.
- `scripts/config.py`: estimator, K, beta/epsilon, generation and sampling config are centrally wired; K is now env-overridable.
- `scripts/train.py`: `GRPOConfig` receives K, estimator, beta, epsilon; metadata now includes K and sampling/loss controls.
- `scripts/data.py`: train/val/test split and shuffle seed are deterministic under `RUN_SEED`; train/test TFDS dirs are env-overridable.
- `scripts/rewards.py`: parser behavior matches implementation; reward scale is the main concern.
- `scripts/evaluate.py`: actor checkpoint restore and greedy CSV export work; empty responses are countable from `model_response` but not explicitly flagged as a CSV column.

Reward sanity:
- File: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/report_diagnostics/reward_sanity_check.txt`
- Correct numeric bad plain format: `-2.5`.
- Correct numeric answer-only tags: `1.0`.
- Correct numeric good format: `10.0`.
- Wrong numeric good format: `6.0`.
- Empty response: `-2.5`.
- Malformed answer closing tag with correct number: `3.0`.
- Interpretation: no parser bug, but format rewards can overpower correctness and should be discussed in the report/control plan.

Reproducibility hygiene:
- `scripts/config.py` supports `NUM_GENERATIONS` env override.
- `scripts/train.py` metadata/logging now records `num_generations`, `beta`, `epsilon`, `temperature`, `top_k`, `top_p`, and `total_generation_steps`.
- D4 run root saved uncommitted diff/status snapshots in `patches/`.

D4 run:
- Run root: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D4-grpo-k8-medium-debug-20260609_100945`
- W&B: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/D4-grpo-k8-medium-debug-20260609_100945`
- Summary: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D4-grpo-k8-medium-debug-20260609_100945/eval/d4_eval_summary.txt`
- CSVs: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D4-grpo-k8-medium-debug-20260609_100945/eval/d4_grpo_k8_step{200,300,400,500}_greedy.csv`
- Logs: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D4-grpo-k8-medium-debug-20260609_100945/logs/`

D4 metrics:
| Step | Exact | Partial | Format | Empty |
| ---: | ---: | ---: | ---: | ---: |
| 100 | missing/pruned | missing/pruned | missing/pruned | missing/pruned |
| 200 | 26/64 (40.62%) | 28/64 (43.75%) | 55/64 (85.94%) | 0/64 |
| 300 | 32/64 (50.00%) | 34/64 (53.12%) | 51/64 (79.69%) | 0/64 |
| 400 | 29/64 (45.31%) | 29/64 (45.31%) | 55/64 (85.94%) | 0/64 |
| 500 | 34/64 (53.12%) | 35/64 (54.69%) | 54/64 (84.38%) | 0/64 |

Interpretation and recommendation:
- K=8 medium debug is the best evidence so far: no empty-response collapse, high format compliance, and step 500 reaches `34/64`, above base greedy `31/64` on the same 64-example setup.
- This supports considering a full K=8 GRPO run after review, with early checkpoint selection and checkpoint-retention hygiene.
- Reward/control patching is still relevant because reward sanity shows format-heavy reward can still pay wrong answers; the full K=8 run should not be treated as guaranteed stable.

## 2026-06-09: Checkpoint retention hygiene before R5

Patch intent:
- `MAX_TO_KEEP` is env-overridable so medium/full diagnostics can retain all planned checkpoints.
- Training metadata/logging records the retention setting alongside generation/sampling controls.

R5 approval and cautions:
- D4 supports launching full GRPO K=8 seed 0 after this hygiene patch.
- R5 should retain enough checkpoints for early-selection review with `SAVE_INTERVAL_STEPS=250` and `MAX_TO_KEEP=20`.
- Do not start R4/R2 and do not change reward weights yet.
- Reward sanity still suggests format-heavy reward can pay wrong answers, so final checkpoint quality should not be assumed; compare retained checkpoints against base/R1/R3/D4.

## 2026-06-09: R5 GRPO K=8 full seed 0 result

Run root:
- `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R5-grpo-k8-full-s0-20260609_114832`

Training:
- Commit: `820fad61060a4260184e71066568af7b28d3109e`
- Config: `ADV_ESTIMATOR=grpo`, `NUM_GENERATIONS=8`, `RUN_SEED=0`, `MAX_STEPS=3364`, `SAVE_INTERVAL_STEPS=250`, `MAX_TO_KEEP=20`, no `MAX_STEPS_OVERRIDE`.
- Training completed successfully with retained checkpoints at `250`, `500`, `750`, `1000`, `1250`, `1500`, `1750`, `2000`, `2250`, `2500`, `2750`, `3000`, `3250`, and `3364`.
- Metadata records the launch commit, run-local data dirs, `num_generations=8`, and `max_to_keep=20`.
- W&B: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R5-grpo-k8-full-s0-20260609_114832`
- TensorBoard: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R5-grpo-k8-full-s0-20260609_114832/tensorboard/events.out.tfevents.1781005731.t1v-n-0339f27d-w-0`
- Warnings: repeated W&B step-order warnings; no fatal/OOM/traceback markers found.

Eval summary:
| Step | Exact | Partial | Format | Empty |
| ---: | ---: | ---: | ---: | ---: |
| 250 | 29/64 (45.31%) | 30/64 (46.88%) | 58/64 (90.62%) | 0/64 |
| 500 | 34/64 (53.12%) | 35/64 (54.69%) | 54/64 (84.38%) | 0/64 |
| 750 | 31/64 (48.44%) | 34/64 (53.12%) | 60/64 (93.75%) | 0/64 |
| 1000 | 32/64 (50.00%) | 34/64 (53.12%) | 60/64 (93.75%) | 0/64 |
| 1250 | 30/64 (46.88%) | 35/64 (54.69%) | 59/64 (92.19%) | 0/64 |
| 1500 | 29/64 (45.31%) | 31/64 (48.44%) | 57/64 (89.06%) | 0/64 |
| 1750 | 32/64 (50.00%) | 34/64 (53.12%) | 56/64 (87.50%) | 0/64 |
| 2000 | 31/64 (48.44%) | 33/64 (51.56%) | 57/64 (89.06%) | 0/64 |
| 2250 | 30/64 (46.88%) | 31/64 (48.44%) | 58/64 (90.62%) | 0/64 |
| 2500 | 33/64 (51.56%) | 34/64 (53.12%) | 58/64 (90.62%) | 0/64 |
| 2750 | 31/64 (48.44%) | 32/64 (50.00%) | 63/64 (98.44%) | 0/64 |
| 3000 | 31/64 (48.44%) | 32/64 (50.00%) | 56/64 (87.50%) | 0/64 |
| 3250 | 35/64 (54.69%) | 36/64 (56.25%) | 55/64 (85.94%) | 0/64 |
| 3364 | 32/64 (50.00%) | 33/64 (51.56%) | 56/64 (87.50%) | 0/64 |

Comparison for report:
- Base greedy: `31/64` exact.
- R1 best retained checkpoint: step 2000, `24/64`; R1 final: `12/64`.
- R3 best retained checkpoint: step 2500, `6/64`.
- D4 K=8 medium step 500: `34/64`.
- R5 best retained checkpoint: step 3250, `35/64`; R5 final: `32/64`.

Interpretation:
- K=8 full GRPO is stable operationally: no empty-output collapse and no OOM.
- R5 improves over base by `4/64` at the best retained checkpoint and over D4 by `1/64`, but the sample is small and final checkpoint is weaker than the best checkpoint.
- For the report, use R5 as the strongest trained K=8 result only with caveats: retained checkpoint selection matters, reward-format pressure remains high, and paired/bootstrap uncertainty should be computed before making strong claims.

Evidence paths:
- All-checkpoint summary: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R5-grpo-k8-full-s0-20260609_114832/eval/r5_all_retained_eval_summary.txt`
- CSVs: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R5-grpo-k8-full-s0-20260609_114832/eval/r5_grpo_k8_step*_greedy.csv`
- Logs: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R5-grpo-k8-full-s0-20260609_114832/logs/`

Recommendation:
- Do not start R4/R2 or another training run yet.
- Next local work: pull this notes commit if pushed, review R5 W&B/TensorBoard curves, compute confidence intervals/paired bootstrap against base and D4/R1/R3, and decide whether the final report should present best-checkpoint R5 or a fixed-step comparison.

## 2026-06-09: Hard-example curriculum idea

Idea:
- Mine difficult GSM8K training examples by probing which train questions the model fails consistently, then train/evaluate a hard-subset curriculum inspired by the deep-research notes.
- Optionally add a small GSM-Hard component later, but treat that as a separate ablation because it changes both difficulty and data distribution.

Recommended selector:
- Use an inference-only probe over GSM8K train with the frozen base model for the clean "hard for the starting policy" experiment.
- Do not use held-out eval/test examples for mining.
- A finetuned checkpoint such as R5 can be used for diagnostic comparison, but using R5 failures to choose training data answers a different question: "hard for the already-adapted policy".

Manifest requirements:
- Save source split, original index or question hash, expected answer, model/revision/checkpoint, prompt template, decoding preset, sample seeds, pass/fail counts, hardness label, and selection rule.

## 2026-06-09: Reproducibility contract patch

Purpose:
- Fix the eval comparability issue where `RUN_SEED` also changed held-out test ordering.
- Prevent moving GitHub/Hugging Face refs from changing the software/model environment between runs.
- Make the launch script and docs encode the experiment contract rather than relying on hand memory.

Changes:
- `scripts/config.py`: adds `MODEL_REVISION`, dependency ref env vars, `EVAL_SEED`, and `EVAL_MANIFEST`.
- `scripts/data.py`: adds manifest-backed eval loading and separate train/test shuffle seeds.
- `scripts/evaluate.py`: creates a missing eval manifest on first use, reloads it on later use, and records eval seed/manifest in per-prompt CSVs.
- `scripts/train.py`: records model revision, dependency refs, eval seed, and manifest path in `run_metadata.json`.
- `scripts/model.py`: refuses to download `google/gemma-3-1b-it` unless `MODEL_REVISION` is an exact revision.
- `bootstrap.sh`: refuses GitHub HEAD installs by requiring `JAX_REF`, `QWIX_REF`, and `FLAX_REF`; `TUNIX_REF` defaults to the known citation commit.
- `scripts/run_tmux.sh`: derives the repo path, requires `RUN_ROOT` and pin variables, sets persistent output/data dirs, and exports `EVAL_MANIFEST`.
- Added `experiments/manifests/experiment_contract.md` and updated launch docs.

Validation:
- `git diff --check` passed.
- `py_compile` passed for `scripts/config.py`, `scripts/data.py`, `scripts/train.py`, `scripts/evaluate.py`, and `scripts/model.py`.
- `bash -n` passed for `bootstrap.sh` and `scripts/run_tmux.sh`.

## 2026-06-10: R6 RLOO K-sweep training completed

Purpose:
- Run a full RLOO K-sweep after the GRPO K=8 results: first RLOO with `NUM_GENERATIONS=8`, then chained RLOO with `NUM_GENERATIONS=2`.

Source-state caveat:
- These R6 runs were not launched from deterministic-platform. W&B metadata for both runs records git commit `8a0f7f266552cb2666710ac589cb6bda5cd40121` on branch `harvey`; git reflog shows checkout to `harvey-grpo-k8-rerun` only at 2026-06-10 10:19 UTC, after both runs had finished.
- The `harvey` launch commit lacks deterministic-platform controls: required `MODEL_REVISION`, `EVAL_SEED`/`EVAL_MANIFEST`, pinned `JAX_REF`/`QWIX_REF`/`FLAX_REF`, and the deterministic branch's `run_metadata.json` writer.

Run table:
| Run | Estimator | K | Start UTC | Finish UTC | Status |
| --- | --- | ---: | --- | --- | --- |
| `R6-rloo-k8-full-s0-20260609_212314` | `rloo` | 8 | 2026-06-09 ~21:57 | 2026-06-10 ~05:37 | training complete |
| `R6-rloo-k2-full-s0-20260609_220042` | `rloo` | 2 | 2026-06-10 05:37:42 | 2026-06-10 ~07:41 | training complete |

Common config:
- `RUN_SEED=0`
- `MAX_STEPS=3364`
- `SAVE_INTERVAL_STEPS=250`
- `MAX_TO_KEEP=20`
- `MAX_STEPS_OVERRIDE` unset
- Runtime paths were persistent under `/home/harvey/tpu-runs/part-i`, not `/tmp`.

Evidence:
- K=8 root: `/home/harvey/tpu-runs/part-i/R6-rloo-k8-full-s0-20260609_212314`
- K=2 root: `/home/harvey/tpu-runs/part-i/R6-rloo-k2-full-s0-20260609_220042`
- K=8 W&B: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R6-rloo-k8-full-s0-20260609_212314`
- K=2 W&B: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R6-rloo-k2-full-s0-20260609_220042`
- Detailed note: `experiments/variants/r6_rloo_k_sweep_full_s0_20260609.md`

Artifacts:
- Both runs logged `Training finished.`.
- Both have final actor checkpoint `ckpts/actor/3364`.
- Both have retained actor checkpoints at step `1` and every `250` steps through `3250`.
- TensorBoard event files and local W&B artifacts are stored under each run root.
- `/tmp` was small when checked (`~61M`); run directories are about `3.5G` each under `/home/harvey/tpu-runs/part-i`.

Caveats:
- No post-training greedy eval CSVs or eval summaries were found for either R6 run at note time.
- No `run_metadata.json` was found in either R6 checkpoint root at note time.
- W&B step-order warnings persisted near final step `3364`.

Next action:
- Run retained-checkpoint evaluation for both R6 runs with a fixed eval manifest/seed before comparing RLOO K=8 vs RLOO K=2 or against R5 GRPO K=8.

## 2026-06-11: R7 deterministic RLOO K=2 launched; K=8 delegated to second TPU

Reason:
- R6 was found to be non-deterministic because both R6 runs were launched from branch `harvey` at commit `8a0f7f266552cb2666710ac589cb6bda5cd40121`.
- R7 repeats the RLOO K comparison on deterministic branch `harvey-grpo-k8-rerun`.

This VM:
- Branch: `harvey-grpo-k8-rerun`.
- Commit: `71aab87dee2d2c78256384d084d063d8b40c9e0c`.
- Verified deterministic-platform ancestry and verified `harvey` is not an ancestor.
- Active run id: `R7-rloo-k2-det-harvey-full-s0-20260611_102009`.
- Run root: `/home/harvey/tpu-runs/part-i/R7-rloo-k2-det-harvey-full-s0-20260611_102009`.
- W&B: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R7-rloo-k2-det-harvey-full-s0-20260611_102009`.
- Tmux attach: `tmux attach -t r7-rloo-k2-det-harvey-full-s0`.
- Log tail: `tail -f /home/harvey/tpu-runs/part-i/R7-rloo-k2-det-harvey-full-s0-20260611_102009/logs/train.log`.

Confirmed config:
- `ADV_ESTIMATOR=rloo`.
- `NUM_GENERATIONS=2`.
- `RUN_SEED=0`.
- `EVAL_SEED=0`.
- `MAX_STEPS=3364`.
- `SAVE_INTERVAL_STEPS=250`.
- `MAX_TO_KEEP=20`.
- `MODEL_REVISION=dcc83ea841ab6100d6b47a070329e1ba4cf78752`.
- `MAX_STEPS_OVERRIDE` unset.

Metadata/evidence:
- `ckpts/run_metadata.json` exists and records commit, model revision, dependency refs, run/eval seeds, eval manifest, run-local data dirs, checkpoint dir, TensorBoard dir, W&B entity/project, estimator, and `num_generations=2`.
- Run-local launcher used: `/home/harvey/tpu-runs/part-i/R7-rloo-k2-det-harvey-full-s0-20260611_102009/launch_train.sh`.
- The launch uses run-local `tmp/` and `wandb/` dirs under the run root, not `/tmp` as the only artifact location.

Failed setup attempts:
- `R7-rloo-k2-det-harvey-full-s0-20260611_101430` failed because `/tmp/libtpu_lockfile` was held by a hung diagnostic JAX probe.
- `R7-rloo-k2-det-harvey-full-s0-20260611_101739` failed at W&B init because the existing tmux server did not inherit the intended run environment.
- Both failed roots were left in place intentionally.

Other TPU:
- Baron has sent the matching K=8 command/prompt to the other TPU VM.
- Expected K=8 settings: same branch/commit lineage, same model revision and dependency refs, `ADV_ESTIMATOR=rloo`, `NUM_GENERATIONS=8`, `RUN_SEED=0`, `EVAL_SEED=0`, `SAVE_INTERVAL_STEPS=250`, `MAX_TO_KEEP=20`, and `MAX_STEPS_OVERRIDE` unset.
- Do not mark K=8 as launched or complete until the other TPU reports its exact run root, W&B URL, and log-confirmed config.
