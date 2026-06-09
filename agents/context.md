# TPU-2026 Coursework Branch Context

This fork is the Part I practical training/evaluation codebase for the Multi-Agent Systems and Agentic AI coursework. It starts from upstream `borisbolliet/tpu-2026` commit `324abbe4b4e229ea812223856393547db4fbb53e` and the local `coursework` branch currently contains the P0-P6 preparation patches needed for reproducible GRPO/RLOO TPU runs. The main coursework/report repository is the sibling directory `../agentic-ai-coursework`.

## Current Status

- Branch: `coursework`.
- Upstream baseline: `324abbe4b4e229ea812223856393547db4fbb53e`.
- Current pulled head: `820fad6` on `coursework` / `origin/coursework`.
- Branch is aligned with `origin/coursework`.
- Local run notes, diagnostics, runbooks, and manifests are now recorded under `experiments/`; read `experiments/README.md` for the map before searching TPU-side logs.
- Only baseline-owned files were touched in the 8 commits: `bootstrap.sh`, `scripts/config.py`, `scripts/data.py`, `scripts/train.py`, and `scripts/evaluate.py`.
- No Tunix source files were edited.
- D1 GRPO 50-step debug completed successfully: step 50 reached, actor checkpoint restored, TensorBoard/W&B emitted evidence, and greedy eval CSV was written.
- D2 RLOO 50-step debug completed successfully: step 50 reached, actor checkpoint restored, TensorBoard/W&B emitted evidence, and greedy eval CSV was written.
- R5 GRPO K=8 seed 0 full training and retained-checkpoint eval completed; do not start R4/R2 or any new run until Baron/local Codex reviews R5.

## What Was Implemented

- P0: `CKPT_DIR`, `INTERMEDIATE_CKPT_DIR`, and `TENSORBOARD_DIR` are environment-overridable.
- P1: `ADV_ESTIMATOR` is passed to `GRPOConfig(advantage_estimator=...)` for `grpo`, `rloo`, or `drgrpo`.
- P2: `RUN_SEED` is threaded into Grain shuffle, rollout config, and `GRPOLearner.data_shuffle_seed`.
- Reproducibility contract follow-up: held-out eval now uses separate `EVAL_SEED` or an explicit `EVAL_MANIFEST` JSONL, so seed-repeat training runs do not silently change the eval prompt set.
- P3: `MAX_STEPS_OVERRIDE` caps debug runs without changing the intended full-run LR schedule. Use `SAVE_INTERVAL_STEPS=50` with 50-step debug runs so the restore check has a checkpoint.
- P4: `evaluate.py` can restore trained LoRA checkpoints through `--ckpt-dir`, optional `--step`, and explicit `--no-restore` for base-model sanity checks.
- P5: `evaluate.py --output-csv` writes per-prompt rows for bootstrap confidence intervals and auditability.
- P6: `train.py` writes `run_metadata.json` into `CKPT_DIR` at run start. Hygiene follow-up now records repo root, launch cwd, and train/test data dirs; git commit resolution is repo-root-aware for run-local launches.
- `bootstrap.sh` pins Tunix to `683256db1a0919b5cfd46cee52cebc96331494fb` and now requires explicit `JAX_REF`, `QWIX_REF`, and `FLAX_REF` commit SHAs to avoid HEAD drift.
- `model.py` now requires exact `MODEL_REVISION` for Hugging Face downloads and refuses moving model HEAD.

## Current Experiment Intent

The planned I.3 controlled comparison is standard GRPO vs RLOO at fixed compute. This is motivated by the coursework theory around group-mean advantage normalisation at small group size `K=2`: RLOO keeps reward-difference magnitude where standard GRPO collapses to mostly sign information.

Target matrix:

| Run | Estimator | Seed | Purpose |
| --- | --- | ---: | --- |
| D1 | grpo | 0 | complete: 50-step debug baseline passed. |
| D2 | rloo | 0 | complete: 50-step debug variant passed. |
| R1 | grpo | 0 | complete: full baseline training and greedy eval finished. |
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


R1 evaluation results:
- Trained checkpoint eval restored `ckpts/actor/3364` with `restored_step=3364`.
- Trained R1 greedy: `correct=12/64`, `acc=18.75%`, `partial=18.75%`, `format=35.94%`.
- Base greedy same setup: `correct=31/64`, `acc=48.44%`, `partial=48.44%`, `format=1.56%`.
- Trained CSV: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231/eval/r1_grpo_step3364_greedy.csv`
- Trained log: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231/logs/eval_r1_grpo_step3364_greedy.log`
- Base CSV: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231/eval/base_greedy_seed0.csv`
- Base log: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231/logs/eval_base_greedy_seed0.log`
- TFDS/protobuf cache issue recurred when reusing existing data caches; successful evals used fresh eval-local TFDS caches under `$RUN_ROOT/eval`.
- Decision: R1 is a valid completed I.1 GRPO baseline control operationally, but its greedy eval underperforms the base model on this 64-prompt sample; review before launching R3.


R1 diagnostic pass:
| Eval | Exact | Partial | Format |
| --- | ---: | ---: | ---: |
| Base greedy | 31/64 (48.44%) | 31/64 (48.44%) | 1/64 (1.56%) |
| R1 step 2000 | 24/64 (37.50%) | 26/64 (40.62%) | 28/64 (43.75%) |
| R1 step 2500 | 11/64 (17.19%) | 11/64 (17.19%) | 23/64 (35.94%) |
| R1 step 3000 | 8/64 (12.50%) | 8/64 (12.50%) | 16/64 (25.00%) |
| R1 step 3364 | 12/64 (18.75%) | 12/64 (18.75%) | 23/64 (35.94%) |

R1 diagnostic interpretation:
- Accuracy degradation is gradual after the best retained checkpoint at step 2000, not only a final-checkpoint artefact.
- TensorBoard scalar summary: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231/eval/r1_tensorboard_scalar_summary.txt`
- Relevant scalar tags found: `actor/train/kl`, `actor/eval/kl`, `actor/*/pg_clipfrac`, `rewards/eval/mean`, `rewards/eval/check_answer`, `rewards/eval/check_numbers`, `rewards/eval/match_format_exactly`, `rewards/eval/match_format_approximately`, and completion length tags.
- KL rises strongly through step 3000: eval KL near 1984 `0.324`, near 2496 `0.545`, near 3008 `0.859`, then near 3328 `0.488`; train KL near 3008 reaches `0.977`.
- `pg_clipfrac` remains `0.0`, so clipping does not flag the issue.
- Eval reward/format metrics improve while greedy exact accuracy worsens versus base, suggesting reward-format optimisation and numeric reasoning degradation rather than an eval-only cache artefact.
- Qualitative CSV comparison: base is correct and trained wrong on 23 prompts; trained is correct and base wrong on 4; both wrong on 29. Trained outputs more often use the requested tags but frequently make arithmetic/interpretation mistakes.
- Recommendation: do not approve R3 immediately; first review R1 curves/CSV and consider changing controls such as early checkpoint selection, shorter training, reward balance, or eval protocol.


R3 RLOO full run:
- Run root: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R3-rloo-full-s0-20260608_230810`
- W&B: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R3-rloo-full-s0-20260608_230810`
- Training completed at `max_steps=3364` with `ADV_ESTIMATOR=rloo`, `RUN_SEED=0`, commit `99cb7f8`.
- Final retained checkpoints exist at `ckpts/actor/2000`, `2500`, `3000`, and `3364`.
- Auto-eval restored all retained checkpoints from `ckpts/actor` with fresh eval-local TFDS caches.
| Eval | Exact | Partial | Format |
| --- | ---: | ---: | ---: |
| R3 step 2000 | 1/64 (1.56%) | 1/64 (1.56%) | 10/64 (15.62%) |
| R3 step 2500 | 6/64 (9.38%) | 7/64 (10.94%) | 16/64 (25.00%) |
| R3 step 3000 | 1/64 (1.56%) | 2/64 (3.12%) | 4/64 (6.25%) |
| R3 step 3364 | 1/64 (1.56%) | 1/64 (1.56%) | 5/64 (7.81%) |
- Best retained R3 checkpoint by exact accuracy is step 2500 at `6/64` (`9.38%`), far below base and R1 step 2000.
- Training and eval logs contain recurring W&B step-order warnings and fresh-cache TFDS variant warnings; no fatal training failure was found.
- Decision: R3 completed the planned RLOO comparison operationally, but results are poor; do not start R4/R2 or K=8 until reviewing reward/training controls.


Report-ready R1/R3 diagnosis:
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
- R1 TensorBoard scalar summary: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231/eval/r1_tensorboard_scalar_summary.txt`
- R3 TensorBoard scalar summary: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R3-rloo-full-s0-20260608_230810/eval/r3_tensorboard_scalar_summary.txt`
- Representative base-correct/trained-wrong examples: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/report_diagnostics/base_correct_r1_r3_wrong_examples.csv`

Diagnosis:
- Evaluation/cache artefact is unlikely as the dominant failure: restores succeeded, evals completed using fresh eval-local TFDS caches, and base remains strong on the same 64-prompt setup.
- R1 failure mode: reward-format over-optimisation plus arithmetic degradation and checkpoint overtraining. Step 2000 is better than final, but still below base; format compliance rises while exact accuracy falls. R1 eval KL rises through step 3008 (`0.859`) and train KL reaches about `0.977`, while `pg_clipfrac` remains `0.0`.
- R3 failure mode: stronger generation collapse/empty-output pathology. Best retained R3 checkpoint is step 2500 at `6/64`; final has `58/64` empty responses. TensorBoard reward/check_answer/check_numbers can look nonzero despite poor external greedy eval, indicating reward/eval mismatch and unstable generation behaviour.
- Qualitative rows: base is correct while both final trained models are wrong on 23 prompts; R1 often uses tags but makes arithmetic/interpretation errors, while R3 frequently emits empty responses.
- D3 K=8 debug recommendation: justified as a cheap diagnostic only, not a full run. It should first expose/configure `NUM_GENERATIONS=8`, run a short debug with fresh eval caches and empty-response checks, and be reviewed before any K=8 full run.

## What To Read First

1. `agents/context.md` in this repository.
2. `agents/plan.md` in this repository.
3. `agents/report_notes.md` in this repository.
4. `experiments/README.md`, then the relevant `experiments/baseline/`, `experiments/variants/`, `experiments/diagnostics/`, `experiments/manifests/`, or `experiments/runbooks/` file.
5. Main coursework repo: `../agentic-ai-coursework/agents/context.md`.
6. Main coursework repo: `../agentic-ai-coursework/experiments/runbooks/tpu_day1_runbook.md`.
7. Patched code: `scripts/config.py`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/data.py`, `bootstrap.sh`.

## Safe Useful Commands

```bash
git status --short --branch
git log --oneline --decorate --max-count=12
git diff --check
env PYTHONPYCACHEPREFIX=/tmp/tpu2026-pycache python3 -m py_compile scripts/config.py scripts/data.py scripts/train.py scripts/evaluate.py
```

On a TPU VM after setup, use the main coursework runbook before any full run.

## Things Not To Do

- Do not push from an automated session unless Baron explicitly asks for a notes/code handoff commit.
- Do not edit Tunix unless a TPU/debug failure proves the pinned Tunix API itself is wrong.
- Do not start R4/R2 or any new full/debug run without explicit approval.
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

## Hard-Example Mining Note

- A hard-example curriculum does not require another training run to identify examples; it requires an inference/probing pass over GSM8K train with a fixed model, prompt template, decoding protocol, and seeds.
- For a clean "hard for the starting policy" experiment, mine failures with the frozen base model (`evaluate.py --no-restore` style), not R5/R1/R3. A finetuned checkpoint can be used for analysis, but selecting data from its failures answers a different question: "hard for the already-adapted policy".
- Do not mine from the held-out eval/test set. Save a manifest containing source split, original index or question hash, expected answer, model/revision/checkpoint, decoding preset, sample seeds, pass/fail counts, and selection rule.
- If adding GSM-Hard, treat it as a separate ablation after GSM8K-train hard mining, because it changes both difficulty and data distribution.

## 2026-06-09: Reproducibility contract patch

- `scripts/config.py` adds `MODEL_REVISION`, `JAX_REF`, `TUNIX_REF`, `QWIX_REF`, `FLAX_REF`, `EVAL_SEED`, and `EVAL_MANIFEST`.
- `scripts/data.py` can load an eval dataset from a JSONL manifest and can use a separate test shuffle seed.
- `scripts/evaluate.py` creates `EVAL_MANIFEST` on first use if missing, reuses it if present, and writes `eval_seed`/`eval_manifest` into CSV metadata.
- `scripts/train.py` records model revision, dependency refs, eval seed, and eval manifest in `run_metadata.json`.
- `bootstrap.sh`, `scripts/run_tmux.sh`, `scripts/README.md`, and `experiments/runbooks/tpu_day1_runbook.md` now encode the launch contract.
- Contract document: `experiments/manifests/experiment_contract.md`.

## 2026-06-09: D3 GRPO K=8 debug completed

Run root:
- `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D3-grpo-k8-debug-20260609_094815`

Code/config:
- `scripts/config.py` now makes `NUM_GENERATIONS` env-overridable: `NUM_GENERATIONS = int(os.environ.get("NUM_GENERATIONS", "2"))`.
- The uncommitted code/doc diff was captured for reproducibility in `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D3-grpo-k8-debug-20260609_094815/patches/`.

Training:
- `ADV_ESTIMATOR=grpo`, `NUM_GENERATIONS=8`, `RUN_SEED=0`, `MAX_STEPS_OVERRIDE=50`, `SAVE_INTERVAL_STEPS=50`.
- Training completed without fatal error or OOM and wrote `ckpts/actor/50`.
- Metadata records commit `99cb7f88b0e49a7d65bb0c5274734a190833aa3f`, estimator `grpo`, seed `0`, `max_steps=50`, and run-local train/test/checkpoint/TensorBoard dirs.
- W&B: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/D3-grpo-k8-debug-20260609_094815`
- Warnings: recurring W&B step-order warnings at steps 49/50; no fatal training warning found.

Eval:
- Restored step `50` from `ckpts/actor`.
- CSV: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D3-grpo-k8-debug-20260609_094815/eval/d3_grpo_k8_step50_greedy.csv`
- Log: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D3-grpo-k8-debug-20260609_094815/logs/eval_d3_grpo_k8_step50_greedy.log`
- Summary: `31/64` exact (`48.44%`), `32/64` partial (`50.00%`), `8/64` format (`12.50%`), `0/64` empty responses.

Decision:
- D3 shows K=8 can train for 50 steps without OOM and avoids the R3 empty-response collapse at this short horizon.
- A full K=8 run is worth considering only after review, because this is a short debug and the result matches base exact accuracy rather than proving full-run stability.

## 2026-06-09: Code audit and D4 GRPO K=8 medium debug

Audit:
- No blocking implementation bug was found in `scripts/config.py`, `scripts/train.py`, `scripts/data.py`, `scripts/rewards.py`, or `scripts/evaluate.py`.
- Confirmed `GRPOConfig` receives `NUM_GENERATIONS`, `ADV_ESTIMATOR`, `BETA`, and `EPSILON`; data uses env-overridden TFDS dirs and `RUN_SEED`; eval resolves `ckpts/actor` and writes per-prompt CSVs suitable for empty-response counting.
- Non-blocking issues: reward scale gives high positive reward to wrong but well-formatted answers; `check_numbers` prints full sampled responses; eval does not store an explicit `empty_response` column; `MAX_TO_KEEP=4` pruned D4 step 100 before eval.
- Audit note: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/report_diagnostics/code_audit_d4_hygiene_note.md`
- Reward sanity output: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/report_diagnostics/reward_sanity_check.txt`

Reproducibility patch:
- `scripts/config.py` has `NUM_GENERATIONS = int(os.environ.get("NUM_GENERATIONS", "2"))`.
- `scripts/train.py` metadata now records `num_generations`, `beta`, `epsilon`, `temperature`, `top_k`, `top_p`, and `total_generation_steps`; startup logging prints these values.
- Validation passed with `git diff --check` and py_compile for `scripts/config.py`, `scripts/train.py`, `scripts/evaluate.py`.

D4 run:
- Run root: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D4-grpo-k8-medium-debug-20260609_100945`
- W&B: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/D4-grpo-k8-medium-debug-20260609_100945`
- Config: `ADV_ESTIMATOR=grpo`, `NUM_GENERATIONS=8`, `RUN_SEED=0`, `MAX_STEPS_OVERRIDE=500`, `SAVE_INTERVAL_STEPS=100`.
- Training completed without fatal error or OOM. W&B step-order warnings persisted.

D4 eval metrics:
| Step | Exact | Partial | Format | Empty |
| ---: | ---: | ---: | ---: | ---: |
| 100 | missing/pruned | missing/pruned | missing/pruned | missing/pruned |
| 200 | 26/64 (40.62%) | 28/64 (43.75%) | 55/64 (85.94%) | 0/64 |
| 300 | 32/64 (50.00%) | 34/64 (53.12%) | 51/64 (79.69%) | 0/64 |
| 400 | 29/64 (45.31%) | 29/64 (45.31%) | 55/64 (85.94%) | 0/64 |
| 500 | 34/64 (53.12%) | 35/64 (54.69%) | 54/64 (84.38%) | 0/64 |

Recommendation:
- K=8 GRPO is worth considering for a full run after review; it is healthier than K=2 at the medium horizon and shows no empty-response collapse.
- Still use early checkpoint review/selection, and consider making `MAX_TO_KEEP` env-overridable before diagnostics that require all retained checkpoints.

## 2026-06-09: Checkpoint-retention hygiene for full K=8

Code/config:
- `MAX_TO_KEEP` is now env-overridable in `scripts/config.py`, defaulting to `4`.
- `scripts/train.py` records `max_to_keep` in `run_metadata.json` and prints `SAVE_INTERVAL_STEPS` plus `MAX_TO_KEEP` in the startup log.

Run approval:
- Full GRPO K=8 seed 0 is approved after D4, but R4/R2 remain blocked.
- Planned R5 should use `NUM_GENERATIONS=8`, `SAVE_INTERVAL_STEPS=250`, and `MAX_TO_KEEP=20` so enough checkpoints are retained for early-selection review.
- Do not change reward weights for R5; D4 reward sanity concerns remain report/control context, not a launch-blocking code bug.

## 2026-06-09: R5 GRPO K=8 full seed 0 completed

Run:
- Run root: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R5-grpo-k8-full-s0-20260609_114832`
- W&B: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R5-grpo-k8-full-s0-20260609_114832`
- Commit: `820fad61060a4260184e71066568af7b28d3109e`
- Config: `ADV_ESTIMATOR=grpo`, `NUM_GENERATIONS=8`, `RUN_SEED=0`, `MAX_STEPS=3364`, `SAVE_INTERVAL_STEPS=250`, `MAX_TO_KEEP=20`, `MAX_STEPS_OVERRIDE` unset.
- Metadata confirms run-local train/test dirs, checkpoint dir, TensorBoard dir, `num_generations=8`, and `max_to_keep=20`.
- Training completed with `Training finished.` in `logs/train.log`; no fatal/OOM/traceback markers were found. W&B step-order warnings persisted.
- TensorBoard event: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R5-grpo-k8-full-s0-20260609_114832/tensorboard/events.out.tfevents.1781005731.t1v-n-0339f27d-w-0`

Retained checkpoints:
- `ckpts/actor/1`, `250`, `500`, `750`, `1000`, `1250`, `1500`, `1750`, `2000`, `2250`, `2500`, `2750`, `3000`, `3250`, `3364`.

Greedy eval with fresh per-step TFDS caches:
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

Evidence:
- All-checkpoint summary: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R5-grpo-k8-full-s0-20260609_114832/eval/r5_all_retained_eval_summary.txt`
- Original automatic summary: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R5-grpo-k8-full-s0-20260609_114832/eval/r5_eval_summary.txt`
- CSVs: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R5-grpo-k8-full-s0-20260609_114832/eval/r5_grpo_k8_step{250,500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250,3364}_greedy.csv`
- Logs: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R5-grpo-k8-full-s0-20260609_114832/logs/`

Comparison:
- Base greedy: `31/64`.
- R1 best retained: step 2000, `24/64`; R1 final: `12/64`.
- R3 best retained: step 2500, `6/64`.
- D4 K=8 medium step 500: `34/64`.
- R5 best retained checkpoint is step 3250 at `35/64` exact (`54.69%`), with no empty responses. Final step 3364 is `32/64`, so early/retained checkpoint selection still matters.

Recommendation:
- R5 is the strongest completed trained result so far and is report-useful as the K=8 GRPO variant.
- Do not start R4/R2 or another full run yet. Local Codex should review R5 CSVs/W&B curves, then decide whether to report R5 best-checkpoint selection, run bootstrap CIs, or change reward controls.
