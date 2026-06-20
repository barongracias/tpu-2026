# TPU-2026 Coursework Branch Context

This fork is the Part I practical training/evaluation codebase for the Multi-Agent Systems and Agentic AI coursework. It starts from upstream `borisbolliet/tpu-2026` commit `324abbe4b4e229ea812223856393547db4fbb53e` and the local `coursework` branch currently contains the P0-P6 preparation patches needed for reproducible GRPO/RLOO TPU runs. The main coursework/report repository is the sibling directory `../agentic-ai-coursework`.

## Current Status

- Active local branch for the current R7 work: `harvey-grpo-k8-rerun`.
- Active local commit for the current K=8 launch: `e3d69a1938fe8e8a2a67a3c84a03331133ed46f1` plus local documentation edits.
- This branch is based on deterministic-platform commit `57c6409add0d81bbdb32ca7f4b3e176b4e044068`; `harvey` is not an ancestor.
- Historical coursework baseline: `324abbe4b4e229ea812223856393547db4fbb53e`; earlier R5 notes refer to `coursework` / `origin/coursework` at `820fad6`.
- Local run notes, diagnostics, runbooks, and manifests are now recorded under `experiments/`; read `experiments/README.md` for the map before searching TPU-side logs.
- Only baseline-owned files were touched in the 8 commits: `bootstrap.sh`, `scripts/config.py`, `scripts/data.py`, `scripts/train.py`, and `scripts/evaluate.py`.
- No Tunix source files were edited.
- D1 GRPO 50-step debug completed successfully: step 50 reached, actor checkpoint restored, TensorBoard/W&B emitted evidence, and greedy eval CSV was written.
- D2 RLOO 50-step debug completed successfully: step 50 reached, actor checkpoint restored, TensorBoard/W&B emitted evidence, and greedy eval CSV was written.
- R5 GRPO K=8 seed 0 full training and retained-checkpoint eval completed.
- R6 RLOO K-sweep training completed on 2026-06-10: K=8 then chained K=2, both seed 0, both full 3364 steps. Eval is pending; do not treat these as performance results until retained-checkpoint eval CSVs exist. These R6 runs were launched from non-deterministic branch `harvey` at commit `8a0f7f266552cb2666710ac589cb6bda5cd40121`, not from deterministic-platform.
- R7 deterministic rerun is now split across TPUs: the separate Harvey VM is running RLOO K=2 from deterministic branch `harvey-grpo-k8-rerun` at commit `71aab87dee2d2c78256384d084d063d8b40c9e0c`; this shared Boris VM is running the matching RLOO K=8 job from `harvey-grpo-k8-rerun` at commit `e3d69a1938fe8e8a2a67a3c84a03331133ed46f1`.

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
| R5 | grpo | 0 | complete: full K=8 training and retained-checkpoint eval finished. |
| R6 K=8 | rloo | 0 | complete: full K=8 training finished; eval pending. |
| R6 K=2 | rloo | 0 | complete: chained full K=2 training finished; eval pending. |

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

## 2026-06-10: R6 RLOO K-sweep full seed 0 training completed

Run pair:
| Run | Estimator | K | Start UTC | Finish UTC | Status |
| --- | --- | ---: | --- | --- | --- |
| `R6-rloo-k8-full-s0-20260609_212314` | `rloo` | 8 | 2026-06-09 ~21:57 | 2026-06-10 ~05:37 | training complete |
| `R6-rloo-k2-full-s0-20260609_220042` | `rloo` | 2 | 2026-06-10 05:37:42 | 2026-06-10 ~07:41 | training complete |

Common config:
- `RUN_SEED=0`, `MAX_STEPS=3364`, `SAVE_INTERVAL_STEPS=250`, `MAX_TO_KEEP=20`, `MAX_STEPS_OVERRIDE` unset.
- Both runs used persistent run-local paths under `/home/harvey/tpu-runs/part-i`, not `/tmp`.
- Launch branch was `harvey` based on the pre-launch `git branch --show-current` check. W&B metadata for both R6 runs records git commit `8a0f7f266552cb2666710ac589cb6bda5cd40121`, which is `harvey` / `origin/harvey`, not deterministic-platform. Git reflog shows the checkout to `harvey-grpo-k8-rerun` happened later, at 2026-06-10 10:19 UTC, after both runs had finished. The R6 launch commit lacks deterministic-platform controls such as required `MODEL_REVISION`, eval manifest/seed wiring, pinned dependency refs, and `run_metadata.json`; no `run_metadata.json` was found in either R6 checkpoint root.

Evidence:
- K=8 root: `/home/harvey/tpu-runs/part-i/R6-rloo-k8-full-s0-20260609_212314`
- K=2 root: `/home/harvey/tpu-runs/part-i/R6-rloo-k2-full-s0-20260609_220042`
- K=8 W&B: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R6-rloo-k8-full-s0-20260609_212314`
- K=2 W&B: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R6-rloo-k2-full-s0-20260609_220042`
- Both final checkpoints exist at `ckpts/actor/3364`; retained checkpoints also exist every 250 steps from 250 through 3250 plus step 1.
- TensorBoard event files exist in each run root's `tensorboard/`.
- Local W&B artifacts are copied/stored under each run root's `wandb/`.
- Detailed paired note: `experiments/variants/r6_rloo_k_sweep_full_s0_20260609.md`.

Caveat and next step:
- These are training-complete runs, not evaluated performance results. No post-training greedy eval CSVs or eval summaries were found for either run at note time.
- Next step is retained-checkpoint eval for both RLOO K=8 and K=2 using the same eval manifest/seed as R5-style comparisons.

## 2026-06-11: R7 deterministic RLOO rerun split across TPUs

Purpose:
- Rerun the R6 RLOO K comparison on the deterministic platform branch lineage after discovering the R6 pair was launched from non-deterministic `harvey`.
- The R7 K-sweep is intentionally split across two TPUs/VMs so the runs do not clash.
- Harvey VM: running the RLOO K=2 deterministic full run recorded below.
- Shared Boris VM: running the matching RLOO K=8 deterministic full run recorded below.

Branch and deterministic checks:
- Branch: `harvey-grpo-k8-rerun`.
- K=2 launch commit on Harvey VM: `71aab87dee2d2c78256384d084d063d8b40c9e0c`.
- K=8 launch commit on shared Boris VM: `e3d69a1938fe8e8a2a67a3c84a03331133ed46f1`.
- Verified `deterministic-platform` is an ancestor and `harvey` is not an ancestor before the K=8 launch.
- Verified deterministic wiring exists for `MODEL_REVISION`, dependency refs, `EVAL_SEED`, `EVAL_MANIFEST`, `ADV_ESTIMATOR`, `NUM_GENERATIONS`, and `ckpts/run_metadata.json`.

K=2 run on separate Harvey VM:
- Run id: `R7-rloo-k2-det-harvey-full-s0-20260611_102009`.
- Tmux session: `r7-rloo-k2-det-harvey-full-s0`.
- Run root: `/home/harvey/tpu-runs/part-i/R7-rloo-k2-det-harvey-full-s0-20260611_102009`.
- W&B: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R7-rloo-k2-det-harvey-full-s0-20260611_102009`.
- Attach: `tmux attach -t r7-rloo-k2-det-harvey-full-s0`.
- Tail: `tail -f /home/harvey/tpu-runs/part-i/R7-rloo-k2-det-harvey-full-s0-20260611_102009/logs/train.log`.
- Config confirmed in log: `ADV_ESTIMATOR=rloo`, `NUM_GENERATIONS=2`, `MAX_STEPS=3364`, `SAVE_INTERVAL_STEPS=250`, `MAX_TO_KEEP=20`, and `MODEL_REVISION=dcc83ea841ab6100d6b47a070329e1ba4cf78752`.
- `RUN_SEED=0`, `EVAL_SEED=0`, and `MAX_STEPS_OVERRIDE` was unset.
- `ckpts/run_metadata.json` exists and records the deterministic metadata.
- Run-local launcher: `/home/harvey/tpu-runs/part-i/R7-rloo-k2-det-harvey-full-s0-20260611_102009/launch_train.sh`. This was used because the existing tmux server did not inherit newly supplied run env vars.

Failed K=2 setup attempts left in place:
- `/home/harvey/tpu-runs/part-i/R7-rloo-k2-det-harvey-full-s0-20260611_101430`: failed before config print because a hung diagnostic JAX process held `/tmp/libtpu_lockfile`; the process was stopped and the stale lockfile removed.
- `/home/harvey/tpu-runs/part-i/R7-rloo-k2-det-harvey-full-s0-20260611_101739`: failed at W&B init with `permission denied` because launching through the existing tmux server lost the intended `WANDB_RUN_ID`/`WANDB_DIR` env. The successful retry used a run-local launcher with explicit exports inside the tmux command.
- These failed roots were intentionally not deleted.

K=8 run on this shared Boris VM:
- Run id: `R7-rloo-k8-det-harvey-full-s0-20260611_105132`.
- Tmux session: `r7-rloo-k8-det-harvey-full-s0`.
- Run root: `/home/ext_harveybermingham1_gmail_com/tpu-runs/part-i/R7-rloo-k8-det-harvey-full-s0-20260611_105132`.
- W&B: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R7-rloo-k8-det-harvey-full-s0-20260611_105132`.
- Attach: `tmux attach -t r7-rloo-k8-det-harvey-full-s0`.
- Tail: `tail -f /home/ext_harveybermingham1_gmail_com/tpu-runs/part-i/R7-rloo-k8-det-harvey-full-s0-20260611_105132/logs/train.log`.
- Config confirmed in log: `ADV_ESTIMATOR=rloo`, `NUM_GENERATIONS=8`, `MAX_STEPS=3364`, `SAVE_INTERVAL_STEPS=250`, `MAX_TO_KEEP=20`, and `MODEL_REVISION=dcc83ea841ab6100d6b47a070329e1ba4cf78752`.
- `RUN_SEED=0`, `EVAL_SEED=0`, and `MAX_STEPS_OVERRIDE` was unset at launch.
- `ckpts/run_metadata.json` exists under the K=8 run root and W&B is syncing.
- Artifact dirs were set under the K=8 run root: `ckpts`, `intermediate_ckpt`, `tensorboard`, `data/train`, `data/test`, `wandb`, and `tmp`; no required artifact is intended to live only in `/tmp`.
- Use the exact K=8 run id in any future K=8 note filename. Do not overwrite the K=2 note from the Harvey VM, and do not update shared rollups until both current R7 runs finish.

R7 post-run lightweight eval collation:
- The two R7 run roots live on different VMs, so raw checkpoints will not automatically be in one physical folder. Keep the large checkpoint trees in their original `$RUN_ROOT`s.
- Choose one collector VM after both runs finish, then store only lightweight comparison evidence under:
  `/home/<user>/tpu-runs/part-i/report_diagnostics/r7_rloo_k_sweep_det_20260611/`
- Required shared eval manifest: `$HOME/tpu-runs/part-i/manifests/gsm8k_test_seed0_n64.jsonl`. If one VM creates this manifest first, copy the exact JSONL to the other VM before eval so K=2 and K=8 CSV rows are prompt-aligned.
- Recommended collector layout:

```bash
PAIR_ROOT="$HOME/tpu-runs/part-i/report_diagnostics/r7_rloo_k_sweep_det_20260611"
mkdir -p "$PAIR_ROOT"/{k2,k8,manifests}
cp "$HOME/tpu-runs/part-i/manifests/gsm8k_test_seed0_n64.jsonl" "$PAIR_ROOT/manifests/"
```

- For each run, copy only these lightweight files into the matching `k2/` or `k8/` folder: `eval/*_greedy.csv`, `eval/*summary*.txt`, `logs/eval_*.log`, `logs/train.log`, and `ckpts/run_metadata.json`. Do not copy `ckpts/actor/` unless eval must be rerun on the collector VM.
- Example on the collector VM after the relevant files have been transferred from the other VM:

```bash
K2_ROOT="/home/harvey/tpu-runs/part-i/R7-rloo-k2-det-harvey-full-s0-20260611_102009"
K8_ROOT="/home/ext_harveybermingham1_gmail_com/tpu-runs/part-i/R7-rloo-k8-det-harvey-full-s0-20260611_105132"
PAIR_ROOT="$HOME/tpu-runs/part-i/report_diagnostics/r7_rloo_k_sweep_det_20260611"

mkdir -p "$PAIR_ROOT"/{k2,k8}/{eval,logs,metadata}
cp "$K2_ROOT"/eval/*_greedy.csv "$PAIR_ROOT/k2/eval/"
cp "$K2_ROOT"/eval/*summary*.txt "$PAIR_ROOT/k2/eval/" 2>/dev/null || true
cp "$K2_ROOT"/logs/eval_*.log "$PAIR_ROOT/k2/logs/" 2>/dev/null || true
cp "$K2_ROOT"/logs/train.log "$PAIR_ROOT/k2/logs/"
cp "$K2_ROOT"/ckpts/run_metadata.json "$PAIR_ROOT/k2/metadata/"
cp "$K8_ROOT"/eval/*_greedy.csv "$PAIR_ROOT/k8/eval/"
cp "$K8_ROOT"/eval/*summary*.txt "$PAIR_ROOT/k8/eval/" 2>/dev/null || true
cp "$K8_ROOT"/logs/eval_*.log "$PAIR_ROOT/k8/logs/" 2>/dev/null || true
cp "$K8_ROOT"/logs/train.log "$PAIR_ROOT/k8/logs/"
cp "$K8_ROOT"/ckpts/run_metadata.json "$PAIR_ROOT/k8/metadata/"
```

- After collation, create a short markdown note in `experiments/variants/` using the exact R7 run ids, W&B URLs, eval summary paths, and best/final checkpoint metrics. Shared rollups should wait until both `k2` and `k8` folders contain eval CSVs from the same manifest.
