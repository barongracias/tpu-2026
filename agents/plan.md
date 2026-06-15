# TPU-2026 Coursework Branch Plan

## Objective

Prepare the `tpu-2026` baseline for reproducible Part I practical runs: baseline GRPO reproduction, controlled RLOO comparison at fixed compute, checkpoint-backed evaluation, and source-backed report evidence.

## Source Of Truth

- Coursework PDF and report planning live in `../agentic-ai-coursework`.
- Main runbook: `../agentic-ai-coursework/experiments/runbooks/tpu_day1_runbook.md`.
- Main patch plan: `../agentic-ai-coursework/experiments/manifests/baseline_patch_plan.md`.
- This repository contains the actual training/evaluation patches for the TPU code.
- Local run notes, diagnostics, manifests, runbooks, and evidence indexes for the current TPU runs live under `experiments/`; start with `experiments/README.md`.

## Milestone 1: Verify Patch Scope

Status: complete; latest pulled workflow head is `4339ba8`.

Goal:
- Confirm P0-P6 are implemented without changing Tunix.

Checks:
- `git diff --name-only main..HEAD` shows only `bootstrap.sh`, `scripts/config.py`, `scripts/data.py`, `scripts/train.py`, `scripts/evaluate.py` for the 8 commits.
- `git diff --check` passes.
- Syntax compilation passes with `PYTHONPYCACHEPREFIX=/tmp/tpu2026-pycache`.

## Milestone 2: Commit/Share Local Follow-Up Fix

Status: complete at latest pulled coursework head.

Goal:
- Preserve the full-run learning-rate schedule during short debug runs, record the resolved W&B run id, and incorporate Fred's debug-checkpoint review.

Files:
- `scripts/config.py`: `FULL_MAX_STEPS`/`LR_DECAY_STEPS` are committed; local follow-up makes `SAVE_INTERVAL_STEPS` env-overridable.
- `scripts/train.py`: `LR_DECAY_STEPS` and resolved W&B id are committed; local follow-up records `lr_decay_steps` and `save_interval_steps` in metadata.
- `scripts/evaluate.py`: local follow-up auto-resolves an `actor/` checkpoint child and records requested/resolved checkpoint paths.

Verification:
- `git diff --check`.
- `env PYTHONPYCACHEPREFIX=/tmp/tpu2026-pycache python3 -m py_compile scripts/config.py scripts/data.py scripts/train.py scripts/evaluate.py`.

Validation:
- `git diff --check`: passed.
- `py_compile` for `scripts/config.py`, `scripts/data.py`, `scripts/train.py`, and `scripts/evaluate.py`: passed.

## Milestone 3: TPU Day-One Debug

Status: complete — D1 GRPO and D2 RLOO debug runs passed.

Goal:
- Run short GRPO and RLOO jobs before committing full TPU time.

Steps:
- Create persistent `$RUN_ROOT` under `$HOME/tpu-runs/...`.
- Export `CKPT_DIR`, `INTERMEDIATE_CKPT_DIR`, `TENSORBOARD_DIR`, `RUN_SEED=0`, `MAX_STEPS_OVERRIDE=50`, and `SAVE_INTERVAL_STEPS=50`.
- D1 with `ADV_ESTIMATOR=grpo`: complete.
- D2 with `ADV_ESTIMATOR=rloo`: complete.
- Record each run in an iteration log.

Checks:
- TPU backend is active.
- Output paths are persistent.
- Checkpoints are saved.
- TensorBoard/W&B metrics appear.
- `run_metadata.json` records estimator, seed, paths, commit, and W&B run id.

## Milestone 4: Evaluation Restore Check

Status: complete for D1 and D2.

Goal:
- Confirm evaluation uses trained LoRA parameters, not an un-restored LoRA wrapper.

Command shape:

```bash
python -u scripts/evaluate.py --preset greedy --source tfds --ckpt-dir "$CKPT_DIR" --output-csv "$RUN_ROOT/eval/d1_grpo_eval.csv"
```

Checks:
- `restored_step` and the resolved actor checkpoint root are printed.
- CSV contains one row per prompt.
- Aggregate accuracy can be recomputed from the CSV.

Results:
- D1 restored step 50 from `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D1-grpo-debug-20260608_142021/ckpts/actor`.
- D2 restored step 50 from `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D2-rloo-debug-20260608_144102/ckpts/actor`.


## Milestone 4.5: Pre-Full-Run Hygiene Patch

Status: complete; committed, pushed, and pulled at `4339ba8`.

Goal:
- Avoid repo-local TFDS/protobuf cache issues in future runs by making train/test data dirs env-overridable.
- Ensure `run_metadata.json` records the real repo commit even when training is launched from a run-local cwd.
- Record `repo_root`, `launch_cwd`, `train_data_dir`, and `test_data_dir` in metadata for auditability.

Checks:
- `git diff --check`.
- `env PYTHONPYCACHEPREFIX=/tmp/tpu2026-pycache python3 -m py_compile scripts/config.py scripts/data.py scripts/train.py scripts/evaluate.py`.

Validation:
- `git diff --check`: passed.
- `py_compile` for `scripts/config.py`, `scripts/data.py`, `scripts/train.py`, and `scripts/evaluate.py`: passed.

## Milestone 5: Full Controlled Runs

Status: R5 GRPO K=8 seed 0 training and retained-checkpoint eval complete; R4/R2 and any new run remain blocked pending review.

Goal:
- Execute the locked GRPO vs RLOO comparison with fixed data, seed controls, and compute budget.

Checks:
- Same held-out split and evaluation preset.
- Same total training step budget unless explicitly justified.
- Per-run metadata, logs, checkpoints, and eval CSVs preserved.

Completed run:
- R1 session: `r1-grpo-full-s0` exited
- R1 run root: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231`
- Launched from repo cwd at HEAD `4339ba8` with run-local train/test data dirs and no `MAX_STEPS_OVERRIDE`.
- W&B: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R1-grpo-full-s0`
- Final monitor: tmux exited; `Training finished.` present; final checkpoint `ckpts/actor/3364` exists; TensorBoard event file exists; W&B emitted step-order warnings.

Launch requirements:
- Launch R1 from repo cwd so `run_metadata.json` records the real git commit.
- Set `TRAIN_DATA_DIR=$RUN_ROOT/data/train` and `TEST_DATA_DIR=$RUN_ROOT/data/test` to avoid repo-local TFDS/protobuf cache issues.
- Keep all run artefacts under `$HOME/tpu-runs/part-i/...`.
- Do not start R3/R4/R2 until Baron approves after R1 review.

## Milestone 5.5: R1 Evaluation

Status: complete; review pending.

Results:
- Trained R1 restored from `ckpts/actor/3364` with `restored_step=3364`.
- Trained R1 greedy: `correct=12/64`, `acc=18.75%`, `partial=18.75%`, `format=35.94%`.
- Base greedy same setup: `correct=31/64`, `acc=48.44%`, `partial=48.44%`, `format=1.56%`.
- Trained CSV: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231/eval/r1_grpo_step3364_greedy.csv`
- Base CSV: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231/eval/base_greedy_seed0.csv`
- Logs: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231/logs/eval_r1_grpo_step3364_greedy.log` and `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231/logs/eval_base_greedy_seed0.log`
- TFDS/protobuf cache issue recurred for reused caches; successful evals used fresh eval-local TFDS caches.

Next:
- Treat R1 as a valid completed I.1 GRPO baseline control operationally.
- Review why trained greedy eval underperforms base before launching R3.
- Do not start additional full runs until Baron approves.


## Milestone 5.6: R1 Diagnostic Pass

Status: complete; review pending.

Metrics:
| Eval | Exact | Partial | Format |
| --- | ---: | ---: | ---: |
| Base greedy | 31/64 (48.44%) | 31/64 (48.44%) | 1/64 (1.56%) |
| R1 step 2000 | 24/64 (37.50%) | 26/64 (40.62%) | 28/64 (43.75%) |
| R1 step 2500 | 11/64 (17.19%) | 11/64 (17.19%) | 23/64 (35.94%) |
| R1 step 3000 | 8/64 (12.50%) | 8/64 (12.50%) | 16/64 (25.00%) |
| R1 step 3364 | 12/64 (18.75%) | 12/64 (18.75%) | 23/64 (35.94%) |

Findings:
- Best retained R1 checkpoint is step 2000, but it still trails base exact accuracy: `37.50%` vs `48.44%`.
- Accuracy degrades after step 2000: step 2500 `17.19%`, step 3000 `12.50%`, step 3364 `18.75%`.
- Format compliance improves versus base, especially at step 2000, but numeric correctness declines.
- TensorBoard scalar summary written to `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231/eval/r1_tensorboard_scalar_summary.txt`.
- Relevant tags include KL, pg_clipfrac, reward components, and completion lengths; no `advantage/nonzero_frac` or ratio-spread tags were found.
- KL is a warning sign: eval KL rises from `0.324` near step 1984 to `0.859` near step 3008, while train KL reaches `0.977` near step 3008.
- `pg_clipfrac` stays `0.0`, so clipping does not explain or catch the degradation.

Recommendation:
- Do not launch R3/R4/R2 yet.
- Review R1 curves and CSV rows first; consider early checkpoint selection, shorter run length, reward-balance changes, or eval-control changes before spending R3 TPU time.


## Milestone 5.7: R3 RLOO Full Run

Status: complete; review pending.

Run:
- Run root: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R3-rloo-full-s0-20260608_230810`
- W&B: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R3-rloo-full-s0-20260608_230810`
- Estimator: `rloo`; seed: `0`; max steps: `3364`; commit: `99cb7f8`.
- Checkpoints: `ckpts/actor/2000`, `2500`, `3000`, `3364`.

Greedy eval metrics:
| Eval | Exact | Partial | Format |
| --- | ---: | ---: | ---: |
| R3 step 2000 | 1/64 (1.56%) | 1/64 (1.56%) | 10/64 (15.62%) |
| R3 step 2500 | 6/64 (9.38%) | 7/64 (10.94%) | 16/64 (25.00%) |
| R3 step 3000 | 1/64 (1.56%) | 2/64 (3.12%) | 4/64 (6.25%) |
| R3 step 3364 | 1/64 (1.56%) | 1/64 (1.56%) | 5/64 (7.81%) |

Findings:
- R3 trained and evaluated without a fatal run failure.
- Best retained checkpoint is step 2500, but exact accuracy is only `9.38%`.
- R3 underperforms both base and R1 retained checkpoints on the 64-prompt greedy eval.
- W&B step-order warnings persist; evals used fresh per-step TFDS caches to avoid the protobuf/cache issue.

Recommendation:
- Do not launch R4/R2 or K=8 yet.
- Review R1/R3 reward curves, checkpoint behaviour, and reward/eval controls before more full TPU runs.


## Milestone 5.8: R1/R3 Report Diagnosis

Status: complete; review pending.

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
- Qualitative examples CSV: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/report_diagnostics/base_correct_r1_r3_wrong_examples.csv`

Findings:
- R1 degrades with training: step 2000 is the best retained checkpoint, final is substantially worse.
- R1 exhibits format optimisation and arithmetic degradation; KL rises substantially before the final checkpoint and `pg_clipfrac` stays `0.0`.
- R3 RLOO performs worse than R1 and shows many empty generations: 45/64 empty at step 2500 and 58/64 empty at final.
- Cache artefact is unlikely to explain the main result because successful evals used fresh eval-local TFDS caches and the base model remains strong.

Decision:
- Do not start another full run yet.
- A cheap D3 K=8 debug is justified, but only as a short diagnostic with `NUM_GENERATIONS=8`, empty-response checks, fresh eval caches, and review before any full K=8 run.

## Milestone 6: Evidence Extraction

Status: pending runs.

Goal:
- Provide report-ready evidence for I.1 and I.3.

Outputs:
- Reward curves.
- KL curves.
- Diagnostic curves such as `advantage/nonzero_frac` if logged.
- Accuracy/score table with bootstrap confidence intervals from per-prompt CSVs.

## Milestone 7: Hard-Example Curriculum Probe

Status: proposed; no training run required for selection.

Goal:
- Mine a reproducible GSM8K-train hard subset for a possible curriculum/ablation inspired by the deep-research notes.

Selection protocol:
- Run an inference-only probe over GSM8K train using the frozen base model, fixed prompt template, fixed decoding presets, and fixed sample seeds.
- Define hardness from pass/fail counts, for example greedy wrong plus all sampled attempts wrong for "hard", and greedy wrong plus at least one sampled success for "medium".
- Save a manifest with source split, original index or question hash, answer, model/revision/checkpoint, decoding settings, seeds, pass/fail counts, and selection rule.

Cautions:
- Do not use held-out eval/test examples for mining.
- Using R5/R1/R3 failures for selection is allowed only if framed as "hard for the already-finetuned policy"; it is not the clean base-difficulty curriculum.
- Add GSM-Hard only as a separate ablation after GSM8K-train hard mining, because it introduces distribution shift as well as difficulty.

## Milestone 8: Reproducibility Contract

Status: implemented locally; validate on TPU before the next run.

Goal:
- Make comparable experiments fail fast unless their eval split, dependency refs, model revision, persistent paths, and launch metadata are explicit.

Implemented:
- Separate `RUN_SEED` and `EVAL_SEED`.
- JSONL `EVAL_MANIFEST` support for held-out eval prompt identity.
- Required exact `MODEL_REVISION` for Hugging Face model download.
- Required exact JAX/Qwix/Flax refs in `bootstrap.sh`; Tunix remains pinned to `683256db1a0919b5cfd46cee52cebc96331494fb`.
- Portable `scripts/run_tmux.sh` that derives repo path and exports run-local paths from `RUN_ROOT`.
- Contract documentation in `experiments/manifests/experiment_contract.md`.

## Milestone 5.9: D3 GRPO K=8 Debug

Status: complete; review pending.

Purpose:
- Test whether increasing GRPO group size from `K=2` to `K=8` is operationally safe and reduces early instability before considering any full K=8 run.

Code/config:
- `scripts/config.py` now supports `NUM_GENERATIONS` via environment override, defaulting to `2`.
- D3 used `NUM_GENERATIONS=8`; no commit or push has been made.
- Reproducibility patch/status files were saved in `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D3-grpo-k8-debug-20260609_094815/patches/`.

Run:
- Run root: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D3-grpo-k8-debug-20260609_094815`
- Estimator: `grpo`; seed: `0`; max steps: `50`; save interval: `50`; group size: `8`.
- Training completed and wrote `ckpts/actor/50` without fatal error or OOM.
- W&B: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/D3-grpo-k8-debug-20260609_094815`

Eval result:
| Eval | Exact | Partial | Format | Empty responses |
| --- | ---: | ---: | ---: | ---: |
| D3 GRPO K=8 step 50 | 31/64 (48.44%) | 32/64 (50.00%) | 8/64 (12.50%) | 0/64 |

Decision:
- D3 passed the cheap debug gate operationally: K=8 fits and checkpoints/eval restore work.
- Do not start a full K=8 run automatically. Review whether the step-50 signal is enough, because it matches base exact accuracy but does not yet demonstrate full-horizon stability.

## Milestone 5.10: Code Audit And D4 K=8 Medium Debug

Status: complete; review pending.

Audit result:
- No blocking code bug found in training/eval wiring.
- Reward sanity check passed mechanically, but showed a reward-control concern: wrong well-formatted answer total `6.0`, correct well-formatted answer total `10.0`, empty response total `-2.5`, malformed answer tag with correct number total `3.0`.
- Audit note: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/report_diagnostics/code_audit_d4_hygiene_note.md`
- Reward sanity output: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/report_diagnostics/reward_sanity_check.txt`

Metadata hygiene:
- `NUM_GENERATIONS` is env-overridable.
- `run_metadata.json` now records generation/sampling and loss-control parameters: `num_generations`, `beta`, `epsilon`, `temperature`, `top_k`, `top_p`, `total_generation_steps`.
- Training startup logs now print those values.

D4 run:
- Run root: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D4-grpo-k8-medium-debug-20260609_100945`
- Config: GRPO, K=8, seed 0, 500 steps, save every 100, fresh run-local data dirs and eval caches.
- Training completed without OOM/fatal error; W&B step-order warnings persisted.
- Step 100 was pruned before eval because `MAX_TO_KEEP=4` and five checkpoints were requested.

Metrics:
| Step | Exact | Partial | Format | Empty |
| ---: | ---: | ---: | ---: | ---: |
| 100 | missing/pruned | missing/pruned | missing/pruned | missing/pruned |
| 200 | 26/64 (40.62%) | 28/64 (43.75%) | 55/64 (85.94%) | 0/64 |
| 300 | 32/64 (50.00%) | 34/64 (53.12%) | 51/64 (79.69%) | 0/64 |
| 400 | 29/64 (45.31%) | 29/64 (45.31%) | 55/64 (85.94%) | 0/64 |
| 500 | 34/64 (53.12%) | 35/64 (54.69%) | 54/64 (84.38%) | 0/64 |

Decision:
- Do not start a full K=8 run automatically.
- Full K=8 GRPO is now reasonable to consider after Baron review, preferably with early checkpoint selection and a `MAX_TO_KEEP`/checkpoint-retention hygiene fix if all planned checkpoints must be evaluated.

## Milestone 5.11: Checkpoint Retention Hygiene And R5 Approval

Status: hygiene patch prepared; R5 launch approved after patch validation/commit.

Patch:
- Make `MAX_TO_KEEP` env-overridable, defaulting to `4`.
- Record `max_to_keep` in `run_metadata.json`.
- Print `SAVE_INTERVAL_STEPS` and `MAX_TO_KEEP` in the startup log.

R5 plan:
- Launch full GRPO K=8 seed 0 only; do not start R4/R2.
- Use `NUM_GENERATIONS=8`, no `MAX_STEPS_OVERRIDE`, `SAVE_INTERVAL_STEPS=250`, and `MAX_TO_KEEP=20`.
- Evaluate retained checkpoints `500`, `1000`, `1500`, `2000`, `2500`, `3000`, and `3364` if present.
- Keep run artifacts under `$HOME/tpu-runs/part-i`, not in the repo.

## Milestone 5.12: R5 GRPO K=8 Full Run

Status: complete; review pending.

Run:
- Run root: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R5-grpo-k8-full-s0-20260609_114832`
- W&B: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R5-grpo-k8-full-s0-20260609_114832`
- Commit: `820fad61060a4260184e71066568af7b28d3109e`
- Config: GRPO, `NUM_GENERATIONS=8`, seed `0`, full `3364` steps, `SAVE_INTERVAL_STEPS=250`, `MAX_TO_KEEP=20`.
- Training completed without fatal error or OOM. W&B step-order warnings persisted.

Retained-checkpoint eval metrics:
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
- Summary: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R5-grpo-k8-full-s0-20260609_114832/eval/r5_all_retained_eval_summary.txt`
- CSVs/logs are under `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R5-grpo-k8-full-s0-20260609_114832/eval` and `logs`.

Decision:
- Best R5 checkpoint is step 3250 at `35/64` exact, slightly above D4 step 500 (`34/64`) and base (`31/64`).
- R5 final is `32/64`, so checkpoint selection remains important.
- Do not launch R4/R2 or any new run until local review. Next report work should compute confidence intervals/paired comparisons and inspect W&B/TensorBoard curves for R5.

## 2026-06-15 Shutdown Plan State

Status: Baron reward-rebalance and full-test evidence are complete and analysis-ready on branch `baron-reward-rebalance-k8`.

Completed work:
- Reward-weight knobs implemented with baseline-preserving defaults and recorded in run metadata.
- Existing R1/R3/R5/D4 lightweight artefacts harvested into `experiments/evidence/<run-id>/` and summarized in `experiments/team_evidence/result_existing-runs-harvest_baron_20260613.md`.
- Reward-rebalance run `B-grpo-k8-rewardrebalance-s0-20260614` trained, evaluated on n=64, scalar-exported, summarized, and committed.
- Shared full-test protocol artefacts and Harvey base CSV synced into this branch.
- Full-test eval completed for Baron report-ready targets R5 and B, with bootstrap CIs committed.

Next analysis steps off-TPU:
- Pull `baron-reward-rebalance-k8` and use `experiments/team_evidence/result_full-test-baron_20260613.md` as the primary Baron full-test evidence file.
- Collate full-test rows into the team register/report using base 625/1319, R5 740/1319, and B 704/1319.
- Treat R5 as the stronger GRPO K=8 result; the reward-rebalance run improved over base but did not beat R5 on the full test.
- Preserve the provenance caveat that the harvested R1/R3/R5/D4 n=64 CSVs are from Baron's seed-0 draw and are separate from the committed n=64/full-test manifests.

No additional TPU work is required for Baron's report-ready evidence unless the team requests new diagnostics from raw checkpoints.
