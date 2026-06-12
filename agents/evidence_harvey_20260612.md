# Evidence: Harvey Experiments, 2026-06-12

## Summary

This audit found two primary deterministic Harvey RLOO runs for the coursework register:

- `R7-rloo-k2-det-harvey-full-s0-20260611_102009`: RLOO, K=2, seed 0, full 3364 steps. Lightweight metadata/eval evidence is committed under `experiments/evidence/`; the original `/home/harvey/...` run root is not present in this local filesystem.
- `R7-rloo-k8-det-harvey-full-s0-20260611_105132`: RLOO, K=8, seed 0, full 3364 steps. The local run root exists under `/home/ext_harveybermingham1_gmail_com/tpu-runs/part-i/` with metadata, train/eval logs, TensorBoard event, checkpoints, and per-checkpoint eval CSVs.

The R7 pair is usable for report claims about RLOO K=2 versus K=8 on the same 64-question GSM8K eval manifest. The main caveat is that no base-model `--no-restore` eval CSV was found for either R7 run on this same manifest, so base-comparison claims are not report-ready from these artefacts alone.

Older related R6 RLOO K-sweep runs were also found in notes, but they were launched from non-deterministic branch `harvey` and lack retained-checkpoint eval CSVs and `run_metadata.json`; they are not usable as final performance evidence now.

W&B was not queried live during this audit. W&B URLs and runtime fields below come from local logs, local W&B metadata/summary JSON, and committed notes.

## Git State

- repo path: `/home/ext_harveybermingham1_gmail_com/tpu-2026`
- current branch: `harvey-grpo-k8-rerun`
- current HEAD: `f3da2c89edc9edecf2e5ae35f7e845ad2c165034` (`Add Harvey report artifacts`)
- status at start of audit: clean, `## harvey-grpo-k8-rerun...origin/harvey-grpo-k8-rerun`
- fetch result: `git fetch --all --prune` completed; it updated `origin/baron` from `57c6409` to `7ab0f1e`.
- relevant branches:
  - local: `coursework`, `harvey-grpo-k8-rerun`, `main`
  - remote: `origin/baron`, `origin/baron_k8`, `origin/coursework`, `origin/deterministic-platform`, `origin/fred`, `origin/harvey`, `origin/harvey-grpo-k8-rerun`, `origin/main`
- relevant commits:
  - `57c6409add0d81bbdb32ca7f4b3e176b4e044068`: `determinism`, on `origin/deterministic-platform`, `origin/coursework`, `origin/baron_k8`, and local `coursework`
  - `71aab87dee2d2c78256384d084d063d8b40c9e0c`: K=2 launch commit metadata; contained by `harvey-grpo-k8-rerun` and `origin/harvey-grpo-k8-rerun`
  - `e3d69a1938fe8e8a2a67a3c84a03331133ed46f1`: K=8 launch commit metadata; contained by `harvey-grpo-k8-rerun` and `origin/harvey-grpo-k8-rerun`
  - `8a0f7f266552cb2666710ac589cb6bda5cd40121`: older R6 `origin/harvey` launch commit, `Enable RLOO K8 experiment on harvey`
  - `f3da2c89edc9edecf2e5ae35f7e845ad2c165034`: current HEAD, `Add Harvey report artifacts`
- ancestry checks:
  - `57c6409` is an ancestor of both R7 launch commits.
  - `8a0f7f2` is not an ancestor of the K=8 R7 launch commit.

## Experiment Register

| Run | Branch | Commit | Estimator | K | Seed | Steps | Status | Best exact | Final exact | W&B | Artefacts usable? |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| `R7-rloo-k2-det-harvey-full-s0-20260611_102009` | `harvey-grpo-k8-rerun` | `71aab87` | RLOO | 2 | 0 | 3364 | completed; eval complete for retained checkpoints | 30/64 at step 500 | 1/64 at step 3364 | `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R7-rloo-k2-det-harvey-full-s0-20260611_102009` | Yes, with caveats |
| `R7-rloo-k8-det-harvey-full-s0-20260611_105132` | `harvey-grpo-k8-rerun` | `e3d69a1` | RLOO | 8 | 0 | 3364 | completed; eval complete for retained checkpoints | 35/64 at step 2000 | 26/64 at step 3364 | `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R7-rloo-k8-det-harvey-full-s0-20260611_105132` | Yes, with caveats |
| `R6-rloo-k8-full-s0-20260609_212314` | `harvey` | `8a0f7f2` | RLOO | 8 | 0 | 3364 | training complete; eval not found locally | n/a | n/a | `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R6-rloo-k8-full-s0-20260609_212314` | No |
| `R6-rloo-k2-full-s0-20260609_220042` | `harvey` | `8a0f7f2` | RLOO | 2 | 0 | 3364 | training complete; eval not found locally | n/a | n/a | `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R6-rloo-k2-full-s0-20260609_220042` | No |

## Run Details

### R7-rloo-k2-det-harvey-full-s0-20260611_102009

Identity:

- run nickname: R7 deterministic RLOO K=2 full seed 0
- run id: `R7-rloo-k2-det-harvey-full-s0-20260611_102009`
- owner/name: Harvey; W&B account in metadata is under team/entity `barongracias-university-of-cambridge`
- team/fork: `barongracias/tpu-2026`
- branch: `harvey-grpo-k8-rerun`
- commit hash: `71aab87dee2d2c78256384d084d063d8b40c9e0c`
- dirty worktree at launch: operator memory says no; no dirty marker is stored in `run_metadata.json`
- W&B project/entity: `agentic-ai-coursework` / `barongracias-university-of-cambridge`
- W&B URL: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R7-rloo-k2-det-harvey-full-s0-20260611_102009`
- local run root path recorded by metadata: `/home/harvey/tpu-runs/part-i/R7-rloo-k2-det-harvey-full-s0-20260611_102009`
- current local availability: original `/home/harvey/...` run root is not found locally; committed lightweight evidence exists at `experiments/evidence/R7-rloo-k2-det-harvey-full-s0-20260611_102009`
- date/time launched: W&B metadata `startedAt=2026-06-11T10:22:56.950168Z`; run metadata timestamp `2026-06-11T10:23:03.053879Z`

Configuration:

- estimator: `rloo`
- `NUM_GENERATIONS`: `2`
- seed: `RUN_SEED=0`, `EVAL_SEED=0`
- max steps: `3364`
- save interval: `250`
- max checkpoints kept: `20`
- dataset/source: GSM8K via `tfds`
- hard-question manifest: not used
- eval manifest path: `/home/harvey/tpu-runs/part-i/manifests/gsm8k_test_seed0_n64.jsonl`; committed copy at `experiments/evidence/R7-rloo-k2-det-harvey-full-s0-20260611_102009/manifests/gsm8k_test_seed0_n64.jsonl`
- eval manifest rows/hash: 64 rows; SHA-256 `9aa1814e295e155f4735c9302a7f9c28c6aaa2319074e053a5888e3f3b2448b0`
- reward function changes: no run-specific reward change found; active reward list in `scripts/rewards.py` is `match_format_exactly`, `match_format_approximately`, `check_answer`, `check_numbers`
- decoding/eval preset: `greedy`
- important env vars from retained launcher: `ADV_ESTIMATOR=rloo`, `NUM_GENERATIONS=2`, `RUN_SEED=0`, `EVAL_SEED=0`, `SAVE_INTERVAL_STEPS=250`, `MAX_TO_KEEP=20`, `MODEL_REVISION=dcc83ea841ab6100d6b47a070329e1ba4cf78752`, `WANDB_PROJECT=agentic-ai-coursework`, `WANDB_ENTITY=barongracias-university-of-cambridge`, `WANDB_MODE=online`; `MAX_STEPS_OVERRIDE` unset

Runtime and platform:

- TPU VM/account: separate Harvey VM according to notes
- host from W&B metadata: `t1v-n-6e5e72ce-w-0`
- TPU type/zone: not found locally
- runtime: local W&B summary `_runtime=20772.573726832` seconds, about 5h 46m 13s
- end time: note reports train log mtime after `Training finished.` at about `2026-06-11T16:09:17Z`; original train log is not present in this local filesystem
- completion status: completed; final checkpoint `ckpts/actor/3364` reported in notes
- failure reason: none for this successful run
- failed setup attempts before this run: notes list `R7-rloo-k2-det-harvey-full-s0-20260611_101430` and `R7-rloo-k2-det-harvey-full-s0-20260611_101739`; they were not deleted, but are not the successful run

Artefacts:

- run metadata copy: `experiments/evidence/R7-rloo-k2-det-harvey-full-s0-20260611_102009/metadata/run_metadata.json`
- original run metadata path: `/home/harvey/tpu-runs/part-i/R7-rloo-k2-det-harvey-full-s0-20260611_102009/ckpts/run_metadata.json`
- launch script copy: `experiments/evidence/R7-rloo-k2-det-harvey-full-s0-20260611_102009/metadata/launch_train.sh`
- W&B metadata/summary copies: `experiments/evidence/R7-rloo-k2-det-harvey-full-s0-20260611_102009/metadata/wandb-metadata.json`, `experiments/evidence/R7-rloo-k2-det-harvey-full-s0-20260611_102009/metadata/wandb-summary.json`
- training log: original path reported as `/home/harvey/tpu-runs/part-i/R7-rloo-k2-det-harvey-full-s0-20260611_102009/logs/train.log`; not found locally in this account and not copied into repo evidence
- TensorBoard scalar export: `experiments/evidence/R7-rloo-k2-det-harvey-full-s0-20260611_102009/eval/r7_rloo_k2_tensorboard_scalars.csv`
- TensorBoard scalar summary: `experiments/evidence/R7-rloo-k2-det-harvey-full-s0-20260611_102009/eval/r7_rloo_k2_tensorboard_scalar_summary.txt`
- checkpoint paths: original checkpoints reported under `/home/harvey/tpu-runs/part-i/R7-rloo-k2-det-harvey-full-s0-20260611_102009/ckpts/actor`; checkpoint directories themselves are not copied into repo evidence and original root is not present locally
- retained/evaluated steps: `250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3364`
- eval CSVs: `experiments/evidence/R7-rloo-k2-det-harvey-full-s0-20260611_102009/eval/r7_rloo_k2_step<step>_greedy.csv`
- eval logs: original logs referenced in notes but not found in current local filesystem/repo evidence
- scalar summaries: local W&B summary JSON plus TensorBoard scalar CSV/summary are present

Metrics:

All CSVs have 64 rows, `preset=greedy`, and `restored_step` matching the filename step. No base `--no-restore` CSV was found in the K=2 eval directory.

| Step | Exact | Partial | Format | Empty | Eval n | Restore confirmed | Base same sample? |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 250 | 23/64 (35.94%) | 25/64 (39.06%) | 34/64 (53.12%) | 0/64 | 64 | yes | no |
| 500 | 30/64 (46.88%) | 31/64 (48.44%) | 53/64 (82.81%) | 0/64 | 64 | yes | no |
| 750 | 25/64 (39.06%) | 27/64 (42.19%) | 44/64 (68.75%) | 0/64 | 64 | yes | no |
| 1000 | 27/64 (42.19%) | 28/64 (43.75%) | 49/64 (76.56%) | 0/64 | 64 | yes | no |
| 1250 | 27/64 (42.19%) | 29/64 (45.31%) | 38/64 (59.38%) | 0/64 | 64 | yes | no |
| 1500 | 25/64 (39.06%) | 28/64 (43.75%) | 58/64 (90.62%) | 0/64 | 64 | yes | no |
| 1750 | 15/64 (23.44%) | 17/64 (26.56%) | 27/64 (42.19%) | 26/64 | 64 | yes | no |
| 2000 | 1/64 (1.56%) | 1/64 (1.56%) | 10/64 (15.62%) | 52/64 | 64 | yes | no |
| 2250 | 0/64 (0.00%) | 0/64 (0.00%) | 2/64 (3.12%) | 62/64 | 64 | yes | no |
| 2500 | 6/64 (9.38%) | 7/64 (10.94%) | 16/64 (25.00%) | 45/64 | 64 | yes | no |
| 2750 | 1/64 (1.56%) | 2/64 (3.12%) | 4/64 (6.25%) | 59/64 | 64 | yes | no |
| 3000 | 1/64 (1.56%) | 2/64 (3.12%) | 4/64 (6.25%) | 60/64 | 64 | yes | no |
| 3250 | 1/64 (1.56%) | 1/64 (1.56%) | 2/64 (3.12%) | 60/64 | 64 | yes | no |
| 3364 | 1/64 (1.56%) | 1/64 (1.56%) | 5/64 (7.81%) | 58/64 | 64 | yes | no |

Caveats:

- Original K=2 run root is not found locally in this account; this audit relies on committed lightweight copies and notes for K=2.
- Training log, eval logs, raw checkpoint directories, and raw TensorBoard event file were not found locally for K=2.
- No base-model `--no-restore` eval CSV was present for this R7 K=2 run.
- W&B was not queried live.
- Dirty launch state is based on operator memory and committed metadata containing a clean git hash; metadata does not store a dirty flag.
- Usability: usable now for K=2 retained-checkpoint performance and collapse claims; not usable for base-comparison claims without a base eval CSV/log on the same manifest.

### R7-rloo-k8-det-harvey-full-s0-20260611_105132

Identity:

- run nickname: R7 deterministic RLOO K=8 full seed 0
- run id: `R7-rloo-k8-det-harvey-full-s0-20260611_105132`
- owner/name: Harvey; W&B account in metadata is under team/entity `barongracias-university-of-cambridge`
- team/fork: `barongracias/tpu-2026`
- branch: `harvey-grpo-k8-rerun`
- commit hash: `e3d69a1938fe8e8a2a67a3c84a03331133ed46f1`
- dirty worktree at launch: operator memory says no; no dirty marker is stored in `run_metadata.json`
- W&B project/entity: `agentic-ai-coursework` / `barongracias-university-of-cambridge`
- W&B URL from train log: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R7-rloo-k8-det-harvey-full-s0-20260611_105132`
- local run root path: `/home/ext_harveybermingham1_gmail_com/tpu-runs/part-i/R7-rloo-k8-det-harvey-full-s0-20260611_105132`
- repo note path: `experiments/evidence/R7-rloo-k8-det-harvey-full-s0-20260611_105132/README.md`
- date/time launched: W&B metadata `startedAt=2026-06-11T10:51:37.716770Z`; run metadata timestamp `2026-06-11T10:51:43.644559Z`

Configuration:

- estimator: `rloo`
- `NUM_GENERATIONS`: `8`
- seed: `RUN_SEED=0`, `EVAL_SEED=0`
- max steps: `3364`
- save interval: `250`
- max checkpoints kept: `20`
- dataset/source: GSM8K via `tfds`
- hard-question manifest: not used
- eval manifest path: `/home/ext_harveybermingham1_gmail_com/tpu-runs/part-i/manifests/gsm8k_test_seed0_n64.jsonl`
- eval manifest rows/hash: 64 rows; SHA-256 `9aa1814e295e155f4735c9302a7f9c28c6aaa2319074e053a5888e3f3b2448b0`, matching the committed K=2 manifest copy
- reward function changes: no run-specific reward change found; active reward list in `scripts/rewards.py` is `match_format_exactly`, `match_format_approximately`, `check_answer`, `check_numbers`
- decoding/eval preset: `greedy`
- important env vars/config from `run_metadata.json` and train log: `ADV_ESTIMATOR=rloo`, `NUM_GENERATIONS=8`, `RUN_SEED=0`, `EVAL_SEED=0`, `SAVE_INTERVAL_STEPS=250`, `MAX_TO_KEEP=20`, `MODEL_REVISION=dcc83ea841ab6100d6b47a070329e1ba4cf78752`, `WANDB_PROJECT=agentic-ai-coursework`, `WANDB_ENTITY=barongracias-university-of-cambridge`; `MAX_STEPS_OVERRIDE` unset per notes

Runtime and platform:

- TPU VM/account: shared Boris VM / current `ext_harveybermingham1_gmail_com` account according to notes
- host from W&B metadata: `t1v-n-0339f27d-w-0`
- TPU type/zone: not found locally
- runtime: local W&B summary `_runtime=27053.481609871` seconds, about 7h 30m 53s
- end time: train log mtime `2026-06-11 18:22:38 +0000`; train log contains `Training finished.`
- completion status: completed
- failure reason: none for this successful run

Artefacts:

- run metadata: `/home/ext_harveybermingham1_gmail_com/tpu-runs/part-i/R7-rloo-k8-det-harvey-full-s0-20260611_105132/ckpts/run_metadata.json`
- training log: `/home/ext_harveybermingham1_gmail_com/tpu-runs/part-i/R7-rloo-k8-det-harvey-full-s0-20260611_105132/logs/train.log`
- W&B metadata/summary: `/home/ext_harveybermingham1_gmail_com/tpu-runs/part-i/R7-rloo-k8-det-harvey-full-s0-20260611_105132/wandb/wandb/run-20260611_105137-R7-rloo-k8-det-harvey-full-s0-20260611_105132/files/wandb-metadata.json`, `.../wandb-summary.json`
- TensorBoard event: `/home/ext_harveybermingham1_gmail_com/tpu-runs/part-i/R7-rloo-k8-det-harvey-full-s0-20260611_105132/tensorboard/events.out.tfevents.1781175131.t1v-n-0339f27d-w-0`
- TensorBoard scalar export/summary: not found locally for K=8; raw event file exists
- checkpoint paths: `/home/ext_harveybermingham1_gmail_com/tpu-runs/part-i/R7-rloo-k8-det-harvey-full-s0-20260611_105132/ckpts/actor/<step>`
- retained checkpoint directories found locally: `1, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3364`
- eval CSVs: `/home/ext_harveybermingham1_gmail_com/tpu-runs/part-i/R7-rloo-k8-det-harvey-full-s0-20260611_105132/eval/r7_rloo_k8_step<step>_greedy.csv`
- eval logs: `/home/ext_harveybermingham1_gmail_com/tpu-runs/part-i/R7-rloo-k8-det-harvey-full-s0-20260611_105132/logs/eval_r7_rloo_k8_step<step>_greedy.log`
- eval summary: `/home/ext_harveybermingham1_gmail_com/tpu-runs/part-i/R7-rloo-k8-det-harvey-full-s0-20260611_105132/eval/r7_rloo_k8_all_retained_eval_summary.txt` and `.csv`
- local collector duplicate: `/home/ext_harveybermingham1_gmail_com/tpu-runs/part-i/report_diagnostics/r7_rloo_k_sweep_det_20260611/k8`

Metrics:

All CSVs have 64 rows, `preset=greedy`, and `restored_step` matching the filename step. Eval logs also confirm `Restored LoRA params ... at step <step>` for each evaluated checkpoint. No base `--no-restore` CSV was found in the K=8 eval directory.

| Step | Exact | Partial | Format | Empty | Eval n | Restore confirmed | Base same sample? |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 250 | 28/64 (43.75%) | 29/64 (45.31%) | 49/64 (76.56%) | 0/64 | 64 | yes | no |
| 500 | 30/64 (46.88%) | 31/64 (48.44%) | 53/64 (82.81%) | 0/64 | 64 | yes | no |
| 750 | 33/64 (51.56%) | 34/64 (53.12%) | 55/64 (85.94%) | 0/64 | 64 | yes | no |
| 1000 | 28/64 (43.75%) | 29/64 (45.31%) | 58/64 (90.62%) | 0/64 | 64 | yes | no |
| 1250 | 27/64 (42.19%) | 29/64 (45.31%) | 61/64 (95.31%) | 0/64 | 64 | yes | no |
| 1500 | 30/64 (46.88%) | 31/64 (48.44%) | 57/64 (89.06%) | 0/64 | 64 | yes | no |
| 1750 | 30/64 (46.88%) | 30/64 (46.88%) | 59/64 (92.19%) | 0/64 | 64 | yes | no |
| 2000 | 35/64 (54.69%) | 36/64 (56.25%) | 58/64 (90.62%) | 0/64 | 64 | yes | no |
| 2250 | 28/64 (43.75%) | 30/64 (46.88%) | 57/64 (89.06%) | 0/64 | 64 | yes | no |
| 2500 | 30/64 (46.88%) | 32/64 (50.00%) | 56/64 (87.50%) | 0/64 | 64 | yes | no |
| 2750 | 31/64 (48.44%) | 32/64 (50.00%) | 53/64 (82.81%) | 0/64 | 64 | yes | no |
| 3000 | 28/64 (43.75%) | 29/64 (45.31%) | 53/64 (82.81%) | 0/64 | 64 | yes | no |
| 3250 | 27/64 (42.19%) | 28/64 (43.75%) | 54/64 (84.38%) | 0/64 | 64 | yes | no |
| 3364 | 26/64 (40.62%) | 29/64 (45.31%) | 54/64 (84.38%) | 0/64 | 64 | yes | no |

Caveats:

- No base-model `--no-restore` eval CSV was present for this R7 K=8 run.
- K8 raw artefacts are present locally in this account, but only a README status note is committed under `experiments/evidence/R7-rloo-k8-det-harvey-full-s0-20260611_105132/`.
- K8 train log contains W&B step-order warnings near the end: attempts to log step 0 after current step 3364 were ignored.
- K8 train/eval logs contain nonfatal TFDS fresh-cache warnings such as missing `dataset_info.json` before data preparation/cache creation.
- TensorBoard raw event exists, but no K8 TensorBoard scalar CSV export/summary was found locally.
- W&B was not queried live.
- Dirty launch state is based on operator memory and committed metadata containing a clean git hash; metadata does not store a dirty flag.
- Usability: usable now for K=8 retained-checkpoint performance and no-empty-response claims; not usable for base-comparison claims without a base eval CSV/log on the same manifest.

### R6-rloo-k8-full-s0-20260609_212314

Identity/config/runtime:

- run nickname: R6 RLOO K=8 full seed 0 predecessor
- run id: `R6-rloo-k8-full-s0-20260609_212314`
- branch: `harvey`
- commit: W&B/notes record `8a0f7f266552cb2666710ac589cb6bda5cd40121`
- estimator/K/seed/steps: `rloo`, K=8, seed 0, 3364 steps
- status: training complete according to notes
- W&B URL: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R6-rloo-k8-full-s0-20260609_212314`
- launch/runtime notes: started about `2026-06-09T21:57:19Z`; K=2 chain records K=8 exit at `2026-06-10T05:37:42Z`; W&B runtime about 27,592 seconds

Artefacts/caveats:

- repo evidence status: `experiments/evidence/R6-rloo-k8-full-s0-20260609_212314/README.md`
- original collector path in notes: `/home/harvey/tpu-runs/part-i/report_diagnostics/R6-rloo-k8-full-s0-20260609_212314`
- current local availability: original `/home/harvey/...` collector/run root not found in this account
- missing locally/repo evidence: `ckpts/run_metadata.json`, base eval CSV, retained-checkpoint eval CSVs, per-checkpoint eval summary table
- provenance caveat: launched from non-deterministic `harvey` branch; lacks deterministic-platform controls from the R7 lineage
- usability: not usable for final performance claims now

### R6-rloo-k2-full-s0-20260609_220042

Identity/config/runtime:

- run nickname: R6 RLOO K=2 full seed 0 predecessor
- run id: `R6-rloo-k2-full-s0-20260609_220042`
- branch: `harvey`
- commit: W&B/notes record `8a0f7f266552cb2666710ac589cb6bda5cd40121`
- estimator/K/seed/steps: `rloo`, K=2, seed 0, 3364 steps
- status: training complete according to notes
- W&B URL: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R6-rloo-k2-full-s0-20260609_220042`
- launch/runtime notes: chained after K=8 at `2026-06-10T05:37:42Z`; train log mtime after completion about `2026-06-10T07:41:38Z`; W&B runtime about 7,425 seconds

Artefacts/caveats:

- repo evidence status: `experiments/evidence/R6-rloo-k2-full-s0-20260609_220042/README.md`
- original collector path in notes: `/home/harvey/tpu-runs/part-i/report_diagnostics/R6-rloo-k2-full-s0-20260609_220042`
- current local availability: original `/home/harvey/...` collector/run root not found in this account
- missing locally/repo evidence: `ckpts/run_metadata.json`, base eval CSV, retained-checkpoint eval CSVs, per-checkpoint eval summary table
- provenance caveat: launched from non-deterministic `harvey` branch; lacks deterministic-platform controls from the R7 lineage
- usability: not usable for final performance claims now

## Hard-Question Mining

Hard-question mining was not used for the R7 K=2 or K=8 runs.

- mining method: none used. Notes under `agents/context.md` and `agents/report_notes.md` discuss hard-example mining as a future inference-only probe, not as an executed run for these experiments.
- source baseline: none for R7; no hard-mined source baseline was found.
- manifest path: no hard-question manifest found. The only manifest used by R7 is the shared GSM8K eval manifest `gsm8k_test_seed0_n64.jsonl`.
- number of questions: not applicable.
- criteria for "hard": not applicable for these runs.
- branch/commit containing code: no hard-mining implementation was found in the inspected code for these runs.
- runs using the manifest: none found.

## Missing Evidence Checklist

- R7 K=2 original run root or copied training log, eval logs, raw TensorBoard event file, and checkpoint directory metadata from `/home/harvey/tpu-runs/part-i/R7-rloo-k2-det-harvey-full-s0-20260611_102009`.
- R7 base-model `--no-restore` eval CSV/log on the same SHA-256 `9aa1814e295e155f4735c9302a7f9c28c6aaa2319074e053a5888e3f3b2448b0` manifest for both K=2 and K=8.
- K8 TensorBoard scalar CSV export/summary, if scalar-curve claims are needed beyond the raw event file and W&B local summary.
- Explicit TPU type and zone metadata for both R7 runs, if the final register requires more than VM host/account.
- Live W&B export/query results, if final claims require server-side chart/runtime verification beyond local W&B metadata and logs.
- R6 retained-checkpoint eval CSVs, eval logs, and metadata, if R6 is to be discussed beyond "non-report-ready predecessor".

## Report-Ready Claims

- The R7 K=2 and K=8 runs were launched from `harvey-grpo-k8-rerun` commits descended from deterministic commit `57c6409`, with `run_metadata.json` recording commit, seed, model revision, dependency refs, run roots, W&B project/entity, and run configuration.
- The R7 K=2 and K=8 retained-checkpoint evals used matching 64-row GSM8K manifests with SHA-256 `9aa1814e295e155f4735c9302a7f9c28c6aaa2319074e053a5888e3f3b2448b0`.
- R7 K=2 best retained checkpoint was step 500 at `30/64` exact (`46.88%`); final step 3364 was `1/64` exact (`1.56%`) with `58/64` empty responses.
- R7 K=8 best retained checkpoint was step 2000 at `35/64` exact (`54.69%`); final step 3364 was `26/64` exact (`40.62%`) with `0/64` empty responses.
- On the saved R7 retained-checkpoint evals, K=8 avoids the empty-response collapse seen in K=2 after step 1750.
- R6 K=2/K=8 predecessor runs are not report-ready performance evidence because deterministic provenance and retained-checkpoint eval artefacts are missing.

## Not Yet Report-Ready

- Any claim that R7 K=8 beats the base model on the same eval sample; no same-manifest base `--no-restore` CSV/log was found.
- Any claim based on live W&B server state; W&B was not queried during this audit.
- Any detailed TPU type/zone claim; only VM hosts/accounts were found locally.
- Any final R6 RLOO K-sweep performance comparison; retained-checkpoint evals and deterministic metadata are missing.
- Any hard-question-mining result; no hard-mined RLOO run or manifest was found.
