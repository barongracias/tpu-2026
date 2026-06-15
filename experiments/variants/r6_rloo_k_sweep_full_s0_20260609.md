# R6 RLOO K-Sweep Full Seed 0 2026-06-09

## Run Identity

These two runs were chained on the Google TPU VM to compare RLOO at different group sizes under the same seed and full training budget. The run ids both use the `R6` prefix; treat them as an R6 RLOO K-sweep pair.

| Run id | K / `NUM_GENERATIONS` | Estimator | Started UTC | Finished UTC | Status |
| --- | ---: | --- | --- | --- | --- |
| `R6-rloo-k8-full-s0-20260609_212314` | 8 | `rloo` | 2026-06-09 ~21:57 | 2026-06-10 ~05:37 | training complete |
| `R6-rloo-k2-full-s0-20260609_220042` | 2 | `rloo` | 2026-06-10 05:37:42 | 2026-06-10 ~07:41 | training complete |

## Source State

- Launch branch: `harvey`.
- W&B-recorded launch commit for both runs: `8a0f7f266552cb2666710ac589cb6bda5cd40121` (`harvey` / `origin/harvey`, commit title `Enable RLOO K8 experiment on harvey`).
- Git reflog shows checkout/pull onto `harvey` before launch and no checkout to `harvey-grpo-k8-rerun` until 2026-06-10 10:19 UTC, after both R6 runs had finished.
- Note-time branch when this record was written: `harvey-grpo-k8-rerun`.
- Note-time HEAD when this record was written: `57c6409add0d81bbdb32ca7f4b3e176b4e044068`.
- Launch code had environment wiring for `ADV_ESTIMATOR` and `NUM_GENERATIONS`.
- `MAX_STEPS_OVERRIDE` was unset for both full runs.
- Important caveat: these runs did **not** use the deterministic-platform code path. The `harvey` launch commit and `deterministic-platform` have merge-base `324abbe`; the R6 launch commit lacks deterministic-platform controls such as required `MODEL_REVISION`, `EVAL_SEED`/`EVAL_MANIFEST`, pinned `JAX_REF`/`QWIX_REF`/`FLAX_REF`, and the deterministic branch's `run_metadata.json` writer.

## Configuration

Common settings:
- Data source: `tfds`
- Estimator: `rloo`
- Seed: `0`
- Max steps: `3364`
- Save interval: `250`
- Max retained checkpoints: `20`
- Train/test data dirs: run-local under each `$RUN_ROOT/data/...`
- Checkpoint, TensorBoard, W&B-local, and log artifacts: persistent `$HOME/tpu-runs/part-i/...`, not `/tmp`.

K=8:
- `ADV_ESTIMATOR=rloo`
- `NUM_GENERATIONS=8`
- W&B: https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R6-rloo-k8-full-s0-20260609_212314

K=2:
- `ADV_ESTIMATOR=rloo`
- `NUM_GENERATIONS=2`
- W&B: https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R6-rloo-k2-full-s0-20260609_220042

## Evidence Paths

K=8:
- Run root: `/home/harvey/tpu-runs/part-i/R6-rloo-k8-full-s0-20260609_212314`
- Train log: `/home/harvey/tpu-runs/part-i/R6-rloo-k8-full-s0-20260609_212314/logs/train.log`
- Final checkpoint: `/home/harvey/tpu-runs/part-i/R6-rloo-k8-full-s0-20260609_212314/ckpts/actor/3364`
- TensorBoard: `/home/harvey/tpu-runs/part-i/R6-rloo-k8-full-s0-20260609_212314/tensorboard/events.out.tfevents.1781042268.t1v-n-6e5e72ce-w-0`
- Local W&B artifacts: `/home/harvey/tpu-runs/part-i/R6-rloo-k8-full-s0-20260609_212314/wandb`

K=2:
- Run root: `/home/harvey/tpu-runs/part-i/R6-rloo-k2-full-s0-20260609_220042`
- Train log: `/home/harvey/tpu-runs/part-i/R6-rloo-k2-full-s0-20260609_220042/logs/train.log`
- Chain script: `/home/harvey/tpu-runs/part-i/R6-rloo-k2-full-s0-20260609_220042/chain_after_k8.sh`
- Final checkpoint: `/home/harvey/tpu-runs/part-i/R6-rloo-k2-full-s0-20260609_220042/ckpts/actor/3364`
- TensorBoard: `/home/harvey/tpu-runs/part-i/R6-rloo-k2-full-s0-20260609_220042/tensorboard/events.out.tfevents.1781069886.t1v-n-6e5e72ce-w-0`
- Local W&B artifacts: `/home/harvey/tpu-runs/part-i/R6-rloo-k2-full-s0-20260609_220042/wandb`

## Results

- Both runs logged `Training finished.`.
- Both runs wrote retained actor checkpoints at `1`, every `250` steps from `250` through `3250`, and final step `3364`.
- Each run directory is about `3.5G`, with checkpoints about `3.3G`.
- `/tmp` was not the artifact location; host `/tmp` was about `61M` when checked, while the run artifacts live under `/home/harvey/tpu-runs/part-i/...`.
- No post-training greedy eval CSVs or eval summaries were found for either R6 run at note time.
- No `run_metadata.json` was found in either R6 checkpoint root at note time.
- W&B step-order warnings occurred near final step `3364`, same class as previous runs.

## Retained Checkpoints

Both K=8 and K=2 have:

`ckpts/actor/{1,250,500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250,3364}`

## Interpretation

Operationally, the RLOO K-sweep training pair completed. The runs are not yet report-ready performance evidence because retained-checkpoint evaluation has not been run or recorded. The immediate next step is to evaluate matching checkpoints for K=8 and K=2 with the same held-out manifest/eval seed used for R5-style comparisons, then compare against R5 GRPO K=8, R3 RLOO K=2, and base.
