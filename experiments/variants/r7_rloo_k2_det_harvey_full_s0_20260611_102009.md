# R7 RLOO K=2 Deterministic Full Seed 0 2026-06-11

## Run Identity

- Run id: `R7-rloo-k2-det-harvey-full-s0-20260611_102009`
- Estimator: `rloo`
- Group size / `NUM_GENERATIONS`: `2`
- Seed: `0`
- Purpose: deterministic rerun of the RLOO K=2 side of the R7 K-sweep after R6 was found to have been launched from a non-deterministic branch.
- W&B: https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R7-rloo-k2-det-harvey-full-s0-20260611_102009

## Source State

- Branch: `harvey-grpo-k8-rerun`
- Launch commit recorded in metadata: `71aab87dee2d2c78256384d084d063d8b40c9e0c`
- Model revision: `dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- Dependency refs recorded in metadata:
  - JAX: `3ff6dc40cdb982921de0ef12c0bf8e5c64311a6f`
  - Tunix: `7fadb3c81e4348f714b2b5a07f6cc2bd10da10de`
  - Qwix: `c2548f06eeca005090329530576623afcd48241c`
  - Flax: `5ab9d1463550a7fc7fe13575af55081e68258dd7`

## Configuration

- Data source: `tfds`
- `RUN_SEED=0`
- `EVAL_SEED=0`
- Eval manifest: `/home/harvey/tpu-runs/part-i/manifests/gsm8k_test_seed0_n64.jsonl`
- Max steps: `3364`
- Save interval: `250`
- Max retained checkpoints: `20`
- `MAX_STEPS_OVERRIDE`: unset
- Run-local artifact dirs were used for checkpoints, TensorBoard, train/test data, W&B local files, and tmp.

## Evidence Paths

- Run root: `/home/harvey/tpu-runs/part-i/R7-rloo-k2-det-harvey-full-s0-20260611_102009`
- Metadata: `/home/harvey/tpu-runs/part-i/R7-rloo-k2-det-harvey-full-s0-20260611_102009/ckpts/run_metadata.json`
- Retained eval summary: `/home/harvey/tpu-runs/part-i/R7-rloo-k2-det-harvey-full-s0-20260611_102009/eval/r7_rloo_k2_all_retained_eval_summary.txt`
- Retained eval metrics CSV: `/home/harvey/tpu-runs/part-i/R7-rloo-k2-det-harvey-full-s0-20260611_102009/eval/r7_rloo_k2_all_retained_eval_metrics.csv`
- TensorBoard scalar CSV: `/home/harvey/tpu-runs/part-i/R7-rloo-k2-det-harvey-full-s0-20260611_102009/eval/r7_rloo_k2_tensorboard_scalars.csv`
- TensorBoard scalar summary: `/home/harvey/tpu-runs/part-i/R7-rloo-k2-det-harvey-full-s0-20260611_102009/eval/r7_rloo_k2_tensorboard_scalar_summary.txt`
- Lightweight collector: `/home/harvey/tpu-runs/part-i/report_diagnostics/r7_rloo_k_sweep_det_20260611/k2`
- Shared manifest copy: `/home/harvey/tpu-runs/part-i/report_diagnostics/r7_rloo_k_sweep_det_20260611/manifests/gsm8k_test_seed0_n64.jsonl`

## Retained-Checkpoint Greedy Eval

| Step | Exact | Partial | Format | Empty | Mean words |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 23/64 (35.94%) | 25/64 (39.06%) | 34/64 (53.12%) | 0/64 | 157.7 |
| 500 | 30/64 (46.88%) | 31/64 (48.44%) | 53/64 (82.81%) | 0/64 | 183.8 |
| 750 | 25/64 (39.06%) | 27/64 (42.19%) | 44/64 (68.75%) | 0/64 | 275.5 |
| 1000 | 27/64 (42.19%) | 28/64 (43.75%) | 49/64 (76.56%) | 0/64 | 273.1 |
| 1250 | 27/64 (42.19%) | 29/64 (45.31%) | 38/64 (59.38%) | 0/64 | 332.3 |
| 1500 | 25/64 (39.06%) | 28/64 (43.75%) | 58/64 (90.62%) | 0/64 | 192.8 |
| 1750 | 15/64 (23.44%) | 17/64 (26.56%) | 27/64 (42.19%) | 26/64 | 168.2 |
| 2000 | 1/64 (1.56%) | 1/64 (1.56%) | 10/64 (15.62%) | 52/64 | 46.9 |
| 2250 | 0/64 (0.00%) | 0/64 (0.00%) | 2/64 (3.12%) | 62/64 | 6.5 |
| 2500 | 6/64 (9.38%) | 7/64 (10.94%) | 16/64 (25.00%) | 45/64 | 77.0 |
| 2750 | 1/64 (1.56%) | 2/64 (3.12%) | 4/64 (6.25%) | 59/64 | 14.7 |
| 3000 | 1/64 (1.56%) | 2/64 (3.12%) | 4/64 (6.25%) | 60/64 | 12.4 |
| 3250 | 1/64 (1.56%) | 1/64 (1.56%) | 2/64 (3.12%) | 60/64 | 15.2 |
| 3364 | 1/64 (1.56%) | 1/64 (1.56%) | 5/64 (7.81%) | 58/64 | 24.5 |

## Result

- Training completed and final checkpoint `ckpts/actor/3364` exists.
- Retained-checkpoint eval completed for steps `250` through `3364` using the shared manifest.
- Best retained checkpoint by exact accuracy: step `500`, `30/64` exact (`46.88%`).
- Final checkpoint: step `3364`, `1/64` exact (`1.56%`), with `58/64` empty responses.
- Collapse begins after the early retained checkpoints: step `1750` has `26/64` empty responses, and steps `2000+` are mostly empty or near-empty.

## Collation Status

This note records only the K=2 side available on the Harvey VM. The documented K=8 R7 run root lives on the separate shared Boris VM and was not present under `/home/harvey/tpu-runs/part-i` during this export. The collector folder has K=2 evidence and empty K=8 placeholders; do not treat the R7 K-sweep comparison as complete until K=8 eval CSVs/logs/metadata are copied into the same collector using the exact shared manifest.
