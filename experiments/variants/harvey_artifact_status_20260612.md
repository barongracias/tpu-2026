# Harvey artifact status 2026-06-12

This note records what was available on `/home/harvey` during the 2026-06-12 collation pass. It does not treat absent files as failed runs; absent means the file was not present in this local filesystem.

## H1: R6 RLOO K sweep

Runs:
- `R6-rloo-k8-full-s0-20260609_212314`
- `R6-rloo-k2-full-s0-20260609_220042`

Provenance:
- Launch branch: `harvey`
- W&B-recorded launch commit: `8a0f7f266552cb2666710ac589cb6bda5cd40121`
- Caveat: these runs were not launched from `deterministic-platform`; they lack the deterministic branch controls (`MODEL_REVISION` requirement at launch time, `EVAL_SEED`/`EVAL_MANIFEST`, pinned dependency refs, and `run_metadata.json` writer).

Collector paths:
- K=8: `/home/harvey/tpu-runs/part-i/report_diagnostics/R6-rloo-k8-full-s0-20260609_212314`
- K=2: `/home/harvey/tpu-runs/part-i/report_diagnostics/R6-rloo-k2-full-s0-20260609_220042`

Copied evidence:
- `logs/train.log`
- TensorBoard event files under `tensorboard/`
- W&B `config.yaml`, `wandb-metadata.json`, and `wandb-summary.json`

Missing on this VM:
- `ckpts/run_metadata.json` for both runs. These predate the metadata writer.
- Per-prompt greedy eval CSVs for the base model and retained checkpoints.
- Per-checkpoint eval summary tables.

W&B:
- https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R6-rloo-k8-full-s0-20260609_212314
- https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R6-rloo-k2-full-s0-20260609_220042

Approximate wall clock from logs/W&B:
- K=8: started from local W&B run directory timestamp `2026-06-09T21:57:19Z`; chained K=2 log records K=8 exit at `2026-06-10T05:37:42Z`; W&B runtime was about 27,592 seconds.
- K=2: started `2026-06-10T05:37:42Z`; train log mtime after `Training finished.` was `2026-06-10T07:41:38Z`; W&B runtime was about 7,425 seconds.

## H2: R7 deterministic RLOO reruns

### K=2

Run: `R7-rloo-k2-det-harvey-full-s0-20260611_102009`

Provenance:
- Launch branch: `harvey-grpo-k8-rerun`
- Launch commit in metadata: `71aab87dee2d2c78256384d084d063d8b40c9e0c`
- Current note branch at collation: `harvey-grpo-k8-rerun`
- Current note HEAD at collation: `a10334df843283aa54827119ede3921230843f96`

Collector path:
- `/home/harvey/tpu-runs/part-i/report_diagnostics/R7-rloo-k2-det-harvey-full-s0-20260611_102009`

Repo evidence path:
- `experiments/evidence/R7-rloo-k2-det-harvey-full-s0-20260611_102009`

Copied evidence:
- `ckpts/run_metadata.json`
- `launch_train.sh`
- all retained-checkpoint greedy eval CSVs found locally
- `r7_rloo_k2_all_retained_eval_metrics.csv`
- `r7_rloo_k2_all_retained_eval_summary.txt`
- `r7_rloo_k2_tensorboard_scalars.csv`
- `r7_rloo_k2_tensorboard_scalar_summary.txt`
- train and eval logs in the local collector
- shared manifest copy: `gsm8k_test_seed0_n64.jsonl`

Eval status:
- Retained checkpoint evals are present for steps `250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3364`.
- CSV metadata has non-empty `restored_step` values matching the checkpoint steps.
- Best retained checkpoint: step `500`, `30/64` exact.
- Final checkpoint: step `3364`, `1/64` exact with `58/64` empty responses.
- Base `--no-restore` CSV was not present locally during this pass.

W&B:
- https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R7-rloo-k2-det-harvey-full-s0-20260611_102009

Approximate wall clock from logs/W&B:
- Started from W&B local run timestamp `2026-06-11T10:22:56Z`; train log mtime after `Training finished.` was `2026-06-11T16:09:17Z`; W&B runtime was about 20,773 seconds.
- Retained-checkpoint eval CSVs were present by `2026-06-12T08:53:27Z` based on final eval CSV mtime.

### K=8

Run: `R7-rloo-k8-det-harvey-full-s0-20260611_105132`

Provenance:
- Launch branch: `harvey-grpo-k8-rerun`
- Launch commit reported in handoff: `e3d69a1938fe8e8a2a67a3c84a03331133ed46f1`
- Run root reported on shared Boris VM: `/home/ext_harveybermingham1_gmail_com/tpu-runs/part-i/R7-rloo-k8-det-harvey-full-s0-20260611_105132`

Local collector path:
- `/home/harvey/tpu-runs/part-i/report_diagnostics/R7-rloo-k8-det-harvey-full-s0-20260611_105132`

Local status:
- Only the markdown retained-checkpoint summary was present in this repo during this pass.
- The underlying K=8 per-prompt CSVs, `run_metadata.json`, TensorBoard scalar export/event file, and train/eval logs were not present under `/home/harvey/tpu-runs/part-i`.

W&B:
- https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R7-rloo-k8-det-harvey-full-s0-20260611_105132

## H3: hard-mined RLOO

Not run on this VM. No local run directory, W&B-named artefact, hard-example RLOO manifest, or note matching RLOO plus hard/hardmedium was found during this pass.

## H4: branch hygiene

No branch cleanup or file deletion was performed. The R6 launch branch remains documented as `harvey`; deterministic rerun notes remain on `harvey-grpo-k8-rerun`.

## Fresh eval blocker

A quick JAX device probe from this shell failed on TPU metadata lookup (`Could not connect to server` for TPU metadata variables). I did not launch new base or checkpoint evals from this environment to avoid producing CPU/fallback or partial results.
