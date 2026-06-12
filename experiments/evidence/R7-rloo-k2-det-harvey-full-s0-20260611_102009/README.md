# R7 RLOO K=2 deterministic evidence

Run id: `R7-rloo-k2-det-harvey-full-s0-20260611_102009`

This directory contains the lightweight CSV/JSON evidence copied from:

`/home/harvey/tpu-runs/part-i/R7-rloo-k2-det-harvey-full-s0-20260611_102009`

Contents:
- `eval/`: retained-checkpoint greedy per-prompt CSVs, metrics summary CSV, TensorBoard scalar CSV, and summary text files.
- `metadata/`: `run_metadata.json`, `launch_train.sh`, W&B metadata, and W&B summary.
- `manifests/gsm8k_test_seed0_n64.jsonl`: shared held-out eval manifest.

Known gap:
- Base `--no-restore` eval CSV was not present on this VM during the 2026-06-12 collation pass.
