# Multi-Agent Systems and Agentic AI — Part I (GRPO finetuning)

Code and logs for the Part I practical: GRPO / RLOO finetuning of `google/gemma-3-1b-it`
on GSM8K with LoRA adapters, built on Tunix / JAX and run on a TPU v6e-1. This is the
team-owned training and evaluation codebase; the individual written report is submitted as
a separate PDF that links back to this repository.

## Layout

- `scripts/` — the pipeline: `train.py` (GRPO/RLOO via the Tunix `GRPOLearner`),
  `evaluate.py` (held-out GSM8K eval, writes per-prompt CSVs), `rewards.py`, `data.py`,
  `config.py` (single source of hyperparameters and env knobs), `model.py`, `chat.py`,
  the run launchers, and `mine_hard_examples.py` / `build_full_test_manifest.py`.
- `experiments/` — run records and saved logs:
  - `evidence/<run-id>/` — per-prompt eval CSVs, exported scalar CSVs, `run_metadata.json`, summaries.
  - `team_evidence/` — per-run result handoffs.
  - `manifests/` — eval manifests (`gsm8k_test_seed0_n64.jsonl`, `gsm8k_test_seed0_full.jsonl`) with SHA-256.
  - `runbooks/`, `variants/`, `diagnostics/`, `baseline/`, `templates/`.
- `bootstrap.sh` — pins the Tunix / JAX / Flax / Qwix commits and builds the TPU environment.

## Reproducing a run

See `experiments/runbooks/tpu_day1_runbook.md`. In brief: run `bootstrap.sh`, then
`scripts/train.py` with the controls in `config.py` — key env knobs are `ADV_ESTIMATOR`
(`grpo`/`rloo`), `NUM_GENERATIONS` (group size *K*), `RUN_SEED`, `EVAL_SEED` / `EVAL_MANIFEST`,
and `REWARD_{FORMAT,ANSWER,NUMBER}_WEIGHT` — then evaluate with
`scripts/evaluate.py --ckpt-dir <run>/ckpts --step <N> --output-csv <path>`. Training scalars
are logged to Weights & Biases (entity `barongracias-university-of-cambridge`, project
`agentic-ai-coursework`).

## Team ownership

The runs and code in this repository are jointly owned by the Part I team
(Frederick Lawrence, Baron Gracias, Harvey Bermingham). The written report is individual.

## AI declaration

Generative AI tools were used to help with LaTeX formatting and plotting, and with code
debugging. All experimental results, code, and analysis were produced and verified by the authors.
