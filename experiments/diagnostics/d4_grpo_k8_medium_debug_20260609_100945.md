# D4 GRPO K=8 Medium Debug 2026-06-09 10:09

## Run Identity

- Run id: `D4-grpo-k8-medium-debug-20260609_100945`
- Run root: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D4-grpo-k8-medium-debug-20260609_100945`
- Machine: Google TPU VM
- Purpose: medium-length diagnostic for GRPO with larger group size `K=8`, after R1/R3 showed K=2 instability, reward-format over-optimisation, and R3 empty-response collapse.
- Hypothesis: increasing `NUM_GENERATIONS` from 2 to 8 may reduce small-group advantage noise and produce healthier training/evaluation behaviour before a full run is attempted.

## Source/Patch State

TPU-side uncommitted patch state:

- `scripts/config.py`: `NUM_GENERATIONS` is environment-overridable.
- `scripts/train.py`: `run_metadata.json` records `num_generations`, `beta`, `epsilon`, `temperature`, `top_k`, `top_p`, and `total_generation_steps`; startup log prints these fields.
- No commit or push was made.

Audit artefacts:

- `/home/ext_barongracias_gmail_com/tpu-runs/part-i/report_diagnostics/code_audit_d4_hygiene_note.md`
- `/home/ext_barongracias_gmail_com/tpu-runs/part-i/report_diagnostics/reward_sanity_check.txt`

## Configuration

- Estimator: `grpo`
- `NUM_GENERATIONS=8`
- Seed: 0
- Medium debug horizon: 500 steps
- Retained checkpoints: `200`, `300`, `400`, `500`
- Step 100 was pruned because `MAX_TO_KEEP=4`

## Results

| Step | Exact | Partial | Format | Empty |
| ---: | ---: | ---: | ---: | ---: |
| 100 | missing/pruned | missing/pruned | missing/pruned | missing/pruned |
| 200 | 26/64, 40.62% | 28/64, 43.75% | 55/64, 85.94% | 0/64 |
| 300 | 32/64, 50.00% | 34/64, 53.12% | 51/64, 79.69% | 0/64 |
| 400 | 29/64, 45.31% | 29/64, 45.31% | 55/64, 85.94% | 0/64 |
| 500 | 34/64, 53.12% | 35/64, 54.69% | 54/64, 84.38% | 0/64 |

W&B:
- https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/D4-grpo-k8-medium-debug-20260609_100945

Validation:

- `git diff --check`: passed.
- `py_compile`: passed for `scripts/config.py`, `scripts/train.py`, and `scripts/evaluate.py`.
- No tmux sessions remain running.

## Interpretation

- D4 is the first run to exceed the base greedy reference on the 64-prompt eval: step 500 reaches 34/64 exact versus base 31/64.
- K=8 also avoids the R3 empty-response collapse: empty count remains 0/64 at all evaluated retained checkpoints.
- Format accuracy is high while numeric accuracy is also above base at step 500, suggesting K=8 may stabilise the reward signal enough to make the existing reward mix usable over a medium horizon.
- This does not prove full-horizon stability. R1 degraded after step 2000, so a full K=8 run still needs frequent checkpointing and early-selection controls.

## Decision

- K=8 GRPO is worth considering for the next full or longer diagnostic run after review.
- Before full K=8, keep/improve reproducibility hygiene: commit or otherwise preserve the `NUM_GENERATIONS` env patch and metadata logging; increase checkpoint retention or choose save intervals so planned diagnostic checkpoints are not pruned.
- If a full K=8 run is started, evaluate multiple checkpoints rather than relying only on the final checkpoint.
