# Baron Results Handoff - 2026-06-15

Purpose: compact handoff of Baron's final Part I evidence for Fred, Harvey, and any reviewing agent. This file is a provenance map, not a replacement for the committed per-run evidence files.

## Current headline

Baron's strongest result is still the original GRPO K=8 full run (R5), not the later reward-rebalance run. On the committed full GSM8K test manifest (`07f0f0fc10580dee941b6f921a3986854a8b0b74529d9bb952662d5daaea6bb2`, n=1319), both R5 and reward-rebalance beat the base model with paired bootstrap intervals excluding zero, but R5 has the larger effect.

| Run | Exact / 1319 | Accuracy | Paired delta vs base |
|---|---:|---:|---:|
| Base, no restore | 625 | 47.38% [44.73, 50.11] | -- |
| R5 GRPO K=8, step 3250 / 3364 | 740 | 56.10% [53.37, 58.83] | +8.72 pp [+5.99, +11.37] |
| B reward-rebalance K=8, step 500 / 3364 | 704 | 53.37% [50.72, 56.10] | +5.99 pp [+3.26, +8.49] |

Bootstrap details: paired by prompt id, 10,000 resamples, seed 12345. Best-retained and final checkpoints scored identically on the full set for both R5 and reward-rebalance.

## Baron runs and evidence

| Run | Evidence status | Notes |
|---|---|---|
| R1 GRPO K=2 | harvested n=64 seed-0-draw scalar/eval evidence | Negative baseline-training result: best 24/64, final 12/64 against same-sample base 31/64. W&B runtime ~4h15m. |
| R3 RLOO K=2 | harvested n=64 seed-0-draw scalar/eval evidence | Negative estimator result: best 6/64, final 1/64, severe empty-response collapse. |
| D4 GRPO K=8 medium | harvested n=64 seed-0-draw scalar/eval evidence | Diagnostic only: step 500 reached 34/64 with no empty collapse. |
| R5 GRPO K=8 | harvested n=64 seed-0-draw + full-test evidence | Main positive result: full-test 740/1319, +8.72 pp over base. W&B runtime ~6h52m. |
| B reward-rebalance K=8 | committed n=64 + full-test evidence | Mechanism test: format rewards x0.3, `check_answer` x2.0, `check_numbers` x1.0. Full-test 704/1319, +5.99 pp over base, below R5. Runtime ~9h09m. |

Primary evidence files in `tpu-2026`:
- `experiments/team_evidence/result_existing-runs-harvest_baron_20260613.md`
- `experiments/team_evidence/result_B-grpo-k8-rewardrebalance-s0-20260614_baron_20260613.md`
- `experiments/team_evidence/result_full-test-baron_20260613.md`
- `experiments/evidence/R5-grpo-k8-full-s0-20260609_114832/summary.md`
- `experiments/evidence/B-grpo-k8-rewardrebalance-s0-20260614/summary.md`
- `experiments/evidence/*/scalars/*_scalars.csv`

Derived submission-report files in `agentic-ai-coursework`:
- `results/part1_full_test_results_20260615.csv`
- `results/part1_report_ready_results_20260613.csv`
- `results/part1_scalar_curves_20260615.csv`
- `results/part1_screening_diagnostics_20260615.csv`
- `report/figures/i3_training_curves_diagnostics.png`

## Provenance rule that must not be broken

There are two different n=64 samples in circulation.

1. Harvested R1/R3/R5/D4 n=64 evals use Baron's original seed-0 draw with empty `EVAL_MANIFEST`. They can be compared with each other, but must not be pooled with committed-manifest evals.
2. Deterministic-platform n=64 evals use committed manifest SHA `9aa1814e295e155f4735c9302a7f9c28c6aaa2319074e053a5888e3f3b2448b0`.

The full-test results use committed manifest SHA `07f0f0fc10580dee941b6f921a3986854a8b0b74529d9bb952662d5daaea6bb2`; base, R5, and B are paired on this manifest.

## What Fred should send back

If the GRPO K=8 seed-1 run completed, send one result markdown plus lightweight evidence with:
- branch and commit;
- W&B URL and runtime;
- `run_metadata.json`;
- scalar CSV with reward/KL/completion tags;
- base and all retained-checkpoint per-prompt eval CSVs;
- full-test eval CSVs on manifest SHA `07f0f0fc...` for best and final checkpoints, or confirmation that full-test eval was not run;
- paired bootstrap vs the shared base if already computed, otherwise the CSVs are enough for Baron to compute it.

If seed-1 did not complete, say so explicitly and include the last useful checkpoint/status. The existing hard/medium result is already treated as negative/inconclusive unless a full-test eval changes it.

## What Harvey should send back

If the K-sweep ran, send one result markdown per run with:
- K value (`K=4`, `K=16`, or other), branch, commit, and any cap/OOM status;
- W&B URL and runtime;
- `run_metadata.json`;
- scalar CSV with reward/KL/completion tags;
- n=64 retained-checkpoint eval summary and CSVs;
- full-test eval CSVs on manifest SHA `07f0f0fc...` for best and final checkpoints, if run;
- explicit note whether K=16 finished, OOMed, or was capped.

For existing RLOO K=2/K=8 evidence, a full-test eval would be useful but is not mandatory for Baron's report unless it is already available. Do not mix these with Baron's seed-0-draw n=64 results.

## Current report status

The main report now uses full-test n=1319 as the primary I.3 evidence and includes a combined scalar/diagnostic figure. Theory and Part II are considered final unless a reviewer finds a concrete issue. Remaining packaging item: replace GitHub links with the final GitLab submission link once available.
