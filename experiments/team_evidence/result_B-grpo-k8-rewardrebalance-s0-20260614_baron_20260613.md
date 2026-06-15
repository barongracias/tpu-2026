# Result: B-grpo-k8-rewardrebalance-s0-20260614 (Baron) -- 2026-06-13
- Experiment: reward-rebalance GRPO K=8
- Branch / commit: baron-reward-rebalance-k8 / 74b648a45ddbab569ace356d6b1d767724b78902
- Estimator / K / seed: grpo / 8 / 0; EVAL_SEED=0
- Reward weights: format=0.3, answer=2.0, number=1.0 (default 1/1/1)
- Eval manifest SHA-256: 9aa1814e295e155f4735c9302a7f9c28c6aaa2319074e053a5888e3f3b2448b0
- W&B run URL + _runtime: https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/B-grpo-k8-rewardrebalance-s0-20260614; 32929.790906207 seconds (~9h09m)
- MAX_STEPS / actual final step: 3364 / 3364
- Base (--no-restore, same manifest): 31/64 exact, partial=31, format=1, empty=0
- Retained-checkpoint table:

| step | exact | partial | format | empty | restored ok |
|---:|---:|---:|---:|---:|---|
| 1 | 31/64 | 31 | 1 | 0 | yes |
| 250 | 30/64 | 32 | 53 | 0 | yes |
| 500 | 34/64 | 34 | 49 | 0 | yes |
| 750 | 30/64 | 31 | 45 | 0 | yes |
| 1000 | 28/64 | 30 | 44 | 0 | yes |
| 1250 | 33/64 | 35 | 51 | 0 | yes |
| 1500 | 27/64 | 27 | 39 | 0 | yes |
| 1750 | 34/64 | 34 | 47 | 0 | yes |
| 2000 | 32/64 | 32 | 40 | 0 | yes |
| 2250 | 28/64 | 30 | 46 | 0 | yes |
| 2500 | 27/64 | 28 | 42 | 0 | yes |
| 2750 | 33/64 | 34 | 45 | 0 | yes |
| 3000 | 32/64 | 32 | 44 | 0 | yes |
| 3250 | 25/64 | 26 | 41 | 0 | yes |
| 3364 | 32/64 | 32 | 45 | 0 | yes |

- Best retained: 34/64 @step 500; Final: 32/64 @step 3364
- Scalars exported: experiments/evidence/B-grpo-k8-rewardrebalance-s0-20260614/scalars/B-grpo-k8-rewardrebalance-s0-20260614_scalars.csv; tags present: jax/core/compile/jaxpr_trace_duration, jax/core/compile/jaxpr_to_mlir_module_duration, jax/core/compile/backend_compile_duration, actor/eval/loss, actor/eval/perplexity, actor/eval/kl, actor/eval/pg_clipfrac, jax/orbax/write/sharded_array_gb, jax/orbax/write/blocking_gbytes_per_sec, jax/checkpoint/write/old_steps_examined_count, jax/orbax/write/gbytes_per_sec, jax/orbax/write/gbytes, rewards/train/sum, rewards/train/mean, rewards/train/min, rewards/train/max, rewards/train/match_format_exactly, rewards/train/match_format_approximately, rewards/train/check_answer, rewards/train/check_numbers, rewards/train/score/mean, rewards/train/score/max, rewards/train/score/min, completions/train/mean_length, completions/train/max_length, completions/train/min_length, rewards/eval/sum, rewards/eval/mean, rewards/eval/min, rewards/eval/max, rewards/eval/match_format_exactly, rewards/eval/match_format_approximately, rewards/eval/check_answer, rewards/eval/check_numbers, rewards/eval/score/mean, rewards/eval/score/max, rewards/eval/score/min, completions/eval/mean_length, completions/eval/max_length, completions/eval/min_length, jax/checkpoint/write/async/commit_future_count, actor/train/loss, actor/train/perplexity, actor/train/grad_norm, actor/train/kl, actor/train/pg_clipfrac
- One-line outcome: reward rebalance best 34/64 vs base 31/64; full-test best/final 704/1319 and 704/1319 vs base 625/1319.

## Full-test (n=1319)

- Manifest SHA-256: `07f0f0fc10580dee941b6f921a3986854a8b0b74529d9bb952662d5daaea6bb2`
- Base: 625/1319, 47.38% [44.73%, 50.11%]
- Best retained step 500: 704/1319, 53.37% [50.72%, 56.10%]; paired delta vs base +5.99pp [+3.26, +8.49]
- Final step 3364: 704/1319, 53.37% [50.64%, 56.10%]; paired delta vs base +5.99pp [+3.03, +8.79]
