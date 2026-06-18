# Result: existing-runs harvest (Baron) -- 2026-06-13

Provenance: R1/R3/R5/D4 were evaluated with Baron seed-0 draw (empty EVAL_MANIFEST). Their base and checkpoint CSVs share one sample and are valid together for paired base-vs-R5 testing. Do not pool these seed-0-draw rows with the committed gsm8k_test_seed0_n64.jsonl manifest sample in one paired test.

## R1-grpo-full-s0-20260608_165231
- TensorBoard event: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231/tensorboard/events.out.tfevents.1780937570.t1v-n-0339f27d-w-0`
- Scalar export: `experiments/evidence/R1-grpo-full-s0-20260608_165231/scalars/R1-grpo-full-s0-20260608_165231_scalars.csv`
- Scalar tags present: `jax/core/compile/jaxpr_trace_duration`, `jax/core/compile/jaxpr_to_mlir_module_duration`, `jax/core/compile/backend_compile_duration`, `actor/eval/loss`, `actor/eval/perplexity`, `actor/eval/kl`, `actor/eval/pg_clipfrac`, `jax/orbax/write/sharded_array_gb`, `jax/orbax/write/blocking_gbytes_per_sec`, `jax/checkpoint/write/old_steps_examined_count`, `jax/orbax/write/gbytes_per_sec`, `jax/orbax/write/gbytes`, `rewards/train/sum`, `rewards/train/mean`, `rewards/train/min`, `rewards/train/max`, `rewards/train/match_format_exactly`, `rewards/train/match_format_approximately`, `rewards/train/check_answer`, `rewards/train/check_numbers`, `rewards/train/score/mean`, `rewards/train/score/max`, `rewards/train/score/min`, `completions/train/mean_length`, `completions/train/max_length`, `completions/train/min_length`, `rewards/eval/sum`, `rewards/eval/mean`, `rewards/eval/min`, `rewards/eval/max`, `rewards/eval/match_format_exactly`, `rewards/eval/match_format_approximately`, `rewards/eval/check_answer`, `rewards/eval/check_numbers`, `rewards/eval/score/mean`, `rewards/eval/score/max`, `rewards/eval/score/min`, `completions/eval/mean_length`, `completions/eval/max_length`, `completions/eval/min_length`, `jax/checkpoint/write/async/commit_future_count`, `actor/train/loss`, `actor/train/perplexity`, `actor/train/grad_norm`, `actor/train/kl`, `actor/train/pg_clipfrac`
- Runtime: ~4h15m (known from run notes)
- Files copied:
  - `experiments/evidence/R1-grpo-full-s0-20260608_165231/scalars/R1-grpo-full-s0-20260608_165231_scalars.csv`
  - `experiments/evidence/R1-grpo-full-s0-20260608_165231/eval/base_greedy_seed0.csv`
  - `experiments/evidence/R1-grpo-full-s0-20260608_165231/eval/r1_grpo_step2000_greedy.csv`
  - `experiments/evidence/R1-grpo-full-s0-20260608_165231/eval/r1_grpo_step2500_greedy.csv`
  - `experiments/evidence/R1-grpo-full-s0-20260608_165231/eval/r1_grpo_step3000_greedy.csv`
  - `experiments/evidence/R1-grpo-full-s0-20260608_165231/eval/r1_grpo_step3364_greedy.csv`
  - `experiments/evidence/R1-grpo-full-s0-20260608_165231/eval/r1_tensorboard_scalar_summary.txt`
  - `experiments/evidence/R1-grpo-full-s0-20260608_165231/run_metadata.json`

## R3-rloo-full-s0-20260608_230810
- TensorBoard event: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R3-rloo-full-s0-20260608_230810/tensorboard/events.out.tfevents.1780960133.t1v-n-0339f27d-w-0`
- Scalar export: `experiments/evidence/R3-rloo-full-s0-20260608_230810/scalars/R3-rloo-full-s0-20260608_230810_scalars.csv`
- Scalar tags present: `jax/core/compile/jaxpr_trace_duration`, `jax/core/compile/jaxpr_to_mlir_module_duration`, `jax/core/compile/backend_compile_duration`, `actor/eval/loss`, `actor/eval/perplexity`, `actor/eval/kl`, `actor/eval/pg_clipfrac`, `jax/orbax/write/sharded_array_gb`, `jax/orbax/write/blocking_gbytes_per_sec`, `jax/checkpoint/write/old_steps_examined_count`, `jax/orbax/write/gbytes_per_sec`, `jax/orbax/write/gbytes`, `rewards/train/sum`, `rewards/train/mean`, `rewards/train/min`, `rewards/train/max`, `rewards/train/match_format_exactly`, `rewards/train/match_format_approximately`, `rewards/train/check_answer`, `rewards/train/check_numbers`, `rewards/train/score/mean`, `rewards/train/score/max`, `rewards/train/score/min`, `completions/train/mean_length`, `completions/train/max_length`, `completions/train/min_length`, `rewards/eval/sum`, `rewards/eval/mean`, `rewards/eval/min`, `rewards/eval/max`, `rewards/eval/match_format_exactly`, `rewards/eval/match_format_approximately`, `rewards/eval/check_answer`, `rewards/eval/check_numbers`, `rewards/eval/score/mean`, `rewards/eval/score/max`, `rewards/eval/score/min`, `completions/eval/mean_length`, `completions/eval/max_length`, `completions/eval/min_length`, `jax/checkpoint/write/async/commit_future_count`, `actor/train/loss`, `actor/train/perplexity`, `actor/train/grad_norm`, `actor/train/kl`, `actor/train/pg_clipfrac`
- Runtime: 20904.61239733 seconds
- Files copied:
  - `experiments/evidence/R3-rloo-full-s0-20260608_230810/scalars/R3-rloo-full-s0-20260608_230810_scalars.csv`
  - `experiments/evidence/R3-rloo-full-s0-20260608_230810/eval/r3_eval_summary.txt`
  - `experiments/evidence/R3-rloo-full-s0-20260608_230810/eval/r3_rloo_step2000_greedy.csv`
  - `experiments/evidence/R3-rloo-full-s0-20260608_230810/eval/r3_rloo_step2500_greedy.csv`
  - `experiments/evidence/R3-rloo-full-s0-20260608_230810/eval/r3_rloo_step3000_greedy.csv`
  - `experiments/evidence/R3-rloo-full-s0-20260608_230810/eval/r3_rloo_step3364_greedy.csv`
  - `experiments/evidence/R3-rloo-full-s0-20260608_230810/eval/r3_tensorboard_scalar_summary.txt`
  - `experiments/evidence/R3-rloo-full-s0-20260608_230810/run_metadata.json`

## R5-grpo-k8-full-s0-20260609_114832
- TensorBoard event: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R5-grpo-k8-full-s0-20260609_114832/tensorboard/events.out.tfevents.1781005731.t1v-n-0339f27d-w-0`
- Scalar export: `experiments/evidence/R5-grpo-k8-full-s0-20260609_114832/scalars/R5-grpo-k8-full-s0-20260609_114832_scalars.csv`
- Scalar tags present: `jax/core/compile/jaxpr_trace_duration`, `jax/core/compile/jaxpr_to_mlir_module_duration`, `jax/core/compile/backend_compile_duration`, `actor/eval/loss`, `actor/eval/perplexity`, `actor/eval/kl`, `actor/eval/pg_clipfrac`, `jax/orbax/write/sharded_array_gb`, `jax/orbax/write/blocking_gbytes_per_sec`, `jax/checkpoint/write/old_steps_examined_count`, `jax/orbax/write/gbytes_per_sec`, `jax/orbax/write/gbytes`, `rewards/train/sum`, `rewards/train/mean`, `rewards/train/min`, `rewards/train/max`, `rewards/train/match_format_exactly`, `rewards/train/match_format_approximately`, `rewards/train/check_answer`, `rewards/train/check_numbers`, `rewards/train/score/mean`, `rewards/train/score/max`, `rewards/train/score/min`, `completions/train/mean_length`, `completions/train/max_length`, `completions/train/min_length`, `rewards/eval/sum`, `rewards/eval/mean`, `rewards/eval/min`, `rewards/eval/max`, `rewards/eval/match_format_exactly`, `rewards/eval/match_format_approximately`, `rewards/eval/check_answer`, `rewards/eval/check_numbers`, `rewards/eval/score/mean`, `rewards/eval/score/max`, `rewards/eval/score/min`, `completions/eval/mean_length`, `completions/eval/max_length`, `completions/eval/min_length`, `jax/checkpoint/write/async/commit_future_count`, `actor/train/loss`, `actor/train/perplexity`, `actor/train/grad_norm`, `actor/train/kl`, `actor/train/pg_clipfrac`
- Runtime: ~6h52m (known from run notes)
- Files copied:
  - `experiments/evidence/R5-grpo-k8-full-s0-20260609_114832/scalars/R5-grpo-k8-full-s0-20260609_114832_scalars.csv`
  - `experiments/evidence/R5-grpo-k8-full-s0-20260609_114832/eval/r5_all_retained_eval_summary.txt`
  - `experiments/evidence/R5-grpo-k8-full-s0-20260609_114832/eval/r5_eval_summary.txt`
  - `experiments/evidence/R5-grpo-k8-full-s0-20260609_114832/eval/r5_grpo_k8_step1000_greedy.csv`
  - `experiments/evidence/R5-grpo-k8-full-s0-20260609_114832/eval/r5_grpo_k8_step1250_greedy.csv`
  - `experiments/evidence/R5-grpo-k8-full-s0-20260609_114832/eval/r5_grpo_k8_step1500_greedy.csv`
  - `experiments/evidence/R5-grpo-k8-full-s0-20260609_114832/eval/r5_grpo_k8_step1750_greedy.csv`
  - `experiments/evidence/R5-grpo-k8-full-s0-20260609_114832/eval/r5_grpo_k8_step2000_greedy.csv`
  - `experiments/evidence/R5-grpo-k8-full-s0-20260609_114832/eval/r5_grpo_k8_step2250_greedy.csv`
  - `experiments/evidence/R5-grpo-k8-full-s0-20260609_114832/eval/r5_grpo_k8_step2500_greedy.csv`
  - `experiments/evidence/R5-grpo-k8-full-s0-20260609_114832/eval/r5_grpo_k8_step250_greedy.csv`
  - `experiments/evidence/R5-grpo-k8-full-s0-20260609_114832/eval/r5_grpo_k8_step2750_greedy.csv`
  - `experiments/evidence/R5-grpo-k8-full-s0-20260609_114832/eval/r5_grpo_k8_step3000_greedy.csv`
  - `experiments/evidence/R5-grpo-k8-full-s0-20260609_114832/eval/r5_grpo_k8_step3250_greedy.csv`
  - `experiments/evidence/R5-grpo-k8-full-s0-20260609_114832/eval/r5_grpo_k8_step3364_greedy.csv`
  - `experiments/evidence/R5-grpo-k8-full-s0-20260609_114832/eval/r5_grpo_k8_step500_greedy.csv`
  - `experiments/evidence/R5-grpo-k8-full-s0-20260609_114832/eval/r5_grpo_k8_step750_greedy.csv`
  - `experiments/evidence/R5-grpo-k8-full-s0-20260609_114832/run_metadata.json`

## D4-grpo-k8-medium-debug-20260609_100945
- TensorBoard event: `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D4-grpo-k8-medium-debug-20260609_100945/tensorboard/events.out.tfevents.1780999805.t1v-n-0339f27d-w-0`
- Scalar export: `experiments/evidence/D4-grpo-k8-medium-debug-20260609_100945/scalars/D4-grpo-k8-medium-debug-20260609_100945_scalars.csv`
- Scalar tags present: `jax/core/compile/jaxpr_trace_duration`, `jax/core/compile/jaxpr_to_mlir_module_duration`, `jax/core/compile/backend_compile_duration`, `actor/eval/loss`, `actor/eval/perplexity`, `actor/eval/kl`, `actor/eval/pg_clipfrac`, `jax/orbax/write/sharded_array_gb`, `jax/orbax/write/blocking_gbytes_per_sec`, `jax/checkpoint/write/old_steps_examined_count`, `jax/orbax/write/gbytes_per_sec`, `jax/orbax/write/gbytes`, `rewards/train/sum`, `rewards/train/mean`, `rewards/train/min`, `rewards/train/max`, `rewards/train/match_format_exactly`, `rewards/train/match_format_approximately`, `rewards/train/check_answer`, `rewards/train/check_numbers`, `rewards/train/score/mean`, `rewards/train/score/max`, `rewards/train/score/min`, `completions/train/mean_length`, `completions/train/max_length`, `completions/train/min_length`, `rewards/eval/sum`, `rewards/eval/mean`, `rewards/eval/min`, `rewards/eval/max`, `rewards/eval/match_format_exactly`, `rewards/eval/match_format_approximately`, `rewards/eval/check_answer`, `rewards/eval/check_numbers`, `rewards/eval/score/mean`, `rewards/eval/score/max`, `rewards/eval/score/min`, `completions/eval/mean_length`, `completions/eval/max_length`, `completions/eval/min_length`, `jax/checkpoint/write/async/commit_future_count`, `actor/train/loss`, `actor/train/perplexity`, `actor/train/grad_norm`, `actor/train/kl`, `actor/train/pg_clipfrac`
- Runtime: 3609.800270175 seconds
- Files copied:
  - `experiments/evidence/D4-grpo-k8-medium-debug-20260609_100945/scalars/D4-grpo-k8-medium-debug-20260609_100945_scalars.csv`
  - `experiments/evidence/D4-grpo-k8-medium-debug-20260609_100945/eval/d4_eval_summary.txt`
  - `experiments/evidence/D4-grpo-k8-medium-debug-20260609_100945/eval/d4_grpo_k8_step200_greedy.csv`
  - `experiments/evidence/D4-grpo-k8-medium-debug-20260609_100945/eval/d4_grpo_k8_step300_greedy.csv`
  - `experiments/evidence/D4-grpo-k8-medium-debug-20260609_100945/eval/d4_grpo_k8_step400_greedy.csv`
  - `experiments/evidence/D4-grpo-k8-medium-debug-20260609_100945/eval/d4_grpo_k8_step500_greedy.csv`
  - `experiments/evidence/D4-grpo-k8-medium-debug-20260609_100945/run_metadata.json`

