# Evidence summary: B-grpo-k8-rewardrebalance-s0-20260614

- Status: Report-ready.
- Branch / commit: `baron-reward-rebalance-k8` / `74b648a45ddbab569ace356d6b1d767724b78902`.
- Experiment: reward-rebalance GRPO K=8; seed 0.
- Reward weights: format=0.3, answer=2.0, number=1.0.
- n=64 manifest SHA-256: `9aa1814e295e155f4735c9302a7f9c28c6aaa2319074e053a5888e3f3b2448b0`.
- Full-test manifest SHA-256: `07f0f0fc10580dee941b6f921a3986854a8b0b74529d9bb952662d5daaea6bb2`.
- W&B: https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/B-grpo-k8-rewardrebalance-s0-20260614; runtime 32929.790906207 seconds (~9h09m).

## n=64 screening

- Base: 31/64 exact, partial=31, format=1, empty=0.
- Best retained: 34/64 at step 500; final: 32/64 at step 3364.

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

## Full-test (n=1319)

- Reused shared base_full.csv: 625/1319, 47.38% [44.73%, 50.11%].
- Best retained step 500: 704/1319, 53.37% [50.72%, 56.10%]; paired delta vs base +5.99pp [+3.26, +8.49].
- Final step 3364: 704/1319, 53.37% [50.64%, 56.10%]; paired delta vs base +5.99pp [+3.03, +8.79].
- Provenance caveat: base_full.csv records Harvey's manifest path string, but the full manifest SHA matches the shared committed manifest used here.
