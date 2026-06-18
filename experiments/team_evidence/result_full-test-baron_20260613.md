# Full-test (n=1319): Baron evidence -- 2026-06-13

- Full manifest SHA-256: `07f0f0fc10580dee941b6f921a3986854a8b0b74529d9bb952662d5daaea6bb2`
- Shared base_full.csv: 625/1319, 47.38% [44.73%, 50.11%]
- Bootstrap: 10000 resamples, seed 12345, paired by prompt_id.
- Path caveat: Harvey base CSV stores a different absolute eval_manifest path, but the manifest SHA matches the committed shared file.

| run | status | selected steps | best exact (95% CI) | final exact (95% CI) | paired delta best-base | paired delta final-base |
|---|---|---|---:|---:|---:|---:|
| R5 GRPO K=8 | Report-ready with provenance caveat | 3250 / 3364 | 740/1319, 56.10% [53.37%, 58.83%] | 740/1319, 56.10% [53.45%, 58.76%] | +8.72pp [+5.99, +11.37] | +8.72pp [+5.91, +11.45] |
| B reward-rebalance GRPO K=8 | Report-ready | 500 / 3364 | 704/1319, 53.37% [50.72%, 56.10%] | 704/1319, 53.37% [50.64%, 56.10%] | +5.99pp [+3.26, +8.49] | +5.99pp [+3.03, +8.79] |

Skipped: non-report-ready deterministic K=2 attempts without local eval remain unusable for this full-test table.
