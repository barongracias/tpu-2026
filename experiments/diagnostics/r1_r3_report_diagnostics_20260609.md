# R1/R3 Report Diagnostics 2026-06-09

## Artefact Location

Generated on the TPU VM under:

`/home/ext_barongracias_gmail_com/tpu-runs/part-i/report_diagnostics/`

Key files:

- `r1_r3_report_diagnostic_summary.md`
- `r1_r3_eval_metrics.csv`
- `r1_r3_tensorboard_scalar_samples.csv`
- `base_correct_r1_r3_wrong_examples.csv`
- `../R3-rloo-full-s0-20260608_230810/eval/r3_tensorboard_scalar_summary.txt`

## Compact Metrics

| Eval | Exact | Partial | Format | Empty |
| --- | ---: | ---: | ---: | ---: |
| Base greedy | 31/64, 48.44% | 31/64, 48.44% | 1/64, 1.56% | 0/64 |
| R1 step 2000 | 24/64, 37.50% | 26/64, 40.62% | 28/64, 43.75% | 0/64 |
| R1 step 3364 | 12/64, 18.75% | 12/64, 18.75% | 23/64, 35.94% | 0/64 |
| R3 step 2500 | 6/64, 9.38% | 7/64, 10.94% | 16/64, 25.00% | 45/64 |
| R3 step 3364 | 1/64, 1.56% | 1/64, 1.56% | 5/64, 7.81% | 58/64 |

## Diagnosis

- R1: reward-format over-optimisation plus arithmetic degradation and checkpoint overtraining. Step 2000 is best; final is much worse.
- R3: generation collapse / unstable decoding. Empty-response count is the strongest signal.
- R1 KL drift is a warning sign: eval KL rises to about 0.859 near step 3008, and train KL reaches about 0.977.
- `pg_clipfrac` stays 0.0 for both R1 and R3, so clipping does not catch the instability.
- Evaluation/cache artefact is unlikely as the dominant explanation: successful evals used fresh eval-local TFDS caches, restore markers are present, and base remains strong.

## Qualitative Pattern

Base is correct while both final trained models are wrong on 23 prompts. R1 usually emits non-empty tagged reasoning but makes arithmetic/interpretation errors. R3 often emits empty responses.

## Next Decision

A cheap D3 K=8 debug is justified as a diagnostic, not as a full run. It should expose/configure `NUM_GENERATIONS=8`, use fresh eval caches, include empty-response checks, and be reviewed before any K=8 full run.
