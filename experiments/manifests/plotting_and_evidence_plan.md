# Plotting And Evidence Plan

Purpose: ensure Part I practical claims are backed by reproducible evidence and report-quality figures.

## Evidence Sources

Preferred raw evidence:

- TensorBoard event files from each run,
- W&B metric export where available,
- training stdout/stderr logs,
- checkpoint metadata,
- per-prompt evaluation CSV or JSONL files,
- iteration logs copied from `experiments/templates/iteration_log_template.md`.

Do not rely on screenshots as primary evidence. Screenshots may be useful while debugging, but report figures should be recreated from saved data.

## Minimum Figures For Part I Practical

| Figure | Purpose | Inputs | Required takeaway |
| --- | --- | --- | --- |
| Reward vs training step | Show learning progress and compare estimators. | TensorBoard or W&B scalar export. | Whether RLOO changes optimisation effectiveness relative to GRPO. |
| KL vs training step | Show policy movement under the KL penalty. | TensorBoard or W&B scalar export. | Whether improvements come with controlled or unstable policy drift. |
| Evaluation accuracy/score with confidence intervals | Report final task performance scientifically. | Per-prompt evaluation output. | Whether observed differences are meaningful relative to uncertainty. |
| Diagnostic metric over training | Explain mechanism or failure mode. | Tunix diagnostics such as `advantage/nonzero_frac`, `pg_clipfrac`, or importance-ratio statistics. | Why an estimator behaved as observed, not just which number was larger. |

Optional if time permits:

- wall-clock cost per useful step,
- response length or format-error rate,
- train/eval gap if both are available.

## Table For Report

Create one compact table with:

- run id,
- estimator,
- seed,
- steps,
- checkpoint step evaluated,
- evaluation sample count,
- accuracy or score,
- confidence interval,
- key caveat.

The table should be generated from saved evaluation outputs where possible, not typed manually from memory.

## Confidence Intervals

For binary correctness:

- use per-prompt correctness from the evaluation output,
- bootstrap prompts with replacement,
- report the mean and a 95 percent interval,
- use the same prompt set for GRPO and RLOO where possible.

For scalar scores:

- bootstrap per-prompt scalar scores in the same way,
- report uncertainty alongside the mean.

If the evaluation script cannot yet save per-prompt results, that is a blocker for confidence intervals and should be addressed before full evaluation.

## Figure Quality Rules

- Use vector PDF or PGF where practical; otherwise export high-resolution PNG from saved data.
- Label axes with units or clear metric names.
- Use identical axes for direct GRPO vs RLOO comparisons.
- Show uncertainty when reporting final evaluation performance.
- Captions should state the conclusion, not only describe the plot contents.
- Keep styling restrained and readable in the final A4 report.

## Extraction Plan

1. Export scalar metrics from TensorBoard event files or W&B for each run.
2. Normalize run metadata using the iteration logs.
3. Join metric traces with run ids, estimator labels, seeds, and checkpoint steps.
4. Generate comparison plots from scripts or notebooks stored under `experiments/`.
5. Save final report assets under `report/figures/` when the report is ready to include them.
6. Record exact plot commands and input paths in the relevant iteration log.

## Claims That Need Evidence

- RLOO improves or worsens final performance relative to GRPO.
- RLOO stabilizes, destabilizes, or otherwise changes training dynamics.
- Any observed improvement is not explained only by run length, seed, checkpoint choice, or evaluation sample noise.
- Any failure mode is visible in logged diagnostics rather than inferred from a single aggregate score.

