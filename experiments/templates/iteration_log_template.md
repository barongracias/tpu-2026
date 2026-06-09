# Iteration Log Template

Copy this file for each run or attempted run. Fill it before launch, then update it during and after the run.

## Run Identity

- Run id:
- Date/time started:
- Operator:
- Machine:
- TPU type/name/zone:
- Purpose:
- Hypothesis:
- Decision this run should enable:

## Source State

- Coursework repo commit:
- `tpu-2026` commit:
- `tunix` commit:
- `cmbagent_lg` commit, if relevant:
- Local patches applied:
- Dirty files intentionally present:

## Configuration

- Data source:
- Estimator:
- Seed:
- Max steps:
- Num generations:
- Num iterations:
- Beta:
- Epsilon:
- Batch size / rollout settings:
- Evaluation preset:
- Checkpoint directory:
- Intermediate checkpoint directory:
- TensorBoard directory:
- W&B entity/project/run id:

## Exact Commands

Training command:

```bash

```

Evaluation command:

```bash

```

Plot/extraction command:

```bash

```

## Live Checks

- TPU backend confirmed:
- W&B run visible:
- TensorBoard events written:
- Checkpoints written:
- Estimator logged correctly:
- Reward metric present:
- KL metric present:
- Diagnostic metrics present:
- Any warnings or crashes:

## Results

- Training completed:
- Final step:
- Wall-clock time:
- Final/representative reward:
- Final/representative KL:
- Evaluation sample count:
- Evaluation accuracy or score:
- Confidence interval:
- Saved per-prompt output:
- Saved plots:
- Saved logs:

## Interpretation

- What happened:
- Whether the result supports the hypothesis:
- Algorithm-effectiveness interpretation:
- Edge cases or instability observed:
- Alternative explanations:
- What changed relative to previous iteration:

## Decision

- Keep / rerun / discard:
- Reason:
- Next run:
- Report relevance:
- Evidence path(s) to cite:
- Caveats for report wording:

