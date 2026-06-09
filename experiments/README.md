# Experiments

Use this directory for lightweight experiment planning and run metadata.

- `baseline/`: metadata for baseline reproduction runs.
- `variants/`: metadata for controlled modification runs.
- `manifests/`: run manifests, patch plans, evidence plans, and comparison matrices.
- `runbooks/`: operational checklists for TPU setup, debug gates, and full-run launch decisions.
- `templates/`: reusable templates for iteration logs and run records.

Current pre-TPU planning files:

- `manifests/i3_sweep_plan.md`: controlled GRPO vs RLOO experiment design.
- `manifests/baseline_patch_plan.md`: minimum baseline changes needed before scientific TPU runs.
- `manifests/plotting_and_evidence_plan.md`: source data, figures, tables, and confidence interval plan.
- `runbooks/tpu_day1_runbook.md`: first TPU session setup, debug, evaluation, and stop/go gates.
- `templates/iteration_log_template.md`: per-run record template.
