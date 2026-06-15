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
- `manifests/experiment_contract.md`: required pins, persistent paths, seeds, and eval manifest contract for comparable runs.
- `manifests/plotting_and_evidence_plan.md`: source data, figures, tables, and confidence interval plan.
- `runbooks/tpu_day1_runbook.md`: first TPU session setup, debug, evaluation, and stop/go gates.
- `templates/iteration_log_template.md`: per-run record template.

Recent run records:

- `variants/r6_rloo_k_sweep_full_s0_20260609.md`: side-by-side record for the completed RLOO K=8 and chained RLOO K=2 full training runs; evaluation pending.
- `variants/h_grpo_k4_full_s0_20260614.md`: completed Harvey GRPO K=4 full run and full-test confirmation eval.
- `variants/h_grpo_k16_cap2500_s0_20260614_r2.md`: capped Harvey GRPO K=16 run, early-stopped with TPU memory exhaustion and evaluated to last-good checkpoint.
