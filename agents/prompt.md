# New Session Prompt

You are working in the `tpu-2026` fork for the Multi-Agent Systems and Agentic AI coursework Part I practical. Read files in this order:

1. `agents/context.md`.
2. `agents/plan.md`.
3. `agents/report_notes.md`.
4. Main coursework repo: `../agentic-ai-coursework/agents/context.md`.
5. Main coursework repo: `../agentic-ai-coursework/experiments/runbooks/tpu_day1_runbook.md`.
6. Main coursework repo: `../agentic-ai-coursework/experiments/manifests/baseline_patch_plan.md`.
7. Patched files: `bootstrap.sh`, `scripts/config.py`, `scripts/data.py`, `scripts/train.py`, `scripts/evaluate.py`.

Workflow rules:

- Do not push.
- Do not commit unless explicitly asked.
- Do not revert user or teammate changes.
- Do not edit Tunix unless a specific verified failure requires it.
- Treat the main coursework PDF as authoritative for assignment requirements.
- Treat per-run logs, metadata, checkpoints, and evaluation CSVs as the evidence source for report claims.
- Keep scope small: debug first, verify checkpoint/eval/logging, then full GRPO/RLOO runs.
- Ask only if blocked; otherwise proceed pragmatically.

Current branch facts:

- Branch: `coursework`.
- Upstream baseline: `324abbe4b4e229ea812223856393547db4fbb53e`.
- Current committed head: `e3ebeef`.
- D1 GRPO debug passed. D2 RLOO debug is pending; inspect `git status --short --branch` first.
