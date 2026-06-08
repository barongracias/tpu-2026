"""Standalone evaluation of a (LoRA) policy on the GSM8K test set.

Reports three numbers:
  * accuracy           — exact numeric match
  * partial_accuracy   — answer within 10% of ground truth
  * format_accuracy    — fraction of completions whose template parses

Run as:
    python evaluate.py --ckpt-dir $CKPT_DIR --output-csv eval_out.csv
    python evaluate.py --ckpt-dir $CKPT_DIR/actor --step 500
    python evaluate.py --no-restore          # base model sanity check
"""
import argparse
import csv as csv_mod
import json
import os

from tqdm.auto import tqdm
from tunix.generate import sampler as sampler_lib
from tunix.sft.checkpoint_manager import CheckpointManager

from config import (
    CKPT_DIR,
    DATA_SOURCE,
    GENERATION_CONFIGS,
    MAX_PROMPT_LENGTH,
    NUM_BATCHES,
    NUM_EPOCHS,
    NUM_TEST_BATCHES,
    RUN_SEED,
    TEST_DATA_DIR,
    TOTAL_GENERATION_STEPS,
    TRAIN_DATA_DIR,
    TRAIN_FRACTION,
    TRAIN_MICRO_BATCH_SIZE,
)
from data import SYSTEM_PROMPT, TEMPLATE, build_train_val_test
from model import build_mesh, download_weights, load_base_model, get_lora_model, load_tokenizer, model_config_for
from rewards import match_format, match_numbers


def resolve_lora_ckpt_root(ckpt_root: str) -> str:
    actor_root = os.path.join(ckpt_root, "actor")
    if os.path.isdir(actor_root):
        print(f"Using actor checkpoint subdirectory: {actor_root}")
        return actor_root
    return ckpt_root


def restore_lora(lora_model, ckpt_root: str, step: int | None) -> tuple[int, str]:
    resolved_root = resolve_lora_ckpt_root(ckpt_root)
    mgr = CheckpointManager(root_directory=resolved_root)
    n, _ = mgr.maybe_restore(model=lora_model, step=step, restore_only_lora_params=True)
    if n == 0:
        raise RuntimeError(
            f"No checkpoint found under {resolved_root}. "
            f"Pass --ckpt-dir or check `ls {resolved_root}`."
        )
    print(f"Restored LoRA params from {resolved_root} at step {n}")
    return n, resolved_root


def generate(question, sampler, eos_tokens, temperature=0.7, top_k=50, top_p=0.95, seed=None):
    if isinstance(question, str):
        batch = [TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=question)]
    else:
        batch = [TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=q) for q in question]

    out = sampler(
        input_strings=batch,
        max_generation_steps=TOTAL_GENERATION_STEPS,
        temperature=temperature, top_k=top_k, top_p=top_p,
        echo=False, seed=seed, eos_tokens=eos_tokens,
    )
    return out.text[0] if isinstance(question, str) else out.text


def evaluate(dataset, sampler, eos_tokens, temperature=0.7, top_k=50, top_p=0.95, num_passes=1):
    corr = partially_corr = corr_format = total = 0
    rows = []  # per-prompt records for P5 CSV export

    for batch in tqdm(dataset):
        answers = batch["answer"]
        questions = batch["question"]
        per_q = [[] for _ in range(len(questions))]
        for p in range(num_passes):
            responses = generate(questions, sampler, eos_tokens, temperature, top_k, top_p, seed=p)
            for i, r in enumerate(responses):
                per_q[i].append(r)

        for q, responses, ans in zip(questions, per_q, answers):
            got_corr = got_partial = got_format = False
            best_response = responses[0] if responses else ""
            for r in responses:
                ext = guess.group(1) if (guess := match_numbers.search(r)) is not None else "-1e9"
                try:
                    if float(ext.strip()) == float(ans.strip()):
                        got_corr = True
                        best_response = r
                    ratio = float(ext.strip()) / float(ans.strip())
                    if 0.9 <= ratio <= 1.1:
                        got_partial = True
                except Exception:
                    pass
                if match_format.search(r) is not None:
                    got_format = True
                if got_corr and got_partial and got_format:
                    best_response = r
                    break

            corr += int(got_corr)
            partially_corr += int(got_partial)
            corr_format += int(got_format)
            total += 1
            rows.append({
                "prompt_id": total,
                "question": q if isinstance(q, str) else q.decode("utf-8"),
                "expected_answer": ans if isinstance(ans, str) else (ans.decode("utf-8") if ans else ""),
                "model_response": best_response,
                "correct": int(got_corr),
                "partial_correct": int(got_partial),
                "format_correct": int(got_format),
            })
            if total % 10 == 0:
                print(f"===> corr={corr} total={total} acc={corr/total*100:.2f}% "
                      f"partial={partially_corr/total*100:.2f}% fmt={corr_format/total*100:.2f}%")

    return corr, total, corr/total*100, partially_corr/total*100, corr_format/total*100, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="greedy", choices=list(GENERATION_CONFIGS))
    ap.add_argument("--source", default=DATA_SOURCE, choices=["tfds", "kaggle"])
    ap.add_argument("--ckpt-dir", default=CKPT_DIR,
                    help="Orbax checkpoint root (per-step subdirs live here).")
    ap.add_argument("--step", type=int, default=None,
                    help="Checkpoint step to restore. Omit for latest.")
    ap.add_argument("--no-restore", action="store_true",
                    help="Skip LoRA restore — evaluates the base model only.")
    ap.add_argument("--output-csv", default=None,
                    help="Write per-prompt results to this CSV path (for bootstrap CI).")
    args = ap.parse_args()

    mesh = build_mesh()
    local_path, eos_tokens = download_weights()
    base, cfg = load_base_model(local_path, mesh)
    lora = get_lora_model(base, mesh)
    tokenizer, eos_tokens = load_tokenizer(eos_tokens)

    if args.no_restore:
        print("Skipping checkpoint restore — evaluating base model.")
        restored_step = None
        resolved_ckpt_dir = args.ckpt_dir
    else:
        restored_step, resolved_ckpt_dir = restore_lora(lora, args.ckpt_dir, args.step)

    _, _, test_ds = build_train_val_test(
        NUM_BATCHES, NUM_TEST_BATCHES, TRAIN_MICRO_BATCH_SIZE, TRAIN_FRACTION,
        NUM_EPOCHS, TRAIN_DATA_DIR, TEST_DATA_DIR, source=args.source,
        shuffle_seed=RUN_SEED,
    )

    sampler = sampler_lib.Sampler(
        transformer=lora,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
            num_layers=cfg.num_layers,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
        ),
    )
    n, t, acc, pacc, facc, rows = evaluate(
        test_ds, sampler, eos_tokens, **GENERATION_CONFIGS[args.preset])
    print(f"\nFINAL: correct={n}/{t}  acc={acc:.2f}%  partial={pacc:.2f}%  format={facc:.2f}%")
    print(
        f"  requested_ckpt_dir={args.ckpt_dir}  resolved_ckpt_dir={resolved_ckpt_dir}  "
        f"restored_step={restored_step}  preset={args.preset}"
    )

    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv_mod.DictWriter(fh, fieldnames=list(rows[0].keys()) + [
                "requested_ckpt_dir", "resolved_ckpt_dir", "restored_step", "preset", "run_seed"])
            writer.writeheader()
            meta = {"requested_ckpt_dir": args.ckpt_dir, "resolved_ckpt_dir": resolved_ckpt_dir,
                    "restored_step": restored_step, "preset": args.preset, "run_seed": RUN_SEED}
            for row in rows:
                writer.writerow({**row, **meta})
        print(f"Per-prompt results written to {args.output_csv}")


if __name__ == "__main__":
    main()
