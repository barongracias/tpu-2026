"""Mine GSM8K-train hard examples with the frozen base model.

This is an inference-only probe. It does not restore LoRA checkpoints and it
does not touch training configuration. Each GSM8K train example is evaluated
with one greedy completion plus sampled completions at fixed seeds.
"""
import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROMPT_TEMPLATE_VERSION = "data.TEMPLATE+SYSTEM_PROMPT"
GENERATION_PRESETS = ["greedy", "standard", "liberal"]


FIELDNAMES = [
    "source",
    "split",
    "source_index",
    "question_sha256",
    "question",
    "expected_answer",
    "model_id",
    "model_revision",
    "probe_model",
    "prompt_template_version",
    "prompt_sha256",
    "greedy_correct",
    "greedy_partial_correct",
    "greedy_format_correct",
    "greedy_extracted_answer",
    "sample_correct_count",
    "sample_partial_correct_count",
    "sample_format_correct_count",
    "num_samples",
    "sample_seeds",
    "sample_extracted_answers",
    "sample_correct_by_seed",
    "hardness_label",
    "completion_path",
    "greedy_completion_preview",
    "sample_completion_previews",
    "command",
    "config_json",
    "error",
]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def run_git(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def git_metadata() -> dict:
    status = run_git(["status", "--short"])
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "dirty": bool(status),
        "status_short": status,
    }


def load_gsm8k_train(data_dir: str):
    """Yield unshuffled GSM8K train examples with TFDS source indices."""
    import tensorflow_datasets as tfds

    os.makedirs(data_dir, exist_ok=True)
    import tensorflow_datasets.text.gsm8k  # noqa: F401  (registers the builder)

    data = tfds.data_source(
        "gsm8k",
        split="train",
        data_dir=data_dir,
        builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
        download=True,
    )
    for source_index, row in enumerate(data):
        question = as_text(row["question"])
        answer = extract_hash_answer(as_text(row["answer"]))
        yield source_index, question, answer


def score_completion(response: str, expected_answer: str | None) -> dict:
    extracted = (
        guess.group(1)
        if response is not None and (guess := match_numbers.search(response)) is not None
        else ""
    )
    exact = False
    partial = False
    if expected_answer is not None and extracted:
        try:
            exact = float(extracted.strip()) == float(expected_answer.strip())
            ratio = float(extracted.strip()) / float(expected_answer.strip())
            partial = 0.9 <= ratio <= 1.1
        except Exception:
            exact = extracted.strip() == expected_answer.strip()
    return {
        "correct": exact,
        "partial_correct": partial,
        "format_correct": match_format.search(response or "") is not None,
        "extracted_answer": extracted,
    }


def preview(text: str, max_chars: int = 240) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def write_jsonl_row(handle, row: dict):
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def write_completion_file(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def build_row(
    *,
    source_index: int,
    question: str,
    expected_answer: str | None,
    greedy_response: str,
    sample_responses: list[dict],
    completion_path: Path,
    command: str,
    config: dict,
    error: str = "",
) -> dict:
    prompt = TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=question)
    greedy_score = score_completion(greedy_response, expected_answer)
    sample_scores = [
        {"seed": item["seed"], **score_completion(item["response"], expected_answer)}
        for item in sample_responses
    ]

    greedy_correct = greedy_score["correct"]
    sample_correct_count = sum(int(item["correct"]) for item in sample_scores)
    if greedy_correct:
        label = "easy"
    elif sample_correct_count:
        label = "medium"
    else:
        label = "hard"

    return {
        "source": "gsm8k",
        "split": "train",
        "source_index": source_index,
        "question_sha256": question_hash(question),
        "question": question,
        "expected_answer": expected_answer or "",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "probe_model": "base",
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "greedy_correct": int(greedy_score["correct"]),
        "greedy_partial_correct": int(greedy_score["partial_correct"]),
        "greedy_format_correct": int(greedy_score["format_correct"]),
        "greedy_extracted_answer": greedy_score["extracted_answer"],
        "sample_correct_count": sample_correct_count,
        "sample_partial_correct_count": sum(int(item["partial_correct"]) for item in sample_scores),
        "sample_format_correct_count": sum(int(item["format_correct"]) for item in sample_scores),
        "num_samples": len(sample_scores),
        "sample_seeds": json_dumps([item["seed"] for item in sample_scores]),
        "sample_extracted_answers": json_dumps([item["extracted_answer"] for item in sample_scores]),
        "sample_correct_by_seed": json_dumps(
            {str(item["seed"]): bool(item["correct"]) for item in sample_scores}
        ),
        "hardness_label": label,
        "completion_path": str(completion_path),
        "greedy_completion_preview": preview(greedy_response),
        "sample_completion_previews": json_dumps(
            {str(item["seed"]): preview(item["response"]) for item in sample_responses}
        ),
        "command": command,
        "config_json": json_dumps(config),
        "error": error,
    }


def write_summary(path: Path, counts: dict, total: int, failures: int, config: dict):
    def pct(n: int) -> str:
        return "0.00%" if total == 0 else f"{n / total * 100:.2f}%"

    lines = [
        "# GSM8K Train Hard-Example Mining Summary",
        "",
        f"- Total processed: {total}",
        f"- Easy: {counts.get('easy', 0)} ({pct(counts.get('easy', 0))})",
        f"- Medium: {counts.get('medium', 0)} ({pct(counts.get('medium', 0))})",
        f"- Hard: {counts.get('hard', 0)} ({pct(counts.get('hard', 0))})",
        f"- Failures/skipped: {failures}",
        "",
        "## Protocol",
        "",
        "- Source: GSM8K train split only",
        "- Probe model: frozen base model, no LoRA checkpoint restore",
        "- Greedy completions: 1",
        f"- Sampled completions: {config['num_samples']} with seeds {config['sample_seeds']}",
        f"- Sample preset: {config['sample_preset']}",
        f"- Model: {MODEL_ID}@{MODEL_REVISION}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-root",
        default=None,
        help="Persistent output root. Defaults to $HOME/tpu-runs/part-i/hard_mining_base_gsm8k_train_<timestamp>.",
    )
    ap.add_argument("--limit", type=int, default=None, help="Limit examples for smoke tests.")
    ap.add_argument("--offset", type=int, default=0, help="Skip the first N train examples.")
    ap.add_argument("--train-data-dir", default=None, help="TFDS cache for GSM8K train.")
    ap.add_argument("--sample-preset", default="standard", choices=GENERATION_PRESETS)
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--sample-seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument(
        "--mining-seed",
        type=int,
        default=int(os.environ.get("MINING_SEED", os.environ.get("EVAL_SEED", "0"))),
        help="Recorded mining/eval seed. Dataset order is unshuffled; sample seeds are controlled separately.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    if not os.environ.get("MODEL_REVISION"):
        raise RuntimeError("MODEL_REVISION must be set to an exact Hugging Face commit SHA.")

    from tqdm.auto import tqdm
    from tunix.generate import sampler as sampler_lib

    from config import (
        EVAL_SEED,
        GENERATION_CONFIGS,
        JAX_REF,
        MAX_PROMPT_LENGTH,
        MODEL_ID,
        MODEL_REVISION,
        QWIX_REF,
        RUN_SEED,
        TOTAL_GENERATION_STEPS,
        TRAIN_DATA_DIR,
        TUNIX_REF,
        FLAX_REF,
    )
    from data import (
        SYSTEM_PROMPT,
        TEMPLATE,
        as_text,
        extract_hash_answer,
        question_hash,
    )
    from evaluate import generate
    from model import build_mesh, download_weights, load_base_model, load_tokenizer
    from rewards import match_format, match_numbers

    globals().update({
        "as_text": as_text,
        "EVAL_SEED": EVAL_SEED,
        "extract_hash_answer": extract_hash_answer,
        "GENERATION_CONFIGS": GENERATION_CONFIGS,
        "JAX_REF": JAX_REF,
        "MAX_PROMPT_LENGTH": MAX_PROMPT_LENGTH,
        "match_format": match_format,
        "match_numbers": match_numbers,
        "MODEL_ID": MODEL_ID,
        "MODEL_REVISION": MODEL_REVISION,
        "question_hash": question_hash,
        "QWIX_REF": QWIX_REF,
        "RUN_SEED": RUN_SEED,
        "TOTAL_GENERATION_STEPS": TOTAL_GENERATION_STEPS,
        "TRAIN_DATA_DIR": TRAIN_DATA_DIR,
        "TUNIX_REF": TUNIX_REF,
        "FLAX_REF": FLAX_REF,
        "SYSTEM_PROMPT": SYSTEM_PROMPT,
        "TEMPLATE": TEMPLATE,
        "generate": generate,
    })

    sample_seeds = [int(x) for x in args.sample_seeds.split(",") if x.strip()]
    if len(sample_seeds) != args.num_samples:
        raise ValueError("--num-samples must match the number of --sample-seeds values.")

    run_root = Path(
        args.run_root
        or Path.home() / "tpu-runs" / "part-i" / f"hard_mining_base_gsm8k_train_{utc_timestamp()}"
    )
    run_root.mkdir(parents=True, exist_ok=False)
    completions_dir = run_root / "completions"
    data_dir = args.train_data_dir or str(run_root / "data" / "train")

    command = shlex.join([sys.executable, *sys.argv])
    config = {
        "source": "tfds",
        "split": "train",
        "run_root": str(run_root),
        "train_data_dir": data_dir,
        "limit": args.limit,
        "offset": args.offset,
        "training_seed_recorded_only": RUN_SEED,
        "mining_seed": args.mining_seed,
        "greedy_preset": "greedy",
        "sample_preset": args.sample_preset,
        "num_samples": args.num_samples,
        "sample_seeds": sample_seeds,
        "generation_configs": {
            "greedy": GENERATION_CONFIGS["greedy"],
            args.sample_preset: GENERATION_CONFIGS[args.sample_preset],
        },
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "total_generation_steps": TOTAL_GENERATION_STEPS,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_template": TEMPLATE,
        "system_prompt": SYSTEM_PROMPT,
    }

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "config": config,
        "git": git_metadata(),
        "env_pins": {
            "MODEL_REVISION": MODEL_REVISION,
            "JAX_REF": JAX_REF,
            "TUNIX_REF": TUNIX_REF,
            "QWIX_REF": QWIX_REF,
            "FLAX_REF": FLAX_REF,
            "RUN_SEED": os.environ.get("RUN_SEED"),
            "EVAL_SEED": os.environ.get("EVAL_SEED"),
            "MINING_SEED": os.environ.get("MINING_SEED"),
            "TRAIN_DATA_DIR": os.environ.get("TRAIN_DATA_DIR", TRAIN_DATA_DIR),
        },
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "probe_model": "base",
    }
    (run_root / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    mesh = build_mesh()
    local_path, eos_tokens = download_weights()
    base, cfg = load_base_model(local_path, mesh)
    tokenizer, eos_tokens = load_tokenizer(eos_tokens)
    sampler = sampler_lib.Sampler(
        transformer=base,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
            num_layers=cfg.num_layers,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
        ),
    )

    all_jsonl = (run_root / "all_examples.jsonl").open("w", encoding="utf-8")
    hard_jsonl = (run_root / "hard_examples.jsonl").open("w", encoding="utf-8")
    medium_jsonl = (run_root / "medium_examples.jsonl").open("w", encoding="utf-8")
    easy_jsonl = (run_root / "easy_examples.jsonl").open("w", encoding="utf-8")
    csv_fh = (run_root / "all_examples.csv").open("w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_fh, fieldnames=FIELDNAMES)
    csv_writer.writeheader()

    label_handles = {"hard": hard_jsonl, "medium": medium_jsonl, "easy": easy_jsonl}
    counts = {"easy": 0, "medium": 0, "hard": 0}
    failures = 0
    processed = 0

    try:
        iterator = load_gsm8k_train(data_dir)
        progress = tqdm(iterator, total=args.limit, desc="Mining GSM8K train")
        for source_index, question, expected_answer in progress:
            if source_index < args.offset:
                continue
            if args.limit is not None and processed >= args.limit:
                break

            completion_path = completions_dir / f"gsm8k_train_{source_index:05d}.json"
            try:
                greedy_response = generate(
                    question,
                    sampler,
                    eos_tokens,
                    **GENERATION_CONFIGS["greedy"],
                    seed=None,
                )
                sample_responses = [
                    {
                        "seed": seed,
                        "response": generate(
                            question,
                            sampler,
                            eos_tokens,
                            **GENERATION_CONFIGS[args.sample_preset],
                            seed=seed,
                        ),
                    }
                    for seed in sample_seeds
                ]
                row = build_row(
                    source_index=source_index,
                    question=question,
                    expected_answer=expected_answer,
                    greedy_response=greedy_response,
                    sample_responses=sample_responses,
                    completion_path=completion_path,
                    command=command,
                    config=config,
                )
                write_completion_file(
                    completion_path,
                    {
                        "source": "gsm8k",
                        "split": "train",
                        "source_index": source_index,
                        "question_sha256": row["question_sha256"],
                        "question": question,
                        "expected_answer": expected_answer or "",
                        "prompt": TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=question),
                        "greedy": {"seed": None, "response": greedy_response},
                        "samples": sample_responses,
                    },
                )
            except Exception as exc:
                failures += 1
                row = build_row(
                    source_index=source_index,
                    question=question,
                    expected_answer=expected_answer,
                    greedy_response="",
                    sample_responses=[],
                    completion_path=completion_path,
                    command=command,
                    config=config,
                    error=repr(exc),
                )
                row["hardness_label"] = "error"
                row["error"] = repr(exc)

            write_jsonl_row(all_jsonl, row)
            csv_writer.writerow(row)
            csv_fh.flush()
            if row["hardness_label"] in label_handles:
                write_jsonl_row(label_handles[row["hardness_label"]], row)
                counts[row["hardness_label"]] += 1
            processed += 1
            progress.set_postfix({**counts, "failures": failures})
    finally:
        for fh in [all_jsonl, hard_jsonl, medium_jsonl, easy_jsonl, csv_fh]:
            fh.close()

    write_summary(run_root / "summary.md", counts, processed, failures, config)
    print(f"Run root: {run_root}")
    print(f"Processed: {processed}; easy={counts['easy']} medium={counts['medium']} hard={counts['hard']} failures={failures}")
    print(f"Manifest JSONL: {run_root / 'all_examples.jsonl'}")
    print(f"Manifest CSV: {run_root / 'all_examples.csv'}")


if __name__ == "__main__":
    main()
