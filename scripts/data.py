"""GSM8K dataset loading and prompt formatting.

GSM8K is a benchmark of grade-school math word problems. Each example is
(question, answer) where the gold answer string ends with `#### <number>`.

We wrap each question in a chat template that asks the model to produce
its reasoning between <reasoning>...</reasoning> and the final numeric
answer between <answer>...</answer>. The reward functions later check
both the format and the number itself.
"""
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import grain
import kagglehub
import tensorflow_datasets as tfds

# Special tokens used by the policy and parsed by the reward fns.
reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"

SYSTEM_PROMPT = (
    f"You are given a problem. First, think about the problem and provide your "
    f"reasoning. Place it between {reasoning_start} and {reasoning_end}. Then, "
    f"provide the final answer (i.e., just one numerical value) between "
    f"{solution_start} and {solution_end}."
)

TEMPLATE = (
    "<start_of_turn>user\n"
    "{system_prompt}\n\n"
    "{question}<end_of_turn>\n"
    "<start_of_turn>model\n"
)


def extract_hash_answer(text: str) -> str | None:
    """GSM8K answers look like '...long explanation... #### 42'."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def as_text(v):
    return v if isinstance(v, str) else v.decode("utf-8")


def question_hash(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def format_example(question: str, answer: str | None):
    return {
        "prompts": TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            question=question,
        ),
        "question": question,
        "answer": answer,
    }


def _download_kaggle_dataset(target_dir: str = "./data/gsm8k") -> str:
    os.makedirs(target_dir, exist_ok=True)
    src = Path(kagglehub.dataset_download("thedevastator/grade-school-math-8k-q-a"))
    dst = Path(target_dir)
    for csv_file in src.glob("*.csv"):
        shutil.copy2(csv_file, dst / csv_file.name)
    return target_dir


def get_dataset(data_dir: str, split: str = "train", source: str = "tfds",
                shuffle_seed: int = 0) -> grain.MapDataset:
    """Return a grain.MapDataset of {prompts, question, answer} dicts."""
    os.makedirs(data_dir, exist_ok=True)

    if source == "tfds":
        import tensorflow_datasets.text.gsm8k  # noqa: F401  (registers the builder)
        data = tfds.data_source(
            "gsm8k",
            split=split,
            data_dir=data_dir,
            builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
            download=True,
        )
    elif source == "kaggle":
        kaggle_dir = _download_kaggle_dataset(data_dir)
        csv_path = os.path.join(kaggle_dir, f"main_{split}.csv")
        data = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({"question": row["question"], "answer": row["answer"]})
    else:
        raise ValueError(f"Unknown source: {source}")

    return (
        grain.MapDataset.source(data)
        .shuffle(seed=shuffle_seed)
        .map(lambda x: {
            "prompts": TEMPLATE.format(
                system_prompt=SYSTEM_PROMPT,
                question=as_text(x["question"]),
            ),
            "question": as_text(x["question"]),
            "answer": extract_hash_answer(as_text(x["answer"])),
        })
    )


def get_dataset_from_manifest(manifest_path: str,
                              batch_size: int) -> grain.MapDataset:
    """Load an explicit eval manifest as a batched Grain dataset."""
    rows = []
    with open(manifest_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            question = row["question"]
            answer = row.get("expected_answer", row.get("answer"))
            rows.append(format_example(question, answer))
    return grain.MapDataset.source(rows).batch(batch_size)


def manifest_row(prompt_id: int,
                 question: str,
                 answer: str | None,
                 source: str,
                 split: str,
                 eval_seed: int) -> dict:
    return {
        "prompt_id": prompt_id,
        "source": source,
        "split": split,
        "eval_seed": eval_seed,
        "question_sha256": question_hash(question),
        "question": question,
        "expected_answer": answer or "",
    }


def build_train_val_test(num_batches: int,
                         num_test_batches: int,
                         train_micro_batch_size: int,
                         train_fraction: float,
                         num_epochs: int,
                         train_dir: str,
                         test_dir: str,
                         source: str = "tfds",
                         shuffle_seed: int = 0,
                         test_shuffle_seed: int | None = None):
    """Materialise (train, val, test) datasets with batching applied."""
    if test_shuffle_seed is None:
        test_shuffle_seed = shuffle_seed
    full = get_dataset(train_dir, "train", source, shuffle_seed=shuffle_seed).batch(
        train_micro_batch_size)[:num_batches]

    if train_fraction == 1.0:
        train_ds = full.repeat(num_epochs)
        val_ds = None
    else:
        cut = int(len(full) * train_fraction)
        train_ds = full[:cut].repeat(num_epochs)
        val_ds = full[cut:].repeat(num_epochs)

    test_ds = get_dataset(test_dir, "test", source, shuffle_seed=test_shuffle_seed).batch(
        train_micro_batch_size)[:num_test_batches]
    return train_ds, val_ds, test_ds
