"""GRPO training entry point.

Run under tmux:
    tmux new -s tunix
    source ~/venvs/tunix/bin/activate
    cd ~/tpu-2026/scripts
    python train.py
    # detach: Ctrl-b d   # reattach: tmux attach -t tunix

Resuming a wandb run: set WANDB_RUN_ID in env (or pass --wandb-run-id).
Resuming from checkpoint: just point CKPT_DIR at the existing directory.
Tunix's RLCluster uses Orbax and will pick up the latest step in CKPT_DIR.
"""
import argparse
import datetime
import json
import os
import subprocess

import nest_asyncio
import optax
import wandb
from dotenv import load_dotenv
from orbax import checkpoint as ocp
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger

from config import (
    ADV_ESTIMATOR,
    B1, B2,
    BETA,
    CKPT_DIR,
    DATA_SOURCE,
    EPSILON,
    EVAL_MANIFEST,
    EVAL_SEED,
    FLAX_REF,
    INTERMEDIATE_CKPT_DIR,
    JAX_REF,
    EVAL_EVERY_N_STEPS,
    LR_DECAY_STEPS,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    MAX_PROMPT_LENGTH,
    MAX_STEPS,
    MAX_TO_KEEP,
    MODEL_ID,
    MODEL_REVISION,
    NUM_BATCHES,
    NUM_EPOCHS,
    NUM_GENERATIONS,
    NUM_ITERATIONS,
    NUM_TEST_BATCHES,
    QWIX_REF,
    RUN_SEED,
    SAVE_INTERVAL_STEPS,
    TEMPERATURE,
    TENSORBOARD_DIR,
    TEST_DATA_DIR,
    TOP_K, TOP_P,
    TOTAL_GENERATION_STEPS,
    TRAIN_DATA_DIR,
    TRAIN_FRACTION,
    TRAIN_MANIFEST,
    TRAIN_MICRO_BATCH_SIZE,
    TUNIX_REF,
    WANDB_ENTITY,
    WANDB_PROJECT,
    WANDB_RUN_ID,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from data import build_train_val_test
from model import build_mesh, download_weights, load_base_model, get_lora_model, load_tokenizer
from rewards import REWARD_FNS


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", _repo_root(), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def save_run_metadata(run_id: str | None, ckpt_dir: str) -> str:
    meta = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "tpu2026_commit": _git_commit(),
        "repo_root": _repo_root(),
        "launch_cwd": os.getcwd(),
        "wandb_run_id": run_id,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dependency_refs": {
            "jax": JAX_REF,
            "tunix": TUNIX_REF,
            "qwix": QWIX_REF,
            "flax": FLAX_REF,
        },
        "advantage_estimator": ADV_ESTIMATOR,
        "run_seed": RUN_SEED,
        "eval_seed": EVAL_SEED,
        "eval_manifest": EVAL_MANIFEST,
        "max_steps": MAX_STEPS,
        "lr_decay_steps": LR_DECAY_STEPS,
        "save_interval_steps": SAVE_INTERVAL_STEPS,
        "max_to_keep": MAX_TO_KEEP,
        "num_generations": NUM_GENERATIONS,
        "beta": BETA,
        "epsilon": EPSILON,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "total_generation_steps": TOTAL_GENERATION_STEPS,
        "data_source": DATA_SOURCE,
        "train_data_dir": TRAIN_DATA_DIR,
        "train_manifest": TRAIN_MANIFEST,
        "test_data_dir": TEST_DATA_DIR,
        "ckpt_dir": ckpt_dir,
        "intermediate_ckpt_dir": INTERMEDIATE_CKPT_DIR,
        "tensorboard_dir": TENSORBOARD_DIR,
        "wandb_project": WANDB_PROJECT,
        "wandb_entity": WANDB_ENTITY,
    }
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, "run_metadata.json")
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Run metadata saved to {path}")
    return path


def login_services():
    load_dotenv()
    nest_asyncio.apply()  # tunix uses async; jupyter-style nesting helps in tmux too
    if os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"])
    if os.environ.get("HF_TOKEN"):
        os.system(f'hf auth login --token "{os.environ["HF_TOKEN"]}"')


def maybe_init_wandb(run_id: str | None):
    """Init wandb. If run_id is given we resume; otherwise a fresh run is created."""
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY not set — skipping wandb.")
        return None
    kwargs = {"project": WANDB_PROJECT, "entity": WANDB_ENTITY}
    if run_id:
        # "allow" => resume if the run exists on the server, otherwise create
        # a new run with this id. "must" errors out if the run was never synced
        # (which is what happens if the previous training crashed before
        # wandb.init was reached).
        kwargs.update({"id": run_id, "resume": "allow"})
    return wandb.init(**kwargs)


def build_optimizer():
    schedule = optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=LR_DECAY_STEPS,
        end_value=0.0,
    )
    opt = optax.adamw(learning_rate=schedule, b1=B1, b2=B2, weight_decay=WEIGHT_DECAY)
    if MAX_GRAD_NORM is not None:
        opt = optax.chain(optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM), opt)
    return opt


def build_cluster_config(mesh, optimizer, eos_tokens):
    return rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optimizer,
            eval_every_n_steps=EVAL_EVERY_N_STEPS,
            max_steps=MAX_STEPS,
            mini_batch_size=TRAIN_MICRO_BATCH_SIZE,
            train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
            metrics_logging_options=metrics_logger.MetricsLoggerOptions(
                log_dir=TENSORBOARD_DIR, flush_every_n_steps=20,
            ),
            checkpoint_root_directory=CKPT_DIR,
            checkpointing_options=ocp.CheckpointManagerOptions(
                save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP,
            ),
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=TOTAL_GENERATION_STEPS,
            max_prompt_length=MAX_PROMPT_LENGTH,
            kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
            temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
            eos_tokens=eos_tokens,
            seed=RUN_SEED,
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DATA_SOURCE, choices=["tfds", "kaggle", "manifest"])
    ap.add_argument("--wandb-run-id", default=WANDB_RUN_ID,
                    help="Pass an existing run id (e.g. bnh9ttlt) to resume.")
    args = ap.parse_args()

    login_services()
    # init wandb BEFORE the trainer because tunix sometimes hangs if wandb is
    # initialised mid-RLCluster construction (known bug).
    wandb_run = maybe_init_wandb(args.wandb_run_id)
    resolved_wandb_run_id = wandb_run.id if wandb_run is not None else args.wandb_run_id
    save_run_metadata(resolved_wandb_run_id, CKPT_DIR)
    if args.source == "manifest" and not TRAIN_MANIFEST:
        raise RuntimeError("DATA_SOURCE=manifest requires TRAIN_MANIFEST to point to a JSONL file.")

    mesh = build_mesh()
    local_path, eos_tokens = download_weights()
    base, _ = load_base_model(local_path, mesh)
    lora = get_lora_model(base, mesh)
    tokenizer, eos_tokens = load_tokenizer(eos_tokens)

    train_ds, val_ds, _ = build_train_val_test(
        NUM_BATCHES, NUM_TEST_BATCHES, TRAIN_MICRO_BATCH_SIZE, TRAIN_FRACTION,
        NUM_EPOCHS, TRAIN_DATA_DIR, TEST_DATA_DIR, source=args.source,
        shuffle_seed=RUN_SEED, test_shuffle_seed=EVAL_SEED,
        train_manifest=TRAIN_MANIFEST if args.source == "manifest" else None,
    )
    print(f"Datasets: train={len(train_ds)} val={len(val_ds) if val_ds else 0}")

    optimizer = build_optimizer()
    cluster_cfg = build_cluster_config(mesh, optimizer, eos_tokens)
    print(f"  ADV_ESTIMATOR={ADV_ESTIMATOR}")
    grpo_cfg = GRPOConfig(
        num_generations=NUM_GENERATIONS,
        num_iterations=NUM_ITERATIONS,
        beta=BETA,
        epsilon=EPSILON,
        advantage_estimator=ADV_ESTIMATOR,
    )

    rl_cluster = rl_cluster_lib.RLCluster(
        actor=lora, reference=base, tokenizer=tokenizer, cluster_config=cluster_cfg,
    )
    trainer = GRPOLearner(
        rl_cluster=rl_cluster, reward_fns=REWARD_FNS, algo_config=grpo_cfg,
        data_shuffle_seed=RUN_SEED,
    )

    print(
        f"Starting GRPO training.\n"
        f"  CKPT_DIR={CKPT_DIR}\n"
        f"  INTERMEDIATE_CKPT_DIR={INTERMEDIATE_CKPT_DIR}\n"
        f"  TENSORBOARD_DIR={TENSORBOARD_DIR}\n"
        f"  MAX_STEPS={MAX_STEPS}\n"
        f"  LR_DECAY_STEPS={LR_DECAY_STEPS}\n"
        f"  SAVE_INTERVAL_STEPS={SAVE_INTERVAL_STEPS}\n"
        f"  MAX_TO_KEEP={MAX_TO_KEEP}\n"
        f"  RUN_SEED={RUN_SEED}\n"
        f"  EVAL_SEED={EVAL_SEED}\n"
        f"  EVAL_MANIFEST={EVAL_MANIFEST}\n"
        f"  TRAIN_MANIFEST={TRAIN_MANIFEST}\n"
        f"  MODEL_REVISION={MODEL_REVISION}\n"
        f"  NUM_GENERATIONS={NUM_GENERATIONS}\n"
        f"  BETA={BETA}\n"
        f"  EPSILON={EPSILON}\n"
        f"  TEMPERATURE={TEMPERATURE}\n"
        f"  TOP_K={TOP_K}\n"
        f"  TOP_P={TOP_P}\n"
        f"  TOTAL_GENERATION_STEPS={TOTAL_GENERATION_STEPS}"
    )
    trainer.train(train_ds, val_ds)
    print("Training finished.")


if __name__ == "__main__":
    main()
