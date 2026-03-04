
import os
import sys
import torch

import data as data_utils
from network import build_model
from train_loop import train_across_tasks
from wrappers import save_json, seed_everything
import json

# Modules (resolved at task generation time to base/edit variants)
from regularizer import Regularizer, RegularizerConfig
from replay import ReplayBuffer, ReplayConfig
from sampler import BatchMixer, SamplerConfig
from scheduler import build_scheduler, SchedulerConfig
from optim import build_optimizer, OptimizerConfig
from make_dataset import default_image_size, get_split_cfg

def main() -> int:
    # CWD agnostic
    # Infer dataset name from directory structure task_src/<DatasetName>/main.py
    dataset_name = os.path.basename(os.path.dirname(__file__))

    # Defaults
    cfg = {
        "training": {
            "epochs_per_task": 1,
            "batch_size": 128,
            "optimizer": {"name": "sgd", "lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4},
            "scheduler": {"kind": "cosine", "warmup_steps": 500},
            "reg_lambda": 1.0,
            "label_smoothing": 0.0,
            "seed": 42,
            "device": "auto",
        },
        "logging": {"output_dir": os.path.dirname(__file__), "checkpoint_every_task": True, "log_every": 100},
        "scoring": {"lambda_forgetting": 0.5},
    }

    seed_everything(int(cfg["training"]["seed"]))

    # Build tasks (small sequences by default)
    n_tasks = 10
    data_root = os.path.join(os.path.dirname(__file__), "data")
    tasks = data_utils.get_task_sequence(
        dataset_name, data_root, cfg["training"]["seed"], n_tasks, get_split_cfg()
    )

    # Model and transforms
    model = build_model()
    img_size = default_image_size()
    data_utils.apply_transforms(tasks, img_size)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["training"]["device"] != "cpu" else "cpu")

    # Optimizer
    optimizer = build_optimizer(
        model.parameters(),
        OptimizerConfig(**cfg.get("training", {}).get("optimizer", {})),
    )

    # Modules
    regularizer = Regularizer(RegularizerConfig(), cfg["training"]["seed"])
    replay = ReplayBuffer(ReplayConfig(), cfg["training"]["seed"], device)
    sampler = BatchMixer(SamplerConfig(), cfg["training"]["seed"])
    scheduler_obj = build_scheduler(optimizer, SchedulerConfig(**cfg.get("training", {}).get("scheduler", {})))

    result = train_across_tasks(
        model=model,
        optimizer=optimizer,
        scheduler_obj=scheduler_obj,
        tasks=tasks,
        modules={"regularizer": regularizer, "replay": replay, "sampler": sampler},
        cfg=cfg,
        device=device,
    )

    out_dir = cfg["logging"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    results_dict =  {**result["metrics"], "acc_matrix": result["acc_matrix"]}
    print(json.dumps(results_dict))
    save_json(os.path.join(out_dir, "metrics.json"), results_dict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
