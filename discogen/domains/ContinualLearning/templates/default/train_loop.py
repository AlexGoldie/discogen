"""Fixed training harness."""

from typing import Any, Dict, List
import os
import torch
from torch.utils.data import DataLoader
from loss import classification_loss
from metrics import final_metrics
from wrappers import JsonlLogger, save_checkpoint


def _evaluate(model, tasks: List[Dict[str, Any]], device: torch.device) -> torch.Tensor:
    model.eval()
    T = len(tasks)
    acc = torch.zeros(T, T, dtype=torch.float32)
    with torch.no_grad():
        # Compute true per-task accuracy using each task's own test set
        acc_vec = torch.zeros(T, dtype=torch.float32)
        for i, task in enumerate(tasks):
            loader = DataLoader(task["test"], batch_size=256, shuffle=False, num_workers=0)
            correct = 0
            total = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.numel())
            acc_vec[i] = (correct / max(1, total))
    return acc_vec


def train_across_tasks(
    model,
    optimizer,
    scheduler_obj,
    tasks: List[Dict[str, Any]],
    modules: Dict[str, Any],
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    os.makedirs(cfg["logging"]["output_dir"], exist_ok=True)
    train_log = JsonlLogger(os.path.join(cfg["logging"]["output_dir"], "train_log.jsonl"))
    model.to(device)
    global_step = 0
    acc_matrix = torch.zeros(len(tasks), len(tasks), dtype=torch.float32)

    for t_idx, task in enumerate(tasks):
        modules["regularizer"].on_task_start(t_idx, model)
        # Grow classifier head to include all seen classes so far
        seen_classes = sorted(set(sum([tasks[i]["class_ids"] for i in range(t_idx + 1)], [])))
        model.ensure_num_classes(max(seen_classes) + 1)
        model.to(device)
        if hasattr(model, "set_active_head"):
            try:
                model.set_active_head(t_idx)
            except Exception:
                pass

        train_loader = DataLoader(task["train"], batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=0)

        for _ in range(cfg["training"]["epochs_per_task"]):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                # Mix with replay
                current = {"x": x, "y": y, "task_id": torch.full_like(y, t_idx)}
                batch = modules["sampler"].mix(current, modules["replay"], cfg["training"]["batch_size"])
                logits = model(batch["x"])  # labels are global ids; head sized to max seen
                loss = classification_loss(logits, batch["y"], cfg["training"].get("label_smoothing", 0.0))
                pen = modules["regularizer"].compute_penalty(model, step=global_step)
                loss = loss + cfg["training"]["reg_lambda"] * pen
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if scheduler_obj is not None:
                    scheduler_obj.step()
                modules["replay"].add(current)

                global_step += 1
                if global_step % max(1, int(cfg["logging"].get("log_every", 100))) == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    train_log.log(
                        {
                            "global_step": global_step,
                            "task_id": t_idx,
                            "loss": float(loss.item()),
                            "penalty": float(pen.item()) if torch.is_tensor(pen) else float(pen),
                            "lr": float(lr),
                            "buffer_size": int(modules["replay"].size()),
                        }
                    )

        modules["regularizer"].on_task_end(t_idx, model, DataLoader(task["val"], batch_size=256, shuffle=False, num_workers=0))
        # Evaluate on all seen tasks
        acc_vec = _evaluate(model, tasks[: t_idx + 1], device)
        acc_matrix[: t_idx + 1, t_idx] = acc_vec

        if cfg["logging"].get("checkpoint_every_task", True):
            save_checkpoint(
                os.path.join(cfg["logging"]["output_dir"], f"latest_task_{t_idx}.pt"),
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "regularizer": modules["regularizer"].state_dict(),
                    "replay": modules["replay"].state_dict(),
                },
            )

    metrics = final_metrics(acc_matrix, cfg["scoring"]["lambda_forgetting"])
    return {"metrics": metrics, "acc_matrix": acc_matrix.tolist()}
