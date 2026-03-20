import os
import random
import json
import time
from functools import partial
from pathlib import Path
from collections import defaultdict

import numpy as np
from torchvision import transforms
import torch
from torch import nn
from torch import multiprocessing as mp
from torch import distributed as dist
from torch.utils.data import Subset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau

from make_dataset import load_dataset
from config import config
from optim import get_optim
from networks import Backbone
from diffusion import Diffusion
from eval import FIDEvaluator


class ClassConditionalSampler:
    def __init__(self, num_classes: int, device: str):
        self.num_classes = num_classes
        self.device = device

    @torch.no_grad()
    def sample(self, model, batch_size):
        classes = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        samples = model.module.sample(classes=classes) if isinstance(model, DDP) else model.sample(classes=classes)
        return samples


class TrainMetrics:
    def __init__(self, train_loss, validation_loss, runtime, steps):
        self.train_loss = train_loss
        self.val_loss = validation_loss
        self.runtime = runtime
        self.steps = steps

    def dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "validation_loss": self.val_loss,
            "runtime": self.runtime,
            "steps": self.steps,
        }


class EvalMetrics:
    def __init__(self, fid_score):
        self.fid_score = fid_score

    def dict(self) -> dict:
        return {"fid_score": self.fid_score}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_seed_all(seed: int = 42):
    set_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc or "")
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def ddp_cleanup():
    dist.destroy_process_group()


def preprocess(example, config: dict):
    image_size = config["transformed_image_size"]
    channels = config["channels"]

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize((0.5,), (0.5,))
        if channels == 1
        else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    processed_images = []
    for image in example["image"]:
        if channels == 3 and image.mode != "RGB":
            image = image.convert("RGB")

        img_tensor = transform(image)
        processed_images.append(img_tensor)

    example["image"] = processed_images
    return example


def load_and_preprocess_data(config: dict) -> tuple:
    image_key = config["image_key"]
    label_key = config.get("label_key")

    dataset = load_dataset()

    train_dataset, eval_dataset = dataset["train"], dataset["test"]

    if image_key != "image":
        train_dataset = train_dataset.rename_column(image_key, "image")
        eval_dataset = eval_dataset.rename_column(image_key, "image")

    if label_key is not None:
        if label_key != "label":
            train_dataset = train_dataset.rename_column(label_key, "label")
    else:
        train_dataset = train_dataset.add_column("label", [0] * len(train_dataset))

    preprocess_fn = partial(preprocess, config=config)
    train_dataset = train_dataset.with_transform(preprocess_fn)
    eval_dataset = eval_dataset.with_transform(preprocess_fn)

    return train_dataset, eval_dataset["image"]


def get_model(config: dict) -> nn.Module:
    backbone = Backbone(config["dim"], config["num_classes"], config)
    diffusion = Diffusion(backbone, config["image_size"], config)

    return diffusion


class Trainer:
    def __init__(
        self,
        model,
        train_dl,
        val_dl,
        optimizer,
        lr_scheduler,
        max_steps,
        rank,
        evaluator=None,
        eval_sampler=None,
        eval_interval=1_000,
        patience=5,
        checkpoint_dir=Path("model"),
    ) -> None:
        self.model = model.to(rank)
        self.model = DDP(model, device_ids=[rank])
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = torch.GradScaler("cuda")
        self.max_steps = max_steps
        self.steps = 0

        if evaluator is not None:
            assert eval_sampler is not None, "`eval_sampler` must be provided with `evaluator`"  # noqa:S101

        self.evaluator = evaluator
        self.eval_sampler = eval_sampler
        self.eval_interval = eval_interval
        self.best_score = float("inf")
        self.score_not_improved_count = 0
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir

        self.metrics = TrainMetrics(None, None, 0, 0)

        self.rank = rank

    def save_checkpoint(self, suffix):
        if self.rank != 0:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filename = f"checkpoint-{suffix}.pth"
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(self.model.module.state_dict(), checkpoint_path)
        print(f"GPU {self.rank}: Saved checkpoint to {checkpoint_path}")

    def _get_image_and_label(self, batch):
        images = batch["image"]
        labels = batch["label"]

        return images.to(self.rank), labels.to(self.rank)

    def _sample_t_and_noise_and_x_t(self, images):
        t = torch.randint(0, self.model.module.num_timesteps, (images.shape[0],), device=self.rank).long()
        alphas_cumprod = self.model.module.alphas_cumprod[t]
        sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod)

        while len(sqrt_alphas_cumprod_t.shape) < len(images.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)

        noise = torch.randn_like(images)
        x_t = sqrt_alphas_cumprod_t * images + sqrt_one_minus_alphas_cumprod_t * noise

        return t, noise, x_t

    def _train(self, epoch):
        self.train_dl.sampler.set_epoch(epoch)
        self.model.train()

        early_stopping = False
        total_loss = 0

        for batch in self.train_dl:
            if self.steps >= self.max_steps:
                break

            self.optimizer.zero_grad()
            images, labels = self._get_image_and_label(batch)
            loss = self.model(images, classes=labels)

            del images
            del labels

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            self.metrics.steps += 1
            self.steps += 1

            if self.steps % 50_000 == 0:
                self.save_checkpoint(f"{self.steps // 50_000}")

            if self.evaluator is not None and self.steps % self.eval_interval == 0 and not self._eval():
                early_stopping = True
                break

        mean_loss = total_loss / len(self.train_dl)
        return mean_loss, early_stopping

    @torch.no_grad()
    def _validate(self, epoch):
        self.val_dl.sampler.set_epoch(epoch)
        self.model.eval()

        total_loss = 0
        loss_fn = nn.MSELoss()

        for batch in self.val_dl:
            images, labels = self._get_image_and_label(batch)
            t, noise, x_t = self._sample_t_and_noise_and_x_t(images)

            noise_pred, _ = self.model.module.model_predictions(x_t, t, labels)
            loss = loss_fn(noise_pred, noise)

            total_loss += loss.item()

        mean_loss = total_loss / len(self.val_dl)
        return mean_loss

    @torch.no_grad()
    def _eval(self) -> bool:
        """Return if should continue."""
        self.model.eval()

        score = self.evaluator.evaluate(self.eval_sampler, self.model)

        if self.rank == 0:
            print(f"Step {self.steps}: FID = {score}")

        if score < self.best_score:
            self.best_score = score
            self.score_not_improved_count = 0
        else:
            self.score_not_improved_count += 1
            if self.score_not_improved_count >= self.patience:
                return False

        return True

    def _run_epoch(self, epoch) -> bool:
        """Return if should continue."""
        train_loss, early_stopping = self._train(epoch)

        if not early_stopping:
            val_loss = self._validate(epoch)

            self.lr_scheduler.step(val_loss)

            if self.rank == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Step {self.steps} (Epoch {epoch}): Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, LR = {current_lr:.3e}"
                )

            self.metrics.val_loss = float(val_loss)

        self.metrics.train_loss = float(train_loss)

        return not early_stopping

    def train(self):
        self.steps = 0
        epoch = 0

        torch.cuda.synchronize()
        start = time.perf_counter()

        while self.steps < self.max_steps:
            if not self._run_epoch(epoch):

                print(f"GPU {self.rank}: Early stopping at step {self.steps}")
                break
            epoch += 1

        torch.cuda.synchronize()
        train_time = time.perf_counter() - start

        if self.rank == 0:
            self.save_checkpoint("final")

        self.metrics.runtime = float(train_time)


def train_model(
    rank,
    model,
    train_dataset,
    eval_dataset,
    features_dir,
    checkpoints_dir,
    train_config,
    eval_config,
    queue,
    world_size,
):
    validation_split = train_config["validation_split"]
    max_steps = train_config["steps"]
    batch_size = train_config["per_device_batch_size"]

    if "early_stopping" in train_config:
        n_fake_samples = train_config["early_stopping"]["n_fake_samples"]
        early_stopping_eval_interval = train_config["early_stopping"]["interval"]
        early_stopping_patience = train_config["early_stopping"]["patience"]

    channels = eval_config["channels"]
    num_classes = eval_config["num_classes"]
    n_real_samples = eval_config["n_real_samples"]

    data_split = 1 - validation_split
    device = f"cuda:{rank}"

    ddp_setup(rank, world_size)
    train_size = int(data_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, drop_last=False)
    val_sampler = DistributedSampler(val_dataset, drop_last=False)

    optimizer = get_optim(model, train_config)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=3,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dl = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=3, pin_memory=True, persistent_workers=True
    )

    if "early_stopping" in train_config:
        if n_real_samples < len(eval_dataset):
            indices = np.random.choice(len(eval_dataset), size=n_real_samples, replace=False)
            eval_dataset = Subset(eval_dataset, indices=indices)

        eval_sampler = DistributedSampler(eval_dataset, drop_last=False)
        eval_dl = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            sampler=eval_sampler,
            num_workers=3,
            pin_memory=True,
            persistent_workers=True,
        )
        evaluator = FIDEvaluator(
            eval_dl,
            channels=channels,
            num_fake_samples=n_fake_samples,
            stats_dir=features_dir,
            rank=rank,
            world_size=world_size,
            batch_size=batch_size,
        )
        eval_sampler = ClassConditionalSampler(num_classes, device=device)
    else:
        evaluator = None
        eval_sampler = None
        early_stopping_eval_interval = None
        early_stopping_patience = None

    trainer = Trainer(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        max_steps=max_steps,
        rank=rank,
        evaluator=evaluator,
        eval_sampler=eval_sampler,
        eval_interval=early_stopping_eval_interval,
        patience=early_stopping_patience,
        checkpoint_dir=checkpoints_dir,
    )
    trainer.train()
    metrics = trainer.metrics.dict()
    queue.put(metrics)

    ddp_cleanup()


def eval_model(rank, model, dataset, features_dir, config, queue, world_size):
    channels = config["channels"]
    num_classes = config["num_classes"]
    n_real_samples = config["n_real_samples"]
    n_fake_samples = config["n_fake_samples"]
    batch_size = config["per_device_batch_size"]

    device = f"cuda:{rank}"

    ddp_setup(rank, world_size)
    set_seed(rank)

    model = model.to(rank)

    if n_real_samples < len(dataset):
        indices = np.random.choice(len(dataset), size=n_real_samples, replace=False)
        dataset = Subset(dataset, indices=indices)

    sampler = DistributedSampler(dataset, drop_last=False)
    dl = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True, persistent_workers=True
    )
    evaluator = FIDEvaluator(
        dl,
        channels=channels,
        num_fake_samples=n_fake_samples,
        stats_dir=features_dir,
        rank=rank,
        world_size=world_size,
        batch_size=batch_size,
    )
    sampler = ClassConditionalSampler(num_classes, device)
    fid_score = evaluator.evaluate(sampler, model)

    if rank == 0:
        metrics = EvalMetrics(float(fid_score))
        queue.put(metrics.dict())

    ddp_cleanup()


def main():
    features_dir = Path("_features")
    checkpoints_dir = Path("_checkpoints")

    dataset_name = config["meta"]["name"]
    seed = config["meta"]["seed"]

    print("Starting training and evaluation routine...")
    set_seed_all(seed)

    features_dir.mkdir(parents=True, exist_ok=True)
    features_dir = features_dir / f"{dataset_name}.pth"
    train_dataset, eval_dataset = load_and_preprocess_data(config["preprocess"])
    model = get_model(config["model"])

    world_size = torch.cuda.device_count()
    manager = mp.Manager()
    queue = manager.Queue()

    mp.spawn(
        train_model,
        args=(
            model,
            train_dataset,
            eval_dataset,
            features_dir,
            checkpoints_dir,
            config["train"],
            config["eval"],
            queue,
            world_size,
        ),
        nprocs=world_size,
    )
    train_metrics_all = [queue.get() for _ in range(world_size)]
    key_values = defaultdict(list)
    for metrics in train_metrics_all:
        for key, value in metrics.items():
            key_values[key].append(value)

    train_metrics = TrainMetrics(
        train_loss=np.mean(key_values["train_loss"]),
        validation_loss=np.mean(key_values["validation_loss"]),
        runtime=np.max(key_values["runtime"]),
        steps=key_values["steps"][0],
    ).dict()

    state_dict = torch.load(f"{checkpoints_dir}/checkpoint-final.pth", map_location="cpu")
    model.load_state_dict(state_dict)

    mp.spawn(eval_model, args=(model, eval_dataset, features_dir, config["eval"], queue, world_size), nprocs=world_size)
    eval_metrics = queue.get()

    print(json.dumps({"train_metrics": train_metrics, "eval_metrics": eval_metrics}))


if __name__ == "__main__":
    main()
