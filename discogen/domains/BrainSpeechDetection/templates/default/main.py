import random
from typing import Dict, Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from make_dataset import load_dataset, baselines
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities import rank_zero_only
from config import config
from loss import compute_loss
from networks import Model
from optim import create_optimizer
from utils import compute_macro_f1_score
import json

class SpeechClassifier(L.LightningModule):

    def __init__(self, config: Dict[str, any]):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.model = Model(config['model'])

        self.loss_fn = compute_loss

        self.val_step_outputs = []
        self.test_step_outputs = {}

    def forward(self, x):
        return self.model(x)

    def _compute_f1_score(self, logits, y):
        """Compute F1 score"""
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        if len(y.shape) > 1:
            y = y.squeeze()

        true_positives = (preds * y).sum().float()
        false_positives = (preds * (1 - y)).sum().float()
        false_negatives = ((1 - preds) * y).sum().float()

        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)

        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        return f1_score

    def _compute_accuracy(self, logits, y):
        """Compute binary accuracy"""
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        if len(y.shape) > 1:
            y = y.squeeze()
        accuracy = (preds.squeeze() == y).float().mean()
        return accuracy

    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]  # (batch, seq_len)
        logits = self(x)
        loss = self.loss_fn(logits, y.unsqueeze(1).float())
        self.log(
            f"train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get("train_loss")
        current_epoch = self.current_epoch + 1
        if train_loss is not None:
            self.print(
                f"Epoch {current_epoch}/{self.trainer.max_epochs} - Train Loss: {train_loss:.4f}"
            )

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val_loss")
        current_epoch = self.current_epoch + 1

        if val_loss is not None:
            self.print(
                f"Epoch {current_epoch}/{self.trainer.max_epochs} - Val Loss: {val_loss:.4f}"
            )

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        logits = self(x)
        loss = self.loss_fn(logits, y.unsqueeze(1).float())
        self.log(
            f"val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]

        logits = self(x)
        probs = torch.sigmoid(logits)
        probs = probs.detach().cpu()
        labels = y.detach().cpu()

        return probs, labels

    def test_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]

        logits = self(x)
        loss = self.loss_fn(logits, y.unsqueeze(1).float())
        probs = torch.sigmoid(logits)

        if "y_probs" not in self.test_step_outputs:
            self.test_step_outputs["y_probs"] = []
        if "y_true" not in self.test_step_outputs:
            self.test_step_outputs["y_true"] = []

        self.test_step_outputs["y_probs"].append(probs.detach().cpu())
        self.test_step_outputs["y_true"].append(y.detach().cpu())

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        return loss

    def on_test_epoch_end(self):
        all_probs = torch.cat(self.test_step_outputs["y_probs"], dim=0)
        all_labels = torch.cat(self.test_step_outputs["y_true"], dim=0)

        best_macro_f1, best_thr = compute_macro_f1_score(all_probs, all_labels)
        self.print(f"Test Results:")
        self.print(f"  Optimal Threshold: {best_thr:.3f}")
        self.print(f"  Best F1-Macro Score: {best_macro_f1:.4f}")

        self.log("test_f1", best_macro_f1, prog_bar=False, logger=True, sync_dist=True)
        self.log("optimal_threshold", best_thr, prog_bar=False, logger=True, sync_dist=True)

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.config['optimizer'])
        return optimizer


def train_and_eval_model(config: Dict, model: SpeechClassifier, train_dataloader, test_dataloader):

    logger = CSVLogger(
        save_dir=config["output_dir"],
        name="speech_model",
        version=None,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=config["early_stopping"]["min_delta"],
        patience=config["early_stopping"]["patience"],
        verbose=False,
        mode="min"
    )

    trainer = L.Trainer(
        devices="auto",
        logger=logger,
        max_epochs=config["max_epochs"],
        enable_checkpointing=config["enable_checkpointing"],
        enable_progress_bar=config["enable_progress_bar"],
        enable_model_summary=config["enable_model_summary"],
        log_every_n_steps=config["log_every_n_steps"],
        callbacks=[early_stopping_callback],
    )

    trainer.fit(model, train_dataloader, test_dataloader)

    test_results = trainer.test(model, test_dataloader)

    final_test_metrics = test_results[0] if test_results else {}

    all_metrics = final_test_metrics

    return trainer, model, all_metrics


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    L.seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@rank_zero_only
def rank0_print(*args, **kwargs):
    print(*args, **kwargs)

@rank_zero_only
def rank0_print_json(*args, **kwargs):
    print(json.dumps(*args, **kwargs))


def main():
    set_seed(config["train"]["seed"])
    processed_dataloaders = load_dataset()
    train_dataloader, test_dataloader = processed_dataloaders["train"], processed_dataloaders["test"]
    baseline_results = baselines(test_dataloader)
    rank0_print(
        "The results of random classifier and optimistic classifier (predict all samples as positive) are:"
    )
    rank0_print(baseline_results)
    rank0_print(
        "If the F1 score of the optimistic classifier is too good, there may be a severe imbalance in the dataset."
    )
    model = SpeechClassifier(config)
    trainer, model, results = train_and_eval_model(config['train'], model, train_dataloader, test_dataloader)

    rank0_print_json(results)


if __name__ == "__main__":
    main()
