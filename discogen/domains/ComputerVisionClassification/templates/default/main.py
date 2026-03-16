from typing import Dict
from preprocess import build_transforms
from make_dataset import load_dataset
from loss import compute_loss
from networks import Model
from optim import create_optimizer
from config import config
from sklearn.metrics import accuracy_score
from transformers import DefaultDataCollator, Trainer, TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback
import random
import numpy as np
import pandas as pd
import torch
import json

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}


def load_and_preprocess_data(config: Dict):

    dataset = load_dataset()
    train_dataset, test_dataset = dataset["train"], dataset["test"]
    train_transforms, eval_transforms = build_transforms(config)
    train_dataset.set_transform(
        lambda batch: {
            "pixel_values": [train_transforms(img) for img in batch[config["image_key"]]],
            "labels": batch[config["label_key"]],
        }
    )
    test_dataset.set_transform(
        lambda batch: {
            "pixel_values": [eval_transforms(img) for img in batch[config["image_key"]]],
            "labels": batch[config["label_key"]],
        }
    )

    return train_dataset, test_dataset


def train_and_eval_model(config, model, optimizer, train_dataset, test_dataset):

    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['num_train_epochs'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        per_device_eval_batch_size=config['per_device_eval_batch_size'],
        weight_decay=config['weight_decay'],
        logging_strategy=config['logging_strategy'],
        save_strategy=config['save_strategy'],
        load_best_model_at_end=config['load_best_model_at_end'],
        disable_tqdm=config['disable_tqdm'],
        remove_unused_columns=config['remove_unused_columns'],
        eval_strategy=config['eval_strategy'],
        metric_for_best_model=config['metric_for_best_model'],
        greater_is_better=config['greater_is_better'],
        lr_scheduler_type=config['lr_scheduler_type'],
        warmup_ratio=config['warmup_ratio'],
        max_grad_norm=config['max_grad_norm']
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        compute_loss_func=compute_loss,
        data_collator=DefaultDataCollator(),
        optimizers=(optimizer, None),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config["early_stopping_patience"],
                early_stopping_threshold=config["early_stopping_threshold"],
            )
        ],
    )

    trainer.train()
    print("Training completed. Evaluating the model...")
    train_metrics = trainer.evaluate(eval_dataset=train_dataset, metric_key_prefix="train")
    eval_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="eval")
    print(json.dumps({"train_metrics":train_metrics, "eval_metrics":eval_metrics}))

def main():
    set_seed(config["train"]["seed"])
    train_dataset, test_dataset = load_and_preprocess_data(config['dataset'])
    model = Model(config['model'])
    optimizer = create_optimizer(model, config['optimizer'])
    train_and_eval_model(config['train'], model, optimizer, train_dataset, test_dataset)


if __name__ == "__main__":
    main()
