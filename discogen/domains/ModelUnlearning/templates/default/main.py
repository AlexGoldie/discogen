import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import os
os.environ["TQDM_DISABLE"] = "1"

import subprocess
import shutil
import json
import logging
from huggingface_hub import login as hf_login
from make_dataset import make_dataset

# The line below adds the src directory to the Python path for convenience.
# Make sure no folders/files in src/ have the same name as those in this root directory.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluators
from trainer.utils import seed_everything

# Copy dataset_config and trainer_config to configs/ before Hydra loads the main config
_base_dir = Path(__file__).resolve().parent
shutil.copy2(_base_dir / "main_config.yaml", _base_dir / "configs" / "main_config.yaml")
shutil.copy2(_base_dir / "trainer_config.yaml", _base_dir / "configs/trainer" / "custom.yaml")
shutil.copy2(_base_dir / "model_config.yaml", _base_dir / "configs/model" / "model_conf.yaml")


@hydra.main(version_base=None, config_path="configs", config_name="main_config.yaml")
def unlearn(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "unlearn")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    # Load Dataset
    data_cfg = cfg.data
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )

    # Load collator
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # This sets the trainer to CustomUnlearnTrainer, as defined in the discovered loss.py file.
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    # Get Evaluators
    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs:
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        tokenizer=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=template_args,
    )

    if trainer_args.do_train:
        # trainer.train() calls the compute_loss function defined in the discovered loss.py file.
        # It overrides the default compute_loss from Transformers.Trainer.
        trainer.train()
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")

    # Final json dump
    print('Dumping final evaluation results...')
    checkpoint_folders = [d for d in os.listdir(trainer_args.output_dir) if d.startswith("checkpoint")]
    latest = max(checkpoint_folders, key=lambda x: int(x.split('-')[-1]))
    eval_dir = os.path.join(trainer_args.output_dir, latest, "evals")

    for _, evaluator in evaluators.items():
        summary_file = evaluator.get_logs_file_path(eval_dir, suffix="SUMMARY")
        with open(summary_file, 'r') as f:
            print(json.dumps(json.load(f), indent=2))


if __name__ == "__main__":
    # Login to Hugging Face
    print('Logging in to Hugging Face...')
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        hf_login(hf_token)
    else:
        print("HF_TOKEN environment variable not set. Aborting.")
        exit(1)

    # Make dataset
    print('Creating dataset...')
    make_dataset()

    # Unlearn
    print('Begin unlearning...')
    unlearn()
