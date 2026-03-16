
import glob
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from networks import Model, ModelConfig
from torch import nn
from make_dataset import load_dataset
from dataloader import DistributedDataLoader

@dataclass
class Hyperparameters:
    # optimization hyperparams
    batch_size: int = 8 * 64  # batch size, in sequences, across all devices
    device_batch_size: int = 16  # batch size, in sequences, per device
    val_tokens: int = (
        10485760  # number of validation tokens - fixed for consistent evaluation
    )
    sequence_length: int = 1024  # sequence length, in tokens

def evaluate_model():

    args = Hyperparameters()

    # Load dataset patterns
    dataset_patterns = load_dataset()
    val_bin_pattern = dataset_patterns["val"]

    args.batch_size = args.device_batch_size * torch.cuda.device_count()

    # set up DDP (distributed data parallel). torchrun sets this env variable
    assert torch.cuda.is_available()
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)

    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.

    # convenience variables
    B, T = args.device_batch_size, args.sequence_length
    # calculate the number of steps to take in the val loop.
    assert args.val_tokens % (B * T * ddp_world_size) == 0
    val_steps = 800  # hardcoded number of validation steps
    num_dataset_passes = val_steps // (args.val_tokens // (B * T * ddp_world_size))


    val_loader = DistributedDataLoader(val_bin_pattern, B, T, ddp_rank, ddp_world_size)

    # Load the saved config and weights
    try:
        import __main__

        setattr(__main__, "ModelConfig", ModelConfig)
        model_config = pickle.load(open("model_config.pt", "rb"))
        state_dict = torch.load("model.pt", weights_only=True)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace("module._orig_mod.", "")] = v
        state_dict = new_state_dict
    except:
        raise Exception("Model not found. Please run the training script first.")


    # Reconstruct the model using the saved configuration
    model = Model(model_config)  # Use the saved config to initialize the model
    model.load_state_dict(state_dict)
    model = model.cuda()

    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # stop the clock
    t0 = time.time()
    # run validation batches
    model.eval()
    val_loader.reset()
    val_loss = 0.0
    total_correct = torch.tensor(0.0, device=device) # Track correct predictions
    total_samples = torch.tensor(0.0, device=device) # Track total predictions made
    criterion = nn.CrossEntropyLoss()
    torch.cuda.empty_cache()

    # set a random seed
    torch.manual_seed(42)

    with torch.no_grad():
        for i in range(val_steps):
            # print(f"Processing batch {i} of {val_steps}")
            x_val, y_val = val_loader.next_batch()

            # The masking and shuffling below prevents the model from cheating by looking ahead in the sequence.
            # Shuffling: Each row is a continuation of the previous row (see dataloader)
            perm = torch.randperm(x_val.size(0))
            x_val = x_val[perm]
            y_val = y_val[perm]

            # Generate random masking indices for each row (between 1 and sequence length-1)
            mask_indices = torch.randint(
                1, x_val.size(1), (x_val.size(0),), device=x_val.device
            )

            # Create mask for each row
            mask = torch.arange(x_val.size(1), device=x_val.device).unsqueeze(
                0
            ) >= mask_indices.unsqueeze(1)

            # Apply mask to input
            x_val_masked = x_val.clone()
            x_val_masked[mask] = 0

            with ctx:
                outputs = model(x_val_masked)

                # Get predictions only at masking positions
                batch_indices = torch.arange(outputs.size(0), device=outputs.device)
                masked_outputs = outputs[
                    batch_indices, mask_indices - 1
                ]  # -1 because we want to predict the first masked token
                masked_targets = y_val[batch_indices, mask_indices - 1]

                loss = criterion(masked_outputs, masked_targets)
                val_loss += loss.detach()

                predictions = torch.argmax(masked_outputs, dim=-1)

                # Count how many match the target
                correct = (predictions == masked_targets).float().sum()

                # Update accumulators
                total_correct += correct
                total_samples += masked_targets.size(0) # Adds batch size
                del loss
    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
    val_loss /= val_steps

    dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    val_accuracy = total_correct / total_samples

    perplexity = torch.exp(val_loss)

    # log val loss to console and to logfile
    if master_process:
        # Create and print the result dictionary
        result = {
                    "val_loss": val_loss.item(),
                    "val_accuracy": val_accuracy.item(),
                    "perplexity": perplexity.item()
                }
        print(json.dumps(result))
