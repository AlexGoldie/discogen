import os
import pickle
import sys
import time
import uuid
from dataclasses import dataclass

import torch
import torch._dynamo
import torch._inductor.config as config
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

torch._dynamo.config.suppress_errors = True

import logging

logging.getLogger("torch._dynamo").setLevel(
    logging.ERROR
)  # Only show errors, not warnings

from dataloader import DistributedDataLoader
from loss import compute_loss
from make_dataset import load_dataset
from networks import Model, ModelConfig
from optim import OptimizerConfig, create_optimizers

# -----------------------------------------------------------------------------
# int main


@dataclass
class Hyperparameters:
    # These are general hyperparameters relating to either the data or the training process.
    # Hyperparameters for the model (networks.py) and optimizer/schedulers (optim.py) can be found in the respective files.
    batch_size: int = 8 * 64  # batch size, in sequences, across all devices
    device_batch_size: int = 16  # batch size, in sequences, per device
    sequence_length: int = 1024  # sequence length, in tokens
    num_iterations: int = 500  # number of iterations to run
    log_frequency: int = 50  # number of iterations to run


def main():
    with open(sys.argv[0]) as f:
        code = f.read()  # read the code of this file ASAP, for logging
    args = Hyperparameters()

    # Load dataset patterns
    dataset_patterns = load_dataset()
    train_bin_pattern = dataset_patterns["train"]

    args.batch_size = args.device_batch_size * torch.cuda.device_count()

    # set up DDP (distributed data parallel). torchrun sets this env variable
    assert torch.cuda.is_available()
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    print(f"using device: {device}")
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.

    # convenience variables
    B, T = args.device_batch_size, args.sequence_length
    # calculate the steps of gradient accumulation required to attain the desired global batch size.
    assert args.batch_size % (B * ddp_world_size) == 0, (
        f"{args.batch_size=}, {B=}, {ddp_world_size=}"
    )
    train_accumulation_steps = args.batch_size // (B * ddp_world_size)

    # load tokens
    train_loader = DistributedDataLoader(
        train_bin_pattern, B, T, ddp_rank, ddp_world_size
    )
    if master_process:
        print(
            f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files"
        )
    x, y = train_loader.next_batch()

    # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
    # this originates from Karpathy's experiments.
    num_vocab = 50304
    model = Model(ModelConfig(vocab_size=num_vocab))
    model = model.cuda()
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True  # suggested by @Chillee
    model = torch.compile(model)
    # here we wrap model into DDP container
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module  # always contains the "raw" unwrapped model
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    optimizers, schedulers = create_optimizers(raw_model, OptimizerConfig())

    # begin logging
    if master_process:
        run_id = str(uuid.uuid4())
        logdir = "logs/%s/" % run_id
        os.makedirs(logdir, exist_ok=True)
        logfile = "logs/%s.txt" % run_id
        # create the log file
        with open(logfile, "w") as f:
            # begin the log by printing this file (the Python code)
            f.write("=" * 100 + "\n")
            f.write(code)
            f.write("=" * 100 + "\n")
            # log information about the hardware/software environment this is running on
            # and print the full `nvidia-smi` to file
            f.write(
                f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n"
            )
            import subprocess

            result = subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            f.write(f"{result.stdout}\n")
            f.write("=" * 100 + "\n")

    training_time_ms = 0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.time()
    # begin training
    train_loader.reset()
    for step in range(args.num_iterations + 1):
        last_step = step == args.num_iterations
        # This effectively ignores timing first 10 steps, which are slower for weird reasons.
        # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
        # steps with dummy data first, and then re-initialize the model and reset the loader.
        if step == 10:
            training_time_ms = 0
            t0 = time.time()
        timed_steps = (
            float("nan") if step <= 11 else (step - 10) + 1
        )  # <= 11 to avoid bug in val

        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        for i in range(1, train_accumulation_steps + 1):
            # forward pass
            with ctx:
                logits = model(x)
                loss = compute_loss(logits, y)
                train_loss = loss.detach()
                logits = None
            # advance the dataset for the next batch
            x, y = train_loader.next_batch()
            # backward pass
            if i < train_accumulation_steps:
                with (
                    model.no_sync()
                ):  # there's no need to sync gradients every accumulation step
                    loss.backward()
            else:
                loss.backward()  # just sync on the last step
            del loss
        for p in model.parameters():
            p.grad /= train_accumulation_steps
        # step the optimizers and schedulers
        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()
        # null the gradients
        model.zero_grad(set_to_none=True)
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        # dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
        if master_process and (step+1)%args.log_frequency == 0:
            approx_time = training_time_ms + 1000 * (time.time() - t0)
            print(
                f"step:{step + 1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time / timed_steps:.2f}ms"
            )
            with open(logfile, "a") as f:
                f.write(
                    f"step:{step + 1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time / timed_steps:.2f}ms\n"
                )

    torch.save(model.state_dict(), "model.pt")
    with open("model_config.pt", "wb") as f:
        pickle.dump(raw_model.get_config(), f)
    f.close()

    # -------------------------------------------------------------------------
    # clean up nice
    dist.destroy_process_group()
