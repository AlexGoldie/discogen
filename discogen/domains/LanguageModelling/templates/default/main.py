import os

import torch
import torch.multiprocessing as mp

# Import the main training function from your train.py
from train import main as train_worker_main
from evaluate import evaluate_model

def train_worker_launcher(rank, world_size):
    """
    This is the wrapper function that mp.spawn will call.
    It sets the environment variables that train.py expects
    and then calls the main training function.
    """

    # --- 1. Set the environment variables ---
    # These are the *exact* variables that torchrun would have set.
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # A free port
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank) # In this setup, RANK and LOCAL_RANK are the same
    os.environ['WORLD_SIZE'] = str(world_size)

    # --- 2. Run the actual training logic ---
    train_worker_main()

def eval_worker_launcher(rank, world_size):
    """
    This is the wrapper function that mp.spawn will call.
    It sets the environment variables that train.py expects
    and then calls the main training function.
    """

    # --- 1. Set the environment variables ---
    # These are the *exact* variables that torchrun would have set.
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # A free port
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank) # In this setup, RANK and LOCAL_RANK are the same
    os.environ['WORLD_SIZE'] = str(world_size)

    # --- 2. Run the actual training logic ---
    evaluate_model()

if __name__ == "__main__":
    # Get the number of available GPUs
    world_size = torch.cuda.device_count()

    if world_size == 0:
        print("No GPUs found. This script requires GPUs to run.")
    else:
        print(f"Found {world_size} GPUs. Spawning worker processes...")

        # --- 3. Launch the workers ---
        # mp.spawn will create `world_size` processes.
        # It calls `worker_launcher` for each process, passing in:
        # - The process's rank (0, 1, 2, ...)
        # - The `args` tuple (in this case, just world_size)
        mp.spawn(
            train_worker_launcher,
            args=(world_size,),
            nprocs=world_size,
            join=True  # Wait for all processes to finish
        )

        mp.spawn(
            eval_worker_launcher,
            args=(world_size,),
            nprocs=world_size,
            join=True  # Wait for all processes to finish
        )
