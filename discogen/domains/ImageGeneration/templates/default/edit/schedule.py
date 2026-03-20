import math

import torch


def beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Generate a beta schedule for diffusion timesteps.

    Args:
        timesteps (int): Total number of diffusion steps.
        s (float, optional): Small offset to prevent numerical instability
            near zero and adjust the curvature of the cosine schedule.
            Defaults to 0.008.

    Returns:
        torch.Tensor: A 1D tensor of shape `(timesteps,)` containing beta
        values clipped to the range [0, 0.999], representing the noise
        variance added at each timestep.
    """
    betas = ...
    return betas
