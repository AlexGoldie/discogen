from typing import Any

import torch
from torch import nn


class Diffusion(nn.Module):
    def __init__(self, backbone: nn.Module, image_size: int, config: dict[str, Any]):
        """
        Initialize the diffusion model and precompute diffusion schedules.

        Args:
            backbone (nn.Module): The neural network used to predict noise or
                denoised signals. Must expose attributes such as `channels`
                and `out_dim`, and implement the required forward methods.
            image_size (int): Spatial size (height and width) of the input images.
            config (dict): Configuration dictionary controlling diffusion behavior.
                Common keys include:
                    - "timesteps" (int): Number of diffusion steps.
                    - "sampling_timesteps" (int): Number of steps used during sampling.
                    - "ddim_sampling_eta" (float): Stochasticity parameter for DDIM.
                    - "offset_noise_strength" (float): Strength of offset noise.
                    - "min_snr_loss_weight" (bool): Whether to clip SNR for loss weighting.
                    - "min_snr_gamma" (float): Maximum SNR value if clipping is enabled.
        """
        super().__init__()
        ...

    @torch.no_grad()
    def sample(self, classes: torch.Tensor, cond_scale: float = 6.0, rescaled_phi: float = 0.7) -> torch.Tensor:
        """
        Generate samples at the configured image size.

        Args:
            classes (torch.Tensor): Class conditioning tensor.
            cond_scale (float, optional): Guidance scale.
            rescaled_phi (float, optional): Rescaling interpolation factor.

        Returns:
            torch.Tensor: Generated images of shape
            `(B, C, image_size, image_size)`.
        """
        img = ...
        return img

    def forward(self, img: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass.

        Args:
            img (torch.Tensor): Input images of shape `(B, C, H, W)`.
            *args: Additional positional arguments forwarded to `p_losses`.
            **kwargs: Additional keyword arguments forwarded to `p_losses`.

        Returns:
            torch.Tensor: Scalar training loss.
        """
        loss = ...
        return loss
