from functools import partial
from typing import Any, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast

from einops import rearrange

from helpers import default, identity, ModelPrediction, extract
from norm import unnormalize_to_zero_to_one, normalize_to_neg_one_to_one
from schedule import beta_schedule
from loss import compute_loss


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
        timesteps = config.get("timesteps", 1000)
        sampling_timesteps = config.get("sampling_timesteps", 50)
        ddim_sampling_eta = config.get("ddim_sampling_eta", 1.0)
        offset_noise_strength = config.get("offset_noise_strength", 0.0)
        min_snr_loss_weight = config.get("min_snr_loss_weight", False)
        min_snr_gamma = config.get("min_snr_gamma", 5)

        assert not (type(self) is Diffusion and backbone.channels != backbone.out_dim)

        self.model = backbone
        self.channels = self.model.channels
        self.objective = None

        self.image_size = image_size

        betas = beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = sampling_timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer(
            "posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        # offset noise strength - 0.1 was claimed ideal

        self.offset_noise_strength = offset_noise_strength

        # loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        loss_weight = maybe_clipped_snr / snr

        register_buffer("loss_weight", loss_weight)

    @property
    def device(self) -> torch.device:
        """
        Return the device on which diffusion buffers are stored.

        Returns:
            torch.device: The device associated with the diffusion process.
        """
        return self.betas.device

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Predict the original clean sample from a noisy sample and predicted noise.

        Args:
            x_t (torch.Tensor): Noisy input tensor at timestep `t`.
            t (torch.Tensor): Timestep tensor of shape `(B,)`.
            noise (torch.Tensor): Predicted noise tensor of the same shape as `x_t`.

        Returns:
            torch.Tensor: Predicted clean sample `x_0`.
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the posterior distribution.

        Args:
            x_start (torch.Tensor): Predicted clean sample `x_0`.
            x_t (torch.Tensor): Noisy sample at timestep `t`.
            t (torch.Tensor): Timestep tensor of shape `(B,)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - posterior mean
                - posterior variance
                - posterior log variance (clipped for numerical stability)
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        classes: torch.Tensor,
        cond_scale: float = 6.0,
        rescaled_phi: float = 0.7,
        clip_x_start: bool = False,
    ) -> ModelPrediction:
        """
        Run the backbone model and derive noise and clean-sample predictions.

        Args:
            x (torch.Tensor): Noisy input tensor at timestep `t`.
            t (torch.Tensor): Timestep tensor of shape `(B,)`.
            classes (torch.Tensor): Class conditioning tensor.
            cond_scale (float, optional): Guidance scale for classifier-free guidance.
            rescaled_phi (float, optional): Rescaling interpolation factor.
            clip_x_start (bool, optional): Whether to clip predicted `x_0`
                to the valid range [-1, 1].

        Returns:
            ModelPrediction: A named tuple containing:
                - pred_noise: The predicted noise tensor.
                - pred_x_start: The predicted clean sample tensor.
        """
        model_output, model_output_null = self.model.forward_with_cond_scale(
            x, t, classes, cond_scale=cond_scale, rescaled_phi=rescaled_phi
        )
        maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity

        pred_noise = model_output

        x_start = self.predict_start_from_noise(x, t, model_output)
        x_start = maybe_clip(x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        classes: torch.Tensor,
        cond_scale: float,
        rescaled_phi: float,
        clip_denoised: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the mean and variance of the reverse diffusion step.

        Args:
            x (torch.Tensor): Noisy input tensor at timestep `t`.
            t (torch.Tensor): Timestep tensor of shape `(B,)`.
            classes (torch.Tensor): Class conditioning tensor.
            cond_scale (float): Guidance scale.
            rescaled_phi (float): Rescaling interpolation factor.
            clip_denoised (bool, optional): Whether to clip predicted `x_0`
                before computing the posterior.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - posterior mean
                - posterior variance
                - posterior log variance
                - predicted clean sample `x_0`
        """
        preds = self.model_predictions(x, t, classes, cond_scale, rescaled_phi)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: int,
        classes: torch.Tensor,
        cond_scale: float = 6.0,
        rescaled_phi: float = 0.7,
        clip_denoised: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a single reverse diffusion step.

        Args:
            x (torch.Tensor): Noisy input tensor at timestep `t`.
            t (int): Current diffusion timestep.
            classes (torch.Tensor): Class conditioning tensor.
            cond_scale (float, optional): Guidance scale.
            rescaled_phi (float, optional): Rescaling interpolation factor.
            clip_denoised (bool, optional): Whether to clip predicted `x_0`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Sampled image at timestep `t-1`
                - Predicted clean sample `x_0`
        """
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=batched_times,
            classes=classes,
            cond_scale=cond_scale,
            rescaled_phi=rescaled_phi,
            clip_denoised=clip_denoised,
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def ddim_sample(
        self,
        classes: torch.Tensor,
        shape: tuple,
        cond_scale: float = 6.0,
        rescaled_phi: float = 0.7,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples using DDIM sampling.

        Args:
            classes (torch.Tensor): Class conditioning tensor.
            shape (tuple): Desired output shape `(B, C, H, W)`.
            cond_scale (float, optional): Guidance scale.
            rescaled_phi (float, optional): Rescaling interpolation factor.
            clip_denoised (bool, optional): Whether to clip predicted `x_0`.

        Returns:
            torch.Tensor: Generated samples in the range [0, 1].
        """
        batch, device, total_timesteps, sampling_timesteps, eta = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
        )

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(
                img, time_cond, classes, cond_scale=cond_scale, rescaled_phi=rescaled_phi, clip_x_start=clip_denoised
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        img = unnormalize_to_zero_to_one(img)
        return img

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
        batch_size, image_size, channels = classes.shape[0], self.image_size, self.channels
        return self.ddim_sample(classes, (batch_size, channels, image_size, image_size), cond_scale, rescaled_phi)

    @torch.no_grad()
    def interpolate(
        self, x1: torch.Tensor, x2: torch.Tensor, classes: torch.Tensor, t: Optional[int] = None, lam: float = 0.5
    ) -> torch.Tensor:
        """
        Interpolate between two images in diffusion latent space.

        Args:
            x1 (torch.Tensor): First input image.
            x2 (torch.Tensor): Second input image.
            classes (torch.Tensor): Class conditioning tensor.
            t (int, optional): Timestep at which to interpolate. Defaults
                to the final timestep.
            lam (float, optional): Interpolation coefficient between x1 and x2.

        Returns:
            torch.Tensor: Interpolated image.
        """
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        for i in reversed(range(0, t)):
            img, _ = self.p_sample(img, i, classes)

        return img

    @autocast("cuda", enabled=False)
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample from the forward diffusion process.

        Args:
            x_start (torch.Tensor): Clean input image.
            t (torch.Tensor): Timestep tensor of shape `(B,)`.
            noise (torch.Tensor, optional): Noise tensor to add. If None,
                standard Gaussian noise is used.

        Returns:
            torch.Tensor: Noisy image at timestep `t`.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.offset_noise_strength > 0.0:
            offset_noise = torch.randn(x_start.shape[:2], device=self.device)
            noise += self.offset_noise_strength * rearrange(offset_noise, "b c -> b c 1 1")

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(
        self, x_start: torch.Tensor, t: torch.Tensor, *, classes: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the training loss for a batch of images.

        Args:
            x_start (torch.Tensor): Clean input images.
            t (torch.Tensor): Timestep tensor of shape `(B,)`.
            classes (torch.Tensor): Class conditioning tensor.
            noise (torch.Tensor, optional): Noise tensor. If None, random
                Gaussian noise is used.

        Returns:
            torch.Tensor: Scalar loss value averaged over the batch.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_out = self.model(x, t, classes)

        target = noise

        loss = compute_loss(model_out, target)

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

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
        b, c, h, w, device, img_size = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f"height and width of image must be {img_size}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)
