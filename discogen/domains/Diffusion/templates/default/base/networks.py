from typing import Optional, Any
from functools import partial

import torch
from torch import nn, einsum
from einops import rearrange, repeat

from helpers import (
    default,
    Residual,
    PreNormalization,
    Downsample,
    Upsample,
    prob_mask_like,
    project,
    Normalize,
    exists,
)
from embedding import PositionEmbedding


class Block(nn.Module):
    """
    Apply a convolutional transformation followed by normalization and activation.
    """

    def __init__(self, dim: int, dim_out: int):
        """
        Initialize the convolutional block.

        Args:
            dim (int): Number of input channels.
            dim_out (int): Number of output channels produced by the convolution.
        """
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = Normalize(dim_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift: Optional[tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        Apply the block's operations to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape `(B, C, H, W)`.
            scale_shift (tuple[torch.Tensor, torch.Tensor], optional):
                A tuple `(scale, shift)` used to modulate the normalized
                activations. Each tensor should be broadcastable to the
                shape of `x`.

        Returns:
            torch.Tensor: The transformed output tensor of shape
            `(B, dim_out, H, W)`.
        """
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """
    Residual convolutional block with optional conditioning.
    """

    def __init__(
        self, dim: int, dim_out: int, *, time_emb_dim: Optional[int] = None, classes_emb_dim: Optional[int] = None
    ):
        """
        Initialize the residual block.

        Args:
            dim (int): Number of input channels.
            dim_out (int): Number of output channels.
            time_emb_dim (int, optional): Dimensionality of the time embedding
                used for conditioning.
            classes_emb_dim (int, optional): Dimensionality of the class
                embedding used for conditioning.
        """

        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2))
            if exists(time_emb_dim) or exists(classes_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(
        self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None, class_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply the residual block to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape `(B, C, H, W)`.
            time_emb (torch.Tensor, optional): Time-step embedding of shape
                `(B, time_emb_dim)` used for conditioning.
            class_emb (torch.Tensor, optional): Class embedding of shape
                `(B, classes_emb_dim)` used for conditioning.

        Returns:
            torch.Tensor: Output tensor of shape `(B, dim_out, H, W)` after
            applying the residual connection.
        """
        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, "b c -> b c 1 1")
            scale_shift = cond_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    """
    Apply linear self-attention over spatial feature maps.
    """

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        """
        Initialize the linear attention module.

        Args:
            dim (int): Number of input and output channels.
            heads (int, optional): Number of attention heads.
            dim_head (int, optional): Dimensionality of each attention head.
        """
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), Normalize(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute linear self-attention over the input feature map.

        Args:
            x (torch.Tensor): Input tensor of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor: Output tensor of shape `(B, C, H, W)` after applying linear attention.
        """
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    """
    Full self-attention module operating over spatial feature maps.
    """

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        """
        Initialize the attention module.

        Args:
            dim (int): Number of input and output feature channels.
            heads (int, optional): Number of attention heads.
            dim_head (int, optional): Dimensionality of each attention head.
                The total hidden dimension is `heads * dim_head`.
        """
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head self-attention over spatial dimensions.

        Args:
            x (torch.Tensor): Input tensor of shape
                `(batch_size, channels, height, width)`.

        Returns:
            torch.Tensor: Output tensor of shape
                `(batch_size, channels, height, width)` with spatial
                self-attention applied.
        """
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)

        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)

        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class Backbone(nn.Module):
    """
    Diffusion backbone with time and class conditioning for conditional multi-channel image generation.
    """

    def __init__(self, dim: int, num_classes: int, config: dict[str, Any]):
        """
        Initialize the backbone network.

        Args:
            dim (int): Base feature dimension used throughout the network.
            num_classes (int): Number of discrete classes for class conditioning.
            config (dict): Configuration dictionary controlling diffusion behavior.
                Common keys include:
                    - "cond_drop_prob" (float): Probability of dropping class conditioning during training.
                    - "init_dim" (int): Number of channels after the initial convolution.
                    - "out_dim" (int): Number of output channels.
                    - "dim_mults" (tuple[int]): Multipliers for feature dimensions at each resolution level.
                    - "channels" (int): Number of input image channels.
                    - "learned_variance" (bool): Whether the model predicts both mean and variance, doubling the output channels.
                    - "attn_dim_head" (int): Dimensionality of each attention head.
                    - "attn_heads" (int): Number of attention heads.
        """
        super().__init__()

        cond_drop_prob = config.get("cond_drop_prob", 0.5)
        init_dim = config.get("init_dim", None)
        out_dim = config.get("out_dim", None)
        dim_mults = config.get("dim_mults", (1, 2, 4, 8))
        channels = config.get("channels", 3)
        learned_variance = config.get("learned_variance", False)
        attn_dim_head = config.get("attn_dim_head", 32)
        attn_heads = config.get("attn_heads", 4)

        self.cond_drop_prob = cond_drop_prob

        # determine dimensions

        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        sinu_pos_emb = PositionEmbedding(dim)
        fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb, nn.Linear(fourier_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
        )

        # class embeddings

        self.classes_emb = nn.Embedding(num_classes, dim)
        self.null_classes_emb = nn.Parameter(torch.randn(dim))

        classes_dim = dim * 4

        self.classes_mlp = nn.Sequential(nn.Linear(dim, classes_dim), nn.GELU(), nn.Linear(classes_dim, classes_dim))

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList([
                    ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                    ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                    Residual(PreNormalization(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                ])
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)
        self.mid_attn = Residual(
            PreNormalization(mid_dim, Attention(mid_dim, dim_head=attn_dim_head, heads=attn_heads))
        )
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList([
                    ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                    ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                    Residual(PreNormalization(dim_out, LinearAttention(dim_out))),
                    Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                ])
            )

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ResnetBlock(init_dim * 2, init_dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    def forward_with_cond_scale(
        self,
        *args: Any,
        cond_scale: float = 1.0,
        rescaled_phi: float = 0.0,
        remove_parallel_component: bool = True,
        keep_parallel_frac: float = 0.0,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with classifier-free guidance scaling.

        Args:
            *args: Positional arguments forwarded to `forward`.
            cond_scale (float, optional): Guidance scale factor. A value of 1.0
                disables guidance. Larger values increase conditioning strength.
            rescaled_phi (float, optional): Interpolation factor between standard
                guided output and a variance-rescaled version. Defaults to 0.0.
            remove_parallel_component (bool, optional): Whether to remove the
                component of the guidance update parallel to the conditioned
                prediction. Defaults to True.
            keep_parallel_frac (float, optional): Fraction of the parallel
                component to keep if removed. Defaults to 0.0.
            **kwargs: Keyword arguments forwarded to `forward`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor] or torch.Tensor:
                - If `cond_scale == 1`, returns the conditioned output only.
                - Otherwise, returns a tuple of
                    `(guided_output, null_condition_output)`.
        """
        logits = self.forward(*args, cond_drop_prob=0.0, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob=1.0, **kwargs)
        update = logits - null_logits

        if remove_parallel_component:
            parallel, orthog = project(update, logits)
            update = orthog + parallel * keep_parallel_frac

        scaled_logits = logits + update * (cond_scale - 1.0)

        if rescaled_phi == 0.0:
            return scaled_logits, null_logits

        std_fn = partial(torch.std, dim=tuple(range(1, scaled_logits.ndim)), keepdim=True)
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))
        interpolated_rescaled_logits = rescaled_logits * rescaled_phi + scaled_logits * (1.0 - rescaled_phi)

        return interpolated_rescaled_logits, null_logits

    def forward(
        self, x: torch.Tensor, time: torch.Tensor, classes: torch.Tensor, cond_drop_prob: Optional[float] = None
    ) -> torch.Tensor:
        """
        Perform a forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape `(B, C, H, W)`, typically
                an image or latent representation.
            time (torch.Tensor): Time-step tensor of shape `(B,)` or `(B, D)`
                used to generate time embeddings.
            classes (torch.Tensor): Class indices tensor of shape `(B,)`.
            cond_drop_prob (float, optional): Probability of dropping class
                conditioning for this forward pass. Defaults to the value
                specified at initialization.

        Returns:
            torch.Tensor: Output tensor of shape `(B, out_dim, H, W)` representing
            the model prediction (e.g., noise, image residual, or distribution
            parameters).
        """
        batch, device = x.shape[0], x.device

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # derive condition, with condition dropout for classifier free guidance

        classes_emb = self.classes_emb(classes)

        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)
            null_classes_emb = repeat(self.null_classes_emb, "d -> b d", b=batch)

            classes_emb = torch.where(rearrange(keep_mask, "b -> b 1"), classes_emb, null_classes_emb)

        c = self.classes_mlp(classes_emb)

        # unet

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t, c)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t, c)
        return self.final_conv(x)
