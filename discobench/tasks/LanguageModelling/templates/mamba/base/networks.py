from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the Mamba 2 language model


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MambaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.mamba = Mamba2(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
        )

    def forward(self, x):
        # Compute residual connection in fp32 for numerical stability
        hidden = self.mamba(self.norm(x))
        return (x.float() + hidden.float()).to(x.dtype)


# -----------------------------------------------------------------------------
# The main Mamba 2 model


@dataclass
class ModelConfig:
    vocab_size: int = 50304
    d_model: int = 768
    n_layer: int = 24
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2


class Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([MambaBlock(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        # idx: [B, T] token indices
        x = self.embedding(idx)  # [B, T, d_model]
        x = F.rms_norm(x, (x.size(-1),))

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]
        logits = logits.float()  # use tf32/fp32 for logits

        return logits

    def get_config(self):
        return self.config
