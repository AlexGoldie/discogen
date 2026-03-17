import flax.linen as nn
from typing import Callable

def get_activation(config) -> Callable:
    if config.get("CONTINUOUS"):
        return nn.tanh
    else:
        return nn.relu
