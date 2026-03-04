from scipy.stats import qmc
from typing import Any, Callable

def build_sampler(obs_dim: int, seed: int, config: dict[str, Any]) -> Callable:
    """
    Builds the sampler for the parameter space.
    Args:
        obs_dim: The dimension of the parameter space.
        seed: The seed for the sampler.
        config: The configuration dictionary. [Must contain 'sampler_type']
    Returns:
        The sampler. Must have a random method that takes a number of samples to draw and returns a numpy array of samples. The samples should be in [0, 1) space.
    """
    return qmc.Sobol(d=obs_dim, scramble=True, seed=seed)
