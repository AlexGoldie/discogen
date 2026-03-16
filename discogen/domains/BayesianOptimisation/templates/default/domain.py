from typing import Sequence, Callable
import jax.numpy as jnp

class Domain:
    def __init__(self, dtype: jnp.dtype):
        self.dtype = dtype

    def transform(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def inverse_transform(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def sample(self, key: jnp.ndarray, shape: Sequence[int]) -> jnp.ndarray:
        raise NotImplementedError


class Real(Domain):
    """
    Continuous real-valued domain with clipping.
    """
    def __init__(self, lower: float | int, upper: float | int) -> None:
        assert isinstance(lower, float) or isinstance(lower, int), "Lower bound must be a float"
        assert isinstance(upper, float) or isinstance(lower, int), "Upper bound must be a float"
        assert lower < upper, "Lower bound must be less than upper bound"
        self.lower = float(lower)
        self.upper = float(upper)
        super().__init__(dtype=jnp.float32)

    def transform(self, x: jnp.ndarray) -> jnp.ndarray:
        return (jnp.clip(x, self.lower, self.upper) - self.lower) / (self.upper - self.lower)

    def inverse_transform(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(x * (self.upper - self.lower) + self.lower, self.lower, self.upper)


class Integer(Domain):
    """
    Discrete integer-valued domain with rounding and clipping.
    """
    def __init__(self, lower: int, upper: int) -> None:
        assert isinstance(lower, int), "Lower bound must be an integer"
        assert isinstance(upper, int), "Upper bound must be an integer"
        assert lower < upper, "Lower bound must be less than upper bound"
        self.lower = int(lower)
        self.upper = int(upper)
        super().__init__(dtype=jnp.int32)

    def transform(self, x: jnp.ndarray) -> jnp.ndarray:
        return (jnp.clip(jnp.round(x), self.lower, self.upper).astype(jnp.float32) - self.lower) / (self.upper - self.lower)

    def inverse_transform(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(jnp.round(x * (self.upper - self.lower) + self.lower), self.lower, self.upper).astype(jnp.int32)


class Ordinal(Domain):
    """
    Ordinal categorical-valued domain.
    Maps ordinal values (ints or floats) to equally spaced floats in [0, 1], assuming equal distance between ordinal categories.
    """
    def __init__(self, values: Sequence[float | int]) -> None:
        assert isinstance(values, Sequence), "Values must be a sequence"
        assert len(values) > 0, "Values must be non-empty"
        self.values = jnp.array(list(values))
        self.upper = self.values.shape[0] - 1
        super().__init__(dtype=type(values[0]))

    def _indices_from_values(self, x: jnp.ndarray) -> jnp.ndarray:
        eq = x[:, None] == self.values[None, :]
        idx = jnp.argmax(eq, axis=1).astype(jnp.int32)
        return idx

    def transform(self, x: jnp.ndarray) -> jnp.ndarray:
        idx = self._indices_from_values(x)
        return idx.astype(jnp.float32) / jnp.float32(self.upper)

    def inverse_transform(self, x: jnp.ndarray) -> jnp.ndarray:
        idx = jnp.clip(jnp.round(x * self.upper), 0, self.upper).astype(jnp.int32)
        return jnp.take(self.values, idx)


class ParamSpace:
    """
    Internal class that manages a collection of named parameter domains.
    Most importantly, handles sampling from the transformed parameter space.
    """
    def __init__(self, space: dict[str, Domain], seed: int, sampler: Callable) -> None:
        self.space = space
        self.obs_dim = len(space)
        self.sampler = sampler

    def sample_params(self, n_samples: int) -> dict[str, jnp.ndarray]:
        """
        Samples from the parameter space using a Sobol sequence.
        Args:
            n_samples: Number of samples to draw.
        Returns:
            A dictionary of sampled parameter values.
        """
        xs = self.sampler.random(n_samples)
        return {k: self.space[k].inverse_transform(xs[:, i]) for i, k in enumerate(self.space)}

    def to_array(self, tree: dict[str, jnp.ndarray]) -> jnp.ndarray:
        # --- transform a batch of parameter values into a 2D array suitable for Surrogate input ---
        return jnp.stack([self.space[k].transform(tree[k]) for k in self.space], axis=1)

    def to_dict(self, xs: jnp.ndarray) -> dict[str, jnp.ndarray]:
        # --- convert stacked parameter matrix back into named parameter trees ---
        return {k: self.space[k].inverse_transform(xs[:, i]) for i, k in enumerate(self.space)}
