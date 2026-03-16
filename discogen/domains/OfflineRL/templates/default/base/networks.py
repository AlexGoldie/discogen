from typing import Any, Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer with variance scaling."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Create an ensemble of modules using vmap.

    Args:
        cls: Module class to ensemblize.
        num_qs: Number of ensemble members.
        in_axes: Input axes specification for vmap.
        out_axes: Output axes specification for vmap.
        **kwargs: Additional arguments passed to vmap.

    Returns:
        Vmapped module class.
    """
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Sequence of hidden layer dimensions.
        activations: Activation function to use.
        activate_final: Whether to apply activation after final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation support."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class Actor(nn.Module):
    """Gaussian actor network for continuous control.

    Outputs a Gaussian distribution over actions, optionally squashed with tanh.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Dimension of the action space.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum log standard deviation.
        log_std_max: Maximum log standard deviation.
        tanh_squash: Whether to squash actions with tanh.
        state_dependent_std: Whether std depends on state.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Scale for final layer initialization.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(self, observations, temperature=1.0):
        """Return action distribution.

        Args:
            observations: Input observations.
            temperature: Scaling factor for standard deviation.

        Returns:
            Action distribution (Normal or TanhNormal).
        """
        outputs = self.actor_net(observations)
        means = self.mean_net(outputs)

        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(
            loc=means,
            scale_diag=jnp.exp(log_stds) * temperature
        )

        if self.tanh_squash:
            distribution = TransformedWithMode(
                distribution,
                distrax.Block(distrax.Tanh(), ndims=1)
            )

        return distribution


class Value(nn.Module):
    """Critic/Value network.

    Can be used for both V(s) and Q(s, a) depending on whether actions are provided.
    Supports ensemble of critics via the num_ensembles parameter.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble members (for clipped double Q-learning).
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        self.value_net = mlp_class(
            (*self.hidden_dims, 1),
            activate_final=False,
            layer_norm=self.layer_norm
        )

    def __call__(self, observations, actions=None):
        """Return value or Q-value estimates.

        Args:
            observations: Input observations.
            actions: Optional actions for Q-value computation.

        Returns:
            Value estimates with shape (batch,) or (num_ensembles, batch).
        """
        inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)
        return v
