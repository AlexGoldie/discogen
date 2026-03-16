from flax.linen.initializers import zeros, constant
from typing import Optional, Tuple, Any
import jax.numpy as jnp
from flax import linen as nn
from tinygp import GaussianProcess, kernels, transforms
import jax
import optax


class SurrogateBase(nn.Module):
    """
    Base class for surrogate models.
    Surrogate models must implement neg_log_likelihood and predict methods.
    Predict method must return the mean and variance of the surrogate model at the test points.
    Non-Parametric surrogate models (e.g. Gaussian Processes) require X and y in the predict method.
    Parametric surrogate models (e.g. Neural Networks) do not require X and y in the predict method.
    """
    config: dict[str, Any]

    def neg_log_likelihood(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def predict(
        self,
        X_test: jnp.ndarray,
        X: Optional[jnp.ndarray | None] = None,
        y: Optional[jnp.ndarray | None] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError


class Surrogate(SurrogateBase):
    """
    Non-parametric (Gaussian Process) surrogate model.
    Args:
        config: Configuration dictionary. [Must contain 'surrogate_min_log_diag' and 'surrogate_max_log_diag', and any constant lengthscale initialisation parameters.]
        obs_dim: Number of observed dimensions.
    Returns:
        Surrogate model.
    """
    config: dict[str, Any]
    obs_dim: int

    def setup(self):
        self.log_amp_1 = self.param("log_amp_1", zeros, ())
        n_scale_params = self.obs_dim + (self.obs_dim * (self.obs_dim - 1) // 2)
        self.log_scale_1 = self.param("log_scale_1", constant(self.config['surrogate_log_scale_1_initialisation']), (n_scale_params,))
        self.log_diag = self.param("log_diag", constant(self.config['surrogate_log_diag_initialisation']), ())

    @nn.compact
    def __call__(self, X: jnp.ndarray, y:jnp.ndarray) -> GaussianProcess:
        assert X.ndim == 2, "Input must be a 2D array"

        # --- kernel with ARD lengthscales ---
        kernel_1 = jnp.exp(self.log_amp_1) * transforms.Cholesky.from_parameters(
            jnp.exp(self.log_scale_1[:self.obs_dim]),
            jnp.exp(self.log_scale_1[self.obs_dim:]),
            kernels.ExpSquared(),
        )
        kernel = kernel_1

        # --- normalise target values ---
        y_mean = jnp.mean(y)
        y_std = jnp.std(y) + 1e-12

        # --- noise initialisation. ---
        diag = jnp.full((X.shape[0],), jnp.exp(jnp.clip(self.log_diag, a_min=self.config['surrogate_min_log_diag'], a_max=self.config['surrogate_max_log_diag'])) + 1e-9)
        return GaussianProcess(kernel, X, diag=diag), y_mean, y_std

    def neg_log_likelihood(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Negative log-likelihood of the surrogate model.
        Args:
            X: Input data.
            y: Target values.
        Returns:
            Negative log-likelihood of target values given the surrogate model.
        """
        gp, y_mean, y_std = self(X, y)
        y_scaled = (y - y_mean) / y_std
        return -gp.log_probability(y_scaled)

    def predict(self, X_test: jnp.ndarray, X: Optional[jnp.ndarray | None] = None, y: Optional[jnp.ndarray | None] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Outputs mean and variance at test points. Non-parametric surrogate models require X and y in the predict method to condition surrogate model on.
        Args:
            X_test: Test points.
            X: Input data.
            y: Target values.
        Returns:
            Predicted mean and variance of the surrogate model at the test points.
        """
        if X is None or y is None:
            raise ValueError("For non-parametric surrogate, X and y must be provided in prediction")
        gp, y_mean, y_std = self(X, y)
        y_scaled = (y - y_mean) / y_std
        _, gp_cond = gp.condition(y=y_scaled, X_test=X_test)
        pred_mean = gp_cond.loc * y_std + y_mean
        pred_var = gp_cond.variance * y_std**2
        return pred_mean, pred_var


def fit_posterior(y: jnp.ndarray,
                  X: jnp.ndarray,
                  surrogate: Surrogate,
                  init_surrogate_params: dict[str, Any],
                  optimizer: optax.GradientTransformation,
                  config: dict[str, Any]) -> dict[str, Any]:
    """
    Fits the surrogate model to the target values.
    Args:
        y: Training targets.
        X: Training inputs.
        surrogate: Surrogate model.
        init_surrogate_params: Initial surrogate model parameters.
        optimizer: Optimizer.
        config: Configuration dictionary. [Must contain 'surrogate_fit_posterior_num_steps']
    Returns:
        Fitted surrogate model parameters.
    """
    train_state = optimizer.init(init_surrogate_params)

    def _loss_fn(params: dict[str, Any]) -> jnp.ndarray:
        return surrogate.apply(params, X, y, method="neg_log_likelihood")

    def _fit_posterior(carry: tuple[dict[str, Any], Any], _: None) -> tuple[tuple[dict[str, Any], Any], jnp.ndarray]:
        surrogate_params, train_state = carry
        loss_val, grads = jax.value_and_grad(_loss_fn)(surrogate_params)
        updates, train_state = optimizer.update(grads, train_state)
        surrogate_params = optax.apply_updates(surrogate_params, updates)
        return (surrogate_params, train_state), loss_val

    (surrogate_params, train_state), losses = jax.lax.scan(_fit_posterior, (init_surrogate_params, train_state), None, length=config['surrogate_fit_posterior_num_steps'])
    return surrogate_params
