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

    def neg_log_likelihood(self, X: jnp.ndarray, y: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def predict(
        self,
        X_test: jnp.ndarray,
        X: Optional[jnp.ndarray | None] = None,
        y: Optional[jnp.ndarray | None] = None,
        mask: Optional[jnp.ndarray | None] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError


class Surrogate(SurrogateBase):
    """
    Non-parametric (Gaussian Process) surrogate model with masking support.
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
    def __call__(self, X: jnp.ndarray, y: jnp.ndarray, mask: jnp.ndarray) -> GaussianProcess:
        assert X.ndim == 2, "Input must be a 2D array"

        # --- FIX 1: SANITIZE INPUTS IMMEDIATELY ---
        # We replace padded values (which might be -inf or garbage) with 0.0.
        # This prevents "inf * 0 = NaN" errors later in variance calculation.
        safe_X = jnp.where(mask[:, None], X, 0.0)
        safe_y = jnp.where(mask, y, 0.0)

        # --- kernel with ARD lengthscales ---
        kernel_1 = jnp.exp(self.log_amp_1) * transforms.Cholesky.from_parameters(
            jnp.exp(self.log_scale_1[:self.obs_dim]),
            jnp.exp(self.log_scale_1[self.obs_dim:]),
            kernels.ExpSquared(),
        )

        # Use safe_X for the GP so we don't calculate covariance on garbage data
        kernel = kernel_1

        # --- normalise target values ---
        valid_count = jnp.sum(mask) + 1e-12
        y_mean = jnp.sum(safe_y) / valid_count

        # Use safe_y here to avoid (inf - mean)**2
        y_var = jnp.sum(((safe_y - y_mean) ** 2) * mask) / valid_count

        # Prevent division by zero if variance is tiny (e.g. initially flat)
        y_std = jnp.where(y_var < 1e-6, 1.0, jnp.sqrt(y_var))

        # --- noise initialisation ---
        base_diag = jnp.exp(jnp.clip(self.log_diag, min=self.config['surrogate_min_log_diag'], max=self.config['surrogate_max_log_diag'])) + 1e-9
        diag = jnp.full((X.shape[0],), base_diag)
        diag = diag + 1e6 * (1.0 - mask)

        return GaussianProcess(kernel, safe_X, diag=diag), y_mean, y_std

    def neg_log_likelihood(self, X: jnp.ndarray, y: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        gp, y_mean, y_std = self(X, y, mask)

        # FIX 3: Sanitize y_scaled before passing to GP
        # If we don't do this, y_scaled has garbage at padded indices.
        # Even with high noise, it's safer to force these to 0.0 (the GP prior mean).
        safe_y = jnp.where(mask, y, y_mean) # Replace padded y with mean so (y-mean) is 0
        y_scaled = (safe_y - y_mean) / y_std

        # Double check mask (redundant but safe)
        y_scaled = jnp.where(mask, y_scaled, 0.0)

        return -gp.log_probability(y_scaled)

    def predict(self, X_test: jnp.ndarray, X: Optional[jnp.ndarray | None] = None, y: Optional[jnp.ndarray | None] = None, mask: Optional[jnp.ndarray | None] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if X is None or y is None:
            raise ValueError("For non-parametric surrogate, X and y must be provided in prediction")
        if mask is None:
            mask = jnp.ones((X.shape[0],))

        gp, y_mean, y_std = self(X, y, mask)

        # Same Fix 3 applied to prediction
        safe_y = jnp.where(mask, y, y_mean)
        y_scaled = (safe_y - y_mean) / y_std
        y_scaled = jnp.where(mask, y_scaled, 0.0)

        _, gp_cond = gp.condition(y=y_scaled, X_test=X_test)
        pred_mean = gp_cond.loc * y_std + y_mean
        pred_var = gp_cond.variance * y_std**2

        # Ensure variance is non-negative (numerical safety)
        pred_var = jnp.maximum(pred_var, 1e-9)

        return pred_mean, pred_var


def fit_posterior(y: jnp.ndarray,
                  X: jnp.ndarray,
                  mask: jnp.ndarray,
                  surrogate: Surrogate,
                  init_surrogate_params: dict[str, Any],
                  optimizer: optax.GradientTransformation,
                  config: dict[str, Any]) -> dict[str, Any]:
    """
    Fits the surrogate model to the target values.
    Args:
        y: Training targets.
        X: Training inputs.
        mask: Binary mask (1 for valid, 0 for padded).
        surrogate: Surrogate model.
        init_surrogate_params: Initial surrogate model parameters.
        optimizer: Optimizer.
        config: Configuration dictionary. [Must contain 'surrogate_fit_posterior_num_steps']
    Returns:
        Fitted surrogate model parameters.
    """
    train_state = optimizer.init(init_surrogate_params)

    def _loss_fn(params: dict[str, Any]) -> jnp.ndarray:
        return surrogate.apply(params, X, y, mask, method="neg_log_likelihood")

    def _fit_posterior(carry: tuple[dict[str, Any], Any], _: None) -> tuple[tuple[dict[str, Any], Any], jnp.ndarray]:
        surrogate_params, train_state = carry
        loss_val, grads = jax.value_and_grad(_loss_fn)(surrogate_params)
        updates, train_state = optimizer.update(grads, train_state)
        surrogate_params = optax.apply_updates(surrogate_params, updates)
        return (surrogate_params, train_state), loss_val

    (surrogate_params, train_state), losses = jax.lax.scan(_fit_posterior, (init_surrogate_params, train_state), None, length=config['surrogate_fit_posterior_num_steps'])
    return surrogate_params
