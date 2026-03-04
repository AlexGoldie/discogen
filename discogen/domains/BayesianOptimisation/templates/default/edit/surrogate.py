from typing import Optional, Tuple, Any
import jax.numpy as jnp
from flax import linen as nn
import optax


class SurrogateBase(nn.Module):
    """
    Base class for surrogate models.
    Surrogate models must implement neg_log_likelihood and predict methods.
    Predict method must return the mean and variance of the surrogate model at the test points.
    Non-Parametric surrogate models (e.g. Gaussian Processes) require X and y in the predict method.
    Even though Parametric surrogate models (e.g. Neural Networks) do not require X and y in the predict method, these are still passed into the predict method, to ensure consistency with the other modules. The predict method will just ignore these inputs.
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
    Surrogate model to be used for Bayesian Optimisation.
    Can be non-parametric or parametric.
    Args:
        config: Configuration dictionary.
        obs_dim: Number of observed dimensions.
    Returns:
        Surrogate model.
    """
    config: dict[str, Any]
    obs_dim: int

    def setup(self):
        # --- Fill in the initialisation of any surrogate model parameters here. ---
        self.surrogate_params = surrogate_params # noqa

    @nn.compact
    def __call__(self, X: jnp.ndarray, y:jnp.ndarray) -> Any:
        assert X.ndim == 2, "Input must be a 2D array"
        # --- Fill in the base surrogate model call here. ---
        return surrogate_call(X, y) # noqa

    def neg_log_likelihood(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Function to compute the negative log-likelihood of the surrogate model.
        Used to fit the parameters of the surrogate model.
        Args:
            X: Training inputs.
            y: Training targets.
        Returns:
            Negative log-likelihood of the data under the surrogate model.
        """
        # --- Fill in the negative log-likelihood calculation here. It should use the surrogate model call, and return a scalar value. ---
        surrogate_call = self(X, y) # noqa
        # --- Use the surrogate call to compute the negative log-likelihood. ---
        # The negative log-likelihood should be a scalar value.
        return neg_log_likelihood # noqa

    def predict(self, X_test: jnp.ndarray, X: Optional[jnp.ndarray | None] = None, y: Optional[jnp.ndarray | None] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Outputs predicted mean and predicted variance at test points.
        If the surrogate model is parametric, X and y can just be ignored, but these are still passed into the predict method, to ensure consistency with the other modules.
        Args:
            X_test: Test inputs.
            X: Training inputs (have as an input, but can be ignored if you choose to make the surrogate model parametric).
            y: Training targets (have as an input, but can be ignored if you choose to make the surrogate model parametric).
        Returns:
            Jax Arrays of the predicted mean and variance of the surrogate model at the test points.
        """
        # --- Fill in the prediction logic here. ---
        # The prediction should return a tuple of two Jax Arrays, the predicted mean and predicted variance at the test points.
        return pred_mean, pred_var # noqa


def fit_posterior(y: jnp.ndarray,
                  X: jnp.ndarray,
                  surrogate: Surrogate,
                  init_surrogate_params: dict[str, Any],
                  optimizer: optax.GradientTransformation,
                  config: dict[str, Any]) -> dict[str, Any]:
    """
    Fits the surrogate model to the training data using the negative log-likelihood loss function.
    If X and y are large, and the surrogate model is parametric, you might want to mini-batch the training data.
    Args:
        y: Training targets.
        X: Training inputs.
        surrogate: Surrogate model.
        init_surrogate_params: Initial surrogate model parameters.
        optimizer: Optimizer.
        config: Configuration dictionary.
    Returns:
        Fitted surrogate model parameters.
    """
    train_state = optimizer.init(surrogate_params) # noqa

    def _loss_fn(params: dict[str, Any]) -> jnp.ndarray:
        # --- Fill in your loss function here. It could be the negative log-likelihood, or some other loss function that is based on the negative log-likelihood. e.g. Maximum A Posteriori (MAP) estimation.
        neg_log_likelihood = surrogate.apply(params, X, y, method="neg_log_likelihood")
        # The loss function should return a scalar value.
        return loss_value # noqa

    # --- Fit the surrogate model to the training data using the negative log-likelihood loss function and the given optimizer. ---
    # Return the fitted surrogate model parameters.
    return surrogate_params # noqa
