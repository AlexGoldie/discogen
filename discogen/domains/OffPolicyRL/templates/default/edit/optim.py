import chex

from typing import Any, NamedTuple, Sequence
from optax._src.base import GradientTransformation

PyTree = Any
Shape = Sequence[int]

Params = chex.ArrayTree  # Parameters are arbitrary nests of `jnp.ndarrays`.
Updates = Params  # Gradient updates are of the same type as parameters.

"""State for the optimisation algorithm"""
class OptState(NamedTuple):
    ... # fill in your OptState here.

"""Optimisation function, following the structure of Optax"""
def scale_by_optimizer(
    # Fill in any hyperparameters here
):

    """Function to initialize all variables in OptState"""
    def init_fn(params):
        # Input = parameters, to help with shape calculations. This is a PyTree

        """Fill in the optimiser initialisation here."""

        # Your function must return an object of class OptState, with all values initialised here.
        return OptState()

    """Function to calculate the actual update"""
    def update_fn(gradients, state, params=None):
        # Inputs:
        # gradients: all calculated gradients, with the same shape as params.
        # state: Instance of OptState. This tracks variables between inputs. You should index using dots.
        # params: the parameters being optimised.

        """Fill in your update logic here. The updates will be applied to each parameter outside of this function as parameter update * (-1) * learning_rate."""

        # Your fuction must return the updates (with the same structure as gradients) and the updated OptState.
        return updates, OptState()

    return GradientTransformation(init_fn, update_fn)
