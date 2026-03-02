from torch import optim
import torch


def create_optimizer(model: torch.nn.Module, config: Dict[str, float]) -> "Optimizer":
    """
    Create an optimizer for the given model and configuration.

    Args:
        model (torch.nn.Module): The model to optimize.
        config (dict): Configuration dictionary containing optimizer parameters.
            - learning_rate (float): The learning rate for the optimizer.
            - weight_decay (float): The weight decay (L2 regularization) factor.

    Returns:
        Optimizer: An optimizer instance with functions including step and zero_grad. A subclass of torch.optim.Optimizer might be a good choice.
    """

    """Fill in your optimizer creation logic here."""
    optimizer = ...

    return optimizer
