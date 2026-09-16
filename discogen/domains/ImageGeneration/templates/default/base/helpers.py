import math
from collections import namedtuple
from typing import Any, Callable, Iterable, SupportsFloat, Optional

import torch
from torch import nn
import torch.nn.functional as F
from einops import pack, unpack
from PIL import Image

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


def exists(x: Any) -> bool:
    """
    Check whether a value is not None.

    Args:
        x (Any)

    Returns:
        bool: True if `x` is not None, otherwise False.
    """
    return x is not None


def default(val: Any, d: Any | Callable[[], Any]) -> Any:
    """
    Return a default value if the input is None.
    If `val` exists (is not None), it is returned directly. Otherwise,
    `d` is returned, or if `d` is callable, the result of calling `d`.

    Args:
        val (Any): The value to return if it exists.
        d (Any or Callable): A fallback value or a callable that produces a fallback value.

    Returns:
        Any: `val` if it exists, otherwise the resolved default value.
    """
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t: Any, *args: Any, **kwargs: Any) -> Any:
    """
    Return the input unchanged.

    Args:
        t (Any): The input value.
        *args: Unused positional arguments.
        **kwargs: Unused keyword arguments.

    Returns:
        Any: The input `t`, unchanged.
    """
    return t


def cycle(dl: Iterable[Any]) -> Any:
    """
    Iterates over the provided iterable repeatedly, yielding elements.

    Args:
        dl (Iterable): A data loader or iterable yielding batches of data.

    Yields:
        Any: The next element produced by the data loader.
    """
    while True:
        for data in dl:
            yield data


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Tensor) -> torch.Tensor:
    """
    Extract values from a tensor at specified indices and reshape.

    Args:
        a (torch.Tensor): A tensor containing values to be indexed.
        t (torch.Tensor): A tensor of indices indicating
            which values to extract for each batch element.
        x_shape (tuple): The shape of the returned tensor.

    Returns:
        torch.Tensor: A tensor suitable for broadcasting to `x_shape`.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def has_int_squareroot(num: SupportsFloat) -> bool:
    """
    Check if a number has an integer square root.

    Args:
        num (int or float): The number to test.

    Returns:
        bool: True if the square root of `num` is an integer, otherwise False.
    """
    return (math.sqrt(num) ** 2) == num


def convert_image_to_fn(img_type: str, image: Image.Image) -> Image.Image:
    """
    Convert an image to a specified image mode.

    Args:
        img_type (str): The desired image mode (e.g., "RGB", "L").
        image (PIL.Image.Image): The input image.

    Returns:
        PIL.Image.Image: The image converted to `img_type`, or the original
        image if it is already in the correct mode.
    """
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def pack_one_with_inverse(
    x: torch.Tensor, pattern: str
) -> tuple[torch.Tensor, Callable[[torch.Tensor, Optional[str]], torch.Tensor]]:
    """
    Pack a tensor according the given pattern and return an inverse unpacking function.

    Args:
        x (torch.Tensor): The input tensor to pack.
        pattern (str): An einops pattern describing how to pack the tensor.

    Returns:
        Tuple[torch.Tensor, Callable]: The packed tensor and a function
        that takes a packed tensor and returns the unpacked original tensor.
    """
    packed, packed_shape = pack([x], pattern)

    def inverse(x: torch.Tensor, inverse_pattern: Optional[str] = None) -> torch.Tensor:
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(x, packed_shape, inverse_pattern)[0]

    return packed, inverse


def uniform(shape: tuple, device: torch.device) -> torch.Tensor:
    """
    Generate a tensor of uniformly distributed random values in [0, 1).

    Args:
        shape (tuple): The desired output tensor shape.
        device (torch.device): The device on which to allocate the tensor.

    Returns:
        torch.Tensor: A float tensor of the given shape with values sampled
        uniformly from [0, 1).
    """
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def prob_mask_like(shape: tuple, prob: float, device: torch.device) -> torch.Tensor:
    """
    Generate a boolean mask with a given probability of True values.

    Args:
        shape (tuple): The desired mask shape.
        prob (float): Probability of an element being True, between 0 and 1.
        device (torch.device): The device on which to allocate the mask.

    Returns:
        torch.Tensor: A boolean tensor of the given shape.
    """

    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def project(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose a tensor into components parallel and orthogonal to another tensor.

    Projects `x` onto the direction defined by `y`, returning both the
    parallel component and the orthogonal residual. The projection is
    performed per batch element after flattening non-batch dimensions.

    Args:
        x (torch.Tensor): The input tensor to be projected.
        y (torch.Tensor): The reference tensor defining the projection direction.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - the component of `x` parallel to `y`
            - the component of `x` orthogonal to `y`
        Both tensors have the same shape as `x`.
    """
    x, inverse = pack_one_with_inverse(x, "b *")
    y, _ = pack_one_with_inverse(y, "b *")

    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = F.normalize(y, dim=-1)

    parallel = (x * unit).sum(dim=-1, keepdim=True) * unit
    orthogonal = x - parallel

    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)


class Residual(nn.Module):
    def __init__(self, fn: Callable):
        """
        Initialize the residual wrapper.

        Args:
            fn (Callable): A module or function that takes an input tensor
                and returns a tensor of the same shape.
        """
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Apply the wrapped function and add the input as a residual.

        Args:
            x (torch.Tensor): The input tensor.
            *args: Additional positional arguments passed to `fn`.
            **kwargs: Additional keyword arguments passed to `fn`.

        Returns:
            torch.Tensor: The result of `fn(x, ...) + x`.
        """
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim: int, dim_out: Optional[int] = None) -> nn.Module:
    """
    Create an upsampling block that doubles spatial resolution.

    Args:
        dim (int): Number of input channels.
        dim_out (int, optional): Number of output channels. Defaults to `dim`.

    Returns:
        nn.Module: A sequential upsampling module.
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim: int, dim_out: Optional[int] = None) -> nn.Module:
    """
    Create a downsampling convolutional block.

    Args:
        dim (int): Number of input channels.
        dim_out (int, optional): Number of output channels. Defaults to `dim`.

    Returns:
        nn.Module: A convolutional downsampling module.
    """

    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class Normalize(nn.Module):
    def __init__(self, dim: int):
        """
        Initialize the normalization layer.

        Args:
            dim (int): Number of channels in the input tensor.
        """
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input tensor across channels.

        Args:
            x (torch.Tensor): Input tensor of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor: The normalized and rescaled tensor.
        """
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class PreNormalization(nn.Module):
    def __init__(self, dim: int, fn: Callable[[torch.Tensor], torch.Tensor]):
        """
        Initialize the pre-normalization wrapper.

        Args:
            dim (int): Number of channels in the input tensor.
            fn (Callable): A module or function applied after normalization.
        """
        super().__init__()
        self.fn = fn
        self.norm = Normalize(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input and apply the wrapped function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output of `fn` applied to the normalized input.
        """
        x = self.norm(x)
        return self.fn(x)
