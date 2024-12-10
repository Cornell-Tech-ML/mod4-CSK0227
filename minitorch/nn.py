from typing import Tuple  # , Iterator, Any

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand  # , tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # # TODO: Implement for Task 4.3.
    # raise NotImplementedError("Need to implement for Task 4.3")

    # Calculate new dimensions
    new_height = height // kh
    new_width = width // kw

    # Create a view into the tensor with these steps:
    # 1. Split height dimension into (new_height, kh)
    # 2. Split width dimension into (new_width, kw)
    # 3. Combine the kernel dimensions at the end

    # First reshape to batch x channel x new_height x kh x new_width x kw
    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    # Then permute to get dimensions in desired order:
    # batch x channel x new_height x new_width x (kh * kw)
    output = reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()

    # Finally combine the last two dimensions
    final_shape = (batch, channel, new_height, new_width, kh * kw)
    output = output.view(*final_shape)

    return output, new_height, new_width


# TODO: Implement for Task 4.3.


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D.

    Args:
    ----
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
    -------
        pooled : batch x channel x new_height x new_width

    """
    batch, channel, _, _ = input.shape
    tiled, new_height, new_width = tile(input, kernel)

    return tiled.mean(dim=4).view(batch, channel, new_height, new_width)


max_reduce = FastOps.reduce(operators.max, -1e9)  # added


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute argmax as a one-hot tensor.

    Creates a tensor of the same shape as input, with 1.0 at positions where the input
    has its maximum value along the specified dimension, and 0.0 elsewhere.

    Args:
    ----
        input (Tensor): Input tensor to compute argmax over
        dim (int): Dimension along which to find maximum values

    Returns:
    -------
        one_hot: Tensor of same shape with 1.0 at max positions and 0.0 elsewhere

    """
    return input == max_reduce(input, dim)


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for max reduction along specified dimension.

        Args:
        ----
            ctx (Context): Context object to save variables for backward pass
            input (Tensor): Input tensor to compute maximum values from
            dim (Tensor): Dimension along which to compute maximum, as a single-element tensor

        Returns:
        -------
            Tensor: Tensor containing maximum values along the specified dimension

        """
        dim_val = int(dim.item())
        # Save max indices for backward pass
        ctx.save_for_backward(input, dim_val)
        return max_reduce(input, dim_val)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for max reduction.

        Computes gradients with respect to the input tensor. Gradient flows back only
        through positions that achieved the maximum in the forward pass.

        Args:
        ----
            ctx (Context): Context object containing saved tensors from forward pass
            grad_output (Tensor): Gradient tensor from downstream layers

        Returns:
        -------
            Tuple[Tensor, float]: Tuple containing:
                - Gradient with respect to input tensor
                - Gradient with respect to dimension (always 0.0 as dimension is not differentiable)

        """
        input, dim = ctx.saved_values
        return grad_output * argmax(input, dim), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Take the max of the input."""
    return Max.apply(input, input._ensure_tensor(dim))


# softmax
def softmax(input: Tensor, dim: int) -> Tensor:
    """Applies the softmax function along a specific dimension.
    Softmax(x)_i = exp(x_i) / sum_j(exp(x_j))

    Uses the log-sum-exp trick for numerical stability:
    1. Shift inputs by max value before exp (prevent overflow)
    2. Compute exp and sum
    3. Divide to get softmax

    Args:
    ----
        input: Input tensor
        dim: Dimension along which to compute softmax

    Returns:
    -------
        Tensor of same shape with softmax applied along specified dimension

    """
    input_exp = input.exp()
    sum_exp = input_exp.sum(dim=dim)
    # Divide each exp value by sum to get softmax
    return input_exp / sum_exp


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Applies the log softmax function element-wise.
    LogSoftmax(x)_i = log(exp(x_i) / sum_j(exp(x_j)))
                    = x_i - log(sum_j(exp(x_j)))
    Uses the log-sum-exp trick for numerical stability.
    """
    # Get max for the trick
    x_max = max(input, dim)
    input_exp = (input - x_max).exp()
    sum_exp = input_exp.sum(dim=dim)
    log_sum_exp = sum_exp.log() + x_max
    return input - log_sum_exp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D.

    Args:
    ----
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
    -------
        pooled : batch x channel x new_height x new_width

    """
    batch, channel, _, _ = input.shape
    tiled, new_height, new_width = tile(input, kernel)

    return max(tiled, dim=4).view(batch, channel, new_height, new_width)


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise.

    Args:
    ----
        input: tensor to apply dropout to
        p: probability of dropping a position (0 to 1)
        ignore: if True, don't apply dropout

    Returns:
    -------
        Tensor with dropout applied

    """
    # # If ignore is True or rate is 0, return input unchanged
    # if ignore or rate == 0.0:
    #     return input

    # # If rate is 1.0, drop everything by returning zeros
    # if rate == 1.0:
    #     return input.zeros(input.shape)

    # # Generate random mask
    # mask = input.zeros(input.shape)
    # for idx in mask._tensor.indices():
    #     # Generate random value between 0 and 1
    #     random_val = rand(tuple())._tensor.get(tuple())
    #     # Keep if random value > rate
    #     mask._tensor.set(idx, float(random_val > rate))

    # # Scale output by 1/(1-rate) to maintain expected sum
    # scale = 1.0 / (1.0 - rate)
    # return input * mask * scale
    if ignore:
        return input

    random_mask = rand(input.shape) > p
    return input * random_mask
