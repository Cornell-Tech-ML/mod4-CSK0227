from typing import Tuple, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """A decorator to apply Numba's `njit` (no-Python just-in-time compilation) to a function with additional options.

    Args:
    ----
        fn (Fn): The function to be compiled with Numba's `njit`.
        **kwargs (Any): Additional keyword arguments to pass to the `njit`
                        compiler, such as optimization settings.

    Returns:
    -------
        Fn: The compiled function with `njit` applied.

    """
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right

    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides

    # # TODO: Implement for Task 4.1.
    # raise NotImplementedError("Need to implement for Task 4.1")

    # # conv1d 1
    # # For each output position
    # for b in prange(batch):  # Parallel processing for batch
    #     for oc in prange(out_channels):  # Parallel for output channels
    #         for w in prange(out_width):
    #             # Initialize accumulator for this output position
    #             acc = 0.0

    #             # For each weight position
    #             for ic in range(in_channels):  # Input channels
    #                 for k in range(kw):  # Kernel width
    #                     if not reverse:
    #                         # Normal convolution - anchor weight on left
    #                         w_p = w + k
    #                     else:
    #                         # Reversed convolution - anchor weight on right
    #                         w_p = w - k

    #                     # Check if we're within bounds of input
    #                     if w_p >= 0 and w_p < width:
    #                         # Calculate input and weight positions
    #                         in_pos = (
    #                             b * s1[0]  # batch stride
    #                             + ic * s1[1]  # input channel stride
    #                             + w_p * s1[2]  # width stride
    #                         )
    #                         w_pos = (
    #                             oc * s2[0]  # output channel stride
    #                             + ic * s2[1]  # input channel stride
    #                             + k * s2[2]  # kernel width stride
    #                         )
    #                         # Accumulate the product
    #                         acc += input[in_pos] * weight[w_pos]

    #             # Calculate output position
    #             out_pos = (
    #                 b * out_strides[0]  # batch stride
    #                 + oc * out_strides[1]  # output channel stride
    #                 + w * out_strides[2]  # width stride
    #             )
    #             # Store result
    #             out[out_pos] = acc

    # Tensor conv1 2
    # For each output position
    for b in prange(batch):
        for oc in prange(out_channels):
            for w in range(out_width):
                # Initialize accumulator
                acc = 0.0

                # For each input channel
                for ic in range(in_channels):
                    # For each position in kernel
                    for k in range(kw):
                        if not reverse:
                            # Forward convolution
                            w_p = w + k
                        else:
                            # Reverse convolution
                            w_p = w - (kw - k - 1)

                        # Only accumulate if we're within input bounds
                        if w_p >= 0 and w_p < width:
                            # Calculate input position
                            in_pos = (
                                b * s1[0]  # batch
                                + ic * s1[1]  # channel
                                + w_p * s1[2]  # width
                            )
                            # Calculate weight position
                            w_pos = (
                                oc * s2[0]  # out channel
                                + ic * s2[1]  # in channel
                                + k * s2[2]  # kernel pos
                            )
                            # Add to accumulator
                            acc += input[in_pos] * weight[w_pos]

                # Store result
                out_pos = (
                    b * out_strides[0]  # batch
                    + oc * out_strides[1]  # channel
                    + w * out_strides[2]  # width
                )
                out[out_pos] = acc


tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Forward pass computing 1D convolution between input and weight tensors.

        Args:
        ----
            ctx (Context): Context object to save tensors for backward pass
            input (Tensor): Input tensor of shape (batch, in_channels, width)
            weight (Tensor): Weight kernel of shape (out_channels, in_channels, kernel_width)

        Returns:
        -------
            Tensor: Output tensor of shape (batch, out_channels, width)

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass computing gradients for 1D convolution.

        Computes gradients with respect to both input and weight tensors using
        transposed convolution operations.

        Args:
        ----
            ctx (Context): Context object containing saved tensors from forward pass
            grad_output (Tensor): Gradient tensor from downstream layers

        Returns:
        -------
            Tuple[Tensor, Tensor]: Tuple containing:
                - grad_input: Gradient with respect to input tensor
                - grad_weight: Gradient with respect to weight tensor

        """
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right

    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    # inners
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

    # # TODO: Implement for Task 4.2.
    # raise NotImplementedError("Need to implement for Task 4.2")

    # # Conv2d 1
    # # For each output position
    # for b in prange(batch):  # Parallel processing for batch
    #     for oc in prange(out_channels):  # Parallel for output channels
    #         for h in prange(height):  # Height of output
    #             for w in prange(width):  # Width of output
    #                 # Initialize accumulator for this output position
    #                 acc = 0.0

    #                 # For each weight position
    #                 for ic in range(in_channels):  # Input channels
    #                     for kh_pos in range(kh):  # Kernel height
    #                         for kw_pos in range(kw):  # Kernel width
    #                             if not reverse:
    #                                 # Normal convolution - anchor weight at top-left
    #                                 h_p = h + kh_pos
    #                                 w_p = w + kw_pos
    #                             else:
    #                                 # Reversed convolution - anchor weight at bottom-right
    #                                 h_p = h - kh_pos
    #                                 w_p = w - kw_pos

    #                             # Check if we're within bounds of input
    #                             if (h_p >= 0 and h_p < height and
    #                                 w_p >= 0 and w_p < width):
    #                                 # Calculate input position
    #                                 in_pos = (
    #                                     b * s10 +    # batch stride
    #                                     ic * s11 +   # input channel stride
    #                                     h_p * s12 +  # height stride
    #                                     w_p * s13    # width stride
    #                                 )
    #                                 # Calculate weight position
    #                                 w_pos = (
    #                                     oc * s20 +     # output channel stride
    #                                     ic * s21 +     # input channel stride
    #                                     kh_pos * s22 + # kernel height stride
    #                                     kw_pos * s23   # kernel width stride
    #                                 )
    #                                 # Accumulate the product
    #                                 acc += input[in_pos] * weight[w_pos]

    for i in prange(out_size):
        out_index = np.empty(4, np.int32)
        to_index(i, out_shape, out_index)
        current_batch, current_out_channel, current_out_height, current_out_width = (
            out_index
        )

        # Accumulator
        acc = 0.0

        # Iterate through kernel
        for current_in_channel in range(in_channels):
            for kernel_height in range(kh):
                for kernel_width in range(kw):
                    # Current offset conv
                    conv_offset_h = (
                        (kh - 1 - kernel_height) if reverse else kernel_height
                    )
                    conv_offset_w = (kw - 1 - kernel_width) if reverse else kernel_width

                    # Curr weight value
                    weight_pos = (
                        current_out_channel * s20
                        + current_in_channel * s21
                        + conv_offset_h * s22
                        + conv_offset_w * s23
                    )

                    # Curr input value
                    input_height = (
                        current_out_height - conv_offset_h
                        if reverse
                        else current_out_height + conv_offset_h
                    )
                    input_width = (
                        current_out_width - conv_offset_w
                        if reverse
                        else current_out_width + conv_offset_w
                    )

                    # Check if input in bounds
                    if 0 <= input_height < height and 0 <= input_width < width:
                        input_pos = (
                            current_batch * s10
                            + current_in_channel * s11
                            + input_height * s12
                            + input_width * s13
                        )
                        acc += input[input_pos] * weight[weight_pos]
        out_pos = index_to_position(out_index, out_strides)
        out[out_pos] = acc


tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Forward pass computing 2D convolution between input and weight tensors.

        Args:
        ----
            ctx (Context): Context object to save tensors for backward pass
            input (Tensor): Input tensor of shape (batch, in_channels, height, width)
            weight (Tensor): Weight kernel of shape (out_channels, in_channels, kernel_height, kernel_width)

        Returns:
        -------
            Tensor: Output tensor of shape (batch, out_channels, height, width)

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass computing gradients for 2D convolution.

        Computes gradients with respect to both input and weight tensors using
        transposed convolution operations.

        Args:
        ----
            ctx (Context): Context object containing saved tensors from forward pass
            grad_output (Tensor): Gradient tensor from downstream layers with shape
                                (batch, out_channels, height, width)

        Returns:
        -------
            Tuple[Tensor, Tensor]: Tuple containing:
                - grad_input: Gradient with respect to input tensor, shape matches input
                - grad_weight: Gradient with respect to weight tensor, shape matches weight

        """
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
