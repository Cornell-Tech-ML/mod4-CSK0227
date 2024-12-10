"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple  # add optional

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Performs the forward pass of the negation operation.

        Args:
        ----
            ctx (Context): Context object to save information for backward pass.
            t1 (Tensor): Input tensor to be negated.

        Returns:
        -------
            Tensor: A new tensor with all elements negated.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Performs the backward pass of the negation operation.

        Args:
        ----
            ctx (Context): Context object (unused in this case).
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
        -------
            Tensor: Gradient of the loss with respect to the input.

        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Performs the forward pass of the inverse operation.

        Args:
        ----
            ctx (Context): Context object to save information for backward pass.
            t1 (Tensor): Input tensor for which to compute the multiplicative inverse.

        Returns:
        -------
            Tensor: A new tensor with all elements inverted (1/x for each element x).

        Note:
        ----
            Saves the input tensor for use in the backward pass.

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Performs the backward pass of the inverse operation.

        Args:
        ----
            ctx (Context): Context object containing saved tensors from the forward pass.
            grad_output (Tensor): Gradient of the loss with respect to the output of the forward pass.

        Returns:
        -------
            Tensor: Gradient of the loss with respect to the input of the forward pass.

        Note:
        ----
            Uses the saved input tensor from the forward pass to compute the gradient.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Performs the forward pass of the addition operation.

        Args:
        ----
            ctx (Context): Context object (unused in this case).
            t1 (Tensor): First input tensor.
            t2 (Tensor): Second input tensor.

        Returns:
        -------
            Tensor: A new tensor containing the element-wise sum of t1 and t2.

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs the backward pass of the addition operation.

        Args:
        ----
            ctx (Context): Context object (unused in this case).
            grad_output (Tensor): Gradient of the loss with respect to the output of the forward pass.

        Returns:
        -------
            Tuple[Tensor, Tensor]: A tuple containing the gradients of the loss with respect to both inputs.
                                   Both gradients are identical to grad_output due to the nature of addition.

        """
        return grad_output, grad_output


##All 1
# class All(Function):
#     @staticmethod
#     def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
#         """Return 1 if all are true"""
#         if dim is not None:
#             return a.f.mul_reduce(a, int(dim.item()))
#         else:
#             return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


# ##All 2
# class All(Function):
#     @staticmethod
#     def forward(ctx: Context, a: Tensor, dim: Optional[int] = None) -> Tensor:
#         """Return 1 if all are true in the tensor or along a dimension.
#         Uses multiplication as a way to perform logical AND.
#         """
#         if dim is not None:
#             return a.f.mul_reduce(a, int(dim))
#         else:
#             # Reshape to 1D and reduce all
#             out = a.contiguous().view(int(operators.prod(a.shape)))
#             return a.f.mul_reduce(out, 0)

#     @staticmethod
#     def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Optional[float]]:
#         """Backward pass for all reduction.
#         This is a logical operation, so technically the gradient is not defined.
#         We return 0 gradients by convention.
#         """
#         return grad_output.zeros(grad_output.shape), None

#     @classmethod
#     def apply(cls, *vals: Tensor) -> Tensor:
#         """Special case application of All reduction that handles the optional dim parameter."""
#         raw_vals = []
#         need_grad = False
#         vals = list(vals)
#         for v in vals:
#             if isinstance(v, Tensor):
#                 if v.requires_grad():
#                     need_grad = True
#                 raw_vals.append(v.detach())
#             else:
#                 raw_vals.append(None)  # for the dim parameter

#         # Create the context.
#         ctx = Context(not need_grad)

#         # Call forward with the variables.
#         dim = None
#         if len(raw_vals) > 1:
#             dim = raw_vals[1]
#         c = cls._forward(ctx, raw_vals[0], dim)

#         # Create a new variable from the result with a new history.
#         back = None
#         if need_grad:
#             back = minitorch.History(cls, ctx, vals)
#         return minitorch.Tensor(c._tensor, back, backend=raw_vals[0].backend)


##All 3
class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor | None = None) -> Tensor:
        """Return 1 if all elements are true in the tensor."""
        # Reshape tensor to 1D and reduce
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


## TODO: Implement for Task 2.3.


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise multiplication of two tensors."""
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for multiplication."""
        a, b = ctx.saved_values
        return (
            grad_output.f.mul_zip(b, grad_output),
            grad_output.f.mul_zip(a, grad_output),
        )


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Applies the sigmoid function element-wise."""
        out = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for sigmoid."""
        # (out,) = ctx.saved_values
        # one = out._ensure_tensor(1.0)
        # return grad_output * out * (one - out)
        # module 2 answer
        sigma: Tensor = ctx.saved_values[0]
        return sigma * (-sigma + 1.0) * grad_output


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Applies the ReLU (Rectified Linear Unit) function element-wise."""
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Applied the ReLU Fuction backward"""
        # (a,) = ctx.saved_tensors
        # return grad_output * (a > 0)
        # Module 2 answer
        (a,) = ctx.saved_tensors
        return grad_output.f.relu_back_zip(a, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Applies the natural logarithm function element-wise."""
        ctx.save_for_backward(t1)
        out = t1.f.log_map(t1)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for log."""
        (a,) = ctx.saved_values
        return grad_output.f.log_back_zip(a, grad_output)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Applies the exponential function element-wise."""
        out = t1.f.exp_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for exp. Given dL/dout, returns dL/dx."""
        (a,) = ctx.saved_values
        return grad_output.f.mul_zip(a, grad_output)


##Sum 9-2
class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Computes the forward pass for summation."""
        # Convert dim to integer or None before saving
        # dim_val = int(dim.item()) if dim is not None else None
        # ctx.save_for_backward(t1.shape, dim_val)
        # if dim is not None:
        #     dim_val = int(dim.item())
        #     result = t1.f.add_reduce(t1, dim_val)
        # else:
        #     flattened = t1.contiguous().view(t1.size)
        #     result = t1.f.add_reduce(flattened, 0)
        # return result
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Computes the backward pass for summation."""
        # shape, dim = ctx.saved_values
        # grad = minitorch.zeros(shape, backend=grad_output.backend)
        # grad_input = grad_output.f.add_zip(grad, grad_output)
        # # Return a tensor for grad_input and an integer/None for dim
        # if dim is not None:
        #     return (grad_input, grad)  # Return 0 instead of None for the dim gradient
        # else:
        #     return (grad_input,)
        a_shape, dim = ctx.saved_values
        return grad_output, 0.0


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise 'less than' comparison of two tensors."""
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for less than."""
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise equality comparison of two tensors."""
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for equals."""
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise comparison of two tensors for approximate equality."""
        return a.f.is_close_zip(a, b)

    # @staticmethod
    # def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
    #     """IsClose doesn't need a backward pass as specified in the requirements."""
    #     return grad_output.zeros(grad_output.shape), grad_output.zeros(
    #         grad_output.shape
    #     )


# permute 2
class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        """Permutes the dimensions of the input tensor according to the given order."""
        ctx.save_for_backward(order)
        # original
        # order_list = [int(order[i]) for i in range(order.size)]
        # # Get the permuted tensor data and shape
        # permuted_tensor = a._tensor.permute(*order_list)
        # # Create new tensor with the correct storage and permuted shape
        # return minitorch.Tensor.make(
        #     permuted_tensor._storage, permuted_tensor.shape, backend=a.backend
        # )
        return a._new(a._tensor.permute(*[int(order[i]) for i in range(order.size)]))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for permute. Reverses the permutation applied in forward."""
        # original
        # (order,) = ctx.saved_values

        # # Create inverse permutation
        # order_list = [int(order[i]) for i in range(order.size)]
        # n = len(order_list)
        # inv_order = [0] * n
        # for i, p in enumerate(order_list):
        #     inv_order[p] = i

        # # Apply inverse permutation to grad_output
        # grad_tensor = grad_output._tensor.permute(*inv_order)
        # return minitorch.Tensor.make(
        #     grad_tensor._storage, grad_tensor.shape, backend=grad_output.backend
        # ), 0.0
        order: Tensor = ctx.saved_values[0]
        order2: List[int] = [
            a[0]
            for a in sorted(
                enumerate([order[i] for i in range(order.size)]), key=lambda a: a[1]
            )
        ]
        return grad_output._new(grad_output._tensor.permute(*order2)), 0.0


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Performs the forward pass of the view (reshape) operation.

        Args:
        ----
            ctx (Context): Context object to save information for backward pass.
            a (Tensor): Input tensor to be reshaped.
            shape (Tensor): A tensor containing the new shape dimensions.

        Returns:
        -------
            Tensor: A new tensor with the same data as 'a' but reshaped to the specified shape.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
            # @staticmethod
            # def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
            #     """Backward pass for view operation."""
            #     (original_shape,) = ctx.saved_values
            #     return (
            #         minitorch.Tensor.make(
            #             grad_output._tensor._storage, original_shape, backend=grad_output.backend
            #         ),
            #         0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def ones(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a ones tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [1.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute the gradient of a function using the central difference method.

    Args:
    ----
        f (Any): The function to differentiate. It should take tensor arguments
                 and return a tensor.
        *vals (Tensor): The input tensors to the function 'f'.
        arg (int, optional): The index of the argument (tensor) to differentiate
                             with respect to. Defaults to 0.
        epsilon (float, optional): The small value used for numerical approximation.
                                   Defaults to 1e-6.
        ind (UserIndex): The index within the tensor specified by 'arg' at which
                         to compute the gradient.

    Returns:
    -------
        float: The approximated gradient (partial derivative) at the specified index.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
