from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the function to the given scalar-like values."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Add two scalar values."""
        # return operators.add(a, b)
        # Module 1 answer
        return a + b

    @staticmethod
    # def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for addition operation."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for log operation."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for log operation."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


# TODO: Implement for Task 1.2.


class Mul(ScalarFunction):
    """Multiplication function $f(x,y) = x*y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for multiplication operation."""
        ctx.save_for_backward(a, b)
        # return operators.mul(a, b)
        # module 1 answer
        c = a * b
        return c

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for multiplication operation."""
        # (a, b) = ctx.saved_values
        # return d_output * b, d_output * a
        # module 1 answer
        a, b = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the inverse of the input."""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for inverse operation."""
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for negation operation."""
        # return operators.neg(a)
        # module 1 answer
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for negation operation."""
        # return operators.neg(d_output)
        # module 1 answer
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1/(1+exp(-x))$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for sigmoid operation."""
        # ctx.save_for_backward(a)
        # return operators.sigmoid(a)
        out = operators.sigmoid(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for sigmoid function.

        Args:
        ----
            ctx (Context): The context object.
            d_output (float): The gradient from the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        # (a,) = ctx.saved_values
        # return d_output * operators.sigmoid_back(a, d_output)
        # module 1 answer
        sigma: float = ctx.saved_values[0]
        return sigma * (1.0 - sigma) * d_output


# class ReLU(ScalarFunction):
class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for ReLU operation."""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for ReLU operation."""
        (a,) = ctx.saved_values
        # return d_output * operators.relu_back(a, d_output)
        # module 1 answer
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function $f(x) = exp(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for exponential operation."""
        # ctx.save_for_backward(a)
        # return operators.exp(a)
        # module 1 answer
        out = operators.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for exponential operation."""
        # (a,) = ctx.saved_values
        # return d_output * operators.exp(a)
        # module 1 answer
        out: float = ctx.saved_values[0]
        return d_output * out


class LT(ScalarFunction):
    """Less than function $f(x,y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for less than operation."""
        # return operators.lt(a, b)
        # module 1 answer
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for less than operation."""
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function $f(x,y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for equal operation."""
        # return operators.eq(a, b)
        # module 1 answer
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for equal operation."""
        return 0.0, 0.0


## maybe GT is not necessary


class GT(ScalarFunction):
    """Greater than function $f(x,y) = x > y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for greater than operation."""
        return operators.GT(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for greater than operation."""
        return 0.0, 0.0
