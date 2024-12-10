"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData

# Comment these out if not yet implemented
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __hash__(self) -> int:
        """Makes the tensor hashable using its unique_id.
        This allows tensors to be used in sets and as dictionary keys.

        Returns: int: A hash value unique to this tensor.
        """
        return hash(self.unique_id)

    # def __eq__(self, other: Any) -> bool:
    #     """Checks equality between tensors based on their unique_id.
    #     Required with __hash__ to make tensors hashable.

    #     Args:other (Any): Object to compare with.

    #     Returns:bool: True if other is a Tensor with the same unique_id, False otherwise.
    #     """
    #     if not isinstance(other, Tensor):
    #         if isinstance(other, (int, float)):
    #             other = Tensor.make([other], (1,), backend=self.backend)
    #         else:
    #             return NotImplemented
    #     return self.unique_id == other.unique_id

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Sets the tensor to track gradients if x is True.

        Args:
        ----
            x (bool): If True, gradients will be computed for this tensor during backpropagation.
                    If False, this tensor will be treated as a constant and not accumulate gradients.

        Returns:
        -------
            None

        """
        self.history = History()

    def requires_grad(self) -> bool:
        """Checks if the tensor requires gradients.

        Returns
        -------
            bool: True if the tensor is tracking gradients, False otherwise.

        """
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Returns
        Converted to numpy array

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Turns a python number into a tensor with the same backend."""
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Convert a 1-element tensor to a float"""
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data"""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data"""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`
        is a different size than the input of `forward`.


        Args:
        ----
            other : backward tensor (must broadcast with self)

        Returns:
        -------
            Expanded version of `other` with the right derivatives

        """
        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Creates a new tensor filled with zeros.

        Args:
        ----
            shape (Optional[UserShape]): The shape of the new tensor. If None, uses the shape
                                        of the current tensor. Default is None.

        Returns:
        -------
            Tensor: A new tensor filled with zeros.

        """

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the tensor data info as a tuple."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detach from backprop"""
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x : value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Determines whether the tensor is a constant.

        Returns
        -------
            bool: True if the tensor is a constant (doesn't require gradients),
                False otherwise.

        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Retrieves the parent variables of this tensor in the computation graph.

        Returns
        -------
            Iterable[Variable]: An iterable (likely a tuple or list) of parent Variables.

        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute gradients for this tensor's inputs.

        Args:
        ----
            d_output (Any): The gradient of the final output with respect to this tensor.
                            Usually a Tensor, but kept as Any to maintain flexibility.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: An iterable of tuples, each containing:
                - A parent tensor (Variable)
                - The gradient of the final output with respect to that parent tensor

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Computes the gradients of this tensor with respect to graph leaves.

        Args:
        ----
            grad_output (Optional[Tensor]): The gradient of the final output with respect
                                            to this tensor. If None, it's assumed to be
                                            a tensor of 1.0 for scalar tensors.

        Returns:
        -------
            None

        """
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    @property
    def shape(self) -> UserShape:
        """Returns
        shape of the tensor

        """
        return self._tensor.shape

    ## Functions
    ## TODO: Implement for Task 2.3.
    # @property
    # def size(self) -> int:
    #     """Returns the total number of elements in the tensor.

    #     Returns
    #     -------
    #         int: The total number of elements in the tensor.

    #     """
    #     return int(operators.prod(self.shape))
    # Module 2 answer
    @property
    def size(self) -> int:
        """Returns the total number of elements in the tensor.

        Returns
        -------
        int: The total number of elements in the tensor.

        """
        return self._tensor.size

    # Module 2 answer
    @property
    def dims(self) -> int:
        """Returns the number of dimensions of the tensor.

        Returns
        -------
             int: The total number of elements in the tensor.

        """
        return self._tensor.dims

    # @property
    # def dims(self) -> int:
    #     """Returns the number of dimensions of the tensor.

    #     Returns
    #     -------
    #         int: The number of dimensions of the tensor.

    #     """
    #     return len(self.shape)

    # #Function before answer
    # def __add__(self, b: TensorLike) -> Tensor:
    #     return Add.apply(self, self._ensure_tensor(b))

    # Module 2 answer
    def __add__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, self._ensure_tensor(b))

    # def __radd__(self, b: TensorLike) -> Tensor:
    #     return Add.apply(self._ensure_tensor(b), self)

    # Module 2 answer
    def __radd__(self, b: TensorLike) -> Tensor:
        return self + b

    # def __sub__(self, b: TensorLike) -> Tensor:
    #     return Add.apply(self, Neg.apply(self._ensure_tensor(b)))

    # Module 2 answer
    def __sub__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, -self._ensure_tensor(b))

    # def __mul__(self, b: TensorLike) -> Tensor:
    #     return Mul.apply(self, self._ensure_tensor(b))

    # Module 2 answer
    def __mul__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, self._ensure_tensor(b))

    # def __rmul__(self, b: TensorLike) -> Tensor:
    #     return Mul.apply(self._ensure_tensor(b), self)

    # Module 2 answer
    def __rmul__(self, b: TensorLike) -> Tensor:
        return self * b

    # def __lt__(self, b: TensorLike) -> Tensor:
    #     return LT.apply(self, self._ensure_tensor(b))

    # Module 2 answer
    def __lt__(self, b: TensorLike) -> Tensor:
        return LT.apply(self, self._ensure_tensor(b))

    # def __eq__(self, b: TensorLike) -> Tensor:
    #     return EQ.apply(self, self._ensure_tensor(b))

    # Module 2 answer
    def __eq__(self, b: TensorLike) -> Tensor:  # type: ignore[override]
        return EQ.apply(self, self._ensure_tensor(b))

    # def __gt__(self, b: TensorLike) -> Tensor:
    #     return LT.apply(self._ensure_tensor(b), self)

    # Module 2 answer
    def __gt__(self, b: TensorLike) -> Tensor:
        return LT.apply(self._ensure_tensor(b), self)

    # def __neg__(self) -> Tensor:
    #     return Neg.apply(self)

    # Module 2 answer
    def __neg__(self) -> Tensor:
        return Neg.apply(self)

    ##all1
    # def all(self) -> Tensor:
    #     """Returns a tensor with True if all elements in the input tensor are True, False otherwise.

    #     Returns:
    #         Tensor: A new tensor containing a single boolean value.
    #                 - True if all elements in the input tensor are True.
    #                 - False if any element in the input tensor is False.

    #     """
    #     return All.apply(self)

    ##all2
    # def all(self) -> Tensor:
    #     """Returns True if all elements in the input tensor are True.

    #     Returns:Tensor: A new scalar tensor containing True if all elements are True,
    #                 False otherwise.

    #     """
    #     # First ensure the tensor is boolean/binary
    #     # For non-zero values to be considered True
    #     result = All.apply(self)

    #     # If we need to compare with the expected tensor
    #     if result.dims > 0:  # If not a scalar
    #         result = result.sum(0)  # Reduce to match expected shape

    #     return result

    ##Module 2 answer for all
    def all(self, dim: Optional[int] = None) -> Tensor:
        """Returns True if all elements in the input tensor are True.

        Returns:Tensor: A new scalar tensor containing True if all elements are True,
                     False otherwise.

        """
        if dim is None:
            return All.apply(self.view(self.size), self._ensure_tensor(0))
        else:
            return All.apply(self, self._ensure_tensor(dim))

    # def is_close(self, b: Tensor) -> Tensor:
    #     """Compares this tensor with another tensor for element-wise approximate equality.

    #     Args:
    #     ----
    #         b (Tensor): The tensor to compare with. Must be broadcastable to the shape
    #                     of this tensor.

    #     Returns:
    #     -------
    #         Tensor: A boolean tensor of the same shape as the broadcasted input tensors.
    #                 Each element is True where the tensors are approximately equal,
    #                 and False otherwise.

    #     """
    #     return IsClose.apply(self, b)

    def is_close(self, y: Tensor) -> Tensor:
        """Compares this tensor with another tensor for element-wise approximate equality.

        Args:
        ----
            y (Tensor): The tensor to compare with. Must be broadcastable to the shape
                        of this tensor.

        Returns:
        -------
            Tensor: A boolean tensor of the same shape as the broadcasted input tensors.
                    Each element is True where the tensors are approximately equal,
                    and False otherwise.

        """
        return IsClose.apply(self, y)

    def sigmoid(self) -> Tensor:
        """Applies the sigmoid activation function element-wise to the tensor.

        Returns
        -------
            Tensor: A new tensor with the sigmoid function applied element-wise.
                    The output tensor has the same shape as the input tensor.

        """
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Applies the Rectified Linear Unit (ReLU) activation function element-wise to the tensor.

        Returns
        -------
            Tensor: A new tensor with the ReLU function applied element-wise.
                    The output tensor has the same shape as the input tensor.

        """
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Applies the natural logarithm function element-wise to the tensor.

        Returns
        -------
            Tensor: A new tensor with the natural logarithm applied element-wise.
                    The output tensor has the same shape as the input tensor.

        """
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Applies the exponential function element-wise to the tensor.

        Returns
        -------
            Tensor: A new tensor with the exponential function applied element-wise.
                    The output tensor has the same shape as the input tensor.

        """
        return Exp.apply(self)

    # def sum(self, dim: Optional[int] = None) -> Tensor:
    #     """Computes the sum of all elements in the tensor along the specified dimension.

    #     Args:
    #     ----
    #         dim (Optional[int]): The dimension along which to compute the sum.
    #                             If None, the sum is computed over all elements.
    #                             Default is None.

    #     Returns:
    #     -------
    #         Tensor: A new tensor containing the sum(s).
    #                 - If dim is None, returns a scalar tensor with the total sum.
    #                 - If dim is specified, returns a tensor with the specified dimension removed.

    #     """
    #     if dim is None:
    #         return Sum.apply(self)
    #     else:
    #         return Sum.apply(self, self._ensure_tensor(dim))  # sal
    #         # return Sum.apply(self, Tensor.make([dim], (1,), backend=self.backend)) #col

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Computes the sum of all elements in the tensor along the specified dimension.

        Args:
        ----
             dim (Optional[int]): The dimension along which to compute the sum.
                                 If None, the sum is computed over all elements.
                                 Default is None.

        Returns:
        -------
             Tensor: A new tensor containing the sum(s).
                     - If dim is None, returns a scalar tensor with the total sum.
                     - If dim is specified, returns a tensor with the specified dimension removed.

        """
        if dim is None:
            return Sum.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
        else:
            return Sum.apply(self, self._ensure_tensor(dim))

    ##Module 2 Answer
    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Computes the mean of all elements in the tensor along the specified dimension.

        Args:
        ----
            dim (Optional[int]): The dimension along which to compute the mean.
                                If None, the mean is computed over all elements.
                                Default is None.

        Returns:
        -------
            Tensor: A new tensor containing the mean(s).
                    - If dim is None, returns a scalar tensor with the overall mean.
                    - If dim is specified, returns a tensor with the specified dimension removed.

        """
        if dim is not None:
            return self.sum(dim) / self.shape[dim]
        else:
            return self.sum() / self.size

    # ##Mean 1
    # def mean(self, dim: Optional[int] = None) -> Tensor:
    #     """Computes the mean of all elements in the tensor along the specified dimension.

    #     Args:
    #     ----
    #         dim (Optional[int]): The dimension along which to compute the mean.
    #                             If None, the mean is computed over all elements.
    #                             Default is None.

    #     Returns:
    #     -------
    #         Tensor: A new tensor containing the mean(s).
    #                 - If dim is None, returns a scalar tensor with the overall mean.
    #                 - If dim is specified, returns a tensor with the specified dimension removed.

    #     """
    #     if dim is None:
    #         return self.sum() / self.size
    #     else:
    #         # Convert dim to positive index if negative
    #         if dim < 0:
    #             dim = self.dims + dim

    #         return self.sum(dim) / self.shape[dim]

    # if dim is None:
    # # Cast total size to int to ensure compatibility with _ensure_tensor
    #     return self.sum() / int(self.size)
    # else:
    # # Cast shape[dim] to int to ensure compatibility with _ensure_tensor
    #     return self.sum(dim) / int(self.shape[dim])

    # ##Mean 2
    # def numel(self) -> int:
    #     """Returns the total number of elements in the tensor."""
    #     total_elements = 1
    #     for dim_size in self.shape:
    #         total_elements *= dim_size
    #     return total_elements

    # def mean(self, dim: Optional[int] = None) -> Tensor:
    #     """Computes the mean of all elements in the tensor along the specified dimension.

    #     Args:
    #         dim (Optional[int]): The dimension along which to compute the mean.
    #                             If None, the mean is computed over all elements.
    #                             Default is None.

    #     Returns:
    #         Tensor: A new tensor containing the mean(s).
    #                 - If dim is None, returns a scalar tensor with the overall mean.
    #                 - If dim is specified, returns a tensor with the specified dimension removed.

    #     """
    #     if dim is None:
    #         # Compute the mean over all elements
    #         return self.sum() / self.numel()
    #     else:
    #         # Handle negative dimension index
    #         if dim < 0:
    #             dim = dim + len(self.shape)

    #     # Compute the mean along the specified dimension
    #     return self.sum(dim) / self.shape[dim]

    ##Module 2 Answer
    def permute(self, *order: int) -> Tensor:
        """Permutes the dimensions of the tensor according to the given order.

        Args:
        ----
            *order (int): A variable-length argument list specifying the new order
                        of dimensions. Must include all dimensions exactly once.

        Returns:
        -------
            Tensor: A new tensor with the same data but with dimensions reordered.

        """
        return Permute.apply(self, tensor(list(order)))

    # def permute(self, *order: int) -> Tensor:
    #     """Permutes the dimensions of the tensor according to the given order.

    #     Args:
    #     ----
    #         *order (int): A variable-length argument list specifying the new order
    #                     of dimensions. Must include all dimensions exactly once.

    #     Returns:
    #     -------
    #         Tensor: A new tensor with the same data but with dimensions reordered.

    #     """
    #     return Permute.apply(self, tensor(list(order)))

    ##Module 2 Answer
    def view(self, *shape: int) -> Tensor:
        """Returns a new tensor with the same data but a different shape.

        Args:
        ----
            *shape (int): A variable-length argument list specifying the new shape.
                        One dimension can be -1, which will be inferred from the
                        remaining dimensions and the number of elements in the tensor.

        Returns:
        -------
            Tensor: A new tensor with the same data but reshaped according to the specified shape.

        """
        return View.apply(self, tensor(list(shape)))

    ##Module 2 Answer
    def zero_grad_(self) -> None:
        """Zeros out the gradient of this tensor.

        Returns
        -------
            None

        """
        self.grad = None
