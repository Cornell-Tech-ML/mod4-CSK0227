# from __future__ import annotations

# from typing import (
#     TYPE_CHECKING,
#     # Union,
#     Optional,
#     Any,
#     Callable,
# )  # , TypeVar #module 3 answer

# import numpy as np

# # from numpy import ndarray, float64  # pre-commit addded
# from numba import prange

# # from numba import njit as _njit #module 3 answer
# from numba import njit  # module 3 answer

# from .tensor_data import (
#     MAX_DIMS,
#     broadcast_index,
#     index_to_position,
#     shape_broadcast,
#     to_index,
# )
# from .tensor_ops import MapProto, TensorOps

# if TYPE_CHECKING:
#     from typing import Any, Callable, Optional  # module 3 answer

#     from .tensor import Tensor
#     from .tensor_data import Index, Shape, Storage, Strides  # module 3 answer

# # TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# # This code will JIT compile fast versions your tensor_data functions.
# # If you get an error, read the docs for NUMBA as to what is allowed
# # in these functions.

# # module 3 answer
# # to_index = njit(to_index)
# # index_to_position = njit(index_to_position)
# # broadcast_index = njit(broadcast_index)

# # module 3 answer
# to_index = njit(inline="always")(to_index)
# index_to_position = njit(inline="always")(index_to_position)
# broadcast_index = njit(inline="always")(broadcast_index)

# # precommit add
# # Number = Union[float, ndarray[Any, float64]]


# class FastOps(TensorOps):
#     @staticmethod
#     def map(fn: Callable[[float], float]) -> MapProto:
#         """See `tensor_ops.py`"""
#         # This line JIT compiles your tensor_map
#         # f = tensor_map(njit(fn))
#         # module 3 answer
#         f = tensor_map(njit()(fn))

#         def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
#             if out is None:
#                 out = a.zeros(a.shape)
#             f(*out.tuple(), *a.tuple())
#             return out

#         return ret

#     @staticmethod
#     def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
#         """See `tensor_ops.py`"""
#         # f = tensor_zip(njit(fn))
#         # module 4 answer
#         f = tensor_zip(njit()(fn))

#         def ret(a: Tensor, b: Tensor) -> Tensor:
#             c_shape = shape_broadcast(a.shape, b.shape)
#             out = a.zeros(c_shape)
#             f(*out.tuple(), *a.tuple(), *b.tuple())
#             return out

#         return ret

#     @staticmethod
#     def reduce(
#         fn: Callable[[float, float], float], start: float = 0.0
#     ) -> Callable[[Tensor, int], Tensor]:
#         """See `tensor_ops.py`"""
#         # f = tensor_reduce(njit(fn))
#         # Module 4 answer
#         f = tensor_reduce(njit()(fn))

#         def ret(a: Tensor, dim: int) -> Tensor:
#             out_shape = list(a.shape)
#             out_shape[dim] = 1

#             # Other values when not sum.
#             out = a.zeros(tuple(out_shape))
#             out._tensor._storage[:] = start

#             f(*out.tuple(), *a.tuple(), dim)
#             return out

#         return ret

#     @staticmethod
#     def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
#         """Batched tensor matrix multiply ::

#             for n:
#               for i:
#                 for j:
#                   for k:
#                     out[n, i, j] += a[n, i, k] * b[n, k, j]

#         Where n indicates an optional broadcasted batched dimension.

#         Should work for tensor shapes of 3 dims ::

#             assert a.shape[-1] == b.shape[-2]

#         Args:
#         ----
#             a : tensor data a
#             b : tensor data b

#         Returns:
#         -------
#             New tensor data

#         """
#         # Make these always be a 3 dimensional multiply
#         both_2d = 0
#         if len(a.shape) == 2:
#             a = a.contiguous().view(1, a.shape[0], a.shape[1])
#             both_2d += 1
#         if len(b.shape) == 2:
#             b = b.contiguous().view(1, b.shape[0], b.shape[1])
#             both_2d += 1
#         both_2d = both_2d == 2

#         ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
#         ls.append(a.shape[-2])
#         ls.append(b.shape[-1])
#         assert a.shape[-1] == b.shape[-2]
#         out = a.zeros(tuple(ls))

#         tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

#         # Undo 3d if we added it.
#         if both_2d:
#             out = out.view(out.shape[1], out.shape[2])
#         return out


# # Implementations


# # Module 3 Answer
# def tensor_map(fn: Callable[[float], float]) -> Any:
#     # ) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
#     """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

#     Optimizations:

#     * Main loop in parallel
#     * All indices use numpy buffers
#     * When `out` and `in` are stride-aligned, avoid indexing

#     Args:
#     ----
#         fn: function mappings floats-to-floats to apply.
#         out (Storage): storage for out tensor.
#         out_strides (Strides): strides for out tensor.
#         in_storage (Storage): storage for in tensor.
#         in_shape (Shape): shape for in tensor.
#         in_strides (Strides): sstrides for in tensor.

#     Returns:
#     -------
#         Tensor map function.

#     """

#     def _map(
#         out: Storage,
#         out_shape: Shape,
#         out_strides: Strides,
#         in_storage: Storage,
#         in_shape: Shape,
#         in_strides: Strides,
#     ) -> None:
#         ## TODO: Implement for Task 3.1.
#         # raise NotImplementedError("Need to implement for Task 3.1")

#         # Module 4 Answer 3.1
#         if (
#             len(out_strides) != len(in_strides)
#             or (out_strides != in_strides).any()
#             or (out_shape != in_shape).any()
#         ):
#             for i in prange(len(out)):
#                 out_index: Index = np.empty(MAX_DIMS, np.int32)
#                 in_index: Index = np.empty(MAX_DIMS, np.int32)
#                 to_index(i, out_shape, out_index)
#                 broadcast_index(out_index, out_shape, in_shape, in_index)
#                 o = index_to_position(out_index, out_strides)
#                 j = index_to_position(in_index, in_strides)
#                 out[o] = fn(in_storage[j])
#         else:
#             for i in prange(len(out)):
#                 out[i] = fn(in_storage[i])

#     return njit(parallel=True)(_map)  # type: ignore

#     # Answer 3.1 end


# # Module 3 Answer
# # def tensor_zip(
# #     fn: Callable[[float, float], float],
# # ) -> Callable[
# #     [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
# # ]:


# def tensor_zip(fn: Callable[[float, float], float]) -> Any:
#     """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

#     Optimizations:

#     * Main loop in parallel
#     * All indices use numpy buffers
#     * When `out`, `a`, `b` are stride-aligned, avoid indexing

#     Args:
#     ----
#         fn: function maps two floats to float to apply.
#         out (array): storage for 'out' tensor.
#         out_shape (array): shape for 'out' tensor.
#         out_strides (array): strides for 'out' tensor.
#         a_storage (array): storage for 'a' tensor.
#         a_shape (array): shape for 'a' tensor.
#         a_strides (array): strides for 'a' tensor.
#         b_storage (array): storage for 'b' tensor.
#         b_shape (array): shape for 'b' tensor.
#         b_strides (array): strides for 'b' tensor.

#     Returns:
#     -------
#         Tensor zip function.

#     """

#     def _zip(
#         out: Storage,
#         out_shape: Shape,
#         out_strides: Strides,
#         a_storage: Storage,
#         a_shape: Shape,
#         a_strides: Strides,
#         b_storage: Storage,
#         b_shape: Shape,
#         b_strides: Strides,
#     ) -> None:
#         ## TODO: Implement for Task 3.1.
#         # raise NotImplementedError("Need to implement for Task 3.1")

#         # Module 3 Answer
#         if (
#             len(out_strides) != len(a_strides)
#             or len(out_strides) != len(b_strides)
#             or (out_strides != a_strides).any()
#             or (out_strides != b_strides).any()
#             or (out_shape != a_shape).any()
#             or (out_shape != b_shape).any()
#         ):
#             for i in prange(len(out)):
#                 out_index: Index = np.empty(MAX_DIMS, np.int32)
#                 a_index: Index = np.empty(MAX_DIMS, np.int32)
#                 b_index: Index = np.empty(MAX_DIMS, np.int32)
#                 to_index(i, out_shape, out_index)
#                 o = index_to_position(out_index, out_strides)
#                 broadcast_index(out_index, out_shape, a_shape, a_index)
#                 j = index_to_position(a_index, a_strides)
#                 broadcast_index(out_index, out_shape, b_shape, b_index)
#                 k = index_to_position(b_index, b_strides)
#                 out[o] = fn(a_storage[j], b_storage[k])
#         else:
#             for i in prange(len(out)):
#                 out[i] = fn(a_storage[i], b_storage[i])

#     return njit(parallel=True)(_zip)

#     # Module 3 Answer 3.1


# # Original
# # def tensor_reduce(
# #     fn: Callable[[float, float], float],
# # ) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:


# # Module 3 Answer
# def tensor_reduce(fn: Callable[[float, float], float]) -> Any:
#     """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

#     Optimizations:

#     * Main loop in parallel
#     * All indices use numpy buffers
#     * Inner-loop should not call any functions or write non-local variables

#     Args:
#     ----
#         fn: reduction function mapping two floats to float.
#         out (Storage): storage for 'out' tensor.
#         out_shape (Shape): shape for 'out' tensor.
#         out_strides (Strides): strides for 'out' tensor.
#         a_storage (Storage): storage for 'a' tensor.
#         a_shape (Shape): shape for 'a' tensor.
#         a_strides (Strides): strides for 'a' tensor.
#         reduce_dim (int): dimension to reduce out

#     Returns:
#     -------
#         Tensor reduce function

#     """

#     def _reduce(
#         out: Storage,
#         out_shape: Shape,
#         out_strides: Strides,
#         a_storage: Storage,
#         a_shape: Shape,
#         a_strides: Strides,
#         reduce_dim: int,
#     ) -> None:
#         ## TODO: Implement for Task 3.1.
#         # raise NotImplementedError("Need to implement for Task 3.1")

#         # Module 3 Answer
#         for i in prange(len(out)):
#             out_index: Index = np.empty(MAX_DIMS, np.int32)
#             reduce_size = a_shape[reduce_dim]
#             to_index(i, out_shape, out_index)
#             o = index_to_position(out_index, out_strides)
#             accum = out[o]
#             j = index_to_position(out_index, a_strides)
#             step = a_strides[reduce_dim]
#             for s in range(reduce_size):
#                 accum = fn(accum, a_storage[j])
#                 j += step
#             out[o] = accum

#     return njit(parallel=True)(_reduce)
#     # End of Module 3 answer 3.1


# def _tensor_matrix_multiply(
#     out: Storage,
#     out_shape: Shape,
#     out_strides: Strides,
#     a_storage: Storage,
#     a_shape: Shape,
#     a_strides: Strides,
#     b_storage: Storage,
#     b_shape: Shape,
#     b_strides: Strides,
# ) -> None:
#     """NUMBA tensor matrix multiply function.

#     Should work for any tensor shapes that broadcast as long as

#     ```
#     assert a_shape[-1] == b_shape[-2]
#     ```

#     Optimizations:

#     * Outer loop in parallel
#     * No index buffers or function calls
#     * Inner loop should have no global writes, 1 multiply.


#     Args:
#     ----
#         out (Storage): storage for `out` tensor
#         out_shape (Shape): shape for `out` tensor
#         out_strides (Strides): strides for `out` tensor
#         a_storage (Storage): storage for `a` tensor
#         a_shape (Shape): shape for `a` tensor
#         a_strides (Strides): strides for `a` tensor
#         b_storage (Storage): storage for `b` tensor
#         b_shape (Shape): shape for `b` tensor
#         b_strides (Strides): strides for `b` tensor

#     Returns:
#     -------
#         None : Fills in `out`

#     """
#     # # Basic compatibility check #added
#     # assert a_shape[-1] == b_shape[-2]

#     # a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
#     # b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

#     ## TODO: Implement for Task 3.2.
#     # raise NotImplementedError("Need to implement for Task 3.2")

#     # Module 3 Answer
#     a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
#     b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

#     for i1 in prange(out_shape[0]):
#         for i2 in prange(out_shape[1]):
#             for i3 in prange(out_shape[2]):
#                 a_inner = i1 * a_batch_stride + i2 * a_strides[1]
#                 b_inner = i1 * b_batch_stride + i3 * b_strides[2]
#                 acc = 0.0
#                 for _ in range(a_shape[2]):
#                     acc += a_storage[a_inner] * b_storage[b_inner]
#                     a_inner += a_strides[2]
#                     b_inner += b_strides[1]
#                 out_position = (
#                     i1 * out_strides[0] + i2 * out_strides[1] + i3 * out_strides[2]
#                 )
#                 out[out_position] = acc


# tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)
# # assert tensor_matrix_multiply is not None  # may need to remove later.


######################################

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Just-in-time compile a Python function using Numba with inline optimization.

    A convenience wrapper around Numba's @njit decorator that enables automatic inlining
    of the compiled function.

    Args:
    ----
       fn: Function to compile
       **kwargs: Additional keyword arguments passed to Numba's njit decorator

    Returns:
    -------
       Compiled function optimized with inlining enabled

    """
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        ## TODO: Implement for Task 3.1.
        # raise NotImplementedError("Need to implement for Task 3.1")

        if np.array_equal(in_strides, out_strides) and np.array_equal(
            in_shape, out_shape
        ):
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
        # Map s
        else:
            for i in prange(len(out)):
                # Create thread-local indices inside the parallel loop
                out_index = np.empty(MAX_DIMS, np.int32)
                in_index = np.empty(MAX_DIMS, np.int32)

                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                o = index_to_position(out_index, out_strides)
                j = index_to_position(in_index, in_strides)
                out[o] = fn(in_storage[j])

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        ## TODO: Implement for Task 3.1.
        # raise NotImplementedError("Need to implement for Task 3.1")

        if (
            np.array_equal(a_strides, b_strides)
            and np.array_equal(a_strides, out_strides)
            and np.array_equal(a_shape, b_shape)
            and np.array_equal(a_shape, out_shape)
        ):
            for i in prange(len(out)):
                out[i] = fn(a_storage[i], b_storage[i])

        # Zip S
        else:
            for i in prange(len(out)):
                # Thread-local indices
                out_index = np.empty(MAX_DIMS, np.int32)
                a_index = np.empty(MAX_DIMS, np.int32)
                b_index = np.empty(MAX_DIMS, np.int32)

                to_index(i, out_shape, out_index)
                o = index_to_position(out_index, out_strides)
                broadcast_index(out_index, out_shape, a_shape, a_index)
                j = index_to_position(a_index, a_strides)
                broadcast_index(out_index, out_shape, b_shape, b_index)
                k = index_to_position(b_index, b_strides)
                out[o] = fn(a_storage[j], b_storage[k])

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        ## TODO: Implement for Task 3.1.
        # raise NotImplementedError("Need to implement for Task 3.1")

        # Reduce S
        out_index = np.zeros(MAX_DIMS, np.int32)
        reduce_size = a_shape[reduce_dim]

        # Parallelize the outer loop over output elements
        for i in prange(len(out)):
            # Thread-local indices
            out_index = np.empty(MAX_DIMS, np.int32)
            local_index = np.empty(MAX_DIMS, np.int32)

            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)

            # Copy indices to local
            for j in range(len(out_shape)):
                local_index[j] = out_index[j]

            # Sequential reduction
            for s in range(reduce_size):
                local_index[reduce_dim] = s
                j = index_to_position(local_index, a_strides)
                out[o] = fn(out[o], a_storage[j])

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    # # Basic compatibility check #added
    # assert a_shape[-1] == b_shape[-2]

    # a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    # b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    ## TODO: Implement for Task 3.2.
    # raise NotImplementedError("Need to implement for Task 3.2")

    # Multi S
    # Basic compatibility check
    assert a_shape[-1] == b_shape[-2]

    # Get batch strides (0 if not batched)
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # Get key dimensions
    batch_size = max(out_shape[0], 1)  # Number of batches
    M = out_shape[1]  # Rows in output
    N = out_shape[2]  # Cols in output
    K = a_shape[-1]  # Shared dimension (cols in A, rows in B)

    # Parallel over both batch and M dimensions for better utilization
    for batch in prange(batch_size):
        for row in range(M):
            for col in range(N):
                a_pos = batch * a_batch_stride + row * a_strides[-2]
                b_pos = batch * b_batch_stride + col * b_strides[-1]

                acc = 0.0
                for _ in range(K):
                    acc += a_storage[a_pos] * b_storage[b_pos]
                    a_pos += a_strides[-1]
                    b_pos += b_strides[-2]

                out_pos = (
                    batch * out_strides[0]
                    + row * out_strides[-2]
                    + col * out_strides[-1]
                )

                out[out_pos] = acc


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
