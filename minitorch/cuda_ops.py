# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Any, Callable, Optional  # , TypeVar #Module 3 Answer

import numba
from numba import cuda

# from numba.cuda import jit as _jit #Module 3 Answer
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

# FakeCUDAKernel = Any #Module 3 Answer

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

# Fn = TypeVar("Fn") #Module 3 Answer

# #Original #Module 3 Answer
# # def device_jit(fn: Fn, **kwargs) -> Fn:
# def device_jit(fn: Fn, **kwargs: Any) -> Fn:
#     """Applies Just-In-Time (JIT) compilation to a function with device-specific optimizations.

#     This function wraps a given function `fn` with JIT compilation, targeting device-specific
#     execution. It uses the underlying `_jit` function with `device=True` to enable optimization
#     for the device where the function will be executed (e.g., GPU or TPU).

#     Args:
#     ----
#     fn : Fn
#         The function to be JIT-compiled with device optimizations.

#     **kwargs : Additional keyword arguments to pass to the `_jit` function.


#     Returns:
#     -------
#     Fn
#         The JIT-compiled version of the input function `fn`, optimized for device execution.

#     """
#     return _jit(device=True, **kwargs)(fn)  # type: ignore


# #Original #Module 3 Answer
# # def jit(fn, **kwargs) -> FakeCUDAKernel:
# def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
#     """Applies Just-In-Time (JIT) compilation to a function, returning a CUDA-compatible kernel.

#     This function wraps a given function `fn` with JIT compilation to optimize it for
#     accelerated execution on CUDA-enabled devices. It leverages the underlying `_jit`
#     function to perform the compilation with additional configuration options provided
#     through `kwargs`.

#     Args:
#     ----
#     fn : Callable
#         The function to be JIT-compiled for CUDA execution.
#     **kwargs : dict
#         Additional keyword arguments to pass to the `_jit` function, allowing for
#         customized compilation settings.

#     Returns:
#     -------
#     FakeCUDAKernel
#         A JIT-compiled CUDA kernel version of the input function `fn`, suitable for
#         execution on CUDA-enabled devices.

#     """
#     return _jit(**kwargs)(fn)  # type: ignore

# #Original #Module 3 Answer
# to_index = device_jit(to_index)
# index_to_position = device_jit(index_to_position)
# broadcast_index = device_jit(broadcast_index)

# Module 3 Answer
to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

THREADS_PER_BLOCK = 32


##Need to work Dec 5th 2024
class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # Original
        # cufn: Callable[[float], float] = device_jit(fn)
        # f = tensor_map(cufn)

        # Module 3 Answer
        f = tensor_map(cuda.jit(device=True)(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Applies a binary function element-wise to two tensors using Just-In-Time (JIT) compilation and CUDA kernel execution.

        This method takes a binary function `fn` that operates on two `float` values and
        returns a new function that applies `fn` element-wise to pairs of elements from two
        tensors, `a` and `b`. The function `fn` is JIT-compiled for device-specific execution,
        and the result is computed in parallel using CUDA kernels.

        Args:
        ----
        fn : Callable[[float, float], float]
            A binary function that takes two float inputs and returns a float output.
            This function is applied element-wise to the tensor inputs `a` and `b`.

        Returns:
        -------
        Callable[[Tensor, Tensor], Tensor]
            A function that takes two tensors as input and returns a tensor resulting from
            the element-wise application of `fn` to corresponding elements in `a` and `b`.

        """
        # #Original
        # cufn: Callable[[float, float], float] = device_jit(fn)
        # f = tensor_zip(cufn)

        # Module 3 Answer
        f = tensor_zip(cuda.jit(device=True)(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Reduces a tensor along a specified dimension using a binary function, optimized for parallel execution on a CUDA-enabled device.

        This method takes a binary function `fn` that combines two `float` values, and
        returns a function that applies `fn` to reduce elements along a specified dimension
        in a tensor. The reduction is performed using JIT-compiled CUDA kernels, allowing
        for efficient computation on large tensors.

        Args:
        ----
        fn : Callable[[float, float], float]
            A binary function that takes two floats and returns a float output. This function
            is used to iteratively combine elements along the specified dimension.
        start : float, optional
            The initial value for the reduction operation, by default `0.0`.

        Returns:
        -------
        Callable[[Tensor, int], Tensor]
            A function that takes a tensor and a dimension index and returns a tensor
            resulting from the reduction along that dimension.

        """
        # #Original
        # cufn: Callable[[float, float], float] = device_jit(fn)
        # f = tensor_reduce(cufn)

        # Module 3 Answer
        f = tensor_reduce(cuda.jit(device=True)(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Performs a matrix multiplication between two tensors, supporting broadcasting and optimized execution on CUDA-enabled devices.

        This method handles both 2D and 3D tensors by internally reshaping 2D tensors
        to 3D format for consistent batch processing, then performing matrix multiplication
        in parallel using CUDA kernels. After computation, the result is reshaped to 2D if
        both inputs were originally 2D.

        Args:
        ----
        a : Tensor
            The first input tensor with shape (..., M, K) or (M, K).
        b : Tensor
            The second input tensor with shape (..., K, N) or (K, N).

            Note: The inner dimensions of `a` and `b` (i.e., `K`) must match for matrix
            multiplication, otherwise an assertion error is raised.

        Returns:
        -------
        Tensor
            The result of matrix multiplication, with shape broadcasted over batch dimensions
            to (..., M, N) if inputs are broadcast-compatible, or (M, N) if both inputs are 2D.

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

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement

# # Original
# def tensor_map(
#     fn: Callable[[float], float],
# ) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:


# Module 3 answer
def tensor_map(fn: Callable[[float], float]) -> Any:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.
        out (Storage): storage for out tensor.
        out_shape (Shape): shape for out tensor.
        out_strides (Strides): strides for out tensor.
        out_size (int) : size for out tensor.
        in_storage (Storage): storage for in tensor.
        in_shape (Shape): shape for in tensor.
        in_strides (Strides): strides for in tensor.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")
        # # Original
        # if i < out_size:
        #     # Convert position to output index
        #     to_index(i, out_shape, out_index)
        #     # Convert output index to input index
        #     broadcast_index(out_index, out_shape, in_shape, in_index)

        #     # Get positions in storage
        #     out_pos = index_to_position(out_index, out_strides)
        #     in_pos = index_to_position(in_index, in_strides)

        #     # Apply function and store result
        #     out[out_pos] = fn(in_storage[in_pos])

        # Module 3 Answer 3.3
        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            o = index_to_position(out_index, out_strides)
            j = index_to_position(in_index, in_strides)
            out[o] = fn(in_storage[j])
        # Module 3 Answer 3.3 End

    return cuda.jit()(_map)  # type: ignore


# #Original
# def tensor_zip(
#     fn: Callable[[float, float], float],
# ) -> Callable[
#     [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
# ]:
def tensor_zip(fn: Callable[[float, float], float]) -> Any:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.
        out (array): storage for 'out' tensor.
        out_shape (array): shape for 'out' tensor.
        out_strides (array): strides for 'out' tensor.
        out_size (array): size for 'out' tensor.
        a_storage (array): storage for 'a' tensor.
        a_shape (array): shape for 'a' tensor.
        a_strides (array): strides for 'a' tensor.
        b_storage (array): storage for 'b' tensor.
        b_shape (array): shape for 'b' tensor.
        b_strides (array): strides for 'b' tensor.

    Returns:
    -------
        None : Fills in 'out'

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")
        # # Original
        # if i < out_size:
        #     # Convert position to indices
        #     to_index(i, out_shape, out_index)
        #     # Broadcast indices
        #     broadcast_index(out_index, out_shape, a_shape, a_index)
        #     broadcast_index(out_index, out_shape, b_shape, b_index)

        #     # Get positions in storage
        #     out_pos = index_to_position(out_index, out_strides)
        #     a_pos = index_to_position(a_index, a_strides)
        #     b_pos = index_to_position(b_index, b_strides)

        #     # Apply function and store result
        #     out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

        # Module 3 answer 3.3
        if i < out_size:
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            j = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            k = index_to_position(b_index, b_strides)
            out[o] = fn(a_storage[j], b_storage[k])
        # Module 3 answer 3.3 end

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """A practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # # TODO: Implement for Task 3.3.
    # raise NotImplementedError("Need to implement for Task 3.3")

    # # Original
    # # Initialize cache
    # cache[pos] = 0.0
    # if i < size:
    #     cache[pos] = a[i]
    # cuda.syncthreads()

    # # Reduction within block using shared memory
    # offset = BLOCK_DIM // 2
    # while offset > 0:
    #     if pos < offset and i + offset < size:
    #         cache[pos] += cache[pos + offset]
    #     cuda.syncthreads()
    #     offset //= 2

    # # Write result for this block
    # if pos == 0:
    #     out[cuda.blockIdx.x] = cache[0]

    # Module 3 Answer 3.3
    if i < size:
        val = float(a[i])
        cache[pos] = val
        cuda.syncthreads()
    else:
        cache[pos] = 0.0

    if i < size:
        for j in [1, 2, 4, 8, 16]:
            if pos % (j * 2) == 0:
                cache[pos] += cache[pos + j]
                cuda.syncthreads()
        if pos == 0:
            out[cuda.blockIdx.x] = cache[0]
    # Module 3 Answer 3.3 End


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Computes the sum of elements in a tensor `a` using a CUDA-enabled kernel for parallel reduction, and stores the result in a new `TensorData` object.

    This function prepares a tensor `out` to store the result, configures CUDA kernel
    parameters, and then launches a JIT-compiled CUDA kernel `jit_sum_practice` to
    perform the summation. The result is returned in a `TensorData` object.

    Args:
    ----
    a : Tensor
        The input tensor containing elements to be summed.

    Returns:
    -------
    TensorData
        A `TensorData` object with shape (2,) containing the result of the sum in the
        first position. The second position in `TensorData` is unused but reserved for
        consistency with CUDA parallelization.

    Notes:
    -----
    - CUDA kernel execution is configured with a number of blocks and threads per block
      based on `THREADS_PER_BLOCK` and the size of `a`.
    - The output tensor `out` is moved to CUDA memory for compatibility with CUDA operations.
    - This function uses a parallel reduction technique to sum elements efficiently on
      CUDA-enabled devices.

    Internal Workflow:
    ------------------
    1. Initializes `out`, a `TensorData` object of shape (2,) to store the summation result.
    2. Configures `blockspergrid` and `threadsperblock` based on the size of `a`.
    3. Moves `out` to CUDA memory.
    4. Executes the JIT-compiled CUDA kernel `jit_sum_practice` to compute the sum.

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


# #Original
# def tensor_reduce(
#     fn: Callable[[float, float], float],
# ) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:


# Module 3 Answer
def tensor_reduce(fn: Callable[[float, float], float]) -> Any:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.
        out (Storage): storage for 'out' tensor.
        out_shape (Shape): shape for 'out' tensor.
        out_strides (Strides): strides for 'out' tensor.
        out_size (int): size for 'out' tensor.
        a_storage (Storage): storage for 'a' tensor.
        a_shape (Shape): shape for 'a' tensor.
        a_strides (Strides): strides for 'a' tensor.
        reduce_dim (int): dimension to reduce out

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x
        cache[pos] = reduce_value

        # # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")

        # # Original
        # # Convert output position to index
        # to_index(out_pos, out_shape, out_index)

        # # Initialize cache
        # #cache[pos] = reduce_value

        # # Calculate number of elements to reduce
        # reduce_size = a_shape[reduce_dim]

        # # Handle case where thread should process data
        # if pos < reduce_size:
        #     # Copy reduce_dim index
        #     out_index[reduce_dim] = pos
        #     a_pos = index_to_position(out_index, a_strides)
        #     cache[pos] = a_storage[a_pos]
        # cuda.syncthreads()

        # # Parallel reduction in shared memory
        # offset = BLOCK_DIM // 2
        # while offset > 0:
        #     if pos < offset:
        #         cache[pos] = fn(cache[pos], cache[pos + offset])
        #     cuda.syncthreads()
        #     offset //= 2

        # # Write final result
        # if pos == 0:
        #     out_pos = index_to_position(out_index, out_strides)
        #     out[out_pos] = cache[0]

        if out_pos < out_size:
            to_index(out_pos, out_shape, out_index)
            o = index_to_position(out_index, out_strides)

            out_index[reduce_dim] = out_index[reduce_dim] * BLOCK_DIM + pos
            if out_index[reduce_dim] < a_shape[reduce_dim]:
                in_a = index_to_position(out_index, a_strides)
                cache[pos] = a_storage[in_a]
                cuda.syncthreads()
                x = 0
                while 2**x < BLOCK_DIM:
                    j = 2**x
                    if pos % (j * 2) == 0:
                        cache[pos] = fn(cache[pos], cache[pos + j])
                        cuda.syncthreads()
                    x += 1
            if pos == 0:
                out[o] = cache[0]

    return cuda.jit()(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """A practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # # TODO: Implement for Task 3.3.
    # raise NotImplementedError("Need to implement for Task 3.3")

    # # Original
    # # Shared memory for tile-based multiplication
    # a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    # b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # # Thread indices
    # tx = cuda.threadIdx.x
    # ty = cuda.threadIdx.y

    # # Output indices
    # i = cuda.blockIdx.x * BLOCK_DIM + tx
    # j = cuda.blockIdx.y * BLOCK_DIM + ty

    # # Initialize output
    # if i < size and j < size:
    #     out[i * size + j] = 0.0

    # # Compute using tiles
    # for k in range(0, size, BLOCK_DIM):
    #     # Load tiles into shared memory
    #     if i < size and k + ty < size:
    #         a_shared[tx, ty] = a[i * size + (k + ty)]
    #     if k + tx < size and j < size:
    #         b_shared[tx, ty] = b[(k + tx) * size + j]
    #     cuda.syncthreads()

    #     # Compute partial dot product
    #     if i < size and j < size:
    #         for kk in range(min(BLOCK_DIM, size - k)):
    #             out[i * size + j] += a_shared[tx, kk] * b_shared[kk, ty]
    #     cuda.syncthreads()

    # Module 3 Answer 3.3
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    i = cuda.threadIdx.x
    j = cuda.threadIdx.y

    if i >= size or j >= size:
        return

    a_shared[i, j] = a[size * i + j]
    b_shared[i, j] = b[size * i + j]
    cuda.syncthreads()

    accum = 0.0
    for k in range(size):
        accum += a_shared[i, k] * b_shared[k, j]

    out[size * i + j] = accum
    # Module 3 Answer 3.3


jit_mm_practice = cuda.jit()(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Performs a matrix multiplication between two square tensors `a` and `b` using a CUDA-enabled kernel for parallel computation, and stores the result in a new `TensorData` object.

    This function prepares an output tensor `out` to store the result, configures the
    CUDA kernel parameters for block and thread counts, and then launches a JIT-compiled
    CUDA kernel `jit_mm_practice` to perform the matrix multiplication.

    Args:
    ----
    a : Tensor
        The first input tensor, a square matrix with shape (size, size).
    b : Tensor
        The second input tensor, also a square matrix with shape (size, size).

        Note: Both `a` and `b` must be square matrices with the same dimensions, as
        assumed by the function.

    Returns:
    -------
    TensorData
        A `TensorData` object with shape (size, size) containing the result of the matrix
        multiplication `a @ b`.

    Notes:
    -----
    - The CUDA kernel execution is configured with `THREADS_PER_BLOCK` threads per
      block and a single block per grid, as the function assumes `a` and `b` are small
      enough to fit within this configuration.
    - The output tensor `out` is moved to CUDA memory to be compatible with CUDA operations.
    - This function uses parallel computation on CUDA-enabled devices to achieve efficient
      matrix multiplication.

    Internal Workflow:
    ------------------
    1. Initializes `out`, a `TensorData` object of shape (size, size) to store the matrix product.
    2. Configures `threadsperblock` and `blockspergrid` based on `THREADS_PER_BLOCK`.
    3. Moves `out` to CUDA memory.
    4. Executes the JIT-compiled CUDA kernel `jit_mm_practice` to compute the matrix product.

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    BLOCK_DIM = 32
    # # Original
    # # Batch dimension - fixed
    # batch = cuda.blockIdx.z

    # a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    # b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # # The final position c[i, j]
    # i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # # The local position in the block.
    # pi = cuda.threadIdx.x
    # pj = cuda.threadIdx.y

    # # Code Plan:
    # # 1) Move across shared dimension by block dim.
    # #    a) Copy into shared memory for a matrix.
    # #    b) Copy into shared memory for b matrix
    # #    c) Compute the dot produce for position c[i, j]
    # # # TODO: Implement for Task 3.4.
    # # raise NotImplementedError("Need to implement for Task 3.4")

    # # Output position in global memory
    # out_pos = (
    #     batch * (out_strides[0] if out_shape[0] > 1 else 0)
    #     + i * out_strides[1]
    #     + j * out_strides[2]
    # )

    # # Initialize output
    # acc = 0.0

    # # Move across shared dimension by block
    # for k in range(0, a_shape[2], BLOCK_DIM):
    #     # Load a tile into shared memory
    #     if i < a_shape[1] and k + pj < a_shape[2]:
    #         a_pos = batch * a_batch_stride + i * a_strides[1] + (k + pj) * a_strides[2]
    #         a_shared[pi, pj] = a_storage[a_pos]
    #     else:
    #         a_shared[pi, pj] = 0.0

    #     # Load b tile into shared memory
    #     if k + pi < b_shape[1] and j < b_shape[2]:
    #         b_pos = batch * b_batch_stride + (k + pi) * b_strides[1] + j * b_strides[2]
    #         b_shared[pi, pj] = b_storage[b_pos]
    #     else:
    #         b_shared[pi, pj] = 0.0
    #     cuda.syncthreads()

    #     # Compute partial dot product
    #     if i < out_shape[1] and j < out_shape[2]:
    #         for kk in range(min(BLOCK_DIM, a_shape[2] - k)):
    #             acc += a_shared[pi, kk] * b_shared[kk, pj]
    #     cuda.syncthreads()

    #     # Write final result
    # if i < out_shape[1] and j < out_shape[2]:
    #     out[out_pos] = acc

    # Module 4 Answer
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    batch = cuda.blockIdx.z
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    accum = 0.0
    for k_start in range(0, a_shape[2], BLOCK_DIM):
        k = k_start + pj
        if i < a_shape[1] and k < a_shape[2]:
            a_shared[pi, pj] = a_storage[
                a_batch_stride * batch + a_strides[1] * i + a_strides[2] * k
            ]

        k = k_start + pi
        if j < b_shared[2] and k < b_shape[1]:
            b_shape[pi, pj] = b_storage[
                b_batch_stride * batch + b_strides[1] * k + b_strides[2] * j
            ]
        cuda.syncthreads()

        for k in range(BLOCK_DIM):
            if (k_start + k) < a_shape[2]:
                accum += a_shared[pi, k] * b_shared[k, pj]

    if i < out_shape[1] and j < out_shape[2]:
        out[out_strides[0] * batch + out_strides[1] * i + out_strides[2] * j] = accum
    # Module 3 answer 3.4


tensor_matrix_multiply = cuda.jit(_tensor_matrix_multiply)
