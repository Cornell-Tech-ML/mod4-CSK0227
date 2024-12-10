"""Deep learning library implementing tensors, automatic differentiation, and neural network modules.

This package provides a complete toolkit for deep learning including:

Core Functionality:
    - Tensor operations and manipulations (tensor, tensor_data, tensor_functions, tensor_ops)
    - Automatic differentiation engine (autodiff)
    - Scalar operations for single-value computations (scalar, scalar_functions)

Neural Networks:
    - Neural network modules and layers (nn)
    - Base module system for building networks (module)

Optimization & Training:
    - Optimization algorithms (optim)
    - Dataset handling and loading (datasets)
    - Testing utilities (testing)

Hardware Acceleration:
    - Fast CPU operations (fast_ops, fast_conv)
    - CUDA GPU support (cuda_ops)

Note:
----
    All submodules are imported and exposed at the package level for convenience.
    Users can access all functionality directly from the root package.

"""

from .testing import MathTest, MathTestVariable  # type: ignore # noqa: F401,F403
from .datasets import *  # noqa: F401,F403
from .optim import *  # noqa: F401,F403
from .tensor import *  # noqa: F401,F403
from .testing import *  # noqa: F401,F403
from .nn import *  # noqa: F401,F403
from .fast_conv import *  # noqa: F401,F403
from .tensor_data import *  # noqa: F401,F403
from .tensor_functions import *  # noqa: F401,F403
from .tensor_ops import *  # noqa: F401,F403
from .scalar import *  # noqa: F401,F403
from .scalar_functions import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .autodiff import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .autodiff import *  # noqa: F401,F403
from .tensor import *  # noqa: F401,F403
from .datasets import *  # noqa: F401,F403
from .testing import *  # noqa: F401,F403
from .optim import *  # noqa: F401,F403
from .tensor_ops import *  # noqa: F401,F403
from .fast_ops import *  # noqa: F401,F403
from .cuda_ops import *  # noqa: F401,F403
from . import fast_ops, cuda_ops  # noqa: F401,F403

from .operators import is_close  # noqa: F401,F403
