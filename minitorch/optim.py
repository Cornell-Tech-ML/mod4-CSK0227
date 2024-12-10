from typing import Sequence

from .module import Parameter
from .scalar import Scalar


class Optimizer:
    def __init__(self, parameters: Sequence[Parameter]):
        self.parameters = parameters


class SGD(Optimizer):
    def __init__(self, parameters: Sequence[Parameter], lr: float = 1.0):
        super().__init__(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        """Zeros out the gradients of all parameters in the optimizer.

        This method iterates through all parameters and sets their gradients
        to None. It handles parameters that may have either a 'derivative'
        or 'grad' attribute as their gradient storage.

        Returns : None

        """
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.value.derivative = None
            if hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.value.grad = None

    def step(self) -> None:
        """Performs a single optimization step by updating all parameters based on their gradients.

        Updates each parameter using the standard SGD update rule:
            param = param - learning_rate * gradient

        The method handles two types of parameters:
        1. Autograd parameters with .derivative attribute:
            - Extracts raw data and creates new Scalar with updated value
        2. Manual gradient parameters with .grad attribute:
            - Performs direct arithmetic update on parameter value

        Returns : None

        """
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.update(Scalar(p.value.data - self.lr * p.value.derivative))
            elif hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.update(p.value - self.lr * p.value.grad)
