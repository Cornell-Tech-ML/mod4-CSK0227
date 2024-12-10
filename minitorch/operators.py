"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.


# Mathematical functions:
# - mul
def mul(a: float, b: float) -> float:
    """Multiplies two real numbers.

    Args:
    ----
        a (float): The first real number.
        b (float): The second real number.

    Returns:
    -------
        float: The product of a and b.

    """
    return a * b


# # add div
# def div(a: float, b: float) -> float:
#     """Divide two floating-point numbers.

#     Args:
#     ----
#         a (float): The first real number.
#         b (float): The second real number.

#     Returns:
#     -------
#         float: The divid a by b.

#     """
#     return float(a/b)


# - id
def id(a: float) -> float:
    """Returns the input unchanged.

    Args:
    ----
        a (float): The input value.

    Returns:
    -------
        float: The same input value, unchanged.

    """
    return a


# - add
def add(a: float, b: float) -> float:
    """Adds two numbers.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        float: The sum of a and b.

    """
    return a + b


# - sub 240926+
def sub(a: float, b: float) -> float:
    """Subtracts two numbers.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        float: The difference between a and b.

    """
    return a - b


# - neg
def neg(a: float) -> float:
    """Negates a float number.

    Args:
    ----
        a (float): The input number.

    Returns:
    -------
        float: The negated value of a.

    """
    return -a


# - lt
def lt(a: float, b: float) -> float:
    """Checks the fisrt number is less than another.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        bool: True if a is less than b, False otherwise.

    """
    return 1.0 if a < b else 0.0


# - GT 092524+
def GT(a: float, b: float) -> float:
    """Checks the fisrt number is greater than another.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        bool: True if a is greater than b, False otherwise.

    """
    return 1.0 if a > b else 0.0


# - eq
def eq(a: float, b: float) -> float:
    """Checks if two numbers are equal.

    Args:
    ----
        a (float): The first number to compare.
        b (float): The second number to compare.

    Returns:
    -------
        bool: True if `a` is equal to `b`, False otherwise.

    """
    return 1.0 if a == b else 0.0


# - max
def max(x: float, y: float) -> float:
    """Returns the larger of two numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The larger of a and b.

    """
    return x if x > y else y


# - is_close
# For is_close:
# $f(x) = |x - y| < 1e-2$
def is_close(a: float, b: float) -> float:
    """Checks if two numbers are close in value within a tolerance.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.
        The tolerance is assgined as 1e-2

    Returns:
    -------
        bool: True if a and b are close within the given tolerance, False otherwise.

    """
    return abs(a - b) < 1e-2


# - sigmoid
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
def sigmoid(a: float) -> float:
    """Calculates the sigmoid function.

    Args:
    ----
        a (float): The input value.

    Returns:
    -------
        float: The result of the sigmoid function: 1 / (1 + exp(-a)), if a >= 0. Or exp(a)/(1+exp(a)), if a <0

    """
    c: float
    if a >= 0:
        c = 1.0 / (1.0 + math.exp(-a))
    else:
        c = math.exp(a) / (1.0 + math.exp(a))
    return c


# - relu
def relu(a: float) -> float:
    """Applies the ReLU (Rectified Linear Unit) activation function.

    Args:
    ----
        a (float): The input value.

    Returns:
    -------
        float: a if a is greater than 0, otherwise 0.

    """
    return a if a > 0 else 0.0


EPS = 1e-6


# - log
def log(a: float) -> float:
    """Calculates the natural logarithm.

    Args:
    ----
        a (float): The input value, must be positive.

    Returns:
    -------
        float: The natural logarithm of a.

    Raises:
    ------
        ValueError: If a is less than or equal to 0.

    """
    # if a > 0:
    return math.log(a + EPS)
    # else:
    #    raise ValueError("Input should be positive")


# - exp
def exp(a: float) -> float:
    """Calculates the exponential function (e^a).

    Args:
    ----
        a (float): The input value.

    Returns:
    -------
        float: The result of exponential to the power of a.

    """
    return math.exp(a)


# - log_back
def log_back(a: float, b: float) -> float:
    """Computes the derivative of the natural logarithm multiplied by a second argument.

    Args:
    ----
        a (float): The input value (must be positive).
        b (float): The second argument for backpropagation.

    Returns:
    -------
        float: The product of the derivative of log(a) and b.

    """
    return b / (a + EPS)


# - inv
def inv(a: float) -> float:
    """Calculates the reciprocal (1 / a).

    Args:
    ----
        a (float): The input value.

    Returns:
    -------
        float: The reciprocal of a.

    Raises:
    ------
        ValueError: If a is 0, undefined.

    """
    return 1.0 / a
    # else:
    #     raise ValueError("Input should not be 0")


# - inv_back
def inv_back(a: float, b: float) -> float:
    """Computes the derivative of the reciprocal function multiplied by a second argument.

    Args:
    ----
        a (float): The input value.
        b (float): The second argument for backpropagation.

    Returns:
    -------
        float: The product of the derivative of 1/a and b.

    """
    return -b / (a * a)
    # else:
    #     raise ValueError("Input should not be 0")


# - relu_back
def relu_back(a: float, b: float) -> float:
    """Computes the derivative of the ReLU function multiplied by a second argument.

    Args:
    ----
        a (float): The input value.
        b (float): The second argument for backpropagation.

    Returns:
    -------
        float: The product of the derivative of ReLU and b.
               If a > 0, returns b; otherwise, returns 0.

    """
    return b if a > 0 else 0.0


#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# - sigmoid_back 092524+
def sigmoid_back(a: float, b: float) -> float:
    """Computes the derivative of the sigmoid function multiplied by a second argument.

    Args:
    ----
        a (float): The input value.
        b (float): The second argument for backpropagation.

    Returns:
    -------
        float: The product of the derivative of sigmoid and b.

    """
    return b * sigmoid(a) * (1 - sigmoid(a))


# - exp_back
def exp_back(a: float, b: float) -> float:
    """Computes the derivative of the exponential function multiplied by a second argument.

    Args:
    ----
        a (float): The input value.
        b (float): The second argument for backpropagation.

    Returns:
    -------
        float: The product of the derivative of exp and b.

    """
    return b * exp(a)


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.


# Implement the following core functions
# - map
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], list[float]]:
    """Map function that returns a function with elements of an iterable.

    Args:
    ----
        fn (Callable[[float], float]): A function that takes a float as a input and returns a float.

    Returns:
    -------
       apply : A function that returns iterable of floats as list .

    """

    def apply(ls: Iterable[float]) -> list[float]:
        """Applies the function to list iterable of floats.

        Args:
        ----
            ls (Iterable[float]): An iterable of floats.

        Returns:
        -------
            [fn(a) for a in ls]: A list containing the elements of `ls` for which `fn` returns True.

        """
        return [fn(a) for a in ls]

    return apply


# - zipWith
def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], list[float]]:
    """ZipWith function that returns a function with elements of an iterable.

    Args:
    ----
        fn (Callable[[float,float], float]): A function that takes two floats as inputs and returns a float.

    Returns:
    -------
       apply : A function that returns iterable of floats as list .

    """

    def apply(ls1: Iterable[float], ls2: Iterable[float]) -> list[float]:
        """Applies the function to list iterable of floats.

        Args:
        ----
            ls1 (Iterable[float]): iterables of floats.
            ls2 (Iterable[float]): iterables of floats.

        Returns:
        -------
            list[float]: A new iterable list containing the elements of `ls` for which `fn` returns True.

        """
        return [fn(a, b) for a, b in zip(ls1, ls2)]

    return apply


# - reduce
def reduce(
    fn: Callable[[float, float], float], initial: float
) -> Callable[[Iterable[float]], float]:
    """Create a reduction function that applies a binary operation to an iterable.

    Args:
    ----
        fn (Callable[[float, float], float]): A binary function that takes two
            floats and returns a float.
        initial (float): The initial value for the reduction.

    Returns:
    -------
        Callable[[Iterable[float]], float]: A function that takes an iterable of
        floats and returns a single float value.

    """

    def apply(ls: Iterable[float]) -> float:
        """Applies the function to list iterable of floats.

        Args:
        ----
            ls (Iterable[float,float]): iterables of floats.

        Returns:
        -------
            ret[float]: A new iterable list containing the elements of `ls` for which `fn` returns True.

        """
        result = initial
        for a in ls:
            result = fn(result, a)
        return result

    return apply


#
# Use these to implement
# - negList : negate a list
def negList(ls: list[float]) -> list[float]:
    """Negate all elements in a list using map.

    Args:
    ----
        ls (list[float]): A list of float values to be negated.

    Returns:
    -------
        list[float]: A new list with all elements negated.

    """
    return map(neg)(ls)


# - addLists : add two lists together
def addLists(ls1: list[float], ls2: list[float]) -> list[float]:
    """Add corresponding elements from two lists using zipWith.

    Args:
    ----
        ls1 (List[float]): The float list as fist input.
        ls2 (List[float]): The float list as second input.

    Returns:
    -------
        zipWith(add)(ls1,ls2): Return a new list with sums of corresponding elements.

    """
    return zipWith(add)(ls1, ls2)


# - sum: sum lists
def sum(ls: Iterable[float]) -> float:
    """Sum all elements in a list using reduce.

    Args:
    ----
        ls (Iterable[float]): The input iterable of floats.

    Returns:
    -------
        return reduce(add,0)(ls): Returns a float which correspond to the sum of all elements in the list.

    """
    return reduce(add, 0)(ls)


# - prod: take the product of lists
def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in a list using reduce.

    Args:
    ----
        ls (Iterable[float]): The float list as input.

    Returns:
    -------
        reduce(mul,1)(ls) : The product of all elements in the list.

    """
    return reduce(mul, 1)(ls)


# TODO: Implement for Task 0.3.
