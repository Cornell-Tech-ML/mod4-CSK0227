import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generate N random 2D points.

    Each point's coordinates are randomly chosen from [0, 1).

    Args:
    ----
        N (int): Number of points to generate.

    Returns:
    -------
        List[Tuple[float, float]]: List of N 2D points.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generate a simple dataset with N points.

    Creates N 2D points and labels them based on the x-coordinate:
    1 if x < 0.5, else 0.

    Args:
    ----
        N (int): Number of data points to generate.

    Returns:
    -------
        Graph: Contains N points and their labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generate a diagonal dataset with N points.

    Creates N 2D points and labels them based on their position relative to the line x + y = 0.5:
    1 if x + y < 0.5, else 0.

    Args:
    ----
        N (int): Number of data points to generate.

    Returns:
    -------
        Graph: Contains N points and their diagonal labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generate a split dataset with N points.

    Creates N 2D points and labels them based on the x-coordinate:
    1 if x < 0.2 or x > 0.8, else 0.

    Args:
    ----
        N (int): Number of data points to generate.

    Returns:
    -------
        Graph: Contains N points and their split labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generate a dataset for the XOR problem.

    Creates N 2D points and labels them according to XOR logic:
    1 if exactly one coordinate > 0.5, else 0.

    Args:
    ----
        N (int): Number of data points to generate.

    Returns:
    -------
        Graph: Contains N points and their XOR labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generate a circular dataset with N points.

    Creates N 2D points and labels them based on their distance from (0.5, 0.5):
    1 if the point is outside a circle with radius sqrt(0.1) centered at (0.5, 0.5), else 0.

    Args:
    ----
        N (int): Number of data points to generate.

    Returns:
    -------
        Graph: Contains N points and their circular boundary labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generate a spiral dataset with N points.

    Creates two intertwined spirals, each with N/2 points:
    - One spiral starts at the center and moves outward clockwise
    - The other spiral starts at the center and moves outward counterclockwise
    Points on the first spiral are labeled 0, and points on the second are labeled 1.

    Args:
    ----
        N (int): Total number of data points to generate (should be even).

    Returns:
    -------
        Graph: Contains N points arranged in two spirals and their labels (0 or 1).

    Note:
    ----
        The spirals are centered at (0.5, 0.5) and scaled to fit within the unit square.

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
