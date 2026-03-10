from __future__ import annotations

from math import acos
from typing import List


EPS = 1e-12


def copy_matrix(a: List[List[float]]) -> List[List[float]]:
    return [row[:] for row in a]


def copy_vector(v: List[float]) -> List[float]:
    return v[:]


def matrix_vector_multiply(a: List[List[float]], x: List[float]) -> List[float]:
    result = []
    for row in a:
        s = 0.0
        for aij, xj in zip(row, x):
            s += aij * xj
        result.append(s)
    return result


def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def vector_norm_2(v: List[float]) -> float:
    return sum(x * x for x in v) ** 0.5


def vector_norm_inf(v: List[float]) -> float:
    return max(abs(x) for x in v) if v else 0.0


def vector_subtract(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def scalar_multiply(c: float, x: List[float]) -> List[float]:
    return [c * xi for xi in x]


def normalize(v: List[float]) -> List[float]:
    norm = vector_norm_2(v)
    if norm < EPS:
        raise ValueError("Нельзя нормировать почти нулевой вектор.")
    return [x / norm for x in v]


def rayleigh_quotient(a: List[List[float]], x: List[float]) -> float:
    ax = matrix_vector_multiply(a, x)
    denominator = dot(x, x)
    if abs(denominator) < EPS:
        raise ValueError("Нулевой знаменатель в отношении Рэлея.")
    return dot(ax, x) / denominator


def residual_vector(a: List[List[float]], x: List[float], lam: float) -> List[float]:
    ax = matrix_vector_multiply(a, x)
    lx = scalar_multiply(lam, x)
    return vector_subtract(ax, lx)


def angle_between_vectors(x: List[float], y: List[float]) -> float:
    nx = vector_norm_2(x)
    ny = vector_norm_2(y)
    if nx < EPS or ny < EPS:
        raise ValueError("Угол с нулевым вектором не определён.")
    value = dot(x, y) / (nx * ny)
    value = max(-1.0, min(1.0, value))
    return acos(value)


def gershgorin_circles(a: List[List[float]]) -> List[tuple[float, float]]:
    circles = []
    n = len(a)
    for i in range(n):
        center = a[i][i]
        radius = sum(abs(a[i][j]) for j in range(n) if j != i)
        circles.append((center, radius))
    return circles


def format_vector(v: List[float], digits: int = 6) -> str:
    return "[" + ", ".join(f"{x:.{digits}g}" for x in v) + "]"


def format_matrix(a: List[List[float]], digits: int = 6) -> str:
    return "\n".join(
        " ".join(f"{x:12.{digits}g}" for x in row)
        for row in a
    )
