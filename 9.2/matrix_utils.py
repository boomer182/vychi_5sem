from __future__ import annotations
from typing import List

EPS = 1e-12


def copy_matrix(a: List[List[float]]) -> List[List[float]]:
    return [row[:] for row in a]


def identity_matrix(n: int) -> List[List[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def matrix_vector_multiply(a: List[List[float]], x: List[float]) -> List[float]:
    result = []
    for row in a:
        s = 0.0
        for aij, xj in zip(row, x):
            s += aij * xj
        result.append(s)
    return result


def matrix_subtract(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    n = len(a)
    m = len(a[0])
    return [[a[i][j] - b[i][j] for j in range(m)] for i in range(n)]


def scalar_matrix_subtract(a: List[List[float]], lam: float) -> List[List[float]]:
    n = len(a)
    result = copy_matrix(a)
    for i in range(n):
        result[i][i] -= lam
    return result


def dot(x: List[float], y: List[float]) -> float:
    return sum(xi * yi for xi, yi in zip(x, y))


def vector_subtract(x: List[float], y: List[float]) -> List[float]:
    return [xi - yi for xi, yi in zip(x, y)]


def scalar_multiply(c: float, x: List[float]) -> List[float]:
    return [c * xi for xi in x]


def vector_norm_2(x: List[float]) -> float:
    return sum(xi * xi for xi in x) ** 0.5


def vector_norm_inf(x: List[float]) -> float:
    return max(abs(xi) for xi in x) if x else 0.0


def normalize(x: List[float]) -> List[float]:
    norm = vector_norm_2(x)
    if norm < EPS:
        raise ValueError("Невозможно нормировать почти нулевой вектор.")
    return [xi / norm for xi in x]


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


def format_vector(x: List[float], digits: int = 6) -> str:
    return "[" + ", ".join(f"{xi:.{digits}g}" for xi in x) + "]"


def format_matrix(a: List[List[float]], digits: int = 6) -> str:
    return "\n".join(
        " ".join(f"{a[i][j]:12.{digits}g}" for j in range(len(a[0])))
        for i in range(len(a))
    )
