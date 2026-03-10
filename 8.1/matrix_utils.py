from __future__ import annotations

from typing import List
import random


EPS = 1e-12


def copy_matrix(a: List[List[float]]) -> List[List[float]]:
    return [row[:] for row in a]


def copy_vector(v: List[float]) -> List[float]:
    return v[:]


def zeros(n: int, m: int) -> List[List[float]]:
    return [[0.0 for _ in range(m)] for _ in range(n)]


def identity_matrix(n: int) -> List[List[float]]:
    e = zeros(n, n)
    for i in range(n):
        e[i][i] = 1.0
    return e


def matrix_vector_multiply(a: List[List[float]], x: List[float]) -> List[float]:
    result = []
    for row in a:
        s = 0.0
        for aij, xj in zip(row, x):
            s += aij * xj
        result.append(s)
    return result


def matrix_multiply(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    n = len(a)
    m = len(b[0])
    p = len(b)
    c = zeros(n, m)

    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(p):
                s += a[i][k] * b[k][j]
            c[i][j] = s

    return c


def vector_add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def vector_subtract(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def vector_norm_inf(v: List[float]) -> float:
    return max(abs(x) for x in v) if v else 0.0


def matrix_norm_inf(a: List[List[float]]) -> float:
    return max(sum(abs(x) for x in row) for row in a)


def matrix_norm_1(a: List[List[float]]) -> float:
    n = len(a)
    m = len(a[0])
    return max(sum(abs(a[i][j]) for i in range(n)) for j in range(m))


def vector_norm_2(v: List[float]) -> float:
    return sum(x * x for x in v) ** 0.5


def matrix_subtract(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    n = len(a)
    m = len(a[0])
    c = zeros(n, m)

    for i in range(n):
        for j in range(m):
            c[i][j] = a[i][j] - b[i][j]

    return c


def diagonal_matrix_inverse(diag: List[float]) -> List[List[float]]:
    n = len(diag)
    d_inv = zeros(n, n)

    for i in range(n):
        if abs(diag[i]) < EPS:
            raise ValueError("На диагонали есть нулевой элемент, приведение к виду x = Bx + c невозможно.")
        d_inv[i][i] = 1.0 / diag[i]

    return d_inv


def extract_diagonal(a: List[List[float]]) -> List[float]:
    return [a[i][i] for i in range(len(a))]


def random_vector(n: int, left: float = -1.0, right: float = 1.0) -> List[float]:
    return [random.uniform(left, right) for _ in range(n)]


def format_vector(v: List[float], digits: int = 6) -> str:
    return "[" + ", ".join(f"{x:.{digits}g}" for x in v) + "]"


def format_matrix(a: List[List[float]], digits: int = 6) -> str:
    return "\n".join(
        " ".join(f"{x:12.{digits}g}" for x in row)
        for row in a
    )


def power_method_spectral_radius(
    a: List[List[float]],
    max_iter: int = 500,
    tol: float = 1e-12,
) -> float:
    n = len(a)
    x = [1.0] * n
    prev_lambda = 0.0

    for _ in range(max_iter):
        y = matrix_vector_multiply(a, x)
        norm_y = vector_norm_inf(y)

        if norm_y < EPS:
            return 0.0

        x = [yi / norm_y for yi in y]

        ax = matrix_vector_multiply(a, x)
        numerator = sum(xi * axi for xi, axi in zip(x, ax))
        denominator = sum(xi * xi for xi in x)
        current_lambda = numerator / denominator

        if abs(current_lambda - prev_lambda) < tol:
            return abs(current_lambda)

        prev_lambda = current_lambda

    return abs(prev_lambda)
