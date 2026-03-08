from __future__ import annotations

from typing import List


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


def transpose(a: List[List[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*a)]


def matrix_multiply(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    n = len(a)
    m = len(b[0])
    p = len(b)

    result = zeros(n, m)
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(p):
                s += a[i][k] * b[k][j]
            result[i][j] = s
    return result


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
    c = zeros(n, m)
    for i in range(n):
        for j in range(m):
            c[i][j] = a[i][j] - b[i][j]
    return c


def vector_subtract(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def vector_norm_inf(v: List[float]) -> float:
    return max(abs(x) for x in v) if v else 0.0


def vector_norm_2(v: List[float]) -> float:
    return sum(x * x for x in v) ** 0.5


def matrix_norm_inf(a: List[List[float]]) -> float:
    return max(sum(abs(x) for x in row) for row in a)


def is_square(a: List[List[float]]) -> bool:
    if not a:
        return False
    n = len(a)
    return all(len(row) == n for row in a)


def back_substitution(r: List[List[float]], y: List[float]) -> List[float]:
    n = len(r)
    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += r[i][j] * x[j]

        if abs(r[i][i]) < EPS:
            raise ValueError("Обратный ход невозможен: диагональный элемент слишком мал.")

        x[i] = (y[i] - s) / r[i][i]

    return x


def format_matrix(a: List[List[float]], digits: int = 6) -> str:
    return "\n".join(
        " ".join(f"{x:12.{digits}g}" for x in row)
        for row in a
    )


def format_vector(v: List[float], digits: int = 6) -> str:
    return "[" + ", ".join(f"{x:.{digits}g}" for x in v) + "]"
