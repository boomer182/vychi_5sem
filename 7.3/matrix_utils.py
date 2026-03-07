from __future__ import annotations

from typing import List


EPS = 1e-12


def copy_matrix(a: List[List[float]]) -> List[List[float]]:
    return [row[:] for row in a]


def is_square(a: List[List[float]]) -> bool:
    if not a:
        return False
    n = len(a)
    return all(len(row) == n for row in a)


def transpose(a: List[List[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*a)]


def matrix_multiply(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    n = len(a)
    m = len(b[0])
    q = len(b)
    result = [[0.0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(q):
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
    return [
        [a[i][j] - b[i][j] for j in range(len(a[0]))]
        for i in range(len(a))
    ]


def vector_subtract(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def matrix_norm_inf(a: List[List[float]]) -> float:
    return max(sum(abs(x) for x in row) for row in a)


def vector_norm_inf(v: List[float]) -> float:
    return max(abs(x) for x in v) if v else 0.0


def is_symmetric(a: List[List[float]], eps: float = EPS) -> bool:
    if not is_square(a):
        return False

    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(a[i][j] - a[j][i]) > eps:
                return False
    return True


def leading_principal_minor(a: List[List[float]], k: int) -> List[List[float]]:
    return [row[:k] for row in a[:k]]


def determinant_gauss(a: List[List[float]], eps: float = EPS) -> float:
    if not is_square(a):
        raise ValueError("Матрица должна быть квадратной.")

    m = copy_matrix(a)
    n = len(m)
    sign = 1.0

    for k in range(n):
        pivot_row = k
        pivot_val = abs(m[k][k])

        for i in range(k + 1, n):
            if abs(m[i][k]) > pivot_val:
                pivot_val = abs(m[i][k])
                pivot_row = i

        if pivot_val < eps:
            return 0.0

        if pivot_row != k:
            m[k], m[pivot_row] = m[pivot_row], m[k]
            sign *= -1.0

        for i in range(k + 1, n):
            factor = m[i][k] / m[k][k]
            for j in range(k, n):
                m[i][j] -= factor * m[k][j]

    det = sign
    for i in range(n):
        det *= m[i][i]
    return det


def format_matrix(a: List[List[float]], digits: int = 6) -> str:
    return "\n".join(
        " ".join(f"{x:12.{digits}g}" for x in row)
        for row in a
    )


def format_vector(v: List[float], digits: int = 6) -> str:
    return "[" + ", ".join(f"{x:.{digits}g}" for x in v) + "]"
