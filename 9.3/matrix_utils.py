from __future__ import annotations

from typing import List


EPS = 1e-12


def copy_matrix(a: List[List[float]]) -> List[List[float]]:
    return [row[:] for row in a]


def identity_matrix(n: int) -> List[List[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def zeros(n: int, m: int) -> List[List[float]]:
    return [[0.0 for _ in range(m)] for _ in range(n)]


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


def matrix_subtract(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    n = len(a)
    m = len(a[0])
    return [[a[i][j] - b[i][j] for j in range(m)] for i in range(n)]


def scalar_identity(n: int, mu: float) -> List[List[float]]:
    e = identity_matrix(n)
    for i in range(n):
        e[i][i] *= mu
    return e


def add_scalar_to_diagonal(a: List[List[float]], mu: float) -> List[List[float]]:
    result = copy_matrix(a)
    for i in range(len(a)):
        result[i][i] += mu
    return result


def subtract_scalar_from_diagonal(a: List[List[float]], mu: float) -> List[List[float]]:
    result = copy_matrix(a)
    for i in range(len(a)):
        result[i][i] -= mu
    return result


def diagonal_entries(a: List[List[float]]) -> List[float]:
    return [a[i][i] for i in range(len(a))]


def off_diagonal_frobenius_norm(a: List[List[float]]) -> float:
    s = 0.0
    n = len(a)
    for i in range(n):
        for j in range(n):
            if i != j:
                s += a[i][j] * a[i][j]
    return s ** 0.5


def symmetry_error_inf(a: List[List[float]]) -> float:
    n = len(a)
    best = 0.0
    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            row_sum += abs(a[i][j] - a[j][i])
        if row_sum > best:
            best = row_sum
    return best


def matrix_norm_inf(a: List[List[float]]) -> float:
    return max(sum(abs(x) for x in row) for row in a)


def is_symmetric(a: List[List[float]], eps: float = EPS) -> bool:
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(a[i][j] - a[j][i]) > eps:
                return False
    return True


def determinant_gauss(a: List[List[float]], eps: float = EPS) -> float:
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

        pivot = m[k][k]
        for i in range(k + 1, n):
            factor = m[i][k] / pivot
            m[i][k] = 0.0
            for j in range(k + 1, n):
                m[i][j] -= factor * m[k][j]

    det = sign
    for i in range(n):
        det *= m[i][i]
    return det


def leading_principal_minor(a: List[List[float]], k: int) -> List[List[float]]:
    return [row[:k] for row in a[:k]]


def is_positive_definite_by_minors(a: List[List[float]], eps: float = EPS) -> bool:
    n = len(a)
    for k in range(1, n + 1):
        if determinant_gauss(leading_principal_minor(a, k), eps) <= eps:
            return False
    return True


def sort_descending(values: List[float]) -> List[float]:
    return sorted(values, reverse=True)


def format_vector(v: List[float], digits: int = 6) -> str:
    return "[" + ", ".join(f"{x:.{digits}g}" for x in v) + "]"


def format_matrix(a: List[List[float]], digits: int = 6) -> str:
    return "\n".join(
        " ".join(f"{x:12.{digits}g}" for x in row)
        for row in a
    )
