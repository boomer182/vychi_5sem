from __future__ import annotations

from typing import List
import random


EPS = 1e-12


def zeros(n: int, m: int) -> List[List[float]]:
    return [[0.0 for _ in range(m)] for _ in range(n)]


def identity_matrix(n: int) -> List[List[float]]:
    a = zeros(n, n)
    for i in range(n):
        a[i][i] = 1.0
    return a


def copy_matrix(a: List[List[float]]) -> List[List[float]]:
    return [row[:] for row in a]


def copy_vector(v: List[float]) -> List[float]:
    return v[:]


def is_square(a: List[List[float]]) -> bool:
    if not a:
        return False
    n = len(a)
    return all(len(row) == n for row in a)


def matrix_multiply(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    n = len(a)
    m = len(b[0])
    k = len(b)

    result = zeros(n, m)
    for i in range(n):
        for j in range(m):
            s = 0.0
            for t in range(k):
                s += a[i][t] * b[t][j]
            result[i][j] = s
    return result


def matrix_vector_multiply(a: List[List[float]], x: List[float]) -> List[float]:
    n = len(a)
    result = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(len(a[i])):
            s += a[i][j] * x[j]
        result[i] = s
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


def matrix_norm_inf(a: List[List[float]]) -> float:
    return max(sum(abs(x) for x in row) for row in a)


def vector_norm_inf(v: List[float]) -> float:
    return max(abs(x) for x in v) if v else 0.0


def lower_triangular_solve(l: List[List[float]], b: List[float]) -> tuple[List[float], int]:
    n = len(l)
    y = [0.0] * n
    ops = 0

    for i in range(n):
        s = 0.0
        for j in range(i):
            s += l[i][j] * y[j]
            ops += 2
        y[i] = (b[i] - s) / l[i][i]
        ops += 2

    return y, ops


def upper_triangular_solve(u: List[List[float]], y: List[float]) -> tuple[List[float], int]:
    n = len(u)
    x = [0.0] * n
    ops = 0

    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += u[i][j] * x[j]
            ops += 2
        x[i] = (y[i] - s) / u[i][i]
        ops += 2

    return x, ops


def permute_vector(p: List[List[float]], b: List[float]) -> List[float]:
    return matrix_vector_multiply(p, b)


def determinant_from_u(u: List[List[float]]) -> float:
    det = 1.0
    for i in range(len(u)):
        det *= u[i][i]
    return det


def permutation_sign_from_matrix(p: List[List[float]]) -> int:
    n = len(p)
    perm = [0] * n
    for i in range(n):
        for j in range(n):
            if abs(p[i][j] - 1.0) < EPS:
                perm[i] = j
                break

    visited = [False] * n
    cycles = 0
    for i in range(n):
        if not visited[i]:
            cycles += 1
            v = i
            while not visited[v]:
                visited[v] = True
                v = perm[v]

    return -1 if (n - cycles) % 2 else 1


def format_matrix(a: List[List[float]], digits: int = 6) -> str:
    return "\n".join(
        " ".join(f"{x:12.{digits}g}" for x in row)
        for row in a
    )


def format_vector(v: List[float], digits: int = 6) -> str:
    return "[" + ", ".join(f"{x:.{digits}g}" for x in v) + "]"


def random_vector_uniform(n: int, left: float = -10.0, right: float = 10.0) -> List[float]:
    return [random.uniform(left, right) for _ in range(n)]
