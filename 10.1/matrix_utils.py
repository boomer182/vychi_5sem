from __future__ import annotations

from math import sqrt
from typing import List, Tuple


EPS = 1e-12


def zeros(n: int, m: int) -> List[List[float]]:
    return [[0.0 for _ in range(m)] for _ in range(n)]


def identity(n: int) -> List[List[float]]:
    a = zeros(n, n)
    for i in range(n):
        a[i][i] = 1.0
    return a


def copy_matrix(a: List[List[float]]) -> List[List[float]]:
    return [row[:] for row in a]


def transpose(a: List[List[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*a)]


def matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
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


def matvec(a: List[List[float]], x: List[float]) -> List[float]:
    y = []
    for row in a:
        s = 0.0
        for aij, xj in zip(row, x):
            s += aij * xj
        y.append(s)
    return y


def dot(x: List[float], y: List[float]) -> float:
    return sum(xi * yi for xi, yi in zip(x, y))


def norm2(x: List[float]) -> float:
    return sqrt(sum(xi * xi for xi in x))


def norm_fro(a: List[List[float]]) -> float:
    s = 0.0
    for row in a:
        for x in row:
            s += x * x
    return sqrt(s)


def norm_inf_matrix(a: List[List[float]]) -> float:
    return max(sum(abs(x) for x in row) for row in a) if a else 0.0


def vec_sub(x: List[float], y: List[float]) -> List[float]:
    return [xi - yi for xi, yi in zip(x, y)]


def mat_sub(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    n = len(a)
    m = len(a[0])
    c = zeros(n, m)
    for i in range(n):
        for j in range(m):
            c[i][j] = a[i][j] - b[i][j]
    return c


def scalar_mul_vec(alpha: float, x: List[float]) -> List[float]:
    return [alpha * xi for xi in x]


def scalar_mul_mat(alpha: float, a: List[List[float]]) -> List[List[float]]:
    return [[alpha * x for x in row] for row in a]


def normalize(x: List[float]) -> List[float]:
    nx = norm2(x)
    if nx < EPS:
        raise ValueError("Нельзя нормировать почти нулевой вектор.")
    return [xi / nx for xi in x]


def diag_rect(sigma: List[float], m: int, n: int) -> List[List[float]]:
    a = zeros(m, n)
    for i in range(min(len(sigma), m, n)):
        a[i][i] = sigma[i]
    return a


def thin_diag(sigma: List[float]) -> List[List[float]]:
    p = len(sigma)
    a = zeros(p, p)
    for i in range(p):
        a[i][i] = sigma[i]
    return a


def offdiag_fro_tridiagonal_like(a: List[List[float]]) -> float:
    n = len(a)
    s = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                s += a[i][j] * a[i][j]
    return sqrt(s)


def matrix_from_columns(cols: List[List[float]]) -> List[List[float]]:
    if not cols:
        return []
    m = len(cols[0])
    n = len(cols)
    a = zeros(m, n)
    for j in range(n):
        for i in range(m):
            a[i][j] = cols[j][i]
    return a


def columns_of_matrix(a: List[List[float]]) -> List[List[float]]:
    return transpose(a)


def sort_svd_desc(
    sigma: List[float],
    u: List[List[float]],
    v: List[List[float]],
) -> Tuple[List[float], List[List[float]], List[List[float]]]:
    idx = list(range(len(sigma)))
    idx.sort(key=lambda i: sigma[i], reverse=True)

    sigma_sorted = [sigma[i] for i in idx]
    u_cols = columns_of_matrix(u)
    v_cols = columns_of_matrix(v)

    u_sorted = matrix_from_columns([u_cols[i] for i in idx])
    v_sorted = matrix_from_columns([v_cols[i] for i in idx])

    return sigma_sorted, u_sorted, v_sorted


def gram_schmidt_columns(a: List[List[float]]) -> List[List[float]]:
    cols = columns_of_matrix(a)
    ortho: List[List[float]] = []

    for col in cols:
        v = col[:]
        for q in ortho:
            proj = dot(v, q)
            for i in range(len(v)):
                v[i] -= proj * q[i]
        nv = norm2(v)
        if nv > EPS:
            ortho.append([vi / nv for vi in v])

    return matrix_from_columns(ortho) if ortho else zeros(len(a), 0)


def format_vector(x: List[float], digits: int = 6) -> str:
    return "[" + ", ".join(f"{xi:.{digits}g}" for xi in x) + "]"


def format_matrix(a: List[List[float]], digits: int = 6) -> str:
    if not a:
        return "[]"
    return "\n".join(
        " ".join(f"{a[i][j]:12.{digits}g}" for j in range(len(a[0])))
        for i in range(len(a))
    )
