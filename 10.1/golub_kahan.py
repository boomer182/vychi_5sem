from __future__ import annotations

from dataclasses import dataclass
from math import copysign
from typing import List

from matrix_utils import EPS, copy_matrix, identity, matmul, zeros


@dataclass
class GolubKahanResult:
    u_left: List[List[float]]
    b: List[List[float]]
    v_right: List[List[float]]
    alpha: List[float]
    beta: List[float]
    steps_text: List[str]


def householder_vector(x: List[float]) -> List[float]:
    norm_x = sum(xi * xi for xi in x) ** 0.5
    if norm_x < EPS:
        return [0.0] * len(x)

    v = x[:]
    v[0] += copysign(norm_x, x[0] if abs(x[0]) > EPS else 1.0)
    norm_v = sum(vi * vi for vi in v) ** 0.5
    if norm_v < EPS:
        return [0.0] * len(x)

    return [vi / norm_v for vi in v]


def apply_householder_left(a: List[List[float]], v: List[float], start_row: int, start_col: int) -> None:
    rows = len(a)
    cols = len(a[0])

    for j in range(start_col, cols):
        s = 0.0
        for i in range(start_row, rows):
            s += v[i - start_row] * a[i][j]
        for i in range(start_row, rows):
            a[i][j] -= 2.0 * v[i - start_row] * s


def apply_householder_right(a: List[List[float]], v: List[float], start_row: int, start_col: int) -> None:
    rows = len(a)
    cols = len(a[0])

    for i in range(start_row, rows):
        s = 0.0
        for j in range(start_col, cols):
            s += a[i][j] * v[j - start_col]
        for j in range(start_col, cols):
            a[i][j] -= 2.0 * s * v[j - start_col]


def embed_householder(v: List[float], size: int, start: int) -> List[List[float]]:
    h = identity(size)
    for i in range(start, size):
        for j in range(start, size):
            h[i][j] -= 2.0 * v[i - start] * v[j - start]
    return h


def golub_kahan_bidiagonalize(a: List[List[float]]) -> GolubKahanResult:
    m = len(a)
    n = len(a[0])
    p = min(m, n)

    b = copy_matrix(a)
    u_acc = identity(m)
    v_acc = identity(n)

    alpha = [0.0] * p
    beta = [0.0] * max(p - 1, 0)
    steps_text: List[str] = []

    for k in range(p):
        x = [b[i][k] for i in range(k, m)]
        v_left = householder_vector(x)

        if any(abs(t) > EPS for t in v_left):
            apply_householder_left(b, v_left, k, k)
            h_left = embed_householder(v_left, m, k)
            u_acc = matmul(u_acc, h_left)

        alpha[k] = b[k][k]
        step_text = [f"Шаг {k + 1}: после левого отражения"]
        step_text.append(f"alpha_{k + 1} = {alpha[k]:.12g}")

        if k < n - 1 and k < p - 1:
            x_right = [b[k][j] for j in range(k + 1, n)]
            v_right = householder_vector(x_right)

            if any(abs(t) > EPS for t in v_right):
                apply_householder_right(b, v_right, k, k + 1)
                h_right = embed_householder(v_right, n, k + 1)
                v_acc = matmul(v_acc, h_right)

            beta[k] = b[k][k + 1]
            step_text.append(f"beta_{k + 2} = {beta[k]:.12g}")

        steps_text.append("\n".join(step_text))

    for i in range(m):
        for j in range(n):
            if not (i == j or j == i + 1):
                if abs(b[i][j]) < 1e-10:
                    b[i][j] = 0.0

    return GolubKahanResult(
        u_left=u_acc,
        b=b,
        v_right=v_acc,
        alpha=alpha,
        beta=beta,
        steps_text=steps_text,
    )
