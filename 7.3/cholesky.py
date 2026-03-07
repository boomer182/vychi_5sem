from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import List

from matrix_utils import (
    EPS,
    is_square,
    is_symmetric,
    matrix_multiply,
    matrix_norm_inf,
    matrix_subtract,
    transpose,
)


@dataclass
class CholeskyResult:
    l: List[List[float]]
    solution: List[float]
    decomposition_operations: int
    solve_operations: int
    total_operations: int
    theoretical_decomposition_ops: float
    theoretical_solve_ops: float
    theoretical_total_ops: float
    factorization_residual: float


def cholesky_decomposition(a: List[List[float]], eps: float = EPS) -> tuple[List[List[float]], int]:
    if not is_square(a):
        raise ValueError("Матрица должна быть квадратной.")
    if not is_symmetric(a, eps):
        raise ValueError("Метод Холецкого применим только к симметричной матрице.")

    n = len(a)
    l = [[0.0 for _ in range(n)] for _ in range(n)]
    ops = 0

    for k in range(n):
        diag_sum = 0.0
        for j in range(k):
            diag_sum += l[k][j] * l[k][j]
            ops += 2

        value = a[k][k] - diag_sum
        ops += 1

        if value <= eps:
            raise ValueError(
                "Матрица не является положительно определённой: "
                "на диагональном шаге получено неположительное значение."
            )

        l[k][k] = sqrt(value)

        for i in range(k + 1, n):
            off_sum = 0.0
            for j in range(k):
                off_sum += l[i][j] * l[k][j]
                ops += 2

            numerator = a[i][k] - off_sum
            ops += 1

            l[i][k] = numerator / l[k][k]
            ops += 1

    return l, ops


def solve_lower(l: List[List[float]], b: List[float]) -> tuple[List[float], int]:
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


def solve_upper_from_transposed(l: List[List[float]], y: List[float]) -> tuple[List[float], int]:
    n = len(l)
    x = [0.0] * n
    ops = 0

    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += l[j][i] * x[j]
            ops += 2

        x[i] = (y[i] - s) / l[i][i]
        ops += 2

    return x, ops


def check_factorization(a: List[List[float]], l: List[List[float]]) -> float:
    lt = transpose(l)
    reconstructed = matrix_multiply(l, lt)
    diff = matrix_subtract(a, reconstructed)
    return matrix_norm_inf(diff)


def cholesky_solve(a: List[List[float]], b: List[float]) -> CholeskyResult:
    l, dec_ops = cholesky_decomposition(a)
    y, ops1 = solve_lower(l, b)
    x, ops2 = solve_upper_from_transposed(l, y)

    solve_ops = ops1 + ops2
    n = len(a)

    return CholeskyResult(
        l=l,
        solution=x,
        decomposition_operations=dec_ops,
        solve_operations=solve_ops,
        total_operations=dec_ops + solve_ops,
        theoretical_decomposition_ops=(n ** 3) / 3.0,
        theoretical_solve_ops=2.0 * (n ** 2),
        theoretical_total_ops=(n ** 3) / 3.0 + 2.0 * (n ** 2),
        factorization_residual=check_factorization(a, l),
    )
