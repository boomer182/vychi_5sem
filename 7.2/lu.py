from __future__ import annotations

from dataclasses import dataclass
from typing import List

from matrix_utils import (
    EPS,
    copy_matrix,
    identity_matrix,
    is_square,
    matrix_multiply,
    matrix_norm_inf,
    matrix_subtract,
    lower_triangular_solve,
    upper_triangular_solve,
    determinant_from_u,
)


@dataclass
class LUResult:
    l: List[List[float]]
    u: List[List[float]]
    determinant: float
    decomposition_operations: int
    solve_operations: int
    total_operations: int
    residual_norm: float


def lu_decomposition(a: List[List[float]]) -> tuple[List[List[float]], List[List[float]], int]:
    if not is_square(a):
        raise ValueError("Матрица должна быть квадратной.")

    n = len(a)
    u = copy_matrix(a)
    l = identity_matrix(n)
    ops = 0

    for k in range(n):
        if abs(u[k][k]) < EPS:
            raise ValueError(
                "Обычное LU-разложение невозможно: встретился нулевой или слишком малый ведущий элемент."
            )

        for i in range(k + 1, n):
            mu = u[i][k] / u[k][k]
            l[i][k] = mu
            ops += 1

            u[i][k] = 0.0
            for j in range(k + 1, n):
                u[i][j] -= mu * u[k][j]
                ops += 2

    return l, u, ops


def solve_with_lu(l: List[List[float]], u: List[List[float]], b: List[float]) -> tuple[List[float], int]:
    y, ops1 = lower_triangular_solve(l, b)
    x, ops2 = upper_triangular_solve(u, y)
    return x, ops1 + ops2


def check_lu(a: List[List[float]], l: List[List[float]], u: List[List[float]]) -> float:
    lu = matrix_multiply(l, u)
    diff = matrix_subtract(a, lu)
    return matrix_norm_inf(diff)


def lu_pipeline(a: List[List[float]], b: List[float]) -> LUResult:
    l, u, dec_ops = lu_decomposition(a)
    x, solve_ops = solve_with_lu(l, u, b)
    residual = check_lu(a, l, u)
    det = determinant_from_u(u)

    return LUResult(
        l=l,
        u=u,
        determinant=det,
        decomposition_operations=dec_ops,
        solve_operations=solve_ops,
        total_operations=dec_ops + solve_ops,
        residual_norm=residual,
    )
