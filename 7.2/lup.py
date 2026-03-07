from __future__ import annotations

from dataclasses import dataclass
from typing import List

from matrix_utils import (
    EPS,
    copy_matrix,
    determinant_from_u,
    identity_matrix,
    is_square,
    lower_triangular_solve,
    matrix_multiply,
    matrix_norm_inf,
    matrix_subtract,
    permute_vector,
    permutation_sign_from_matrix,
    upper_triangular_solve,
)


@dataclass
class LUPResult:
    p: List[List[float]]
    l: List[List[float]]
    u: List[List[float]]
    determinant: float
    decomposition_operations: int
    solve_operations: int
    total_operations: int
    residual_norm: float


def swap_rows(a: List[List[float]], i: int, j: int) -> None:
    a[i], a[j] = a[j], a[i]


def lup_decomposition(a: List[List[float]]) -> tuple[List[List[float]], List[List[float]], List[List[float]], int]:
    if not is_square(a):
        raise ValueError("Матрица должна быть квадратной.")

    n = len(a)
    u = copy_matrix(a)
    l = identity_matrix(n)
    p = identity_matrix(n)
    ops = 0

    for k in range(n):
        pivot_row = k
        pivot_value = abs(u[k][k])

        for i in range(k + 1, n):
            if abs(u[i][k]) > pivot_value:
                pivot_value = abs(u[i][k])
                pivot_row = i

        if pivot_value < EPS:
            raise ValueError("LUP-разложение невозможно: матрица вырождена или почти вырождена.")

        if pivot_row != k:
            swap_rows(u, k, pivot_row)
            swap_rows(p, k, pivot_row)

            for j in range(k):
                l[k][j], l[pivot_row][j] = l[pivot_row][j], l[k][j]

        for i in range(k + 1, n):
            mu = u[i][k] / u[k][k]
            l[i][k] = mu
            ops += 1

            u[i][k] = 0.0
            for j in range(k + 1, n):
                u[i][j] -= mu * u[k][j]
                ops += 2

    return p, l, u, ops


def solve_with_lup(p: List[List[float]], l: List[List[float]], u: List[List[float]], b: List[float]) -> tuple[List[float], int]:
    pb = permute_vector(p, b)
    y, ops1 = lower_triangular_solve(l, pb)
    x, ops2 = upper_triangular_solve(u, y)
    return x, ops1 + ops2


def check_lup(a: List[List[float]], p: List[List[float]], l: List[List[float]], u: List[List[float]]) -> float:
    pa = matrix_multiply(p, a)
    lu = matrix_multiply(l, u)
    diff = matrix_subtract(pa, lu)
    return matrix_norm_inf(diff)


def lup_pipeline(a: List[List[float]], b: List[float]) -> LUPResult:
    p, l, u, dec_ops = lup_decomposition(a)
    x, solve_ops = solve_with_lup(p, l, u, b)
    residual = check_lup(a, p, l, u)
    det = permutation_sign_from_matrix(p) * determinant_from_u(u)

    return LUPResult(
        p=p,
        l=l,
        u=u,
        determinant=det,
        decomposition_operations=dec_ops,
        solve_operations=solve_ops,
        total_operations=dec_ops + solve_ops,
        residual_norm=residual,
    )
