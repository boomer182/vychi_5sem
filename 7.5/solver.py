from __future__ import annotations

from dataclasses import dataclass
from typing import List

from matrix_utils import (
    back_substitution,
    identity_matrix,
    matrix_multiply,
    matrix_norm_inf,
    matrix_subtract,
    matrix_vector_multiply,
    transpose,
    vector_norm_2,
    vector_norm_inf,
    vector_subtract,
)


@dataclass
class SolveReport:
    x: List[float]
    residual: List[float]
    residual_inf: float
    residual_2: float
    solution_error: List[float] | None
    solution_error_inf: float | None
    solution_error_2: float | None
    decomposition_error_inf: float
    orthogonality_error_inf: float
    relative_residual_inf: float
    relative_solution_error_inf: float | None


def solve_via_qr(
    a: List[List[float]],
    b: List[float],
    q: List[List[float]],
    r: List[List[float]],
    x_exact: List[float] | None = None,
) -> SolveReport:
    qt = transpose(q)
    y = matrix_vector_multiply(qt, b)
    x = back_substitution(r, y)

    ax = matrix_vector_multiply(a, x)
    residual = vector_subtract(ax, b)

    qr = matrix_multiply(q, r)
    decomp_error = matrix_norm_inf(matrix_subtract(a, qr))

    qtq = matrix_multiply(qt, q)
    orth_error = matrix_norm_inf(matrix_subtract(qtq, identity_matrix(len(a))))

    residual_inf = vector_norm_inf(residual)
    residual_2 = vector_norm_2(residual)
    b_inf = max(vector_norm_inf(b), 1e-15)
    relative_residual_inf = residual_inf / b_inf

    if x_exact is not None:
        solution_error = vector_subtract(x, x_exact)
        solution_error_inf = vector_norm_inf(solution_error)
        solution_error_2 = vector_norm_2(solution_error)
        x_exact_inf = max(vector_norm_inf(x_exact), 1e-15)
        relative_solution_error_inf = solution_error_inf / x_exact_inf
    else:
        solution_error = None
        solution_error_inf = None
        solution_error_2 = None
        relative_solution_error_inf = None

    return SolveReport(
        x=x,
        residual=residual,
        residual_inf=residual_inf,
        residual_2=residual_2,
        solution_error=solution_error,
        solution_error_inf=solution_error_inf,
        solution_error_2=solution_error_2,
        decomposition_error_inf=decomp_error,
        orthogonality_error_inf=orth_error,
        relative_residual_inf=relative_residual_inf,
        relative_solution_error_inf=relative_solution_error_inf,
    )
