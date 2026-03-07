from __future__ import annotations

from typing import List

from analysis_utils import matrix_vector_product, relative_error_inf
from gauss_core import solve_gauss


def build_hilbert_matrix(n: int) -> List[List[float]]:
    return [
        [1.0 / (i + j + 1) for j in range(n)]
        for i in range(n)
    ]


def ones_vector(n: int) -> List[float]:
    return [1.0] * n


def run_hilbert_experiment(
    n_values: List[int],
    pivot_strategy: str = "full"
) -> List[dict]:
    results = []

    for n in n_values:
        a = build_hilbert_matrix(n)
        x_exact = ones_vector(n)
        b = matrix_vector_product(a, x_exact)

        try:
            result = solve_gauss(a, b, pivot_strategy=pivot_strategy)
            error = relative_error_inf(x_exact, result.solution)

            results.append(
                {
                    "n": n,
                    "success": True,
                    "determinant": result.determinant,
                    "operations_total": result.operations_total,
                    "theoretical_operations": result.theoretical_operations,
                    "error_inf": error,
                    "solution": result.solution,
                }
            )
        except ValueError as exc:
            results.append(
                {
                    "n": n,
                    "success": False,
                    "message": str(exc),
                }
            )

    return results
