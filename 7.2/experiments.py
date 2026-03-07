from __future__ import annotations

from typing import List

from matrix_utils import matrix_vector_multiply, random_vector_uniform, vector_norm_inf, vector_subtract
from lu import lu_decomposition, solve_with_lu
from lup import lup_decomposition, solve_with_lup


def repeated_lu_experiment(a: List[List[float]], num_tests: int) -> List[dict]:
    n = len(a)
    l, u, dec_ops = lu_decomposition(a)

    results = []
    for _ in range(num_tests):
        x_exact = random_vector_uniform(n, -10.0, 10.0)
        b = matrix_vector_multiply(a, x_exact)
        x_num, solve_ops = solve_with_lu(l, u, b)

        err = vector_norm_inf(vector_subtract(x_num, x_exact))
        results.append(
            {
                "x_exact": x_exact,
                "b": b,
                "x_num": x_num,
                "error_inf": err,
                "decomposition_operations": dec_ops,
                "solve_operations": solve_ops,
                "total_if_reuse": dec_ops + solve_ops,
            }
        )

    return results


def repeated_lup_experiment(a: List[List[float]], num_tests: int) -> List[dict]:
    n = len(a)
    p, l, u, dec_ops = lup_decomposition(a)

    results = []
    for _ in range(num_tests):
        x_exact = random_vector_uniform(n, -10.0, 10.0)
        b = matrix_vector_multiply(a, x_exact)
        x_num, solve_ops = solve_with_lup(p, l, u, b)

        err = vector_norm_inf(vector_subtract(x_num, x_exact))
        results.append(
            {
                "x_exact": x_exact,
                "b": b,
                "x_num": x_num,
                "error_inf": err,
                "decomposition_operations": dec_ops,
                "solve_operations": solve_ops,
                "total_if_reuse": dec_ops + solve_ops,
            }
        )

    return results


def compare_operation_counts(n: int, num_rhs: int, decomposition_ops: int, solve_ops: int) -> dict:
    direct_gauss_each_time = num_rhs * ((2.0 / 3.0) * n**3 + 2.0 * n**2)
    decomposition_once_then_solves = decomposition_ops + num_rhs * solve_ops

    return {
        "gauss_each_time_estimate": direct_gauss_each_time,
        "reuse_decomposition_estimate": decomposition_once_then_solves,
        "gain": direct_gauss_each_time - decomposition_once_then_solves,
    }
