from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from matrix_utils import (
    EPS,
    matrix_norm_1,
    matrix_norm_inf,
    matrix_vector_multiply,
    power_method_spectral_radius,
    residual,
    vector_add,
    vector_norm_inf,
    vector_subtract,
    zeros,
)


@dataclass
class IterationStep:
    iteration: int
    x_prev: List[float]
    x_curr: List[float]
    diff_inf: float
    residual_inf: float
    aposterior_estimate: Optional[float]


@dataclass
class IterationResult:
    solution: List[float]
    steps: List[IterationStep]
    iterations: int
    operations: int
    stop_reason: str
    norm_b_inf: float
    norm_b1: float
    spectral_radius: float


def build_iteration_form(a: List[List[float]], b: List[float]) -> tuple[List[List[float]], List[float]]:
    n = len(a)
    b_matrix = zeros(n, n)
    c = [0.0] * n

    for i in range(n):
        if abs(a[i][i]) < EPS:
            raise ValueError("На диагонали матрицы A есть нулевой элемент.")
        c[i] = b[i] / a[i][i]

        for j in range(n):
            if i == j:
                b_matrix[i][j] = 0.0
            else:
                b_matrix[i][j] = -a[i][j] / a[i][i]

    return b_matrix, c


def one_simple_iteration(b_matrix: List[List[float]], c: List[float], x: List[float]) -> List[float]:
    return vector_add(matrix_vector_multiply(b_matrix, x), c)


def simple_iteration(
    a: List[List[float]],
    b: List[float],
    x0: List[float],
    eps: float,
    max_iter: int = 10_000,
) -> IterationResult:
    b_matrix, c = build_iteration_form(a, b)
    norm_b_inf = matrix_norm_inf(b_matrix)
    norm_b1 = matrix_norm_1(b_matrix)
    spectral_radius = power_method_spectral_radius(b_matrix)

    x_prev = x0[:]
    steps: List[IterationStep] = []
    operations = 0

    for k in range(1, max_iter + 1):
        x_curr = one_simple_iteration(b_matrix, c, x_prev)
        n = len(x_prev)
        operations += 2 * n * n

        diff = vector_subtract(x_curr, x_prev)
        diff_inf = vector_norm_inf(diff)

        res = residual(a, x_curr, b)
        res_inf = vector_norm_inf(res)

        aposterior = None
        if norm_b_inf < 1.0:
            aposterior = norm_b_inf / (1.0 - norm_b_inf) * diff_inf

        steps.append(
            IterationStep(
                iteration=k,
                x_prev=x_prev[:],
                x_curr=x_curr[:],
                diff_inf=diff_inf,
                residual_inf=res_inf,
                aposterior_estimate=aposterior,
            )
        )

        if aposterior is not None and aposterior < eps:
            return IterationResult(
                solution=x_curr,
                steps=steps,
                iterations=k,
                operations=operations,
                stop_reason="Апостериорная оценка стала меньше eps.",
                norm_b_inf=norm_b_inf,
                norm_b1=norm_b1,
                spectral_radius=spectral_radius,
            )

        x_prev = x_curr

    return IterationResult(
        solution=x_prev,
        steps=steps,
        iterations=max_iter,
        operations=operations,
        stop_reason="Достигнуто максимальное число итераций.",
        norm_b_inf=norm_b_inf,
        norm_b1=norm_b1,
        spectral_radius=spectral_radius,
    )
